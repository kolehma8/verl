from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10), device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world % args.tp or args.vocab_size % args.tp or any(n % args.tp for n in args.tokens):
        raise ValueError("world, vocabulary and tokens must be divisible by TP")
    group = None
    for first in range(0, world, args.tp):
        candidate = dist.new_group(list(range(first, first + args.tp)))
        if first <= rank < first + args.tp:
            group = candidate
    assert group is not None
    tp_rank = dist.get_rank(group)

    from liger_cute_kernels import nvshmem
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
    from verl.utils.kernel.linear_cross_entropy import configure_liger_tp_flsce, linear_cross_entropy

    frontend = importlib.import_module("liger_kernel.ops.fused_linear_scaled_cross_entropy")

    def fallback_forbidden(*unused_args, **unused_kwargs):
        raise RuntimeError("Native LCK required: fallback is forbidden")

    frontend._apply_tp_fallback = fallback_forbidden
    if not configure_liger_tp_flsce(
        max_tokens=8192, hidden_size=args.hidden_size, local_vocab_size=args.vocab_size // args.tp,
        process_group=group, device=torch.device("cuda", local_rank),
    ):
        raise RuntimeError("Native LCK runtime unavailable")
    if rank == 0:
        print("CONFIG " + json.dumps({
            **vars(args), "world": world, "dtype": "bfloat16", "workspace_tokens": 8192,
            "device": torch.cuda.get_device_name(), "temperature": 1.0, "entropy": True,
            "tp4_includes_sequence_parallel_gather_and_backward": True,
            "statistic": "maximum CUDA-event latency across all eight ranks per iteration",
        }), flush=True)

    for tokens in args.tokens:
        for block in range(args.blocks):
            seed = 42 + block // 2
            gen = torch.Generator(device="cuda").manual_seed(seed * 100 + rank)
            weight = (torch.randn(
                args.vocab_size // args.tp, args.hidden_size, generator=gen, device="cuda", dtype=torch.bfloat16,
            ) * 0.02).requires_grad_(True)
            gen.manual_seed(seed * 1000 + rank // args.tp)
            full_hidden = torch.randn(tokens, args.hidden_size, generator=gen, device="cuda", dtype=torch.bfloat16) * 0.02
            hidden = full_hidden.chunk(args.tp)[tp_rank].clone().requires_grad_(True)
            del full_hidden
            labels = (torch.arange(tokens, device="cuda", dtype=torch.int64) * 17 + seed) % args.vocab_size
            grads = (
                torch.linspace(-0.7, 0.9, tokens, device="cuda"),
                torch.linspace(0.4, -0.2, tokens, device="cuda"),
            )

            def forward(backend):
                gathered = (
                    gather_from_sequence_parallel_region(
                        hidden, tensor_parallel_output_grad=backend == "triton", group=group,
                    ) if args.tp > 1 else hidden
                )
                return linear_cross_entropy(
                    gathered, weight, labels, 1.0, "none", group, impl_backend=backend,
                )

            # Validate the complete output-head path, including SP gradient communication.
            if block == 0:
                reference = None
                errors = {}
                for backend in ("triton", "liger_tp"):
                    hidden.grad = weight.grad = None
                    output = forward(backend)
                    torch.autograd.backward(output, grads)
                    actual = (*[v.detach().clone() for v in output], hidden.grad.clone(), weight.grad.clone())
                    if reference is None:
                        reference = actual
                    else:
                        for name, lhs, rhs in zip(("log_probs", "entropy", "d_hidden", "d_weight"), actual, reference):
                            torch.testing.assert_close(lhs.float(), rhs.float(), atol=0.03, rtol=0.05)
                            errors[name] = float((lhs.float() - rhs.float()).abs().max())
                    del output
                all_errors = [None] * world
                dist.all_gather_object(all_errors, errors)
                if rank == 0:
                    print("PARITY " + json.dumps({"tokens": tokens, "ranks": all_errors}), flush=True)
                del reference, actual

            order = ("triton", "liger_tp") if block % 2 == 0 else ("liger_tp", "triton")
            for mode in ("inference_forward", "forward", "backward", "full"):
                for backend in order:
                    cuda_samples, wall_samples = [], []
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    for iteration in range(args.warmups + args.iterations):
                        hidden.grad = weight.grad = None
                        output = forward(backend) if mode == "backward" else None
                        torch.cuda.synchronize()
                        dist.barrier()
                        torch.cuda.synchronize()
                        wall_start = time.perf_counter()
                        start.record()
                        if mode == "inference_forward":
                            with torch.no_grad():
                                output = forward(backend)
                        elif mode == "forward":
                            output = forward(backend)
                        elif mode == "backward":
                            torch.autograd.backward(output, grads)
                        else:
                            output = forward(backend)
                            torch.autograd.backward(output, grads)
                        end.record()
                        end.synchronize()
                        elapsed_wall = (time.perf_counter() - wall_start) * 1000
                        elapsed_cuda = start.elapsed_time(end)
                        if iteration >= args.warmups:
                            cuda_samples.append(elapsed_cuda)
                            wall_samples.append(elapsed_wall)
                        del output
                    rank_samples = [None] * world
                    dist.all_gather_object(rank_samples, {"cuda_ms": cuda_samples, "wall_ms": wall_samples})
                    if rank == 0:
                        critical = [
                            max(item["cuda_ms"][i] for item in rank_samples) for i in range(args.iterations)
                        ]
                        print("TIMING " + json.dumps({
                            "tp": args.tp, "hidden_size": args.hidden_size, "vocab_size": args.vocab_size,
                            "tokens": tokens, "block": block, "seed": seed, "mode": mode, "backend": backend,
                            "median_max_rank_ms": statistics.median(critical),
                            "rank_samples": rank_samples,
                        }), flush=True)
            del weight, hidden
    dist.barrier()
    nvshmem.pool_clear_all()
    dist.barrier()
    nvshmem.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
