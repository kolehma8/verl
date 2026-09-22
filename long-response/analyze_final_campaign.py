from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_verl_ablation import BACKENDS, SEEDS, _read_steps

CASES = {
    "h200-tp1": ("microbatch-h200-qwen3-1.7b-v8", 1, "H200 / Qwen3-1.7B / TP1"),
    "h200-tp4": ("microbatch-h200-qwen3-14b-v8", 4, "H200 / Qwen3-14B / TP4"),
    "b300-tp4": ("microbatch-b300-qwen3-32b-v9", 4, "B300 / Qwen3-32B / TP4"),
    "b300-tp1": ("microbatch-b300-qwen3-4b-v8", 1, "B300 / Qwen3-4B / TP1"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("cases", nargs="+", choices=CASES)
    parser.add_argument("--long-response", action="store_true")
    parser.add_argument("--response-cap", type=int, default=64)
    parser.add_argument("--min-actor-tokens", type=int, default=3000)
    parser.add_argument("--max-actor-tokens", type=int, default=6500)
    args = parser.parse_args()
    if args.long_response:
        for key, prefix in (
            ("h200-tp1", "long-response-h200-qwen3-1.7b-format-v11"),
            ("h200-tp4", "long-response-h200-qwen3-14b-v10"),
            ("b300-tp1", "long-response-b300-qwen3-4b-v10"),
        ):
            _, tp, title = CASES[key]
            CASES[key] = (prefix, tp, title)
    rows = []
    audit = {}
    for case in args.cases:
        label, tp, title = CASES[case]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
        run_audits = {}
        reward_values = []
        for seed, ax in zip(SEEDS, axes.flat, strict=True):
            for backend, color, linestyle in (
                ("triton", "#0072B2", "-"),
                ("liger_tp", "#D55E00", "--"),
            ):
                run = args.root / f"{label}-tp{tp}-seed{seed}-{backend}"
                log = run / "train.log"
                steps = _read_steps(log)
                if set(steps) != set(range(1, 101)) or "run_complete=1" not in log.read_text():
                    raise RuntimeError(f"incomplete run: {run.name}")
                native = [
                    json.loads(path.read_text())
                    for path in (run / "native-markers").glob("rank-*.json")
                ]
                backward = [
                    json.loads(path.read_text())
                    for path in (run / "native-markers").glob("actor-backward-rank-*.json")
                ]
                if backend == "liger_tp":
                    for markers in (native, backward):
                        if len(markers) != 8 or {m["global_rank"] for m in markers} != set(range(8)):
                            raise RuntimeError(f"missing per-rank native markers: {run.name}")
                    if not all(
                        m["fallback_forbidden"] and m["backend"] == "liger_cute_kernels"
                        for m in native
                    ):
                        raise RuntimeError(f"native execution not confirmed: {run.name}")
                    if not all(args.min_actor_tokens <= m["tokens"] <= args.max_actor_tokens for m in backward):
                        raise RuntimeError(f"unexpected first actor shape: {run.name}")
                rewards = [steps[i]["critic/rewards/mean"] for i in range(1, 101)]
                reward_values.extend(rewards)
                run_audits[run.name] = {
                    "steps": len(steps),
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    "native_ranks": sorted(m["global_rank"] for m in native),
                    "first_actor_backward_tokens": {
                        str(m["global_rank"]): m["tokens"] for m in backward
                    },
                    "reward_mean": statistics.fmean(rewards),
                    "reward_min": min(rewards),
                    "reward_max": max(rewards),
                    "nonzero_reward_steps": sum(r != 0 for r in rewards),
                    "reward_first10_mean": statistics.fmean(rewards[:10]),
                    "reward_last20_mean": statistics.fmean(rewards[-20:]),
                    "response_clip_last20_mean": statistics.fmean(
                        steps[i]["response_length/clip_ratio"] for i in range(81, 101)
                    ),
                    "grad_norm_mean": statistics.fmean(m["actor/grad_norm"] for m in steps.values()),
                }
                for step, metrics in sorted(steps.items()):
                    rows.append({
                        "case": case,
                        "seed": seed,
                        "backend": backend,
                        "step": step,
                        "reward": metrics["critic/rewards/mean"],
                        "entropy": metrics["actor/entropy"],
                        "grad_norm": metrics["actor/grad_norm"],
                        "actor_update_ms_per_token": metrics["timing_per_token_ms/update_actor"],
                        "actor_update_seconds": metrics["timing_s/update_actor"],
                        "whole_step_throughput": metrics["perf/throughput"],
                    })
                ax.plot(
                    range(1, 101), rewards, label="Verl Triton" if backend == "triton" else "Liger native",
                    color=color, linestyle=linestyle, linewidth=1.4, alpha=0.85,
                )
            ax.set_title(f"Seed {seed}")
            ax.grid(alpha=0.25)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Mean training reward")
        axes[0, 0].legend(loc="upper right")
        if all(r == 0 for r in reward_values):
            axes[0, 0].set_ylim(-0.005, 0.025)
            fig.text(
                0.5, 0.035, "All rewards are zero; the two curves overlap. This is not evidence of learning parity.",
                ha="center", fontsize=10,
            )
        prompt_cap = 384 if args.long_response and case == "h200-tp1" else 256
        fig.suptitle(
            f"{title}\nUncached GSM8K; 4 rollouts/prompt; prompt <={prompt_cap}, response <={args.response_cap} tokens"
        )
        fig.tight_layout(rect=(0, 0.065, 1, 0.93))
        fig.savefig(args.root / f"{case}-rewards.png", dpi=170)
        plt.close(fig)
        audit[case] = run_audits
        (args.root / f"{case}-audit.json").write_text(json.dumps(run_audits, indent=2) + "\n")
        summary = json.loads((args.root / f"{case}-summary.json").read_text())
        print(title)
        for metric in ("actor_update_ms_per_token", "whole_step_throughput"):
            result = summary["metrics"][metric]
            print(
                f"  {metric}: {result['triton_mean']:.6f} -> {result['liger_mean']:.6f}; "
                f"gain {result['mean_gain_percent']:+.2f}%, "
                f"95% CI [{result['gain_ci95_low_percent']:+.2f}, {result['gain_ci95_high_percent']:+.2f}], "
                f"p={result['paired_t_p_value_two_sided']:.4f}"
            )
        tokens = [t for run in run_audits.values() for t in run["first_actor_backward_tokens"].values()]
        print(f"  first actor tokens/rank: {min(tokens)}..{max(tokens)}")
        print(f"  nonzero reward steps: {sum(r != 0 for r in reward_values)}/800")
        print(
            "  rollout mismatch steps by seed:",
            {s: len(v["rollout_mismatch_steps"]) for s, v in summary["trajectory_checks"].items()},
        )
    with (args.root / "step-metrics.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
