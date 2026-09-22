# Verl / Liger TP output-head results

Final September 20, 2026 results for
[verl-project/verl#7945](https://github.com/verl-project/verl/pull/7945).
This results-only branch is separate from the implementation branch.

**September 22 update:** [longer-response results](long-response/README.md)
supersede the short-response training results for B300 Qwen3-4B TP1 and
H200 Qwen3-14B TP4. H200 Qwen3-1.7B is still in progress. The original
artifacts below are preserved unchanged as historical results.

## Training campaign

Each of four GPU/model/topology cases has four seeds (42-45), two backends,
and 100 completed steps: 32 runs and 3,200 steps. Timing summaries exclude
steps 1-10. Rollouts were generated independently, without caching or replay.

The cases are H200 Qwen3-1.7B TP1 and Qwen3-14B TP4, plus B300 Qwen3-4B TP1
and Qwen3-32B TP4. Each uses eight GPUs, BF16, PP=CP=1, four rollouts per
prompt, prompt cap 256, response cap 64, and an 8192-token native workspace.

TP1 uses train/PPO minibatch 128 and microbatch 32 generated sequences/GPU;
TP4 uses 24 and 24. Native-only dispatch and first actor-backward token
counts were recorded for each rank of every Liger run.

`*-rewards.png` shows every seed/backend pair. `*-summary.json` contains
paired-seed statistics and trajectory-comparison results. `*-audit.json`
contains completion, native-rank, first-backward-shape, reward, gradient,
and source-log-hash records. `step-metrics.csv` contains all 3,200 steps.

`b300-*-reward-mean-ci.png` shows one mean curve per backend with pointwise
95% Student-t confidence intervals across the four seeds (df=3), without
smoothing or clipping. These are pointwise intervals for the seed mean,
not simultaneous confidence bands or intervals for a paired backend
difference. Lower endpoints can fall below zero; the underlying rewards
remain nonnegative. `b300-reward-mean-ci.csv` records every plotted mean and
interval endpoint. `plot_b300_reward_confidence.py` regenerates the figures.

Reward learning is evident for Qwen3-32B, but rewards are nearly all zero
for the other cases under the short response cap. These runs do not
establish policy equivalence. No whole-step throughput confidence interval
excludes zero. Nominal intervals are not adjusted for multiple comparisons.

## Matched output-head benchmarks

`*-head-summary.csv` and `.json` contain forward, backward, combined, and
inference-only timings and effective TFLOP/s. `*-benchmark-records.txt`
contains configuration, parity differences, and raw per-rank timing samples.

Both GPU environments were idle before benchmarking. All eight GPUs ran
eight TP1 groups or two TP4 groups. The harness uses four seeds, eight
alternating-backend blocks, ten warmups and twenty timed iterations per
block/phase. Timings use the maximum CUDA-event latency across ranks per
iteration, then block medians averaged within seed and across four seeds.
The raw records also include wall-clock timings.

Per-GPU effective projection FLOPs are 2*M*H*(V/TP) for forward,
4*M*H*(V/TP) for backward, and 6*M*H*(V/TP) for combined execution.
These rates include wrapper/launch and TP/SP communication time, omit
additional recomputation/non-GEMM arithmetic from the numerator, and are
not profiler-isolated hardware GEMM utilization. M counts all packed tokens.

The TP1 grid is 4096, 4800, 5120 tokens; TP4 is 3328, 3648, 4096.
H200 hidden sizes are 2048 (TP1) and 5120 (TP4); B300 hidden sizes are 2560
and 5120. Global vocabulary is 151936 in every case. TP4 includes the real
sequence-parallel gather and each backend's required gradient communication.
Native fallback is forbidden.

Example, after installing the matching Verl, Liger/LCK and Megatron stack:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --standalone --nproc-per-node=8 \
  benchmark_h200_campaign_heads.py --tp 4 --hidden-size 5120 \
  --tokens 3328 3648 4096 > head-benchmark.log
python summarize_campaign_head_benchmarks.py head-benchmark.log \
  --output-prefix head-summary
```

Despite its historical filename, the harness is shared by H200 and B300.
Configure NVSHMEM transports for the machine. These single-node experiments
used `NVSHMEM_DISABLE_NCCL=1`, `NVSHMEM_REMOTE_TRANSPORT=none`, and
`NVSHMEM_SYMMETRIC_SIZE=3G`. Do not blindly reuse those transport settings
for multi-node experiments.

H200 uses the compatible CUDA 12 LCK 0.8.3 environment. B300 uses the
temporary CUDA 13 LCK 0.8.3 validation build; published CUDA-versioned
packaging is separate follow-up work.

Interpretation: the full output head is repeatably faster in the measured
shapes, but individual phases can regress. H200 TP4 backward at 3328 tokens
has lower Liger throughput; B300 TP1 backward is markedly slower at 4800
than 4096 tokens. Full-actor profiling would be needed to assign end-to-end
timing differences to particular operations.

GitHub Copilot assisted with the experiments and analysis.
