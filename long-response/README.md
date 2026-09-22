# Longer-response results, September 22, 2026

This snapshot contains two complete four-seed comparisons: B300 Qwen3-4B
TP1 and H200 Qwen3-14B TP4. Each backend/seed arm completed exactly 100
steps (16 runs, 1,600 steps total). The H200 Qwen3-1.7B rerun is still
in progress and is not included in these aggregate files or plots.

## Changed workload

The response cap is 2048, not 64. The prompt cap is 256. Each GPU's fixed
PPO/log-prob microbatch contains two generated sequences, not the previous
32 (TP1) or 24 (TP4). Four rollouts per prompt, global train/PPO batches
128 (TP1) and 24 (TP4), native workspace capacity 8192, learning rate 1e-6,
entropy coefficient 0.001, KL coefficient 0.001, and the original GSM8K
prompts/strict reward scorer are unchanged. Full determinism is enabled.

Rollouts are uncached, seeds are 42-45, and backend order alternates.
Both cases use eight GPUs, BF16, PP=CP=1, and remove-padding.

First actor-backward calls contain 749-1085 packed tokens/rank for B300
4B and 716-1008 for H200 14B. These are first-call observations, not a
whole-run token-count distribution, and are not approximately 4K.
The previously published 3K-5K output-head benchmarks characterize the
earlier workload and should not be presented as exact shape matches for
these new runs.

## Outcomes

All 800 step rewards per case are nonzero. Averaging within each seed,
then across four seeds, first-10 to last-20 mean rewards are:

| Case | Verl Triton | Liger |
|---|---:|---:|
| B300 Qwen3-4B TP1 | 0.643 to 0.929 | 0.640 to 0.927 |
| H200 Qwen3-14B TP4 | 0.462 to 0.878 | 0.475 to 0.889 |

Actor-update token-throughput changes are +2.58% for B300 (95% interval
-6.42% to +11.57%) and -0.70% for H200 (-7.16% to +5.76%). Neither is
statistically clear. Neither whole-step throughput interval excludes zero.
These are training rewards, not held-out evaluations or proof of policy
equivalence. Reward increases can reflect answer-format compliance as
well as mathematical correctness.

## Files

- `*-summary.json`: timing means, four-seed paired statistics, and trajectory
  comparisons across independently generated rollouts.
- `*-audit.json`: completion/native-rank evidence, first actor-backward
  token counts, source-log hashes, rewards, clipping, and gradient summaries.
- `step-metrics.csv`: all 1,600 training steps.
- `*-reward-mean-ci.png`: one curve per backend, averaged over four seeds.
- `long-response-reward-mean-ci.csv`: plotted means and interval endpoints.

Shaded bands are pointwise 95% Student-t confidence intervals for the
four-seed mean (df=3), without smoothing or clipping. They are not
simultaneous bands or paired backend-difference intervals.

Regenerate plots with:

```bash
python plot_b300_reward_confidence.py . --cases b300-tp1 h200-tp4 \
  --response-cap 2048 --interval-prefix long-response
```

Native-only dispatch was enforced, and all eight ranks of all eight Liger
runs have both dispatch and actor-backward markers. Full logs/markers are
archived separately; rollout records remain in the experiment storage.
