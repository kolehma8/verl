# Longer-response results, September 23, 2026

This snapshot contains three complete four-seed comparisons: B300 Qwen3-4B
TP1, H200 Qwen3-14B TP4, and H200 Qwen3-1.7B TP1. Each backend/seed arm
completed exactly 100 steps (24 runs, 2,400 steps total). The final 1.7B
run finished September 22 at 19:56 PDT.

## Changed workload

The response cap is 2048, not 64. The prompt cap is 256 for 4B/14B and
384 for 1.7B. Each GPU's fixed
PPO/log-prob microbatch contains two generated sequences, not the previous
32 (TP1) or 24 (TP4). Four rollouts per prompt, global train/PPO batches
128 (TP1) and 24 (TP4), native workspace capacity 8192, learning rate 1e-6,
entropy coefficient 0.001, KL coefficient 0.001, and the strict GSM8K
reward scorer are unchanged. Full determinism is enabled.

The 4B/14B cases retain the original prompts. The 1.7B pilot was
answer-format limited, so its copied dataset adds an explicit final-line
`#### <number>` instruction rather than accepting a boxed answer. All
7,473 rows and non-prompt columns are preserved. Both 1.7B backends use
this same strengthened prompt; do not conflate it with the original
prompt protocol.

Rollouts are uncached, seeds are 42-45, and backend order alternates.
All three cases use eight GPUs, BF16, PP=CP=1, and remove-padding.

First actor-backward calls contain 749-1085 packed tokens/rank for B300
4B, 716-1008 for H200 14B, and 810-1174 for H200 1.7B. These are first-call observations, not a
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
| H200 Qwen3-1.7B TP1 | 0.857 to 0.896 | 0.852 to 0.896 |

Actor-update token-throughput changes are +2.58% for B300 4B (95% interval
-6.42% to +11.57%), -0.70% for H200 14B (-7.16% to +5.76%), and +2.32%
for H200 1.7B (-1.80% to +6.44%). None is statistically clear.
No whole-step throughput interval excludes zero; H200 1.7B is -0.10%
(-4.72% to +4.51%). Timing statistics use steps 11-100 and four paired
seed-level ratios, not individual steps as replicates.

H200 1.7B actor time per step is 5.230 to 5.089 seconds, a +2.78% inverse-time
gain (95% interval +1.71% to +3.85%). Unlike the token-normalized metric,
this raw-time comparison excludes zero, but includes different sequence
lengths across independently generated trajectories. It does not establish
a token-throughput or whole-step improvement. Late response truncation
averages approximately 5% for both 1.7B backends.
These are training rewards, not held-out evaluations or proof of policy
equivalence. Reward increases can reflect answer-format compliance as
well as mathematical correctness.

## Files

- `*-summary.json`: timing means, four-seed paired statistics, and trajectory
  comparisons across independently generated rollouts.
- `*-audit.json`: completion/native-rank evidence, first actor-backward
  token counts, source-log hashes, rewards, clipping, and gradient summaries.
- `step-metrics.csv`: all 2,400 training steps.
- `*-reward-mean-ci.png`: one curve per backend, averaged over four seeds.
- `long-response-reward-mean-ci.csv`: plotted means and interval endpoints.

Shaded bands are pointwise 95% Student-t confidence intervals for the
four-seed mean (df=3), without smoothing or clipping. They are not
simultaneous bands or paired backend-difference intervals.

Regenerate plots with:

```bash
python plot_b300_reward_confidence.py . --cases b300-tp1 h200-tp4 h200-tp1 \
  --response-cap 2048 --interval-prefix long-response
```

Native-only dispatch was enforced, and all eight ranks of all twelve Liger
runs have both dispatch and actor-backward markers. Full logs/markers are
archived separately; rollout records remain in the experiment storage.
