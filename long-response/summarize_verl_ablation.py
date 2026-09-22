from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics

from pathlib import Path

from scipy import stats


STEP_RE = re.compile(r"(?:^|\s)step:(\d+) - (.*)")
SEEDS = (42, 43, 44, 45)
BACKENDS = ("triton", "liger_tp")


def _read_steps(path: Path) -> dict[int, dict[str, float]]:
    steps = {}
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match is None:
            continue
        metrics = {}
        for item in match.group(2).split(" - "):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
        steps[int(match.group(1))] = metrics
    if not steps:
        raise RuntimeError(f"no step metrics found in {path}")
    return steps


def _rollout_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    normalized = []
    for line in path.read_text().split("\n"):
        if not line:
            continue
        row = json.loads(line)
        trajectory = {key: row.get(key) for key in ("input", "output", "gts")}
        normalized.append(json.dumps(trajectory, sort_keys=True, separators=(",", ":")))
    normalized.sort()
    return hashlib.sha256("\n".join(normalized).encode()).hexdigest()


def _mean_after_burn_in(
    steps: dict[int, dict[str, float]], key: str, burn_in: int
) -> float:
    values = [metrics[key] for step, metrics in sorted(steps.items()) if step > burn_in]
    if not values:
        raise RuntimeError(f"no values for {key!r} after burn-in step {burn_in}")
    return statistics.fmean(values)


def _paired_stats(
    triton: list[float], liger: list[float], *, higher_is_better: bool
) -> dict[str, float]:
    if len(triton) != len(liger):
        raise ValueError("paired inputs must have equal length")
    if higher_is_better:
        gains = [
            (new / old - 1.0) * 100.0 for old, new in zip(triton, liger, strict=True)
        ]
    else:
        gains = [
            (old / new - 1.0) * 100.0 for old, new in zip(triton, liger, strict=True)
        ]

    mean_gain = statistics.fmean(gains)
    stdev = statistics.stdev(gains)
    sem = stdev / math.sqrt(len(gains))
    critical = stats.t.ppf(0.975, df=len(gains) - 1)
    t_stat, p_value = stats.ttest_rel(liger, triton)
    wilcoxon = stats.wilcoxon(liger, triton, alternative="two-sided")
    ci_low = mean_gain - critical * sem
    ci_high = mean_gain + critical * sem
    return {
        "mean_gain_percent": mean_gain,
        "median_gain_percent": statistics.median(gains),
        "gain_stdev_percent": stdev,
        "gain_ci95_low_percent": ci_low,
        "gain_ci95_high_percent": ci_high,
        "positive_gain_pairs": sum(gain > 0 for gain in gains),
        "paired_t_statistic": float(t_stat),
        "paired_t_p_value_two_sided": float(p_value),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p_value_two_sided": float(wilcoxon.pvalue),
        "paired_t_significant_at_0_05": bool(p_value < 0.05 and ci_low > 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_label")
    parser.add_argument("tp", type=int)
    parser.add_argument(
        "--result-root", type=Path, default=Path("/home/jobuser/verl-ablation-results")
    )
    parser.add_argument("--burn-in", type=int, default=1)
    parser.add_argument("--expected-steps", type=int)
    args = parser.parse_args()

    runs = {}
    for seed in SEEDS:
        runs[seed] = {}
        for backend in BACKENDS:
            name = f"{args.model_label}-tp{args.tp}-seed{seed}-{backend}"
            path = args.result_root / name / "train.log"
            runs[seed][backend] = _read_steps(path)
            if args.expected_steps is not None:
                expected = set(range(1, args.expected_steps + 1))
                actual = set(runs[seed][backend])
                if actual != expected:
                    raise RuntimeError(
                        f"{name} has {len(actual)} steps; "
                        f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                    )

    metrics = {
        "actor_update_time": ("timing_s/update_actor", False),
        "actor_update_ms_per_token": ("timing_per_token_ms/update_actor", False),
        "whole_step_throughput": ("perf/throughput", True),
        "whole_step_time": ("perf/time_per_step", False),
        "old_log_prob_time": ("timing_s/old_log_prob", False),
        "reference_log_prob_time": ("timing_s/ref", False),
    }
    summary = {
        "model_label": args.model_label,
        "tp": args.tp,
        "seeds": list(SEEDS),
        "burn_in_steps": args.burn_in,
        "expected_steps": args.expected_steps,
        "measured_steps_per_run": (
            None if args.expected_steps is None else args.expected_steps - args.burn_in
        ),
        "metrics": {},
        "trajectory_checks": {},
        "step1_numerics": {},
    }

    for label, (key, higher_is_better) in metrics.items():
        triton = [
            _mean_after_burn_in(runs[seed]["triton"], key, args.burn_in)
            for seed in SEEDS
        ]
        liger = [
            _mean_after_burn_in(runs[seed]["liger_tp"], key, args.burn_in)
            for seed in SEEDS
        ]
        summary["metrics"][label] = {
            "triton_by_seed": dict(zip(SEEDS, triton, strict=True)),
            "liger_by_seed": dict(zip(SEEDS, liger, strict=True)),
            "triton_mean": statistics.fmean(triton),
            "liger_mean": statistics.fmean(liger),
            **_paired_stats(triton, liger, higher_is_better=higher_is_better),
        }

    for seed in SEEDS:
        triton_steps = runs[seed]["triton"]
        liger_steps = runs[seed]["liger_tp"]
        common_steps = sorted(set(triton_steps) & set(liger_steps))
        token_mismatches = []
        rollout_mismatches = []
        missing_rollouts = []
        for step in common_steps:
            if triton_steps[step].get("perf/total_num_tokens") != liger_steps[step].get(
                "perf/total_num_tokens"
            ):
                token_mismatches.append(step)
            triton_digest = _rollout_digest(
                args.result_root
                / f"{args.model_label}-tp{args.tp}-seed{seed}-triton"
                / "rollouts"
                / f"{step}.jsonl"
            )
            liger_digest = _rollout_digest(
                args.result_root
                / f"{args.model_label}-tp{args.tp}-seed{seed}-liger_tp"
                / "rollouts"
                / f"{step}.jsonl"
            )
            if triton_digest is None or liger_digest is None:
                missing_rollouts.append(step)
            elif triton_digest != liger_digest:
                rollout_mismatches.append(step)
        summary["trajectory_checks"][seed] = {
            "steps_checked": len(common_steps),
            "token_count_mismatch_steps": token_mismatches,
            "rollout_mismatch_steps": rollout_mismatches,
            "missing_rollout_steps": missing_rollouts,
            "all_token_counts_equal": not token_mismatches,
            "all_rollout_records_equal": not rollout_mismatches
            and not missing_rollouts,
        }
        summary["step1_numerics"][seed] = {
            key: {
                "triton": triton_steps[1].get(key),
                "liger": liger_steps[1].get(key),
                "absolute_delta": (
                    None
                    if triton_steps[1].get(key) is None
                    or liger_steps[1].get(key) is None
                    else liger_steps[1][key] - triton_steps[1][key]
                ),
            }
            for key in ("actor/entropy", "actor/grad_norm", "actor/loss")
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
