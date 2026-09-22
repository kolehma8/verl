from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    titles = {
        "b300-tp1": "B300 / Qwen3-4B / TP1",
        "b300-tp4": "B300 / Qwen3-32B / TP4",
        "h200-tp4": "H200 / Qwen3-14B / TP4",
        "h200-tp1": "H200 / Qwen3-1.7B / TP1",
    }
    parser.add_argument("--cases", nargs="+", choices=titles, default=["b300-tp1", "b300-tp4"])
    parser.add_argument("--response-cap", type=int, default=64)
    parser.add_argument("--interval-prefix", default="b300")
    args = parser.parse_args()
    with (args.root / "step-metrics.csv").open() as source:
        rows = list(csv.DictReader(source))
    intervals = []
    critical = float(t.ppf(0.975, 3))
    for case in args.cases:
        title = titles[case]
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for backend, label, color, style in (
            ("triton", "Verl Triton", "#0072B2", "-"),
            ("liger_tp", "Liger native", "#D55E00", "--"),
        ):
            means, lower, upper = [], [], []
            for step in range(1, 101):
                matched = [
                    r for r in rows
                    if r["case"] == case and r["backend"] == backend and int(r["step"]) == step
                ]
                if len(matched) != 4 or {int(r["seed"]) for r in matched} != {42, 43, 44, 45}:
                    raise RuntimeError(f"Missing seed data: {case}/{backend}/{step}")
                values = [float(r["reward"]) for r in matched]
                mean = statistics.fmean(values)
                margin = critical * statistics.stdev(values) / 2
                means.append(mean)
                lower.append(mean - margin)
                upper.append(mean + margin)
                intervals.append({
                    "case": case, "backend": backend, "step": step, "n_seeds": 4,
                    "mean_reward": mean, "ci95_lower": mean - margin, "ci95_upper": mean + margin,
                })
            ax.plot(range(1, 101), means, color=color, linestyle=style, label=label, linewidth=1.7)
            ax.fill_between(range(1, 101), lower, upper, color=color, alpha=0.18, linewidth=0)
        ax.set(
            title=f"{title}\nMean reward and pointwise 95% confidence interval across four seeds",
            xlabel="Training step", ylabel="Mean training reward",
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
        fig.text(
            0.5, 0.025,
            f"Student-t intervals (df=3); no smoothing or clipping. Uncached; response cap {args.response_cap} tokens.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(args.root / f"{case}-reward-mean-ci.png", dpi=180)
        plt.close(fig)
    with (args.root / f"{args.interval_prefix}-reward-mean-ci.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)


if __name__ == "__main__":
    main()
