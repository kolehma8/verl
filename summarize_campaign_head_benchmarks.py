from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for log in args.logs:
        lines = log.read_text().splitlines()
        configs = [json.loads(line.removeprefix("CONFIG ")) for line in lines if line.startswith("CONFIG ")]
        if len(configs) != 1:
            raise RuntimeError(f"missing benchmark configuration: {log}")
        config = configs[0]
        records = [json.loads(line.removeprefix("TIMING ")) for line in lines if line.startswith("TIMING ")]
        expected = len(config["tokens"]) * config["blocks"] * 4 * 2
        if len(records) != expected:
            raise RuntimeError(f"incomplete timings: {log}: {len(records)}/{expected}")
        for tokens in config["tokens"]:
            for mode, multiplier in (("inference_forward", 2), ("forward", 2), ("backward", 4), ("full", 6)):
                values = {}
                wall_values = {}
                for backend in ("triton", "liger_tp"):
                    matched = [r for r in records if (r["tokens"], r["mode"], r["backend"]) == (tokens, mode, backend)]
                    seeds = sorted({r["seed"] for r in matched})
                    values[backend] = [
                        statistics.fmean(r["median_max_rank_ms"] for r in matched if r["seed"] == seed)
                        for seed in seeds
                    ]
                    wall_values[backend] = statistics.fmean(
                        statistics.median(
                            max(rank["wall_ms"][i] for rank in r["rank_samples"])
                            for i in range(config["iterations"])
                        )
                        for r in matched
                    )
                triton, liger = (statistics.fmean(values[b]) for b in ("triton", "liger_tp"))
                gains = [(old / new - 1) * 100 for old, new in zip(values["triton"], values["liger_tp"], strict=True)]
                margin = float(t.ppf(0.975, len(gains) - 1)) * statistics.stdev(gains) / math.sqrt(len(gains))
                flops = multiplier * tokens * config["hidden_size"] * config["vocab_size"] / config["tp"]
                row = {
                    "device": config["device"],
                    "tp": config["tp"],
                    "tokens": tokens,
                    "mode": mode,
                    "triton_ms": triton,
                    "liger_ms": liger,
                    "triton_effective_tflops_per_gpu": flops / (triton * 1e9),
                    "liger_effective_tflops_per_gpu": flops / (liger * 1e9),
                    "mean_paired_gain_percent": statistics.fmean(gains),
                    "gain_ci95_low": statistics.fmean(gains) - margin,
                    "gain_ci95_high": statistics.fmean(gains) + margin,
                    "triton_seed_cv_percent": statistics.stdev(values["triton"]) / triton * 100,
                    "liger_seed_cv_percent": statistics.stdev(values["liger_tp"]) / liger * 100,
                    "triton_wall_ms": wall_values["triton"],
                    "liger_wall_ms": wall_values["liger_tp"],
                    "triton_seed_ms": values["triton"],
                    "liger_seed_ms": values["liger_tp"],
                }
                rows.append(row)
                print(
                    f"{row['device']} TP{row['tp']} M={tokens} {mode}: "
                    f"{triton:.3f} -> {liger:.3f} ms; "
                    f"{row['triton_effective_tflops_per_gpu']:.1f} -> "
                    f"{row['liger_effective_tflops_per_gpu']:.1f} TFLOP/s/GPU; "
                    f"gain {row['mean_paired_gain_percent']:+.1f}% "
                    f"[{row['gain_ci95_low']:+.1f}, {row['gain_ci95_high']:+.1f}]"
                )
    args.output_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
