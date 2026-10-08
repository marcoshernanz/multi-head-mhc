"""Aggregate the runs of one or more batches: a table per variant (mean ± std over seeds) and paired differences.

Usage: uv run python analysis/aggregate.py results/<batch> [results/<batch2> ...] [--baseline <variant>]
A run named "<variant>-s<seed>" is grouped by <variant>.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def load_runs(dirs: list[Path]) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    for d in dirs:
        for summary_path in sorted(d.glob("**/runs/*/summary.json")):
            name = summary_path.parent.name
            match = re.match(r"(.*)-s(\d+)$", name)
            variant, seed = (match.group(1), int(match.group(2))) if match else (name, 0)
            summary = json.loads(summary_path.read_text())
            curve = []
            for line in (summary_path.parent / "metrics.jsonl").read_text().splitlines():
                record = json.loads(line)
                if "val_loss" in record:
                    curve.append((record["step"], record["val_loss"]))
            summary["val_curve"] = curve
            summary["dir"] = summary_path.parent
            runs[variant][seed] = summary
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    m = sum(values) / len(values)
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else float("nan")
    return m, s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--baseline", default=None)
    args = parser.parse_args()
    runs = load_runs(args.dirs)
    print(f"{'variant':34s} {'seeds':>5s} {'val loss':>16s} {'tok/s':>8s} {'params':>9s} {'hc params':>9s} {'mem GB':>6s}")
    for variant, by_seed in sorted(runs.items(), key=lambda kv: mean_std([r["final_val_loss"] for r in kv[1].values()])[0]):
        losses = [r["final_val_loss"] for r in by_seed.values()]
        m, s = mean_std(losses)
        any_run = next(iter(by_seed.values()))
        tok = mean_std([r["tok_s"] for r in by_seed.values() if r["tok_s"]])[0]
        print(f"{variant:34s} {len(losses):5d} {m:9.4f} ± {s:5.4f} {tok:8.0f} {any_run['params']:9d} {any_run['hc_params']:9d} {any_run.get('peak_mem_gb') or 0:6.2f}")
    if args.baseline and args.baseline in runs:
        base = runs[args.baseline]
        print(f"\npaired differences vs {args.baseline} (negative = better), mean ± std over shared seeds, and t = mean / (std / sqrt(k))")
        for variant, by_seed in sorted(runs.items()):
            if variant == args.baseline:
                continue
            shared = sorted(set(by_seed) & set(base))
            diffs = [by_seed[s]["final_val_loss"] - base[s]["final_val_loss"] for s in shared]
            if not diffs:
                continue
            m, s = mean_std(diffs)
            t = m / (s / math.sqrt(len(diffs))) if len(diffs) > 1 and s > 0 else float("nan")
            print(f"{variant:34s} k={len(diffs)} diff {m:+.4f} ± {s:.4f}  t={t:+.2f}  per seed {[round(d, 4) for d in diffs]}")


if __name__ == "__main__":
    main()
