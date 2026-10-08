"""Plots for one or more result batches.

Usage: uv run python analysis/plots.py results/<batch> [...] --out figures/<name> [--baseline <variant>] [--only <regex>]
Writes <out>_curves.png (validation loss against tokens; with a baseline, the paired difference to it, averaged over the seeds
both have)
and <out>_train.png (smoothed training loss, seed 0 of every variant).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_runs  # noqa: E402


def mean_curve(by_seed: dict) -> list[tuple[int, float]]:
    steps = None
    for run in by_seed.values():
        s = [step for step, _ in run["val_curve"]]
        steps = s if steps is None else [x for x in steps if x in s]
    out = []
    for step in steps:
        values = [dict(run["val_curve"])[step] for run in by_seed.values()]
        out.append((step, sum(values) / len(values)))
    return out


def paired_curve(run: dict, base: dict) -> list[tuple[int, float]]:
    base_at = dict(base["val_curve"])
    return [(step, value - base_at[step]) for step, value in run["val_curve"] if step in base_at]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()
    runs = load_runs(args.dirs)
    runs = {k: v for k, v in runs.items() if not k.startswith("bench-")}  # throughput benchmarks, not training runs
    if args.only:
        runs = {k: v for k, v in runs.items() if re.search(args.only, k) or k == args.baseline}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    base = runs.get(args.baseline)
    for variant, by_seed in sorted(runs.items()):
        any_run = next(iter(by_seed.values()))
        tokens_per_step = int(any_run["args"]["batch"]) * int(any_run["args"]["seq_len"])
        if base is None:
            curve = mean_curve(by_seed)
            label = f"{variant} ({len(by_seed)})"
        else:
            shared = sorted(set(by_seed) & set(base))  # paired: only seeds the baseline also has
            if not shared:
                continue
            diffs = {seed: paired_curve(by_seed[seed], base[seed]) for seed in shared}
            curve = mean_curve({seed: {"val_curve": d} for seed, d in diffs.items()})
            label = f"{variant} (seeds {shared})"
        ax.plot([s * tokens_per_step / 1e6 for s, _ in curve], [v for _, v in curve], marker=".", label=label)
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel(f"val loss − {args.baseline} (paired by seed)" if base else "val loss")
    if base:
        ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}_curves.png", dpi=130)

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, by_seed in sorted(runs.items()):
        path = by_seed[min(by_seed)]["dir"]
        records = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        train = [(r["step"], r["loss"]) for r in records if "loss" in r]
        smooth, out = None, []
        for step, loss in train:
            smooth = loss if smooth is None else 0.9 * smooth + 0.1 * loss
            out.append((step, smooth))
        ax.plot([s for s, _ in out], [v for _, v in out], label=variant, lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("train loss (EMA of logged steps)")
    ax.set_ylim(top=min(ax.get_ylim()[1], 6))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}_train.png", dpi=130)
    print(f"wrote {args.out}_curves.png and {args.out}_train.png")


if __name__ == "__main__":
    main()
