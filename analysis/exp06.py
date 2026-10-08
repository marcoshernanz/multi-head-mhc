"""The comparisons exp06 was run for, pooled with exp04 (same code, same settings; pairs may span the two sessions).

Usage: cd analysis && uv run python exp06.py ../results/exp04_main ../results/exp06_controls [../results/exp07_scale]
Prints: H2 (global heads vs mhc-h1 with a wider MLP of the same parameter count), the d768 scale check, H7 (4x LR), and the
cross-session repeat of mhc-h1 seed 0. Writes ../figures/exp06_scale_curves.png (d768, paired over training).
"""

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_runs, mean_std  # noqa: E402
from sweep import paired  # noqa: E402

runs = load_runs([Path(d) for d in sys.argv[1:]])


def show(a: str, b: str) -> None:
    if a not in runs or b not in runs:
        print(f"| {a} − {b} | missing | | | |")
        return
    p = paired(runs[a], runs[b])
    if p is None:
        print(f"| {a} − {b} | no shared seeds | | | |")
        return
    star = " *" if p["exists"] else ""
    diffs = ", ".join(f"{d:+.4f}" for d in p["diffs"])
    print(f"| {a} − {b} | {len(p['seeds'])} | {p['mean']:+.4f} ± {p['se']:.4f}{star} | {p['better']}/{len(p['seeds'])} | {diffs} |")


def header(title: str) -> None:
    print(f"\n{title}\n\n| comparison | seeds | Δ (SE) | a better | per seed |\n|---|---|---|---|---|")


header("H2: heads vs parameters (ctrl-mlpW = mhc-h1 with SwiGLU width W, same parameter count as the global heads)")
for a, b in [("a-global-h4", "mhc-h1"), ("ctrl-mlp1216", "mhc-h1"), ("a-global-h4", "ctrl-mlp1216"),
             ("a-global-h16", "mhc-h1"), ("ctrl-mlp1992", "mhc-h1"), ("a-global-h16", "ctrl-mlp1992"),
             ("ctrl-mlp1992", "ctrl-mlp1216")]:
    show(a, b)

header("Scale check: d768, 12 layers (112M parameters), LR 1.5e-3")
for a, b in [("d768-a-global-h4", "d768-mhc-h1"), ("d768-ctrl-mlp2240", "d768-mhc-h1"), ("d768-a-global-h4", "d768-ctrl-mlp2240")]:
    show(a, b)

header("H7: 4x the learning rate (1.2e-2)")
for a, b in [("mhc-h1-lr4x", "mhc-h1"), ("a-global-h4-lr4x", "a-global-h4"), ("a-global-h4-lr4x", "mhc-h1-lr4x"),
             ("mhc-h1-lr2x", "mhc-h1"), ("a-global-h4-lr2x", "a-global-h4"), ("a-global-h4-lr2x", "mhc-h1-lr2x")]:
    show(a, b)
print()
for variant in ["mhc-h1-lr4x", "a-global-h4-lr4x", "mhc-h1-lr2x", "a-global-h4-lr2x", "mhc-h1", "a-global-h4"]:
    for seed, run in sorted(runs.get(variant, {}).items()):
        if seed > 1:
            continue
        records = [json.loads(line) for line in (run["dir"] / "metrics.jsonl").read_text().splitlines()]
        grads = [r["grad_norm"] for r in records if "grad_norm" in r and r["step"] >= 200]
        print(f"{variant}-s{seed}: final {run['final_val_loss']:.4f}, diverged {run['diverged']}, "
              f"max grad norm after warmup {max(grads):.2f}, median {sorted(grads)[len(grads) // 2]:.2f}")

print("\nCross-session repeat (exp04 mhc-h1-s0 vs exp06 repeat-mhc-h1-s0)")
if "repeat-mhc-h1" in runs:
    first = dict(runs["mhc-h1"][0]["val_curve"])
    again = dict(runs["repeat-mhc-h1"][0]["val_curve"])
    for step in sorted(set(first) & set(again)):
        print(f"step {step}: {first[step]:.6f} vs {again[step]:.6f} (diff {again[step] - first[step]:+.6f})")

print("\nThroughput (tok/s over the full run)")
for variant in ["mhc-h1", "a-global-h4", "a-global-h16", "ctrl-mlp1216", "ctrl-mlp1992", "d768-mhc-h1", "d768-a-global-h4",
                "d768-ctrl-mlp2240"]:
    values = [r["tok_s"] for r in runs.get(variant, {}).values() if r.get("tok_s")]
    if values:
        m, s = mean_std(values)
        seconds = [r["seconds"] for r in runs[variant].values()]
        print(f"{variant}: {m:,.0f} tok/s ({len(values)} runs), {min(seconds):.0f}-{max(seconds):.0f} s per run, "
              f"{next(iter(runs[variant].values()))['params']:,} parameters")

if "d768-mhc-h1" in runs and "d768-a-global-h4" in runs:
    base = runs["d768-mhc-h1"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, color in [("d768-a-global-h4", "#2a78d6"), ("d768-ctrl-mlp2240", "#0b0b0b")]:
        for seed in sorted(set(base) & set(runs.get(variant, {}))):
            at = dict(base[seed]["val_curve"])
            curve = [(s, v - at[s]) for s, v in runs[variant][seed]["val_curve"] if s in at]
            ax.plot([s * 16 * 1024 / 1e6 for s, _ in curve], [v for _, v in curve], marker="o", ms=4, lw=2, color=color,
                    ls="-" if seed == 0 else "--", label=f"{variant[5:]} − mhc-h1, seed {seed}")
    ax.axhline(0, color="#52514e", lw=1)
    ax.set_xlabel("tokens (M)")
    ax.set_ylabel("val loss − mHC h1 (d768, paired by seed)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("../figures/exp07_scale_curves.png", dpi=130)
    print("wrote ../figures/exp07_scale_curves.png")
