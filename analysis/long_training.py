"""exp13's read-out: do the d384 comparisons change with 4x the tokens (393M instead of 98M, TPU v5e, LR 3e-3, seeds 0-2)?

Usage: cd analysis && uv run python long_training.py ../results/exp13_d384_long ../results/exp04_main ../results/exp06_controls \
       ../results/exp08_local
The read-out (notes/plan.md): global h4 − control and local h4 − mHC h1 within exp13, an effect if |mean| > 2 SE and all 3 seeds
agree. Also, on the same seeds and chip, each comparison at 98M tokens (exp04/exp06/exp08 runs, same LR) and the per-seed change
from 98M to 393M tokens (the runs' schedules differ, so only final losses are compared).
"""

import sys
from pathlib import Path

from aggregate import load_runs
from sweep import paired

COMPARISONS = [("a-global-h4", "ctrl-mlp1216"), ("a-local-h4", "mhc-h1"), ("a-global-h4", "mhc-h1"), ("ctrl-mlp1216", "mhc-h1"),
               ("mhc-h1", "residual")]
NAMES = {"a-global-h4": "global h4", "ctrl-mlp1216": "control (MLP 1216)", "a-local-h4": "local h4", "mhc-h1": "mHC h1",
         "residual": "residual"}


def fmt(p: dict | None) -> str:
    if p is None:
        return "–"
    se = f" ± {p['se']:.4f}" if len(p["diffs"]) > 1 else ""
    return f"{p['mean']:+.4f}{se} ({p['better']}/{len(p['diffs'])}){' *' if p['exists'] else ''}"


all_runs = load_runs([Path(d) for d in sys.argv[1:]])
long = {k.removeprefix("v5e-long-"): v for k, v in all_runs.items() if k.startswith("v5e-long-")}
seeds = sorted(set.intersection(*(set(v) for v in long.values())))
long = {k: {s: v[s] for s in seeds} for k, v in long.items()}
short = {k: {s: all_runs[k][s] for s in seeds if s in all_runs[k]} for k in long if k in all_runs}

print(f"seeds {seeds}\n\n| variant | 98M tokens | 393M tokens |\n|---|---|---|")
mean = lambda by_seed: sum(r["final_val_loss"] for r in by_seed.values()) / len(by_seed)  # noqa: E731
for k in NAMES:
    print(f"| {NAMES[k]} | {mean(short[k]):.4f} ({len(short[k])}) | {mean(long[k]):.4f} ({len(long[k])}) |")

print("\n| comparison | 98M tokens | 393M tokens | change (393M − 98M, per seed) |\n|---|---|---|---|")
for a, b in COMPARISONS:
    at_short, at_long = paired(short[a], short[b]), paired(long[a], long[b])
    both = sorted(set(short[a]) & set(short[b]) & set(long[a]))
    change = paired({s: {"final_val_loss": long[a][s]["final_val_loss"] - long[b][s]["final_val_loss"]} for s in both},
                    {s: {"final_val_loss": short[a][s]["final_val_loss"] - short[b][s]["final_val_loss"]} for s in both})
    print(f"| {NAMES[a]} − {NAMES[b]} | {fmt(at_short)} | {fmt(at_long)} | {fmt(change)} |")
print("\n(change: negative = the first variant gains relative to the second with longer training)")
