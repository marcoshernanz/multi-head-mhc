"""exp14's read-outs: which blocks of the head split (pre / post / res) carry global h4's gain and its LR tolerance (TPU v6e, d384).

Usage: cd analysis && uv run python head_parts.py ../results/exp11_v6e_check ../results/exp12_d384_lr_grid ../results/exp14_head_parts \
       ../results/exp17_parts_best_lr
Read-outs fixed in experiments/exp14_head_parts/jobs.py:
(1) at 6e-3 (2x LR): each part variant's Δ to mHC h1, and the share of global h4's lead over mHC h1 it keeps (on the seeds both have);
(2) at 3e-3: each part variant against mHC h1, global h4, and the MLP line at its own parameter count (mHC h1 to ctrl-mlp1216,
    linear in parameters; excess > 0 = a worse use of the parameters than widening the MLP).
exp17 adds the same read-out as (2) at 2.1e-3, d384's best LR on v6e (exp12), with the same line test as the 02:15 rule.
"""

import sys
from pathlib import Path

from aggregate import load_runs
from sweep import paired

PARTS = ["pre", "post", "prepost", "res"]
NAMES = {"pre": "pre (read)", "post": "post (write)", "prepost": "pre,post (shared mix)", "res": "res (mix)"}


def fmt(p: dict | None) -> str:
    if p is None:
        return "–"
    se = f" ± {p['se']:.4f}" if len(p["diffs"]) > 1 else ""
    return f"{p['mean']:+.4f}{se} ({p['better']}/{len(p['diffs'])}){' *' if p['exists'] else ''}"


runs = {k.removeprefix("v6e-"): v for k, v in load_runs([Path(d) for d in sys.argv[1:]]).items() if k.startswith("v6e-")}
for lr, tag in [("2.1e-3", "-lr0.7x"), ("3e-3", ""), ("6e-3", "-lr2x")]:
    base, full = runs.get(f"mhc-h1{tag}", {}), runs.get(f"a-global-h4{tag}", {})
    print(f"\n## {lr}{dict(zip(['-lr0.7x', '', '-lr2x'], [' (best LR on v6e, exp17)', ' (exp14)', ' (2x LR, exp14)']))[tag]}\n")
    params = lambda v: next(iter(runs[v].values()))["params"] / 1e6  # noqa: E731
    line = None
    if tag != "-lr2x" and f"ctrl-mlp1216{tag}" in runs and base:
        ctrl = paired(runs[f"ctrl-mlp1216{tag}"], base)
        line = (params("mhc-h1"), params("ctrl-mlp1216"), ctrl["mean"])
        print(f"MLP line: mHC h1 {line[0]:.2f}M -> ctrl-mlp1216 {line[1]:.2f}M, {ctrl['mean']:+.4f} ({len(ctrl['diffs'])} seeds)\n")
    print("| variant | params | seeds | Δ vs mHC h1 | Δ vs global h4 | " + ("MLP line | excess |" if line else "share of global h4's lead kept |"))
    print("|---|---|---|---|---|" + ("---|---|" if line else "---|"))
    rules = []
    for part in PARTS:
        name = f"a-global-h4-{part}{tag}"
        if name not in runs:
            print(f"| {NAMES[part]} | | 0 | | | |")
            continue
        vs_base, vs_full = paired(runs[name], base), paired(runs[name], full)
        row = f"| {NAMES[part]} | {params(name):.2f}M | {len(runs[name])} | {fmt(vs_base)} | {fmt(vs_full)} |"
        if line:
            at = line[2] * (params(name) - line[0]) / (line[1] - line[0])
            row += f" {at:+.4f} | {vs_base['mean'] - at:+.4f} |" if vs_base else " | |"
            if vs_base and (part in ("pre", "prepost") or tag):  # notes/plan.md 2026-10-04 02:15 UTC (3e-3) and 05:20 UTC (2.1e-3)
                below = sum(d < at for d in vs_base["diffs"])
                meets = vs_base["mean"] - at < -2 * vs_base["se"] and below == len(vs_base["diffs"])
                rules.append(f"{NAMES[part]}: excess {vs_base['mean'] - at:+.4f} vs 2 SE {2 * vs_base['se']:.4f}, "
                             f"{below}/{len(vs_base['diffs'])} seeds below the line -> {'MEETS' if meets else 'does not meet'} the rule"
                             + ("" if len(vs_base["diffs"]) == 3 else " (not read yet: the rule needs seeds 0-2)"))
        else:
            lead = paired(full, base)
            shared = sorted(set(runs[name]) & set(base) & set(full))
            if lead and shared:
                part_gain = sum(runs[name][s]["final_val_loss"] - base[s]["final_val_loss"] for s in shared)
                full_gain = sum(full[s]["final_val_loss"] - base[s]["final_val_loss"] for s in shared)
                row += f" {part_gain / full_gain:.2f} (seeds {','.join(map(str, shared))}) |"
            else:
                row += " – |"
        print(row)
    if full and base:
        print(f"\nglobal h4 − mHC h1 at {lr}: {fmt(paired(full, base))}")
    if rules:
        what = "the 02:15 d768 follow-up rule" if not tag else "exp17's line test (the 02:15 rule's test, at 2.1e-3)"
        print(f"\n{what} (excess below zero by more than 2 SE of the Δ to mHC h1, every seed below the line):")
        print("\n".join(f"- {r}" for r in rules))
