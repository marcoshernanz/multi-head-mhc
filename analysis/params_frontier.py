"""Every d384 variant against what the same parameters buy in the MLP.

Usage: cd analysis && uv run python params_frontier.py ../results/exp04_main ../results/exp06_controls [...] [--out PATH] [--paper]
The "MLP line" joins mhc-h1 and mhc-h1 with wider SwiGLU layers (ctrl-mlp1216, ctrl-mlp1992). For each variant with at least
mhc-h1's parameters it prints the paired Δ to mhc-h1, the line's Δ at the same parameter count (linear in parameters), and the
excess (Δ − line: negative = a better use of the parameters than widening the MLP). Writes ../figures/params_frontier.png;
--paper: the paper's look (paper_style.py), no title, and only the variants that add parameters (the rest share mHC's count and
are in the paper's table), for a PDF --out.
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paper_style  # noqa: E402
from aggregate import load_runs  # noqa: E402
from sweep import paired  # noqa: E402

BASE = "mhc-h1"
LINE = ["mhc-h1", "ctrl-mlp1216", "ctrl-mlp1992"]
FAMILIES = {"global": "#2a78d6", "local": "#eb6834", "other": "#1baf7a"}  # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {  # variant -> (label, x offset in points, y offset in points, alignment)
    "a-global-h2": ("global h2", 7, 9, "left"), "a-global-h4": ("global h4", 8, -2, "left"),
    "a-global-h8": ("global h8", 8, 2, "left"), "a-global-h16": ("global h16", -8, 10, "right"),
    "a-local-h2": ("local h2", -8, -4, "right"), "a-local-h4": ("local h4, h16", 8, 2, "left"),
    "a-local-h8": ("local h8", 8, 0, "left"),
    "static-h1": ("static h1, h4, h8", -8, 2, "right"), "d-joint-h4-static": ("joint h4 static", -8, -3, "right"),
    "d-joint-h4-global": ("joint h4 global", 8, 2, "left"), "mhc-n8": ("mHC n = 8", 8, 2, "left"),
    "residual": ("residual", -8, 0, "right"), "mhc-h1": ("mHC h1", -8, 5, "right"),
    "ctrl-mlp1216": ("MLP 1216", -8, -12, "right"), "ctrl-mlp1992": ("MLP 1992", -8, -4, "right"),
}


def family(variant: str) -> str:
    return "global" if variant.startswith("a-global") else "local" if variant.startswith("a-local") else "other"


def line_at(params: float, line: list[tuple[float, float]]) -> float | None:
    if params < line[0][0]:
        return None
    for (x0, y0), (x1, y1) in zip(line, line[1:]):
        if params <= x1 or (x1, y1) == line[-1]:  # past the last point: extend the last segment (global h16 is 0.03M past it)
            return y0 + (y1 - y0) * (params - x0) / (x1 - x0)


parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs="+", type=Path)
parser.add_argument("--out", type=Path, default=Path("../figures/params_frontier.png"))
parser.add_argument("--paper", action="store_true")
cli = parser.parse_args()
runs = {k: v for k, v in load_runs(cli.dirs).items()
        if not k.startswith(("bench-", "d768-", "repeat-")) and "-lr" not in k}
points = {}
for variant, by_seed in runs.items():
    params = next(iter(by_seed.values()))["params"] / 1e6
    p = {"mean": 0.0, "se": 0.0, "seeds": sorted(by_seed)} if variant == BASE else paired(by_seed, runs[BASE])
    if p is not None:
        points[variant] = (params, p)
line = [(points[v][0], points[v][1]["mean"]) for v in LINE]

print("| variant | params | seeds | Δ vs mhc-h1 | MLP line at these params | excess |\n|---|---|---|---|---|---|")
for variant, (params, p) in sorted(points.items(), key=lambda kv: kv[1][0]):
    at = line_at(params, line)
    excess = f"{p['mean'] - at:+.4f}" if at is not None and variant not in LINE else "–"
    at_text = f"{at:+.4f}" if at is not None else "–"
    print(f"| {variant} | {params:.2f}M | {len(p['seeds'])} | {p['mean']:+.4f} ± {p['se']:.4f} | {at_text} | {excess} |")

if cli.paper:
    paper_style.use()
    SURFACE = paper_style.SURFACE
else:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
label_size, ms, bars = (paper_style.SMALL, 6, 2) if cli.paper else (8, 7, 1)  # the paper's figures show ±2 SE, the effect threshold
if cli.paper:  # the paper's names, and room for the narrower figure
    LABELS = {v: (text.replace("mHC h1", "mHC").replace("n = 8", "$n=8$"), dx, dy, ha) for v, (text, dx, dy, ha) in LABELS.items()}
    LABELS = {v: (re.sub(r"\bh(\d+)", r"$h=\1$", text), dx, dy, ha) for v, (text, dx, dy, ha) in LABELS.items()}
    LABELS = {"mhc-h1": ("mHC", 0, 9, "center"), "a-global-h2": ("global $h=2$", -8, 0, "right"),
              "a-global-h4": ("global $h=4$", 8, 6, "left"), "mhc-n8": ("mHC, $n=8$", -8, 0, "right"),  # left of its point: on the right it crossed the global h=8 bar
              "a-global-h8": ("global $h=8$", 8, 0, "left"), "d-joint-h4-global": ("joint mix, $h=4$", 8, 0, "left"),
              "a-global-h16": ("global $h=16$", 0, 0, "center")}
    points = {v: pt for v, pt in points.items() if pt[0] >= points[BASE][0] + 0.1 or v == BASE}
fig, ax = plt.subplots(figsize=(paper_style.TEXT_WIDTH, 3.0) if cli.paper else (8, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(color=GRID, lw=0.6 if cli.paper else 1)
ax.set_axisbelow(True)
ax.axhline(0, color=MUTED, lw=1)
ax.plot([x for x, _ in line], [y for _, y in line], color=INK, lw=1.4 if cli.paper else 2, zorder=2,
        label="mHC with a wider MLP" if cli.paper else "mHC h1 with a wider MLP")
for variant, (params, p) in points.items():
    if variant in LINE:
        color, label = INK, None
    else:
        color, label = FAMILIES[family(variant)], family(variant)
    ax.errorbar(params, p["mean"], yerr=bars * p["se"], fmt="o", ms=ms, color=color, ecolor=color,
                elinewidth=1.4 if cli.paper else 1, capsize=0,
                mec=SURFACE, mew=1.5 if cli.paper else 2, zorder=3, label=label)
    if variant in LABELS and not variant.startswith(("static-h4", "static-h8")):
        text, dx, dy, ha = LABELS[variant]
        at = (params, p["mean"])
        if cli.paper and variant == "a-global-h16":  # above its bar: the joint-mix label takes the space to its left
            at, dy = (params, p["mean"] + bars * p["se"]), 8
        ax.annotate(text, at, xytext=(dx, dy), textcoords="offset points", color=MUTED, fontsize=label_size, va="center", ha=ha)
handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))
order = ["mHC h1 with a wider MLP", "mHC with a wider MLP", "global", "local", "other"]
names = {"global": "global-predictor heads", "local": "local-predictor heads", "other": "other connections"}
names["other"] = "other variants" if cli.paper else names["other"]
if cli.paper:  # plain markers in the key, not error-bar glyphs
    from matplotlib.lines import Line2D
    marker = {"global": FAMILIES["global"], "local": FAMILIES["local"], "other": FAMILIES["other"]}
    unique = {k: (Line2D([], [], color=INK, lw=1.4, marker="o", ms=ms, mec=SURFACE, mew=1.5) if k not in marker else
                  Line2D([], [], color=marker[k], ls="none", marker="o", ms=ms, mec=SURFACE, mew=1.5)) for k in unique}
ax.legend([unique[k] for k in order if k in unique], [names.get(k, k) for k in order if k in unique], frameon=False,
          loc="lower left" if cli.paper else "upper right", fontsize=9 if cli.paper else 8)
ax.set_xlim(*((26.4, 37.0) if cli.paper else (24.9, 37.2)))  # room for the labels left of the smallest models
ax.set_xlabel("parameters (M)")
ax.set_ylabel("loss − mHC (paired by seed, ± 2 SE)" if cli.paper else "val loss − mHC h1 (paired by seed, ± SE)")
if not cli.paper:
    ax.set_title("What the extra parameters buy: connection variants vs a wider MLP (d384, 98M tokens)", color=INK, fontsize=10,
                 loc="left")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(cli.out, dpi=150, facecolor=SURFACE)
print(f"wrote {cli.out}")
