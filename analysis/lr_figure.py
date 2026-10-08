"""The paper's learning-rate figure: final val loss above each width's best, against the LR, one panel per width (one chip per call).

Usage: cd analysis && uv run python lr_figure.py ../results/exp11_v6e_check ../results/exp12_d384_lr_grid ../results/exp15_d768_v6e
       [--seeds 0,1,2] [--out ../figures/lr_v6e.png] [--paper]
The y axis is symlog (linear up to 0.05), so the near-optimum differences and the failures past the stability edge share one
panel. Means over --seeds only, at the LRs that have them all; dots are the single seeds. --paper: the paper's look (paper_style.py),
for a PDF --out.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import paper_style  # noqa: E402
from lr_sweep import CONTROL, GRID, INK, MUTED, STYLE, SURFACE, USABLE, load, mean_loss  # noqa: E402

SHOWN = ["residual", "mhc-h1", "control", "a-local-h4", "a-global-h4"]
LINEAR = 0.05

parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs="+", type=Path)
parser.add_argument("--seeds", default="0,1,2")
parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures" / "lr_v6e.png")
parser.add_argument("--paper", action="store_true")
cli = parser.parse_args()
if cli.paper:
    paper_style.use()
    SURFACE = paper_style.SURFACE
size, title_size, label_size, tick_size, lw, ms = (9, 9, 9, 8.5, 1.4, 22) if cli.paper else (8, 10, 9, 8, 1.8, 36)
seeds = {int(s) for s in cli.seeds.split(",")}
table = load(cli.dirs)
widths = sorted({width for width, v in table if v == "mhc-h1"})

figsize = (paper_style.TEXT_WIDTH, 3.0) if cli.paper else (4.0 * len(widths), 3.3)
fig, axes = plt.subplots(1, len(widths), figsize=figsize, facecolor=SURFACE, squeeze=False)
for ax, width in zip(axes[0], widths):
    series = {}
    for v in SHOWN:
        name = CONTROL[width] if v == "control" else v
        by_lr = {lr: {s: r for s, r in by_seed.items() if s in seeds} for lr, by_seed in table.get((width, name), {}).items()}
        by_lr = {lr: by_seed for lr, by_seed in by_lr.items() if set(by_seed) == seeds}
        if by_lr:
            series[name] = by_lr
    floor = min(mean_loss(by_seed) for by_lr in series.values() for by_seed in by_lr.values())
    ax.set_facecolor(SURFACE)
    ax.axhspan(0, USABLE, color=GRID, alpha=0.6, lw=0, zorder=0)
    for name, by_lr in series.items():
        label, color, dash, marker = STYLE[name]
        label = paper_style.NAMES[name] if cli.paper else label
        lrs = sorted(by_lr)
        ys = [mean_loss(by_lr[lr]) - floor for lr in lrs]
        ax.plot(lrs, ys, dash, color=color, lw=lw, zorder=2, label=label)
        ax.scatter(lrs, ys, s=ms, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        dots = [(lr, r["final_val_loss"] - floor) for lr in lrs for r in by_lr[lr].values()]
        ax.scatter(*zip(*dots), s=8, color=color, alpha=0.4, linewidth=0, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINEAR, linscale=1.5)
    lrs = sorted({lr for by_lr in series.values() for lr in by_lr})
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{lr * 1e3:g}" for lr in lrs])
    ax.minorticks_off()
    ax.set_yticks([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.02", "0.05", "0.1", "0.2", "0.5", "1"])
    ax.set_ylim(-0.005, 1.2)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, length=0, labelsize=tick_size)
    params = next(iter(next(iter(series["mhc-h1"].values())).values()))["params"] / 1e6
    title = f"{params:.0f}M parameters ($D={width}$), best mean {floor:.3f}" if cli.paper else \
        f"D = {width} ({params:.0f}M), best mean {floor:.3f}"
    ax.set_title(title, color=INK, fontsize=title_size, loc="left")
    unit = r"$\times 10^{-3}$" if cli.paper else "×10⁻³"  # STIX has no superscript minus
    ax.set_xlabel(f"peak learning rate ({unit})", color=MUTED, fontsize=label_size)
axes[0][0].set_ylabel("final val loss − best (symlog)", color=MUTED, fontsize=label_size)
legend = {label: handle for ax in axes[0] for handle, label in zip(*ax.get_legend_handles_labels())}
fig.legend(legend.values(), legend.keys(), loc="lower center", ncol=len(legend), frameon=False, fontsize=size, labelcolor=INK, handlelength=3)
fig.tight_layout(rect=(0, 0.09 if cli.paper else 0.07, 1, 1))
fig.savefig(cli.out, dpi=200, facecolor=SURFACE)
print(f"wrote {cli.out}")
