"""The paper's stability figure (exp19): final val loss against the peak LR on the whole grid, without and with QK-norm, the
largest attention logit that marks the failure, and how often the gradient norm crosses the clipping threshold.

Usage: cd analysis && uv run python stability_figure.py [--out ../paper/figures/stability.pdf] [--partial]
The paper's figure is drawn with --partial: twelve runs without QK-norm at 8.5e-3 and above could not be run, so there the
lines show the mean over the seeds that ran.
Top: final val loss above each panel's best mean (symlog, linear up to 0.05; a diverged run counts as ln 16384, exp19's rule),
means over seeds 0-2 at the LRs that have them all, dots the single seeds. Bottom: the largest attention logit over layers at step
1000, mean over seeds (log axis), from the diagnostics on 4 fixed validation windows. Third row: the share of steps in the last
80% of training whose grad norm before clipping exceeds 1.0, mean over seeds (symlog, linear up to 0.1%).
"""

import argparse
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt

import paper_style
from lr_sweep import STYLE
from paper_style import GRID, INK, MUTED, SURFACE, TEXT_WIDTH
from stability import GRID as LRS
from stability import SEEDS, capped, late_clipped, load

ORDER = ["residual", "mhc", "control", "local-h4", "global-h4"]
RUN_NAME = {"mhc": "mhc-h1", "global-h4": "a-global-h4", "local-h4": "a-local-h4", "control": "ctrl-mlp1216", "residual": "residual"}
LINEAR = 0.05


def largest_logit(run: dict, step: int = 1000) -> float | None:
    d = dict(run["diag"])
    return max(d[step]["attn_max_logit"]) if step in d else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "paper" / "figures" / "stability.pdf")
    parser.add_argument("--partial", action="store_true", help="use whichever seeds ran (the paper's figure)")
    cli = parser.parse_args()
    paper_style.use()
    runs = load()
    fig, axes = plt.subplots(3, 2, figsize=(TEXT_WIDTH, 6.0), height_ratios=(1.6, 1, 1), sharex=True)
    for col, (qk, title) in enumerate([(False, "Without QK-norm"), (True, "With QK-norm")]):
        complete = {(v, lr): [runs[(qk, v, lr, s)] for s in SEEDS if (qk, v, lr, s) in runs] for v in ORDER for lr in LRS
                    if all((qk, v, lr, s) in runs for s in SEEDS) or (cli.partial and any((qk, v, lr, s) in runs for s in SEEDS))}
        floor = min(st.mean(capped(r) for r in rs) for rs in complete.values())
        top, bottom, clip = axes[0][col], axes[1][col], axes[2][col]
        top.axhspan(0, 0.02, color=GRID, alpha=0.6, lw=0, zorder=0)
        for v in ORDER:
            label, color, dash, marker = STYLE[RUN_NAME[v]]
            lrs = [lr for lr in LRS if (v, lr) in complete]
            if not lrs:
                continue
            ys = [st.mean(capped(r) for r in complete[(v, lr)]) - floor for lr in lrs]
            top.plot(lrs, ys, dash, color=color, lw=1.4, zorder=2, label=paper_style.NAMES[RUN_NAME[v]])
            top.scatter(lrs, ys, s=22, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            dots = [(lr, capped(r) - floor) for lr in lrs for r in complete[(v, lr)]]
            top.scatter(*zip(*dots), s=8, color=color, alpha=0.4, linewidth=0, zorder=2)
            logits = [(lr, st.mean(x)) for lr in lrs if (x := [y for r in complete[(v, lr)] if (y := largest_logit(r)) is not None])]
            if logits:
                bottom.plot(*zip(*logits), dash, color=color, lw=1.4, zorder=2)
                bottom.scatter(*zip(*logits), s=22, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            clipped = [(lr, st.mean(x)) for lr in lrs if (x := [y for r in complete[(v, lr)] if (y := late_clipped(r)) is not None])]
            if clipped:
                clip.plot(*zip(*clipped), dash, color=color, lw=1.4, zorder=2)
                clip.scatter(*zip(*clipped), s=22, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        top.set_yscale("symlog", linthresh=LINEAR, linscale=1.2)
        top.set_yticks([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
        top.set_yticklabels(["0", "0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
        top.set_ylim(-0.005, 6.5)
        top.set_title(f"{title}: best mean {floor:.3f}", color=INK, loc="left")
        bottom.set_yscale("log")
        bottom.set_ylim(5, 2e4)
        clip.set_yscale("symlog", linthresh=0.1, linscale=0.6)
        clip.set_yticks([0, 0.1, 1, 10, 100])
        clip.set_yticklabels(["0", "0.1", "1", "10", "100"])
        clip.set_ylim(-0.01, 150)
        for ax in (top, bottom, clip):
            ax.set_xscale("log")
            ax.set_xticks(LRS)
            ax.set_xticklabels([f"{lr * 1e3:g}" for lr in LRS])
            ax.minorticks_off()
            ax.grid(True, color=GRID, lw=0.6, zorder=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(length=0)
        clip.set_xlabel(r"peak learning rate ($\times 10^{-3}$)")
    axes[0][0].set_ylabel("final val loss − best (symlog)")
    axes[1][0].set_ylabel("largest attention\nlogit at step 1000")
    axes[2][0].set_ylabel("% of late steps\nclipped (symlog)")
    handles, labels = axes[0][1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, handlelength=3)
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=1.5)
    fig.savefig(cli.out)
    print(f"wrote {cli.out}")


if __name__ == "__main__":
    main()
