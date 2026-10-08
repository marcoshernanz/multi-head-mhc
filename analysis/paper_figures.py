"""The paper's figures (paper/figures/*.pdf), computed from the raw results.

Usage: cd analysis && uv run python paper_figures.py
Writes summary.pdf (Figure 1), parts.pdf, and, through lr_figure.py and params_frontier.py with --paper, lr.pdf and frontier.pdf.
Comparisons follow REPORT.md: paired by seed at one LR; each variant at its own best LR (lowest mean over seeds 0-2 on v6e, 0-1
on v5e) with the two-sample SE when the LRs differ, and always at d768 (exp15's rule); the two chips pooled by inverse variance
as independent estimates. Error bars are ±2 SE, the threshold of the effect rule.
"""

import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import lr_sweep
import paper_style
from aggregate import load_runs
from lr_sweep import best_lr, load, two_sample
from paper_style import AQUA, BLUE, GRID, INK, MUTED, ORANGE, SMALL, SURFACE, TEXT_WIDTH
import stability
from sweep import paired

ROOT = Path(__file__).resolve().parents[1]
RESULTS, OUT = ROOT / "results", ROOT / "paper" / "figures"
S3, S2 = (0, 1, 2), (0, 1)


def at_lr(by_lr: dict, lr: float) -> dict:
    return by_lr[min(by_lr, key=lambda x: abs(math.log(x / lr)))]


def compare(table: dict, width: int, a: str, b: str, seeds: tuple, two: bool = False, lr: float | None = None) -> dict:
    """a − b, both at `lr`, or each at its own best LR over `seeds`."""
    lr_sweep.PLOT_SEEDS = seeds
    ta, tb = table[(width, a)], table[(width, b)]
    la, lb = (lr, lr) if lr else (best_lr(ta), best_lr(tb))
    ra, rb = at_lr(ta, la), at_lr(tb, lb)
    return two_sample(ra, rb) if two or la != lb else paired(ra, rb)


def pool(estimates: list[dict]) -> dict:
    """Inverse-variance mean of independent estimates (the two chips); established if > 2 SE and every seed agrees."""
    w = [1 / e["se"] ** 2 for e in estimates]
    mean = sum(wi * e["mean"] for wi, e in zip(w, estimates)) / sum(w)
    se = 1 / math.sqrt(sum(w))
    better, n = sum(e["better"] for e in estimates), sum(len(e["diffs"]) for e in estimates)
    return {"mean": mean, "se": se, "better": better, "diffs": [None] * n, "exists": abs(mean) > 2 * se and better in (0, n)}


def qk_at_best(a: str, b: str) -> dict:
    """exp19 with QK-norm (d384, v6e, seeds 0-2): a − b, each at its own best LR on the 9-point grid (stability.py's rule)."""
    runs = stability.load()
    best = stability.best_lrs(runs, True)
    return stability.compare(runs, True, a, best[a], b, best[b])


def main_figure() -> None:
    v6e = load([RESULTS / d for d in ("exp11_v6e_check", "exp12_d384_lr_grid", "exp15_d768_v6e")])
    v5e = load([RESULTS / d for d in ("exp07_scale", "exp09_d768_lr")])
    long = {k.removeprefix("v5e-long-"): v for k, v in load_runs([RESULTS / "exp13_d384_long"]).items()}
    g, loc = ("a-global-h4", {384: "ctrl-mlp1216", 768: "ctrl-mlp2240"}), ("a-local-h4", "mhc-h1")
    both = lambda w, a, b, **kw: pool([compare(v6e, w, a, b, S3, True, **kw), compare(v5e, w, a, b, S2, True, **kw)])  # noqa: E731
    groups = [
        ("At each variant’s best learning rate", [
            ("global", "27M", compare(v6e, 384, g[0], g[1][384], S3)),
            ("global", "112M", both(768, g[0], g[1][768])),
            ("local", "27M", compare(v6e, 384, *loc, S3)),
            ("local", "112M", both(768, *loc)),
        ]),
        ("With QK-norm, at each variant’s best learning rate", [
            ("global", "27M", qk_at_best("global-h4", "control")),
            ("local", "27M", qk_at_best("local-h4", "mhc")),
        ]),
        ("At 4× the tokens (393M, one learning rate)", [
            ("global", "27M", paired(long[g[0]], long[g[1][384]])),
            ("local", "27M", paired(long[loc[0]], long[loc[1]])),
        ]),
        ("At 2× mHC’s best learning rate", [
            ("global", "27M", compare(v6e, 384, g[0], g[1][384], S3, lr=4.2e-3)),
            ("global", "112M", pool([compare(v6e, 768, g[0], g[1][768], S3, lr=1.5e-3),
                                     compare(v5e, 768, g[0], g[1][768], S2, lr=1.5e-3)])),
            ("local", "27M", compare(v6e, 384, *loc, S3, lr=4.2e-3)),
            ("local", "112M", pool([compare(v6e, 768, *loc, S3, lr=1.5e-3), compare(v5e, 768, *loc, S2, lr=1.5e-3)])),
        ]),
        ("Reference: what mHC itself gains", [
            ("ref", "27M", compare(v6e, 384, "mhc-h1", "residual", S3)),
            ("ref", "112M", compare(v6e, 768, "mhc-h1", "residual", S3, True)),
        ]),
    ]
    style = {"global": (BLUE, "o", "global $h=4$ − MLP control"), "local": (ORANGE, "s", "local $h=4$ − mHC"),
             "ref": (INK, "D", "mHC − residual")}

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.65))
    y, ticks, labels = 0.0, [], []
    for title, rows in groups:
        ax.text(0.0, y, title, transform=ax.get_yaxis_transform(), fontweight="bold", color=INK, va="center", zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0})  # hide the zero line behind long titles
        y -= 1
        for kind, size, e in rows:
            color, marker, _ = style[kind]
            ax.errorbar(e["mean"], y, xerr=2 * e["se"], fmt=marker, ms=5.5, color=color, ecolor=color, elinewidth=1.4,
                        capsize=0, mfc=color if e["exists"] else SURFACE, mec=color, mew=1.4, zorder=3)
            n = len(e["diffs"])
            value = f"{e['mean']:+.3f}".replace("-", "−")
            ax.text(1.02, y, f"{value} ± {e['se']:.3f}", transform=ax.get_yaxis_transform(), fontsize=SMALL, color=MUTED,
                    va="center")
            ax.text(1.3, y, f"{e['better']}/{n}", transform=ax.get_yaxis_transform(), fontsize=SMALL, color=MUTED, va="center",
                    ha="right")
            ticks.append(y)
            labels.append(size)
            y -= 1
        y -= 0.4
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.6, 0.6)
    ax.set_xlim(-0.115, 0.06)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("difference in final validation loss (negative: the first model is better)")
    ax.text(1.02, 0.6, "Δ ± SE", transform=ax.get_yaxis_transform(), fontsize=SMALL, color=MUTED, va="bottom")
    ax.text(1.3, 0.6, "better", transform=ax.get_yaxis_transform(), fontsize=SMALL, color=MUTED, va="bottom", ha="right")
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=5.5, mew=1.4) for c, m, _ in style.values()]
    ax.legend(handles, [s[2] for s in style.values()], loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False,
              handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(left=0.075, right=0.755, top=0.905, bottom=0.115)
    fig.savefig(OUT / "summary.pdf")
    for title, rows in groups:
        for kind, size, e in rows:
            print(f"{title} | {kind} {size}: {e['mean']:+.4f} ± {e['se']:.4f} ({e['better']}/{len(e['diffs'])})"
                  f"{' *' if e['exists'] else ''}")
    print(f"wrote {OUT / 'summary.pdf'}")


def parts_figure() -> None:
    table = load([RESULTS / d for d in ("exp11_v6e_check", "exp12_d384_lr_grid", "exp14_head_parts", "exp17_parts_best_lr")])
    variants = [("a-global-h4-pre", "read only"), ("a-global-h4-post", "write only"), ("a-global-h4-prepost", "read and write"),
                ("a-global-h4-res", "mix only"), ("a-global-h4", "all three")]
    params = lambda v: next(iter(next(iter(table[(384, v)].values())).values()))["params"] / 1e6  # noqa: E731

    fig, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.9), gridspec_kw={"width_ratios": [1.15, 1]}, sharey=True)
    ys = range(len(variants), 0, -1)
    for lr, color, dy, name in [(2.1e-3, BLUE, 0.13, r"$2.1\times10^{-3}$ (the optimum)"), (3e-3, ORANGE, -0.13, r"$3\times10^{-3}$ (1.4×)")]:
        base = at_lr(table[(384, "mhc-h1")], lr)
        ctrl = paired(at_lr(table[(384, "ctrl-mlp1216")], lr), base)
        slope = ctrl["mean"] / (params("ctrl-mlp1216") - params("mhc-h1"))
        for (v, _), y in zip(variants, ys):
            p = paired(at_lr(table[(384, v)], lr), base)
            line = slope * (params(v) - params("mhc-h1"))
            meets = p["mean"] - line < -2 * p["se"] and all(d < line for d in p["diffs"])
            left.errorbar(p["mean"] - line, y + dy, xerr=2 * p["se"], fmt="o", ms=5, color=color, elinewidth=1.4, capsize=0,
                          mfc=color if meets else SURFACE, mec=color, mew=1.4, label=name if v == variants[0][0] else None)
    for (v, _), y in zip(variants, ys):
        p = paired(at_lr(table[(384, v)], 6e-3), at_lr(table[(384, "mhc-h1")], 6e-3))
        right.errorbar(p["mean"], y, xerr=2 * p["se"], fmt="o", ms=5, color=AQUA, elinewidth=1.4, capsize=0,
                       mfc=AQUA if p["exists"] else SURFACE, mec=AQUA, mew=1.4, label=r"$6\times10^{-3}$ (2.9×)" if v == variants[0][0] else None)
    left.set_yticks(list(ys), [name for _, name in variants])
    left.set_xlabel("loss − MLP line at the same parameter count")
    right.set_xlabel("loss − mHC at the same LR")
    left.set_title("(a) Against the MLP line", loc="left", color=INK)
    right.set_title("(b) Against mHC, past the optimum", loc="left", color=INK)
    handles = [h for ax in (left, right) for h in ax.get_legend_handles_labels()[0]]
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=3, frameon=False, handletextpad=0.2,
               columnspacing=1.6)
    for ax in (left, right):
        ax.axvline(0, color=INK, lw=0.8, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(w_pad=2, rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / "parts.pdf")
    print(f"wrote {OUT / 'parts.pdf'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    paper_style.use()
    main_figure()
    parts_figure()
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, "lr_figure.py", "--paper", "--out", str(OUT / "lr.pdf"),
                    *(f"../results/{d}" for d in ("exp11_v6e_check", "exp12_d384_lr_grid", "exp15_d768_v6e"))], cwd=here, check=True)
    subprocess.run([sys.executable, "params_frontier.py", "--paper", "--out", str(OUT / "frontier.pdf"),
                    *(f"../results/{d}" for d in ("exp04_main", "exp06_controls", "exp07_scale", "exp08_local"))],
                   cwd=here, check=True, stdout=subprocess.DEVNULL)
