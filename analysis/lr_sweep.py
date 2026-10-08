"""Final val loss against the learning rate, per model size and variant (exp06 lr4x, exp07 lr2x, exp09 d768, exp10 d384).

Usage: cd analysis && uv run python lr_sweep.py ../results/exp04_main ../results/exp06_controls ../results/exp07_scale [...]
       [--out ../figures/NAME.png] [--seeds 0,1]
The LR and the model width are read from each run's run.json, so run names only need "<variant>[-<tag>]-s<seed>" with an
optional "v6e-" and "d768-" prefix (pass v5e and v6e batches in separate calls: the chips differ in floating-point noise). Prints, per width: the loss table (variant x LR), the paired Δ to mhc-h1 at each LR, global h4
against its MLP control at each LR, and each variant at its own best LR. Writes ../figures/lr_sweep.png.
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_runs  # noqa: E402
from sweep import paired  # noqa: E402

BASE = "mhc-h1"
SWEPT = ["mhc-h1", "a-global-h4", "ctrl-mlp1216", "ctrl-mlp2240", "a-local-h4", "residual", "mhc-h1-conn0.25",
         "a-global-h4-conn4", "mhc-h1-conn4",  # every connection parameter's LR x0.25 / x4 (exp12, exp16)
         "a-global-h4-pre", "a-global-h4-post", "a-global-h4-res", "a-global-h4-prepost"]  # only those blocks per head (exp14)
PARTS = ["a-global-h4-pre", "a-global-h4-post", "a-global-h4-res", "a-global-h4-prepost"]
CONTROL = {384: "ctrl-mlp1216", 768: "ctrl-mlp2240"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {  # variant -> (label, color, line style, marker); colors as in params_frontier.py (MLP line and mHC h1 in ink)
    "mhc-h1": ("mHC h1", INK, "-", "o"),
    "a-global-h4": ("global h4", "#2a78d6", "-", "o"),
    "a-local-h4": ("local h4", "#eb6834", "-", "o"),
    "ctrl-mlp1216": ("mHC h1, wider MLP", INK, "--", "s"),
    "ctrl-mlp2240": ("mHC h1, wider MLP", INK, "--", "s"),
    "residual": ("residual", "#1baf7a", "-", "o"),
    "mhc-h1-conn0.25": ("mHC h1, connection LR x0.25", INK, ":", "D"),
    "a-global-h4-conn4": ("global h4, connection LR x4", "#2a78d6", ":", "D"),
    "mhc-h1-conn4": ("mHC h1, connection LR x4", INK, "-.", "D"),
    "a-global-h4-pre": ("h4, read per head", "#8a5cd6", "-", "^"),
    "a-global-h4-post": ("h4, write per head", "#c4a000", "-", "^"),
    "a-global-h4-res": ("h4, mix per head", "#d6336c", "-", "^"),
    "a-global-h4-prepost": ("h4, read+write per head", "#2a78d6", "--", "^"),
}
WINDOW = 0.15  # the loss panels show [best − 0.01, best + WINDOW]; worse means sit on the top edge as ▲ with their value
DELTA_WINDOW = (-0.1, 0.05)  # the Δ panels
USABLE = 0.02  # exp12 read-out (c)
H7B = (0.05, 0.15)  # exp12 read-out (d): mHC h1 conn x0.25 − global h4 at 6e-3, "without heads" / "the split matters"
PLOT_SEEDS = (0, 1)  # the figure uses only the seeds every LR has, so a line never joins means over different seeds (--seeds)


def base_variant(name: str) -> str:
    name = name.removeprefix("v6e-").removeprefix("d768-")
    name = re.sub(r"-mb2$", "", name)
    return name.split("-lr")[0]  # the tag can hold dashes itself: "-lr2.1e-3"


def load(dirs: list[Path]) -> dict:
    """(width, variant) -> lr -> seed -> summary."""
    table: dict = defaultdict(lambda: defaultdict(dict))
    for name, by_seed in load_runs(dirs).items():
        if name.startswith(("bench-", "repeat-", "repro-", "chk-", "v6e-long-")):
            continue
        variant = base_variant(name)
        if variant not in SWEPT:
            continue
        for seed, summary in by_seed.items():
            args = json.loads((summary["dir"] / "run.json").read_text())["args"]
            table[(int(args["dim"]), variant)][float(args["lr"])][seed] = summary
    return table


def fmt(p: dict | None) -> str:
    if p is None:
        return "–"
    se = "" if math.isnan(p["se"]) else f" ± {p['se']:.4f}"
    star = " *" if p["exists"] else ""
    return f"{p['mean']:+.4f}{se} ({p['better']}/{len(p['diffs'])}){star}"


def two_sample(by_seed: dict, base: dict) -> dict | None:
    """Difference of means with the two-sample SE (exp15's rule across LRs); "better" and "exists" still use the per-seed signs."""
    p = paired(by_seed, base)
    xa, xb = [r["final_val_loss"] for r in by_seed.values()], [r["final_val_loss"] for r in base.values()]
    if p is None or len(xa) < 2 or len(xb) < 2:
        return None
    var = lambda x: sum((v - sum(x) / len(x)) ** 2 for v in x) / (len(x) - 1)  # noqa: E731
    m = sum(xa) / len(xa) - sum(xb) / len(xb)
    se = math.sqrt(var(xa) / len(xa) + var(xb) / len(xb))
    return {**p, "mean": m, "se": se, "exists": abs(m) > 2 * se and p["better"] in (0, len(p["diffs"]))}


def late_max_grad(run_dir: Path, frac: float = 0.2) -> float:
    """The largest logged grad norm after the first 20% of training (as hardware_check.py)."""
    grads = [(r["step"], r["grad_norm"]) for r in map(json.loads, (run_dir / "metrics.jsonl").read_text().splitlines())
             if "grad_norm" in r]
    return max(g for step, g in grads if step >= frac * grads[-1][0])


def mean_loss(by_seed: dict) -> float:
    return sum(r["final_val_loss"] for r in by_seed.values()) / len(by_seed)


def best_lr(by_lr: dict) -> float:
    """Lowest mean loss over the --seeds (exp15's rule: seeds 0-2), among the LRs that have all of them, so extra seeds at one LR
    do not bias the choice and a half-run LR does not shrink the comparison to fewer seeds. If no LR has all of them: the seeds
    run at every LR."""
    full = {lr: by_seed for lr, by_seed in by_lr.items() if set(PLOT_SEEDS) <= set(by_seed)}
    if full:
        return min(full, key=lambda lr: mean_loss({s: r for s, r in full[lr].items() if s in PLOT_SEEDS}))
    common = set.intersection(*[set(by_seed) for by_seed in by_lr.values()])
    return min(by_lr, key=lambda lr: mean_loss({s: r for s, r in by_lr[lr].items() if s in common} or by_lr[lr]))


def report(table: dict, width: int) -> None:
    variants = [v for v in SWEPT if (width, v) in table]
    lrs = sorted({lr for v in variants for lr in table[(width, v)]})
    print(f"\n## d{width}\n\nfinal val loss, mean over seeds (seeds)\n")
    print("| variant | " + " | ".join(f"{lr:g}" for lr in lrs) + " |")
    print("|---|" + "---|" * len(lrs))
    for v in variants:
        cells = [f"{mean_loss(table[(width, v)][lr]):.4f} ({len(table[(width, v)][lr])})" if lr in table[(width, v)] else "–"
                 for lr in lrs]
        print(f"| {v} | " + " | ".join(cells) + " |")
    print("\npaired Δ at the same LR (negative = better), mean ± SE (better seeds), * = the effect exists\n")
    print("| comparison | " + " | ".join(f"{lr:g}" for lr in lrs) + " |")
    print("|---|" + "---|" * len(lrs))
    comparisons = [(v, BASE) for v in variants if v != BASE] + [("a-global-h4", CONTROL[width])]
    comparisons += [("a-global-h4-conn4", "a-global-h4")] if "a-global-h4-conn4" in variants else []
    comparisons += [("mhc-h1-conn0.25", "a-global-h4")] if "mhc-h1-conn0.25" in variants else []
    comparisons += [("a-global-h4-conn4", "mhc-h1-conn4")] if "mhc-h1-conn4" in variants else []  # exp16 read-out (3)
    comparisons += [(v, "a-global-h4") for v in PARTS if v in variants]
    for a, b in comparisons:
        cells = []
        for lr in lrs:
            ra = table.get((width, a), {}).get(lr)
            rb = table.get((width, b), {}).get(lr)
            cells.append(fmt(paired(ra, rb)) if ra and rb else "–")
        print(f"| {a} − {b} | " + " | ".join(cells) + " |")
    best = {v: best_lr(table[(width, v)]) for v in variants}
    print(f"\neach variant at its own best LR (lowest mean over seeds {SEED_TEXT} among the LRs that have them all), paired by seed\n")
    for v in variants:
        at_best = table[(width, v)][best[v]]
        on_seeds = mean_loss({s: r for s, r in at_best.items() if s in PLOT_SEEDS} or at_best)
        print(f"- {v}: best LR {best[v]:g}, {on_seeds:.4f} on seeds {SEED_TEXT} ({mean_loss(at_best):.4f} on all {len(at_best)})")
    for a, b in comparisons:
        if a in best and b in best:
            ra, rb = table[(width, a)][best[a]], table[(width, b)][best[b]]
            print(f"- {a} at {best[a]:g} − {b} at {best[b]:g}: {fmt(paired(ra, rb))}; two-sample: {fmt(two_sample(ra, rb))}")
    print(f"\nlargest grad norm in the last 80% of training (spikes), mean over seeds\n")
    print("| variant | " + " | ".join(f"{lr:g}" for lr in lrs) + " |")
    print("|---|" + "---|" * len(lrs))
    for v in variants:
        cells = []
        for lr in lrs:
            runs = table[(width, v)].get(lr, {})
            cells.append(f"{sum(late_max_grad(r['dir']) for r in runs.values()) / len(runs):.2f}" if runs else "–")
        print(f"| {v} | " + " | ".join(cells) + " |")
    print(f"\nusable range (exp12 read-out c): the LRs whose mean is within {USABLE} of the variant's best (same seeds)\n")
    for v in variants:
        by_lr = table[(width, v)]
        full = {lr: by_seed for lr, by_seed in by_lr.items() if set(PLOT_SEEDS) <= set(by_seed)}
        means = {lr: mean_loss({s: r for s, r in by_seed.items() if s in PLOT_SEEDS}) for lr, by_seed in full.items()}
        usable = [lr for lr in sorted(means) if means[lr] <= means[best[v]] + USABLE] if best[v] in means else []
        partial = sorted(set(by_lr) - set(full))
        print(f"- {v}: {', '.join(f'{lr:g}' for lr in usable)} (of {', '.join(f'{lr:g}' for lr in sorted(means))} on seeds "
              f"{SEED_TEXT}{'; not all seeds yet: ' + ', '.join(f'{lr:g}' for lr in partial) if partial else ''})")
    slow, heads = table.get((width, "mhc-h1-conn0.25"), {}), table.get((width, "a-global-h4"), {})
    for lr in sorted(set(slow) & set(heads)):
        p = paired(slow[lr], heads[lr])
        if p and lr >= 5e-3:
            verdict = ("the tolerance can be had without heads" if p["mean"] <= H7B[0] else
                       "the split itself matters" if p["mean"] >= H7B[1] else "partial")
            print(f"\nH7b (read-out d) at {lr:g}: mHC h1 with connection LR x0.25 − global h4 = {fmt(p)}: {verdict} "
                  f"(thresholds {H7B[0]} / {H7B[1]})")


def draw(ax, series: dict, dots: dict, lo: float, hi: float) -> None:
    """series: variant -> [(lr, y)]; dots: variant -> [(lr, y)] per seed. Values outside [lo, hi] become ▲/▼ on the edge."""
    outside = defaultdict(list)  # (lr, edge) -> [(label, y)]
    for v, points in series.items():
        label, color, dash, marker = STYLE[v]
        ax.plot(*zip(*points), dash, color=color, lw=2, zorder=2)
        inside = [(lr, y) for lr, y in points if lo <= y <= hi]
        if inside:
            ax.scatter(*zip(*inside), s=64, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3)
        for lr, y in points:
            if not lo <= y <= hi:
                edge = hi if y > hi else lo
                ax.scatter([lr], [edge], s=64, color=color, marker="^" if y > hi else "v", edgecolor=SURFACE, linewidth=2,
                           zorder=3, clip_on=False)
                outside[(lr, edge)].append((label, y))
        dot_points = [(lr, y) for lr, y in dots.get(v, []) if lo <= y <= hi]
        if dot_points:
            ax.scatter(*zip(*dot_points), s=12, color=color, alpha=0.45, linewidth=0, zorder=2)
    lrs = sorted({lr for points in series.values() for lr, _ in points})
    for (lr, edge), items in outside.items():
        text = "\n".join(f"{label} {y:+.2f}" if lo < 0 else f"{label} {y:.2f}" for label, y in sorted(items, key=lambda i: i[1]))
        first = lr == lrs[0]
        top = edge == hi
        ax.annotate(text, (lr, edge), textcoords="offset points", xytext=(8 if first else -8, -10 if top else 10),
                    ha="left" if first else "right", va="top" if top else "bottom", fontsize=8, color=MUTED,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    ax.set_xscale("log")
    ax.set_xticks(lrs)
    ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.minorticks_off()
    ax.set_ylim(lo, hi)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)


def plot(table: dict, widths: list[int], path: Path) -> None:
    """Top: final loss; bottom: paired Δ to mhc-h1 at the same LR. PLOT_SEEDS only, the pairs every LR has."""
    fig, axes = plt.subplots(2, len(widths), figsize=(6 * len(widths), 8.4), facecolor=SURFACE, squeeze=False)
    for col, width in enumerate(widths):
        variants = [v for v in SWEPT if (width, v) in table]
        pick = {v: {lr: {s: r for s, r in by_seed.items() if s in PLOT_SEEDS} for lr, by_seed in table[(width, v)].items()}
                for v in variants}
        pick = {v: {lr: by_seed for lr, by_seed in by_lr.items() if by_seed} for v, by_lr in pick.items()}
        loss = {v: [(lr, mean_loss(pick[v][lr])) for lr in sorted(pick[v])] for v in variants if pick[v]}
        loss_dots = {v: [(lr, r["final_val_loss"]) for lr in pick[v] for r in pick[v][lr].values()] for v in variants}
        floor = min(y for points in loss.values() for _, y in points)
        ax = axes[0][col]
        ax.set_facecolor(SURFACE)
        draw(ax, loss, loss_dots, floor - 0.01, floor + WINDOW)
        ax.set_title(f"d{width}: final val loss", color=INK, fontsize=11, loc="left")
        ax.set_ylabel(f"val loss (mean of seeds {SEED_TEXT}; dots = seeds)", color=MUTED)
        delta, delta_dots = {}, {}
        for v in variants:
            if v == BASE:
                continue
            points, dots = [], []
            for lr in sorted(pick[v]):
                base = pick[BASE].get(lr)
                if not base:
                    continue
                diffs = [pick[v][lr][s]["final_val_loss"] - base[s]["final_val_loss"] for s in pick[v][lr] if s in base]
                if diffs:
                    points.append((lr, sum(diffs) / len(diffs)))
                    dots += [(lr, d) for d in diffs]
            if points:
                delta[v], delta_dots[v] = points, dots
        ax = axes[1][col]
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=INK, lw=0.8, zorder=1)
        draw(ax, delta, delta_dots, DELTA_WINDOW[0], DELTA_WINDOW[1])
        ax.set_title(f"d{width}: Δ to mHC h1 at the same LR (negative = better)", color=INK, fontsize=11, loc="left")
        ax.set_ylabel(f"paired Δ val loss (mean of seeds {SEED_TEXT})", color=MUTED)
        ax.set_xlabel("peak learning rate", color=MUTED)
    shown = [v for v in SWEPT if any((width, v) in table for width in widths)]
    legend = [plt.Line2D([], [], color=color, ls=dash, lw=2, marker=marker, markersize=7, markeredgecolor=SURFACE)
              for label, color, dash, marker in {STYLE[v][0]: STYLE[v] for v in shown}.values()]
    labels = list({STYLE[v][0]: None for v in shown})
    fig.legend(legend, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False, fontsize=9, labelcolor=INK, handlelength=4)
    fig.suptitle("Learning-rate dependence (98M tokens; ▲▼ = off the scale, value listed)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures" / "lr_sweep.png")
    parser.add_argument("--seeds", default="0,1")
    cli = parser.parse_args()
    PLOT_SEEDS = tuple(int(s) for s in cli.seeds.split(","))
    SEED_TEXT = f"{PLOT_SEEDS[0]}-{PLOT_SEEDS[-1]}" if len(PLOT_SEEDS) > 1 else str(PLOT_SEEDS[0])
    table = load(cli.dirs)
    widths = sorted({width for width, _ in table})
    for width in widths:
        report(table, width)
    plot(table, widths, cli.out)
