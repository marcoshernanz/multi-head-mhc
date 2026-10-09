"""Are TPU v6e results (micro-batch 16) on the same scale as the v5e ones (micro-batch 8 x 2)? Checks exp11 against its v5e twins.

Usage: cd analysis && uv run python hardware_check.py ../results/exp11_v6e_check ../results/exp04_main ../results/exp06_controls \
           ../results/exp07_scale ../results/exp08_local ../results/exp09_d768_lr ../results/exp10_d384_lr
The first directory holds the v6e runs (named "v6e-<variant>[-<tag>]-s<seed>"); the others the v5e runs with the same name minus
"v6e-". Prints, per variant and LR: each seed's v5e and v6e final val loss and their difference, the v5e seed-to-seed spread of
that variant (all its v5e seeds), the mean v6e − v5e shift over all pairs; then the paired gaps the report uses, on each chip;
then the largest late grad norm (spikes) on each chip. Writes ../figures/hardware_check.png (v6e against v5e loss, one dot per run).
The pass criteria are in experiments/exp11_v6e_check/jobs.py.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_runs, mean_std  # noqa: E402
from sweep import paired  # noqa: E402

GAPS = [  # (variant, reference, LR tag): the paired comparisons REPORT.md rests on (d384, then exp07's d768)
    ("a-global-h4", "mhc-h1", ""), ("a-global-h4", "ctrl-mlp1216", ""), ("ctrl-mlp1216", "mhc-h1", ""),
    ("a-local-h4", "mhc-h1", ""), ("mhc-h1", "residual", ""),
    ("a-global-h4", "mhc-h1", "-lr2x"), ("ctrl-mlp1216", "mhc-h1", "-lr2x"), ("a-local-h4", "mhc-h1", "-lr2x"),
    ("a-global-h4", "ctrl-mlp1216", "-lr2x"),
    ("d768-a-global-h4", "d768-mhc-h1", ""), ("d768-a-global-h4", "d768-ctrl-mlp2240", ""), ("d768-ctrl-mlp2240", "d768-mhc-h1", ""),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def late_max_grad(run_dir: Path, frac: float = 0.2) -> float:
    import json
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    grads = [(r["step"], r["grad_norm"]) for r in records if "grad_norm" in r]
    last = grads[-1][0]
    return max(g for s, g in grads if s >= frac * last)


def fmt_gap(p: dict | None) -> str:
    if p is None:
        return "–"
    se = "" if math.isnan(p["se"]) else f" ± {p['se']:.4f}"
    return f"{p['mean']:+.4f}{se} ({p['better']}/{len(p['diffs'])})"


def main() -> None:
    v6e_runs = load_runs([Path(sys.argv[1])])
    v5e_runs = load_runs([Path(d) for d in sys.argv[2:]])
    v6e = {name.removeprefix("v6e-"): by_seed for name, by_seed in v6e_runs.items() if name.startswith("v6e-")}  # not chk-*
    shifts, dots = [], []
    print("variant              seed   v5e      v6e      v6e−v5e   | v5e spread of the variant (all seeds)")
    for name in sorted(v6e):
        base = v5e_runs.get(name, {})
        values = [r["final_val_loss"] for r in base.values()]
        m, s = mean_std(values) if values else (float("nan"), float("nan"))
        for seed, run in sorted(v6e[name].items()):
            new = run["final_val_loss"]
            old = base.get(seed, {}).get("final_val_loss")
            diff = "" if old is None else f"{new - old:+.4f}"
            if old is not None:
                shifts.append(new - old)
                dots.append((old, new, name))
            print(f"{name:20s} {seed:4d}   {old if old is not None else float('nan'):.4f}   {new:.4f}   {diff:9s} "
                  f"| mean {m:.4f}, std {s:.4f}, range {min(values, default=float('nan')):.4f}-{max(values, default=float('nan')):.4f} "
                  f"(n={len(values)})")
    if shifts:
        m, s = mean_std(shifts)
        print(f"\nv6e − v5e over {len(shifts)} paired runs: mean {m:+.4f} ± {s / math.sqrt(len(shifts)):.4f} (std {s:.4f}), "
              f"|diff| max {max(abs(x) for x in shifts):.4f}")

    print("\npaired gaps (same seeds on both chips)")
    print(f"{'gap':42s} {'v5e, all its seeds':30s} {'v5e, exp11 seeds':30s} {'v6e':30s}")
    for variant, ref, tag in GAPS:
        a, b = variant + tag, ref + tag
        if a not in v6e or b not in v6e:
            continue
        on_v6e = paired(v6e[a], v6e[b])
        seeds = set(v6e[a]) & set(v6e[b])
        on_v5e_all = paired(v5e_runs.get(a, {}), v5e_runs.get(b, {}))
        on_v5e = paired({s: r for s, r in v5e_runs.get(a, {}).items() if s in seeds}, v5e_runs.get(b, {}))
        print(f"{a + ' − ' + b:42s} {fmt_gap(on_v5e_all):30s} {fmt_gap(on_v5e):30s} {fmt_gap(on_v6e):30s}")

    print("\nlargest grad norm in the last 80% of training, mean over the exp11 seeds")
    for name in sorted(v6e):
        new = [late_max_grad(r["dir"]) for r in v6e[name].values()]
        old = [late_max_grad(v5e_runs[name][s]["dir"]) for s in v6e[name] if s in v5e_runs.get(name, {})]
        print(f"{name:20s} v5e {sum(old) / len(old) if old else float('nan'):.3f}   v6e {sum(new) / len(new):.3f}")

    if dots:
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        lo = min(min(o, n) for o, n, _ in dots) - 0.02
        hi = max(max(o, n) for o, n, _ in dots) + 0.02
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)
        for old, new, name in dots:
            ax.scatter(old, new, s=18, color="#eb6834" if "lr2x" in name else INK, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("final val loss on TPU v5e (micro-batch 8 × 2)")
        ax.set_ylabel("final val loss on TPU v6e (micro-batch 16)")
        ax.set_title("Same run, two chips (orange: 2× LR)", fontsize=10)
        ax.grid(color=GRID, lw=0.6)
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "figures" / "hardware_check.png"
        fig.savefig(out, dpi=160)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
