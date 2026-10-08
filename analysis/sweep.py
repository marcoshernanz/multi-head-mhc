"""Tables and the head-count figure for a sweep: paired differences to mHC h1, throughput and probe readings.

Usage: uv run python analysis/sweep.py results/<batch> [results/<batch2> ...] --out figures/<name> [--baseline mhc-h1]
Prints markdown tables (paired differences, benchmarks) and writes <out>_hcurve.png and <out>_analysis.json.
A difference "exists" (notes/plan.md) if |mean| > 2 SE and every seed agrees in sign.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aggregate import load_runs, mean_std  # noqa: E402

CURVES = {  # predictor -> {h: variant}; h = 1 of the global and local curves is mHC itself
    "global": {1: "mhc-h1", 2: "a-global-h2", 4: "a-global-h4", 8: "a-global-h8", 16: "a-global-h16"},
    "local": {1: "mhc-h1", 2: "a-local-h2", 4: "a-local-h4", 8: "a-local-h8", 16: "a-local-h16"},
    "static": {1: "static-h1", 4: "static-h4", 8: "static-h8"},
}


def paired(by_seed: dict, base: dict) -> dict | None:
    shared = sorted(set(by_seed) & set(base))
    diffs = [by_seed[s]["final_val_loss"] - base[s]["final_val_loss"] for s in shared]
    if not diffs:
        return None
    m, s = mean_std(diffs)
    se = s / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    exists = len(diffs) > 1 and abs(m) > 2 * se and (all(d < 0 for d in diffs) or all(d > 0 for d in diffs))
    return {"seeds": shared, "diffs": diffs, "mean": m, "std": s, "se": se, "t": m / se if se and se > 0 else float("nan"),
            "better": sum(d < 0 for d in diffs), "exists": exists}


def probe(by_seed: dict) -> dict:
    out = {}
    for key in ("stream_copy_cosine", "depth_head_spread"):
        values = [r["coefficients"][key] for r in by_seed.values() if r.get("coefficients") and r["coefficients"].get(key) is not None]
        out[key] = sum(values) / len(values) if values else None
    stds = [sum(layer["res_token_std"] for layer in r["coefficients"]["layers"]) / len(r["coefficients"]["layers"])
            for r in by_seed.values() if r.get("coefficients")]
    out["res_token_std"] = sum(stds) / len(stds) if stds else None
    return out


def fmt(value, spec: str) -> str:
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="mhc-h1")
    args = parser.parse_args()
    runs = load_runs(args.dirs)
    bench = {k: v for k, v in runs.items() if k.startswith("bench-")}
    runs = {k: v for k, v in runs.items() if not k.startswith("bench-")}
    base = runs[args.baseline]
    rows = {}
    for variant, by_seed in runs.items():
        losses = [r["final_val_loss"] for r in by_seed.values()]
        m, s = mean_std(losses)
        tok = [r["tok_s"] for r in by_seed.values() if r.get("tok_s")]
        rows[variant] = {"n": len(losses), "loss": m, "loss_std": s, "tok_s": sum(tok) / len(tok) if tok else None,
                         "hc_params": next(iter(by_seed.values()))["hc_params"],
                         "paired": None if variant == args.baseline else paired(by_seed, base), **probe(by_seed)}

    print(f"Paired differences to {args.baseline} (negative = better); SE over seeds; * = exists (|mean| > 2 SE, all seeds agree)\n")
    print("| variant | seeds | val loss | Δ vs baseline | t | better seeds | tok/s | copy cosine | depth spread | res token std |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for variant, row in sorted(rows.items(), key=lambda kv: kv[1]["loss"]):
        p = row["paired"]
        delta = "–" if p is None else f"{p['mean']:+.4f} ± {fmt(p['se'], '.4f')}{' *' if p['exists'] else ''}"
        t = "–" if p is None else fmt(p["t"], "+.1f")
        better = "–" if p is None else f"{p['better']}/{len(p['diffs'])}"
        print(f"| {variant} | {row['n']} | {row['loss']:.4f} ± {fmt(row['loss_std'], '.4f')} | {delta} | {t} | {better} | "
              f"{fmt(row['tok_s'], ',.0f')} | {fmt(row['stream_copy_cosine'], '.3f')} | {fmt(row['depth_head_spread'], '.3f')} | "
              f"{fmt(row['res_token_std'], '.3f')} |")

    if bench:
        print("\nBenchmarks (150 steps; first step includes XLA compilation)\n")
        print("| run | first step (s) | tok/s | val loss |")
        print("|---|---|---|---|")
        for name, by_seed in sorted(bench.items()):
            r = by_seed[0]
            first = json.loads((r["dir"] / "metrics.jsonl").read_text().splitlines()[0])["elapsed"]
            print(f"| {name} | {first:.0f} | {fmt(r.get('tok_s'), ',.0f')} | {r['final_val_loss']:.3f} |")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for offset, (predictor, points) in zip((-0.04, 0.0, 0.04), CURVES.items()):
        hs, means, ses = [], [], []
        for h, variant in points.items():
            if variant == args.baseline and args.baseline in runs:
                hs.append(h), means.append(0.0), ses.append(0.0)
            elif variant in rows and rows[variant]["paired"]:
                p = rows[variant]["paired"]
                hs.append(h), means.append(p["mean"]), ses.append(0.0 if math.isnan(p["se"]) else p["se"])
        if len(hs) > 1:
            ax.errorbar([h * 2**offset for h in hs], means, yerr=ses, marker="o", capsize=3, label=predictor)
    for variant, marker in (("d-joint-h4-global", "s"), ("d-joint-h4-static", "D"), ("mhc-n8", "^")):
        if variant in rows and rows[variant]["paired"]:
            p = rows[variant]["paired"]
            ax.errorbar([4 if "joint" in variant else 1], [p["mean"]], yerr=[0.0 if math.isnan(p["se"]) else p["se"]],
                        marker=marker, linestyle="none", capsize=3, label=variant)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlabel("connection heads h (channel groups)")
    ax.set_ylabel(f"val loss − {args.baseline} (paired, ± SE)")
    if "residual" in rows and rows["residual"]["paired"]:
        ax.set_title(f"residual: {rows['residual']['paired']['mean']:+.3f} (off the chart)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{args.out}_hcurve.png", dpi=130)
    Path(f"{args.out}_analysis.json").write_text(json.dumps(rows, indent=1, default=str))
    print(f"\nwrote {args.out}_hcurve.png and {args.out}_analysis.json")


if __name__ == "__main__":
    main()
