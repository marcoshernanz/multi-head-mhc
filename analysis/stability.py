"""exp19's read-outs (notes/plan.md, 2026-10-05): LR sensitivity, the comparisons at each LR, spikes and instability diagnostics,
with and without QK-norm, at d384 on TPU v6e; with --width 768, exp20's (d768, QK-norm) on the same terms.

Usage: uv run python analysis/stability.py [--width 768] [--json figures/stability.json]
Reads, at d384, results/exp11_v6e_check, exp12_d384_lr_grid and exp19_stability; at d768, exp11_v6e_check and exp15_d768_v6e (no QK-norm) and
exp20_d768_qknorm (runs grouped by their run.json, not their names; the earlier batches' runs have no per-step logs or diagnostics,
exp19's replays of them do).
- LR sensitivity (Wortsman et al. 2023): S = mean over the grid of min(final val loss, l0) - l*, l0 = ln(vocab), l* the variant's
  best mean final loss on the grid; a diverged run counts as l0. Per seed, paired across variants.
- Spike score (OLMo 2, arXiv 2501.00656 §3.2): the percentage of steps at least 7 standard deviations from the mean of the previous
  1000 (mean and SD of that window; the first 1000 steps are not scored).
- Loss spikes per 10k steps (Qwen3.8-Next, arXiv 2608.30320 §3.3): steps whose loss exceeds the median of the 201 steps centred on
  them by more than 0.1. Clip crossings: the share of steps whose grad norm (logged before clipping) exceeds the threshold, 1.0;
  late clip crossings: the same over the last 80% of training (exp20's E1; from every 25th step for runs without per-step logs).
The edge: the lowest grid LR above the best whose mean final loss is more than 0.1 above the best (where the cliff is).
An effect "exists" if |mean| > 2 SE and every seed agrees in sign (notes/plan.md).
"""

import argparse
import itertools
import json
import math
import statistics as st
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = {  # width -> batches, LR grid, the feed-forward width of every variant but the control, and the control's
    384: (["exp11_v6e_check", "exp12_d384_lr_grid", "exp19_stability"], [1.5e-3, 2.1e-3, 3e-3, 4.2e-3, 6e-3, 8.5e-3, 1.2e-2, 1.7e-2, 2.4e-2],
          1024, 1216),
    768: (["exp11_v6e_check", "exp15_d768_v6e", "exp20_d768_qknorm"], [5.3e-4, 7.5e-4, 1.06e-3, 1.5e-3, 2.1e-3], 2048, 2240),
}
WIDTH = 384
BATCHES, GRID, MLP, CONTROL_MLP = WIDTHS[WIDTH]
VARIANTS = ["mhc", "global-h4", "local-h4", "control", "residual"]
NAMES = {"mhc": "mHC", "global-h4": "global h4", "local-h4": "local h4", "control": f"control (MLP {CONTROL_MLP})", "residual": "residual"}
SEEDS = [0, 1, 2]
L0 = math.log(16384)
SPIKE_WINDOW, SPIKE_SIGMAS = 1000, 7.0
MEDIAN_WINDOW, MEDIAN_JUMP, CLIP = 201, 0.1, 1.0
EDGE = 0.1


def set_width(width: int) -> None:
    global WIDTH, BATCHES, GRID, MLP, CONTROL_MLP
    WIDTH = width
    BATCHES, GRID, MLP, CONTROL_MLP = WIDTHS[width]
    NAMES["control"] = f"control (MLP {CONTROL_MLP})"


def variant_of(config: dict) -> str | None:
    if config["conn"] == "residual":
        return "residual" if config["mlp_dim"] == MLP else None
    if config.get("hc_head_parts", "pre,post,res") != "pre,post,res" or config.get("hc_joint") or config["hc_mult"] != 4:
        return None
    if config["hc_heads"] == 1 and config["hc_predictor"] == "global":
        return {MLP: "mhc", CONTROL_MLP: "control"}.get(config["mlp_dim"])
    if config["hc_heads"] == 4 and config["mlp_dim"] == MLP:
        return {"global": "global-h4", "local": "local-h4"}.get(config["hc_predictor"])
    return None


def load() -> dict:
    """(qk, variant, lr, seed) -> run; a later batch's run of the same setting replaces an earlier one only if it has diagnostics."""
    runs = {}
    for batch in BATCHES:
        for summary_path in sorted((ROOT / "results" / batch).glob("runs/*/summary.json")):
            d = summary_path.parent
            if d.name.startswith("chk-"):
                continue
            info = json.loads((d / "run.json").read_text())
            a, c = info["args"], info["config"]
            if c["dim"] != WIDTH or "v6e" not in info["device"] or a["steps"] != "6000" or float(a.get("conn_lr_mult", 1)) != 1:
                continue
            v = variant_of(c)
            if v is None:
                continue
            summary = json.loads(summary_path.read_text())
            run = {"dir": d, "final": summary["final_val_loss"], "diverged": summary["diverged"], "steps_done": summary["steps_done"],
                   "loss": [], "grad_norm": [], "logged": [], "diag": []}
            for line in (d / "metrics.jsonl").read_text().splitlines():
                rec = json.loads(line)
                if "losses" in rec:
                    run["loss"] += rec["losses"]
                    run["grad_norm"] += rec["grad_norms"]
                if "loss" in rec:
                    run["logged"].append((rec["step"], rec["loss"], rec["grad_norm"]))
                if "diag" in rec:
                    run["diag"].append((rec["step"], rec["diag"]))
            key = (bool(c.get("qk_norm", False)), v, float(a["lr"]), int(a["seed"]))
            if key in runs and not run["diag"]:
                continue
            if key in runs:
                run["replaces"] = runs[key]
            runs[key] = run
    return runs


def capped(run: dict) -> float:
    f = run["final"]
    return L0 if run["diverged"] or f is None or not math.isfinite(f) else min(f, L0)


def stats(diffs: list[float]) -> dict:
    m = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
    agree = all(x < 0 for x in diffs) or all(x > 0 for x in diffs)
    return {"mean": m, "se": se, "better": sum(x < 0 for x in diffs), "n": len(diffs),
            "exists": len(diffs) > 1 and abs(m) > 2 * se and agree, "diffs": diffs}


def compare(runs: dict, qk: bool, a: str, lr_a: float, b: str, lr_b: float) -> dict:
    """a at lr_a − b at lr_b over SEEDS: paired by seed at one LR, two-sample SE when the LRs differ (notes/plan.md)."""
    xa = [capped(runs[(qk, a, lr_a, s)]) for s in SEEDS]
    xb = [capped(runs[(qk, b, lr_b, s)]) for s in SEEDS]
    s = stats([x - y for x, y in zip(xa, xb)])
    if lr_a != lr_b:
        s["se"] = math.sqrt(st.variance(xa) / len(xa) + st.variance(xb) / len(xb))
        s["exists"] = abs(s["mean"]) > 2 * s["se"] and s["better"] in (0, s["n"])
        s["two_sample"] = True
    return s


def best_lrs(runs: dict, qk: bool) -> dict:
    """Each variant's best LR: the lowest mean final loss over SEEDS on the grid (LRs without every seed are skipped)."""
    best = {}
    for v in VARIANTS:
        have = [lr for lr in GRID if all((qk, v, lr, s) in runs for s in SEEDS)]
        if have:
            best[v] = min(have, key=lambda lr: st.mean(capped(runs[(qk, v, lr, s)]) for s in SEEDS))
    return best


def sensitivity(values: dict, v: str) -> dict:
    """S per seed from values[(v, lr, seed)], capped final losses on the full grid; l* the variant's best mean."""
    lstar = min(st.mean(values[(v, lr, s)] for s in SEEDS) for lr in GRID)
    return {s: st.mean(values[(v, lr, s)] for lr in GRID) - lstar for s in SEEDS}


def s3_verdict(with_qk: dict, without_qk: dict) -> str:
    """S3's rule (notes/plan.md, 2026-10-05 11:47 UTC) from the with-QK dS stats and the without-QK dS means, per pair. "Neither is
    an effect" is read as "dS < 0 is an effect for neither": a reversal counts as removal ("or of opposite sign")."""
    pairs = ["global-h4 - mhc", "local-h4 - mhc"]
    if any(with_qk[p]["exists"] and with_qk[p]["mean"] < 0 for p in pairs):
        return "survives"
    if all(with_qk[p]["mean"] * without_qk[p] < 0 or abs(with_qk[p]["mean"]) <= abs(without_qk[p]) / 3 for p in pairs):
        return "removes"
    return "partial"


def fmt(s: dict | None, digits: int = 3) -> str:
    if s is None:
        return "–"
    return f"{s['mean']:+.{digits}f} ± {s['se']:.{digits}f} ({s['better']}/{s['n']}){' *' if s['exists'] else ''}"


def spike_score(values: list[float]) -> float | None:
    """Percentage of steps at least SPIKE_SIGMAS standard deviations from the mean of the previous SPIKE_WINDOW steps."""
    if len(values) <= SPIKE_WINDOW:
        return None
    x = np.asarray(values, dtype=np.float64)
    windows = sliding_window_view(x[:-1], SPIKE_WINDOW)  # windows[i] = x[i : i + W], the W values before x[i + W]
    mean, sd = windows.mean(1), windows.std(1)
    return 100.0 * float(np.mean(np.abs(x[SPIKE_WINDOW:] - mean) >= SPIKE_SIGMAS * sd))


def median_spikes(values: list[float]) -> float | None:
    """Steps per 10k whose value exceeds the median of the MEDIAN_WINDOW steps centred on them by more than MEDIAN_JUMP."""
    if len(values) < MEDIAN_WINDOW:
        return None
    x = np.asarray(values, dtype=np.float64)
    half = MEDIAN_WINDOW // 2
    med = np.median(sliding_window_view(x, MEDIAN_WINDOW), axis=1)
    return 1e4 * float(np.mean(x[half:len(x) - half] - med > MEDIAN_JUMP))


def late_max_grad(run: dict) -> float | None:
    """Largest grad norm in the last 80% of training (exp12's measure), from the per-step log if there is one."""
    if run["grad_norm"]:
        g = run["grad_norm"]
        return max(g[len(g) // 5:]) if g else None
    logged = [g for step, _, g in run["logged"] if step >= 1200]
    return max(logged) if logged else None


def late_clipped(run: dict) -> float | None:
    """Percentage of steps in the last 80% of training whose grad norm exceeds CLIP (every 25th step if there is no per-step log)."""
    if run["grad_norm"]:
        g = np.asarray(run["grad_norm"])
        return 100.0 * float(np.mean(g[len(g) // 5:] > CLIP))
    if not run["logged"]:
        return None
    last = run["logged"][-1][0]
    logged = [g for step, _, g in run["logged"] if step >= 0.2 * last]
    return 100.0 * float(np.mean(np.asarray(logged) > CLIP)) if logged else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--width", type=int, default=384, choices=sorted(WIDTHS))
    args = parser.parse_args()
    set_width(args.width)
    runs = load()
    out = {"grid": GRID, "recipes": {}}

    # S1: replays must repeat the earlier runs' logged losses bit for bit
    replays = [(k, r) for k, r in runs.items() if "replaces" in r]
    same = [k for k, r in replays if [x[1] for x in r["logged"]] == [x[1] for x in r["replaces"]["logged"]]
            and r["final"] == r["replaces"]["final"]]
    print(f"S1 bit-identity: {len(same)} of {len(replays)} replays repeat the earlier run's logged losses and final loss exactly")
    for k, r in replays:
        if k not in same:
            a, b = [x[1] for x in r["logged"]], [x[1] for x in r["replaces"]["logged"]]
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            print(f"  differs: {k} first at logged point {first}, final {r['final']} vs {r['replaces']['final']}")
    out["bit_identity"] = {"replays": len(replays), "identical": len(same)}

    for qk in (False, True):
        recipe = "qk-norm" if qk else "no qk-norm"
        res = {}
        print(f"\n===== {recipe}: final val loss, mean over seeds (seeds); l0 = {L0:.3f} for diverged runs")
        means = {}
        print(f"{'variant':20s} " + " ".join(f"{lr:>9.2g}" for lr in GRID))
        for v in VARIANTS:
            row = []
            for lr in GRID:
                vals = [capped(runs[(qk, v, lr, s)]) for s in SEEDS if (qk, v, lr, s) in runs]
                if len(vals) == len(SEEDS):
                    means[(v, lr)] = st.mean(vals)
                row.append(f"{st.mean(vals):9.4f}" if vals else f"{'–':>9s}")
                if vals and len(vals) < len(SEEDS):
                    row[-1] = row[-1][:-1] + "?"
            print(f"{NAMES[v]:20s} " + " ".join(row))
        # best LR, usable range
        best = {}
        for v in VARIANTS:
            have = [lr for lr in GRID if (v, lr) in means]
            if not have:
                continue
            b = min(have, key=lambda lr: means[(v, lr)])
            best[v] = b
            usable = [lr for lr in have if means[(v, lr)] <= means[(v, b)] + 0.02]
            print(f"  {NAMES[v]:20s} best {b:g} ({means[(v, b)]:.4f}); within 0.02: {min(usable):g}-{max(usable):g}"
                  f"{' (grid incomplete)' if len(have) < len(GRID) else ''}")
        res["best_lr"] = best
        res["means"] = {f"{v}@{lr:g}": m for (v, lr), m in means.items()}
        # S4's edge: the lowest grid LR above the best whose mean final loss is more than EDGE above the best (where the cliff is)
        res["edge"] = {}
        for v, b in best.items():
            above = [lr for lr in GRID if lr > b and (v, lr) in means and means[(v, lr)] > means[(v, b)] + EDGE]
            res["edge"][v] = min(above) if above else None
        print("  edge (lowest LR more than 0.1 above the best):", {NAMES[v]: (f"{e:g}" if e else "none on grid") for v, e in res["edge"].items()})

        # S3: LR sensitivity on the full grid
        sens = {}
        for v in VARIANTS:
            if not all((qk, v, lr, s) in runs for lr in GRID for s in SEEDS):
                continue
            lstar = min(st.mean(capped(runs[(qk, v, lr, s)]) for s in SEEDS) for lr in GRID)
            sens[v] = {s: st.mean(capped(runs[(qk, v, lr, s)]) for lr in GRID) - lstar for s in SEEDS}
        print("  LR sensitivity S (mean over seeds):", {NAMES[v]: round(st.mean(x.values()), 4) for v, x in sens.items()})
        pairs = [("global-h4", "mhc"), ("local-h4", "mhc"), ("global-h4", "control"), ("control", "mhc"), ("mhc", "residual")]
        res["sensitivity"] = {v: x for v, x in sens.items()}
        res["dS"] = {}
        res["dS_complete_seeds"] = {}
        for a, b in pairs:
            if a in sens and b in sens:
                s = stats([sens[a][k] - sens[b][k] for k in SEEDS])
                res["dS"][f"{a} - {b}"] = s
                print(f"  dS {NAMES[a]} − {NAMES[b]}: {fmt(s, 4)}")
            elif a in best and b in best:  # a grid with runs missing: dS on the seeds whose grids are complete for both
                done = [s for s in SEEDS if all((qk, v, lr, s) in runs for v in (a, b) for lr in GRID)]
                lstar = {v: means[(v, best[v])] for v in (a, b)}
                d = {s: st.mean(capped(runs[(qk, a, lr, s)]) - lstar[a] - capped(runs[(qk, b, lr, s)]) + lstar[b] for lr in GRID)
                     for s in done}
                res["dS_complete_seeds"][f"{a} - {b}"] = d
                print(f"  dS {NAMES[a]} − {NAMES[b]} on the seeds with complete grids: " +
                      (", ".join(f"s{s} {x:+.4f}" for s, x in d.items()) or "none"))
        # exp20's E3 (descriptive): S over the lowest five grid LRs, the same 4x span as d768's grid
        if len(GRID) > 5:
            low = GRID[:5]
            sub = {}
            for v in VARIANTS:
                if all((qk, v, lr, s) in runs for lr in low for s in SEEDS):
                    lstar = min(st.mean(capped(runs[(qk, v, lr, s)]) for s in SEEDS) for lr in low)
                    sub[v] = {s: st.mean(capped(runs[(qk, v, lr, s)]) for lr in low) - lstar for s in SEEDS}
            res["dS_low5"] = {}
            for a, b in pairs[:3]:
                if a in sub and b in sub:
                    s = stats([sub[a][k] - sub[b][k] for k in SEEDS])
                    res["dS_low5"][f"{a} - {b}"] = s
                    print(f"  dS over {low[0]:g}-{low[-1]:g} {NAMES[a]} − {NAMES[b]}: {fmt(s, 4)}")

        # per-LR paired differences
        print("  paired differences at each LR (negative = the first is better):")
        res["paired"] = {}
        for a, b in pairs[:4]:
            cells = []
            for lr in GRID:
                shared = [s for s in SEEDS if (qk, a, lr, s) in runs and (qk, b, lr, s) in runs]
                if len(shared) >= 2:
                    s = stats([capped(runs[(qk, a, lr, k)]) - capped(runs[(qk, b, lr, k)]) for k in shared])
                    res["paired"][f"{a} - {b} @ {lr:g}"] = s
                    cells.append(f"{lr:g}: {fmt(s)}")
            print(f"    {NAMES[a]} − {NAMES[b]}: " + "; ".join(cells))

        # S4: parameter-matched comparisons at each variant's own best LR
        res["at_best"] = {}
        for a, b in [("local-h4", "mhc"), ("global-h4", "control"), ("global-h4", "mhc"), ("control", "mhc"), ("mhc", "residual")]:
            if a in best and b in best:
                s = compare(runs, qk, a, best[a], b, best[b])
                res["at_best"][f"{a} - {b}"] = s
                print(f"  at own best LR: {NAMES[a]} ({best[a]:g}) − {NAMES[b]} ({best[b]:g}): {fmt(s, 4)}"
                      f"{' two-sample' if best[a] != best[b] else ' paired'}")

        # S5: spikes, and S2: diagnostics
        print("  spikes, mean over seeds: OLMo 2 spike score of loss % / of grad norm % ; Qwen loss spikes per 10k ; "
              "% of steps clipped ; largest late grad norm ; % of late steps clipped")
        res["spikes"] = {}
        for lr in GRID:
            cells = []
            for v in VARIANTS:
                rs = [runs[(qk, v, lr, s)] for s in SEEDS if (qk, v, lr, s) in runs]
                if not rs:
                    continue
                per = {"loss_spike": [spike_score(r["loss"]) for r in rs], "grad_spike": [spike_score(r["grad_norm"]) for r in rs],
                       "qwen_spikes": [median_spikes(r["loss"]) for r in rs],
                       "clipped": [100.0 * float(np.mean(np.asarray(r["grad_norm"]) > CLIP)) if r["grad_norm"] else None for r in rs],
                       "late_max_grad": [late_max_grad(r) for r in rs], "late_clipped": [late_clipped(r) for r in rs]}
                entry = {k: (st.mean(x) if (x := [y for y in vals if y is not None]) else None) for k, vals in per.items()}
                entry["per_seed"] = per
                entry["n"] = len(rs)
                res["spikes"][f"{v}@{lr:g}"] = entry
                f = lambda k, spec: format(entry[k], spec) if entry[k] is not None else "–"
                cells.append(f"{NAMES[v].split(' (')[0]} {f('loss_spike', '.2f')}/{f('grad_spike', '.2f')}; {f('qwen_spikes', '.1f')}; "
                             f"{f('clipped', '.1f')}; {f('late_max_grad', '.3g')}; {f('late_clipped', '.2f')}")
            if cells:
                print(f"    {lr:g}: " + " | ".join(cells))
        # exp20's E1: late clip crossings, paired by seed at each LR
        print("  late clip crossings (% of steps in the last 80% above 1.0), paired by seed (negative = the first crosses less):")
        res["late_clipped"] = {}
        for a, b in [("global-h4", "mhc"), ("local-h4", "mhc"), ("global-h4", "control"), ("residual", "mhc")]:
            cells = []
            for lr in GRID:
                shared = [s for s in SEEDS if (qk, a, lr, s) in runs and (qk, b, lr, s) in runs]
                xs = [(late_clipped(runs[(qk, a, lr, k)]), late_clipped(runs[(qk, b, lr, k)])) for k in shared]
                xs = [(x, y) for x, y in xs if x is not None and y is not None]
                if len(xs) >= 2:
                    s = stats([x - y for x, y in xs])
                    res["late_clipped"][f"{a} - {b} @ {lr:g}"] = s
                    cells.append(f"{lr:g}: {fmt(s, 2)}")
            print(f"    {NAMES[a]} − {NAMES[b]}: " + "; ".join(cells))
        print("  diagnostics (mean over seeds) at steps 500 / 1000 / last: largest attention logit (max over layers); "
              "lowest layer attention entropy; z^2")
        res["diag"] = {}
        for lr in GRID:
            for v in VARIANTS:
                rs = [runs[(qk, v, lr, s)] for s in SEEDS if (qk, v, lr, s) in runs and runs[(qk, v, lr, s)]["diag"]]
                if not rs:
                    continue
                row = {}
                for label, pick in (("500", 500), ("1000", 1000), ("last", None)):
                    vals = []
                    for r in rs:
                        d = dict(r["diag"])
                        step = pick if pick in d else max(d)
                        if pick is not None and pick not in d:
                            continue
                        vals.append((max(d[step]["attn_max_logit"]), min(d[step]["attn_entropy"]), d[step]["z2"]))
                    if vals:
                        row[label] = [st.mean(x[i] for x in vals) for i in range(3)]
                res["diag"][f"{v}@{lr:g}"] = row
                print(f"    {lr:g} {NAMES[v]:20s} " + "  ".join(f"{k}: {x[0]:7.1f} {x[1]:.2f} {x[2]:6.1f}" for k, x in row.items()))
        out["recipes"][recipe] = res

    # S3's verdict; with runs missing without QK-norm (exp19: 12 at 8.5e-3 and above could not be run), for every outcome of them,
    # each diverged or finite at 4.5, 5.5 or 7.0
    with_qk = out["recipes"]["qk-norm"]["dS"]
    if all(p in with_qk for p in ("global-h4 - mhc", "local-h4 - mhc")):
        missing = [(v, lr, s) for v in VARIANTS for lr in GRID for s in SEEDS if (False, v, lr, s) not in runs]
        known = {(v, lr, s): capped(r) for (q, v, lr, s), r in runs.items() if not q}
        verdicts = {}
        if len(missing) <= 16:
            for fill in (4.5, 5.5, 7.0) if missing else (None,):
                for outcome in itertools.product((L0, fill), repeat=len(missing)):
                    values = known | dict(zip(missing, outcome))
                    sens = {v: sensitivity(values, v) for v in ("global-h4", "local-h4", "mhc")}
                    without = {f"{a} - mhc": st.mean(sens[a][s] - sens["mhc"][s] for s in SEEDS) for a in ("global-h4", "local-h4")}
                    verdict = s3_verdict(with_qk, without)
                    verdicts[verdict] = verdicts.get(verdict, 0) + 1
        print(f"\nS3 verdict ({len(missing)} runs without QK-norm missing; every outcome of them): {verdicts}")
        out["s3_verdict"] = {"missing": [list(m) for m in missing], "verdicts": verdicts}

    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=str))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
