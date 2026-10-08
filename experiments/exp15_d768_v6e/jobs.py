# d768 learning-rate grid on TPU v6e: does global h4's lead over its parameter-matched control at d768 (exp07: −0.034, 2 seeds, at the
# untuned 1.5e-3) survive when every variant gets its own best LR? And do local heads (exactly mHC h1's parameters) gain anything there?
#
# Why on v6e, and why now (changed from notes/plan.md's "best LR from exp09, then 5 seeds on v6e"): exp09 runs the same LR grid on v5e
# (2 seeds, exact pairing with exp07), but a d768 run takes 2.6 h on v5e and spot capacity for 4-chip v5e VMs was not available, so
# exp09 ends around 03:00 UTC. On v6e a d768 run takes ~35 min (51.9k tok/s), so the grid itself runs here with 3 seeds, on one chip
# type, and exp09 becomes an independent second chip for the same curve. Same settings as exp11's d768 runs (micro-batch 16, the same
# 819k validation tokens); 1.5e-3 seeds 0-1 of mHC h1, global h4 and the control are exp11's `v6e-d768-*` runs.
#
# Decision rule (written before the runs): each variant's best LR = the lowest mean final val loss over seeds 0-2 on this grid. Then
# seeds 3-4 at each variant's best LR are added (a second list, JOBS_BEST, filled in once the grid is in), and the scale condition of
# notes/plan.md is read on the 5 seeds: met only if global h4 at its best beats the control at its best with an effect that exists
# (|mean| > 2 SE, all seeds agree; unpaired across LRs, so the SE is the two-sample one), or local h4 at its best beats mHC h1 at its
# best. Also read: global h4 − control and local h4 − mHC h1 paired at each LR (as at d384, exp12), and whether the d384 pattern (gain
# only at and above the tuned LR) repeats at d768. d1024 only if the scale condition is met.
# 2026-10-03 21:51 UTC (notes/plan.md), after one result (exp09: mHC h1 at 7.5e-4 is 0.21 better than at 1.5e-3 on seed 0): 5.3e-4
# added for every variant, and on downward in sqrt(2) steps while any variant's mean at the lowest LR beats the next one up, so
# every variant's best LR is bracketed. The queue now runs the three lowest LRs first (all seeds), then 1.5e-3 and 2.1e-3.
D768 = ["--steps", "6000", "--warmup", "200", "--eval-every", "500", "--log-every", "25", "--hc-layout", "streams",
        "--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50"]
VARIANTS = {
    "mhc-h1": [],
    "residual": ["--conn", "residual"],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp2240": ["--mlp-dim", "2240"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
}
IN_EXP11 = {(name, "1.5e-3", seed) for name in ["mhc-h1", "a-global-h4", "ctrl-mlp2240"] for seed in [0, 1]}


def job(name: str, lr: str, seed: int) -> dict:
    tag = "" if lr == "1.5e-3" else f"-lr{lr}"  # as exp07 / exp09 / exp11: 1.5e-3 has no LR tag
    return {"name": f"v6e-d768-{name}{tag}-s{seed}", "args": [*D768, "--seed", str(seed), "--lr", lr, *VARIANTS[name]],
            "expected_seconds": 2400, "timeout": 4 * 3600}


SWEPT = ["mhc-h1", "a-global-h4", "ctrl-mlp2240", "a-local-h4"]
JOBS_LOW = [job(name, lr, seed) for seed in [0, 1, 2] for lr in ["7.5e-4", "5.3e-4", "1.06e-3"] for name in SWEPT]
JOBS_HIGH = [job(name, lr, seed) for seed in [0, 1, 2] for lr in ["1.5e-3", "2.1e-3"] for name in SWEPT
             if (name, lr, seed) not in IN_EXP11]
# 2026-10-04 01:00 UTC, once the three lowest LRs had all seeds 0-2 (1.5e-3 and 2.1e-3 partly run, every mean there 0.1 or more
# worse): best LR by the rule above = mHC h1 7.5e-4 (3.9458), global h4 1.06e-3 (3.9422; 3.9429 at 7.5e-4), control 7.5e-4
# (3.9378), local h4 7.5e-4 (3.9435). No variant's mean at 5.3e-4 beats its mean at 7.5e-4, so 3.75e-4 is not added. Seeds 3-4 at
# those LRs go ahead of the rest of 1.5e-3 / 2.1e-3. Added beside the rule, not part of it: global h4 at 7.5e-4, seeds 3-4, so
# that all four variants also have 5 seeds at one LR and can be compared paired there (a secondary read-out; the decision is
# the rule's two-sample comparison at each variant's best LR).
BEST_LR = {"mhc-h1": "7.5e-4", "a-global-h4": "1.06e-3", "ctrl-mlp2240": "7.5e-4", "a-local-h4": "7.5e-4"}
JOBS_BEST = [job(name, BEST_LR[name], seed) for seed in [3, 4] for name in SWEPT]
JOBS_SAME_LR = [job("a-global-h4", "7.5e-4", seed) for seed in [3, 4]]
# 2026-10-04 04:00 UTC, after the rest of exp15 and exp09 were in: the residual at d768 had run only at 1.5e-3 (exp09, v5e), far
# from any tuned LR, so mHC's own gain at d768 was unknown. Context for the heads' gaps, not part of any decision: the residual at
# 5.3e-4 .. 1.5e-3, seeds 0-2 (short runs).
JOBS_RESIDUAL = [{**job("residual", lr, seed), "expected_seconds": 1200} for seed in [0, 1, 2]
                 for lr in ["7.5e-4", "1.06e-3", "5.3e-4", "1.5e-3"]]
JOBS = JOBS_LOW + JOBS_BEST + JOBS_SAME_LR + JOBS_HIGH + JOBS_RESIDUAL
