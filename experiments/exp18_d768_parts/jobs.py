# exp18: the pre-registered d768 follow-up of exp14 (notes/plan.md, 2026-10-04 02:15 UTC, written before any exp14 result): a per-head-blocks
# variant that beat the MLP line at its own parameter count at d384 (3e-3, v6e, an effect that exists, all 3 seeds below the line) is
# run at d768 on v6e at 7.5e-4 (the best LR of mHC h1 and the control in exp15), seeds 0-4, against exp15's mHC h1 and control at
# 7.5e-4, paired by seed, with the same rule: it counts only if its Δ to mHC h1 lies below the MLP line (mHC h1 -> ctrl-mlp2240,
# linear in parameters) by more than 2 SE of that Δ, with all 5 seeds below the line.
#
# VARIANTS holds the variant(s) that met the rule at d384 (filled in when exp14's 3e-3 runs were read; see the dated line below).
# Settings exactly as exp15 (micro-batch 16, evals on the same 819k tokens via 10 / 50 batches of 16).
D768 = ["--steps", "6000", "--warmup", "200", "--eval-every", "500", "--log-every", "25", "--hc-layout", "streams",
        "--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50", "--lr", "7.5e-4"]
PARTS = {"pre": "pre", "prepost": "pre,post"}
# 2026-10-04 05:50 UTC, exp14's 3e-3 runs complete (seeds 0-2): pre: excess −0.0186 against 2 SE 0.0221 (3/3 below the line), does
# not meet the rule; pre,post: excess −0.0155 against 2 SE 0.0089 (3/3 below the line), meets it. So pre,post runs here.
VARIANTS = ["prepost"]
JOBS = [{"name": f"v6e-d768-a-global-h4-{tag}-lr7.5e-4-s{seed}",
         "args": [*D768, "--seed", str(seed), "--hc-heads", "4", "--hc-head-parts", PARTS[tag]],
         "expected_seconds": 2400, "timeout": 4 * 3600}
        for seed in [0, 1, 2, 3, 4] for tag in VARIANTS]
