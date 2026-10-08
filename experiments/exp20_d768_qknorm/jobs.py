# d768 with QK-norm on TPU v6e: does what exp19 found with QK-norm at d384 hold at 112M? Exploratory.
#
# Why (2026-10-05, designed after reading most of exp19): exp19's pre-registered rule S6 runs d768 only if the heads' LR tolerance
# survives QK-norm, and it does not (with QK-norm heads are more LR-sensitive than mHC h1, not less). This batch asks two other
# questions, both raised by exp19's data, so it is exploratory and is reported as such:
#   (1) Gradient norm. With QK-norm at d384, mHC h1 and its control cross the clipping threshold (1.0) on 0.1-1.3% of the steps in
#       the last 80% of training at every LR from 2.1e-3 to 8.5e-3, global h4 on none up to 8.5e-3, local h4 and the residual on
#       almost none; without QK-norm the same at 2.1e-3 and 4.2e-3. HC-family papers, mHC's included, argue stability from exactly
#       this kind of gradient-norm curve. Exp15's d768 runs without QK-norm show no such difference near their optimum (every
#       variant under 1% of logged steps), but they log every 25th step only. Here every step is logged.
#   (2) The main result. Exp19 shows the parameter-matched null holds with QK-norm at d384; this checks it at d768.
# Settings as exp15 (d768, 12 layers, micro-batch 16, the same validation tokens), plus --qk-norm 1, per-step logs and the exp19
# diagnostics every 250 steps. Grid: sqrt(2) steps from 5.3e-4 (exp15's lowest) to 2.1e-3 (about 2.8x exp15's optimum, 7.5e-4,
# the multiple of exp19's 6e-3 over its 2.1e-3). Read-outs in notes/plan.md (2026-10-05), fixed before any run of this batch.
D768 = ["--steps", "6000", "--warmup", "200", "--eval-every", "500", "--log-every", "25", "--hc-layout", "streams",
        "--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50",
        "--qk-norm", "1", "--diag-every", "250", "--log-steps", "1"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp2240": ["--mlp-dim", "2240"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}
GRID = ["5.3e-4", "7.5e-4", "1.06e-3", "1.5e-3", "2.1e-3"]
SEEDS = [0, 1, 2]


def job(name: str, lr: str, seed: int) -> dict:
    expected = 1200 if name == "residual" else 2400
    return {"name": f"v6e-d768-{name}-qk-lr{lr}-s{seed}", "args": [*D768, "--seed", str(seed), "--lr", lr, *VARIANTS[name]],
            "expected_seconds": expected, "timeout": 4 * 3600}


JOBS = [job(name, lr, seed) for seed in SEEDS for lr in GRID for name in VARIANTS]
