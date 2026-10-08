# d384 learning-rate grid on TPU v6e, to compare every variant at its own best LR and to test where the heads' LR tolerance comes from.
#
# Why: at d384 the heads' gain over mHC h1 depends on the LR (exp10: zero at 0.5x, equal to the parameter-matched control at 1x,
# 0.2 ahead of both at 2x, where every variant is badly degraded: ~4.9-5.2 vs ~4.1). Two LR points per variant on 2 seeds cannot
# say whether heads move the optimum, widen the stable range, or only fail less badly once everything fails. A sqrt(2) grid with
# 3 seeds per point can: 1.5e-3, 2.1e-3, 3e-3, 4.2e-3, 6e-3 for mhc-h1, global h4, ctrl-mlp1216 (its parameter match) and local h4
# (mHC h1's exact parameters), plus the residual. The 3e-3 points (seeds 0-2) and 6e-3 points (seeds 0-1) are exp11's runs on the
# same chip and settings, so they are not repeated here; 6e-3 seed 2 is added.
#
# Mechanism test (H7b): is the tolerance just slower connection dynamics? --conn-lr-mult scales the LR of every connection parameter
# (phi, biases, alpha). mHC h1 with its connection at 0.25x at 3e-3 / 4.2e-3 / 6e-3: if it degrades like global h4 (or less) at
# 6e-3, the heads' tolerance can be had without heads; if it breaks like mHC h1, the head split itself matters. Reverse check: global
# h4 with its connection at 4x at 6e-3.
#
# Settings as exp11 (v6e-1, micro-batch 16, evals on the same 819k tokens via 10 / 50 batches of 16). Never fused accumulation on v6e.
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}
LR_TAG = {"1.5e-3": "-lrhalf", "2.1e-3": "-lr0.7x", "4.2e-3": "-lr1.4x", "6e-3": "-lr2x"}  # exp10's / exp07's tags where they exist
SEEDS = [0, 1, 2]


def job(name: str, seed: int, lr: str, extra: tuple = (), tag: str = "") -> dict:
    expected = 500 if name == "residual" else 1500
    return {"name": f"v6e-{name}{tag}{LR_TAG.get(lr, '')}-s{seed}", "args": [*D384, "--seed", str(seed), "--lr", lr,
            *VARIANTS[name], *extra], "expected_seconds": expected, "timeout": 3 * 3600}


SWEEP = ["mhc-h1", "a-global-h4", "ctrl-mlp1216", "a-local-h4"]
JOBS = (
    [job(name, seed, lr) for seed in SEEDS for lr in ["1.5e-3", "2.1e-3", "4.2e-3"] for name in SWEEP]
    + [job(name, 2, "6e-3") for name in SWEEP]
    + [job("mhc-h1", seed, lr, ("--conn-lr-mult", "0.25"), "-conn0.25") for seed in SEEDS for lr in ["3e-3", "4.2e-3", "6e-3"]]
    + [job("a-global-h4", seed, "6e-3", ("--conn-lr-mult", "4"), "-conn4") for seed in SEEDS]
    + [job("residual", seed, lr) for seed in SEEDS for lr in ["1.5e-3", "2.1e-3", "4.2e-3"]]
)
