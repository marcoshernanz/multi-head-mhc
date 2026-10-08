# d768 learning-rate sweep. exp07 found global h4 ahead of its parameter-matched control at d768 (−0.034, 2/2 seeds), but at an
# untuned LR (1.5e-3) at which mhc-h1 learned no faster per token than at d384. At d384 global heads were more robust to a too
# high LR (2x: −0.21, 4x: −0.79 vs mhc-h1), so the d768 gap could be an LR effect.
# Queue revised on 2026-09-30 after exp10 (d384): at 0.5x LR the heads' gain vanished (+0.002 vs mhc-h1, +0.013 vs the control,
# 2/2), and at 2x the control broke like mhc-h1. So the decisive side is the LOW one: here 7.5e-4 (0.5x) with two seeds first,
# then 1.06e-3 (0.7x) and 2.1e-3 (1.4x); 3e-3 (2x) is dropped (at d384 2x broke every variant). 1.5e-3 is in exp07.
# Local h4 (exactly mhc-h1's parameters) is added: the cleanest structural test at this scale. Residual gives mHC's own gain.
# Everything else as in exp07 (6000 steps, micro-batch 4 with --micro-sync 1, eval every 500 steps on the 819k tokens).
# One session of 3 waves x 8 chips (~2.5 h per d768 run); the deadline guard cuts whatever does not fit (the residual runs last).
# 2026-10-03: run on Google Cloud TPU v5e (v5litepod-1 spot VMs, cloud/controller.py) instead of Kaggle, whose TPU queue took up to 11 h.
# Same chip and the same software as Kaggle's v5e image (torch 2.8.0, torch_xla 2.8.0, libtpu 0.0.17): a replay of exp04's mhc-h1-s0
# on GCP v5e matched the Kaggle log bit for bit, so these runs pair exactly with exp07's. The full-length repeat of exp07's
# d768-mhc-h1-s0 below checks that over a whole run. Added: 2.1e-3 seed 1 (the budget allows two seeds at every LR).
# The Kaggle push of this batch (2026-10-03, still queued) is made redundant by this.
# 2026-10-03 21:51 UTC (notes/plan.md), after its first result (mHC h1 at 7.5e-4: 3.938 on seed 0, 0.21 better than at 1.5e-3): 5.3e-4
# added for every variant (seeds 0-1), ahead of 2.1e-3 in the queue, so that every variant's best LR is bracketed.
SLUG = "mhmhc-exp09-d768-lr"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20
DEADLINE_SECONDS = int(7.4 * 3600)  # hard stop 8.4 h (Kaggle's TPU limit is 9 h): wave 3 fits if waves 1-2 end by 5.6 h; needs the 10-03 reset
D768 = ["--steps", "6000", "--warmup", "200", "--log-every", "25", "--hc-layout", "streams", "--dim", "768", "--layers", "12",
        "--heads", "12", "--mlp-dim", "2048", "--micro-batch", "4", "--micro-sync", "1", "--eval-every", "500",
        "--eval-batches", "40", "--final-eval-batches", "200"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp2240": ["--mlp-dim", "2240"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}
SWEEP = ["mhc-h1", "a-global-h4", "ctrl-mlp2240", "a-local-h4"]


def job(name: str, lr: str, seed: int, expected: int = 9400) -> dict:
    tag = "" if lr == "1.5e-3" else f"-lr{lr}"  # exp07's runs (1.5e-3) have no LR tag
    return {"name": f"d768-{name}{tag}-s{seed}", "args": [*D768, "--lr", lr, "--seed", str(seed), *VARIANTS[name]],
            "expected_seconds": expected, "timeout": 8 * 3600}


# Two d384 runs (~1 h each) in wave 3, before the residual runs: local h4 at 2x LR. exp10 showed the LR tolerance is not the
# parameters' (the MLP control broke like mhc-h1 at 2x); local heads have exactly mhc-h1's parameters and also damp gradient spikes,
# so they separate "head structure" from "bigger predictor". Same settings as exp07's d384 lr2x runs.
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams"]
D384_JOBS = [{"name": f"a-local-h4-lr2x-s{seed}",
              "args": [*D384, "--seed", str(seed), "--lr", "6e-3", "--hc-heads", "4", "--hc-predictor", "local"],
              "expected_seconds": 3700, "timeout": 4200} for seed in [0, 1]]

JOBS = (
    [job(name, "7.5e-4", seed) for seed in [0, 1] for name in SWEEP]
    + [job(name, "1.06e-3", 0) for name in SWEEP]
    + [job(name, "5.3e-4", 0) for name in SWEEP]
    + [job(name, "1.06e-3", 1) for name in SWEEP]
    + [job(name, "5.3e-4", 1) for name in SWEEP]
    + [job("a-local-h4", "1.5e-3", seed) for seed in [0, 1]]
    + [job(name, "2.1e-3", 0) for name in SWEEP]
    + D384_JOBS
    + [job("residual", "1.5e-3", seed, expected=6000) for seed in [0, 1]]
    + [job(name, "2.1e-3", 1) for name in SWEEP]
    + [{**job("mhc-h1", "1.5e-3", 0), "name": "repro-d768-mhc-h1-s0"}]  # = exp07's d768-mhc-h1-s0, to compare bit for bit
)
