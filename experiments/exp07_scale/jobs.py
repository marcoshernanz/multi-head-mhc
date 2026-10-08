# The scale check again, now with the parameter control exp06 showed is needed, plus more seeds for H2 and H7 at 2x LR.
#
# exp06: d768 x micro-batch 4 failed at compile time: HBM out of memory (22.2 GB needed for mhc-h1, 20.4 GB for global h4, of
# 15.75 GB), because the whole step (4 micro-steps) is one XLA graph. Fix: --micro-sync 1 (a sync after every micro-step,
# so each micro-step is its own graph). Fallback: if a d768 job still fails, the same run with micro-batch 2 takes its chip
# (the launcher's "only_if_failed"); same 6000-step schedule, so if it is cut by the hard stop its curve still pairs by step.
# exp06 also showed global h4's gain over mhc-h1 equals that of the same parameters in a wider MLP, so the d768 check has
# the matched control too: mhc-h1 with SwiGLU 2240 (117.21M vs global h4's 117.25M). Seeds 0-1 (6 chips; ~3.5 h expected).
#
# The other 2 chips (and all 8 once the d768 runs end) run d384 jobs exactly as in exp04/exp06 (runs are bit-reproducible
# across sessions, so they pair with the earlier ones): more seeds for global h4 vs its control (H2), and H7 at 2x LR (4x
# broke both variants in exp06). The deadline guard cuts the tail.
SLUG = "mhmhc-exp07-scale"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20
DEADLINE_SECONDS = int(4.75 * 3600)  # hard stop at 5.75 h: the TPU quota has 7.0 h left this week
STEPS = 6000
LR = "3e-3"
STREAMS = ["--hc-layout", "streams"]
COMMON = ["--steps", str(STEPS), "--warmup", "200", "--log-every", "25", *STREAMS]
D768 = ["--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048", "--eval-every", "500",
        "--eval-batches", "40", "--final-eval-batches", "200", "--lr", "1.5e-3", "--micro-sync", "1"]
D768_VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp2240": ["--mlp-dim", "2240"],  # overrides D768's 2048 (argparse keeps the last)
}
D384_VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
}


def d768(name: str, seed: int, micro_batch: int) -> dict:
    args = [*COMMON, "--seed", str(seed), *D768, "--micro-batch", str(micro_batch), *D768_VARIANTS[name]]
    return {"name": f"d768-{name}-s{seed}", "args": args, "expected_seconds": 4 * 3600, "timeout": 6 * 3600}


def d384(name: str, seed: int, lr: str = LR, tag: str = "") -> dict:
    args = [*COMMON, "--eval-every", "1000", "--seed", str(seed), "--lr", lr, *D384_VARIANTS[name]]
    return {"name": f"{name}{tag}-s{seed}", "args": args, "expected_seconds": 4200}


SCALE_JOBS = [d768(name, seed, 4) for seed in [0, 1] for name in D768_VARIANTS]
FALLBACK_JOBS = [{**d768(name, seed, 2), "name": f"d768-{name}-mb2-s{seed}", "only_if_failed": f"d768-{name}-s{seed}"}
                 for seed in [0, 1] for name in D768_VARIANTS]
D384_JOBS = []
for seed in [6, 7, 8, 9]:
    D384_JOBS += [d384(name, seed) for name in D384_VARIANTS]
    if seed in (6, 7):
        D384_JOBS += [d384(name, seed - 6, "6e-3", "-lr2x") for name in ["mhc-h1", "a-global-h4"]]
JOBS = SCALE_JOBS + FALLBACK_JOBS + D384_JOBS
