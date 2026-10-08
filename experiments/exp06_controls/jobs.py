# Follow-up to exp04 (global-predictor heads beat mHC h1: h4 -0.031, h16 -0.023, 4/4 seeds each). Four questions, in queue order:
#
# 1. Scale check (plan: "the effect does not shrink to nothing at the larger scale check"): d768, 12 layers, 12 attention heads,
#    SwiGLU 2048 (112M parameters), mhc-h1 vs a-global-h4, seeds 0-2. Micro-batch 4 so the stream fits a v5e chip; the
#    evaluation uses 200 x 4 windows, the same 819k validation tokens as 100 x 8 at d384. LR 1.5e-3 (3e-3 halved for twice the
#    width; not tuned). These go first because they are the longest (~4 h each by extrapolating exp04's per-micro-step cost).
# 2. H2, "is it the heads or just more parameters?": global heads add predictor parameters (h4 +6.5%, h16 +32.7% at d384), so
#    mhc-h1 with a wider MLP matched to each: 1216 (29.11M vs global h4's 29.13M) and 1992 (36.26M vs global h16's 36.29M).
#    Seeds 0-3 pair with exp04's four finished seeds. Two chips run these from the start, alongside the six d768 jobs.
# 3. More seeds for the comparisons that matter: seed 4 for mhc-h1, global h4, global h16 and both controls.
# 4. H7 (stability): mhc-h1 and global h4 at 4x the tuned LR (1.2e-2), seeds 0-1.
# 5. A repeat of exp04's mhc-h1-s0: the pairs above span two Kaggle sessions, so this measures how far the same seed and config
#    drift between sessions (if the TPU run is bit-reproducible, the two final losses match).
# 6. Seed 5 for the same five variants as 3, if time is left.
# Everything else matches exp04 (6000 steps x 16 x 1024 tokens, streams layout, micro-batch 8 at d384).
SLUG = "mhmhc-exp06-controls"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20
DEADLINE_SECONDS = int(6.5 * 3600)  # no job starts after 6.5 h; hard stop at 7.5 h
STEPS = 6000
LR = "3e-3"
STREAMS = ["--hc-layout", "streams"]
COMMON = ["--steps", str(STEPS), "--warmup", "200", "--eval-every", "1000", "--log-every", "25", *STREAMS]
D384_SECONDS = 4200  # exp04: 3190-3990 s per run
D768_SECONDS = 4 * 3600
D768 = ["--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048", "--micro-batch", "4",
        "--eval-batches", "40", "--final-eval-batches", "200", "--lr", "1.5e-3"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "a-global-h16": ["--hc-heads", "16"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
    "ctrl-mlp1992": ["--mlp-dim", "1992"],
}


def d384(name: str, seed: int, lr: str = LR, tag: str = "") -> dict:
    args = [*COMMON, "--seed", str(seed), "--lr", lr, *VARIANTS[name]]
    return {"name": f"{name}{tag}-s{seed}", "args": args, "expected_seconds": D384_SECONDS}


SCALE_JOBS = [
    {"name": f"d768-{name}-s{seed}", "args": [*COMMON, "--seed", str(seed), *D768, *VARIANTS[name]],
     "expected_seconds": D768_SECONDS, "timeout": 7 * 3600}
    for seed in [0, 1, 2]
    for name in ["mhc-h1", "a-global-h4"]
]
CONTROL_JOBS = [d384(name, seed) for seed in [0, 1, 2, 3] for name in ["ctrl-mlp1216", "ctrl-mlp1992"]]
SEED4_JOBS = [d384(name, 4) for name in VARIANTS]
LR_JOBS = [d384(name, seed, "1.2e-2", "-lr4x") for seed in [0, 1] for name in ["mhc-h1", "a-global-h4"]]
SEED5_JOBS = [d384(name, 5) for name in VARIANTS]
REPEAT_JOBS = [{**d384("mhc-h1", 0), "name": "repeat-mhc-h1-s0"}]
JOBS = SCALE_JOBS + CONTROL_JOBS + SEED4_JOBS + LR_JOBS + REPEAT_JOBS + SEED5_JOBS
