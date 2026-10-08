# Uses the 1.39 TPU hours left before the 2026-10-03 reset (one wave of 8 d384 runs, ~1 h each); pushed before exp09.
# Question: does the heads' gain over mhc-h1 depend on the LR, and is it the heads or their parameters?
# exp07's d768 gap (global h4 − ctrl-mlp2240 −0.034) came at an untuned LR, and at 2x LR (d384) global h4 degrades less
# than mhc-h1 (−0.21). Two gaps in that picture, both at d384 where 3e-3 is tuned:
#   - 2x LR (6e-3): the matched control ctrl-mlp1216 was never run. If it breaks like mhc-h1, the robustness is the heads';
#     if it holds like global h4, it is just the parameters.
#   - 0.5x LR (1.5e-3, the same ratio as d768's halved LR): mhc-h1, global h4 and ctrl-mlp1216. If the heads beat the
#     control only off the tuned LR, the d768 result is probably an LR effect.
# Seeds 0-1, pairing with exp04/exp06/exp07 (bit-reproducible across sessions). Same settings as exp04 otherwise.
SLUG = "mhmhc-exp10-d384-lr"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20
DEADLINE_SECONDS = 900  # every job starts at ~2 min; hard stop at 1.25 h (quota left: 1.39 h; exp08's wave took 1.07 h)
STEPS = 6000
COMMON = ["--steps", str(STEPS), "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
}


def job(name: str, seed: int, lr: str, tag: str) -> dict:
    args = [*COMMON, "--seed", str(seed), "--lr", lr, *VARIANTS[name]]
    return {"name": f"{name}{tag}-s{seed}", "args": args, "expected_seconds": 3700, "timeout": 4200}


JOBS = [job("ctrl-mlp1216", seed, "6e-3", "-lr2x") for seed in [0, 1]]
JOBS += [job(name, seed, "1.5e-3", "-lrhalf") for seed in [0, 1] for name in VARIANTS]
