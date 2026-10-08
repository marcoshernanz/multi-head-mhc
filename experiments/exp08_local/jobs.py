# A short batch that fits the 2.46 TPU hours left before the 2026-10-03 reset (one wave of 8 d384 runs, ~1 h each).
# It tests the only equal-parameter hint left at d384: local-predictor heads have exactly mHC h1's parameters, and local h2
# was −0.018 ± 0.010 (3/4 seeds) vs mhc-h1, below the "wider MLP" line (analysis/params_frontier.py). Seeds 4-9 of local h2
# pair with mhc-h1's seeds 0-9 (runs are bit-reproducible across sessions); seeds 4-5 of local h4 go with the d768 local-h4
# runs planned after the reset. Same settings as exp04.
SLUG = "mhmhc-exp08-local"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20
DEADLINE_SECONDS = 1800  # every job starts at ~2 min; no job starts after 30 min; hard stop at 1.5 h (quota left: 2.46 h)
STEPS = 6000
COMMON = ["--steps", str(STEPS), "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
          "--lr", "3e-3"]
JOBS = [
    {"name": f"a-local-h{h}-s{seed}", "args": [*COMMON, "--seed", str(seed), "--hc-heads", str(h), "--hc-predictor", "local"],
     "expected_seconds": 3900}
    for h, seeds in [(2, [4, 5, 6, 7, 8, 9]), (4, [4, 5])]
    for seed in seeds
]
