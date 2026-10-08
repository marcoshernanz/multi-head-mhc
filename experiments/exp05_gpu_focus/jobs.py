# Focused follow-up to the pilot (GPU, while the TPU queue is long), same scale and budget as exp02 (d384 L8, 3000 steps, 49M tokens).
# 1. Is h8-local's deficit vs mHC h1 real? (seeds 1, 2 of both; seed 0 is in exp02)
# 2. Is it the local predictor's slow learning (each head reads 1/h of the stream)? h8-local with logit gain h = 8, seeds 0-2.
# 3. The structure without any predictor: static h1 vs static h8 (seed 0). The fair-fan-in predictor: global h8 (seed 0).
# 4. One more seed of the residual for H0. Runs use the reduction-form Sinkhorn under regional compile (tok/s check).
SLUG = "mhmhc-exp05-gpu-focus"
TRAIN_SHARDS = 10
DEADLINE_SECONDS = 6 * 3600
COMMON = ["--steps", "3000", "--warmup", "200", "--eval-every", "500", "--lr", "3e-3"]
VARIANTS = {
    "mhc-h1": [],
    "a-local-h8": ["--hc-heads", "8", "--hc-predictor", "local"],
    "a-local-h8-gain8": ["--hc-heads", "8", "--hc-predictor", "local", "--hc-local-gain", "8"],
    "a-global-h8": ["--hc-heads", "8"],
    "static-h1": ["--hc-predictor", "static"],
    "static-h8": ["--hc-heads", "8", "--hc-predictor", "static"],
    "residual": ["--conn", "residual"],
}
RUNS = [("a-global-h8", 0), ("a-local-h8-gain8", 0), ("a-local-h8-gain8", 1), ("a-local-h8-gain8", 2), ("a-local-h8", 1),
        ("a-local-h8", 2), ("static-h8", 0), ("mhc-h1", 1), ("mhc-h1", 2), ("static-h1", 0), ("residual", 1)]
JOBS = [{"name": f"{name}-s{seed}", "args": [*COMMON, "--seed", str(seed), *VARIANTS[name]]} for name, seed in RUNS]
