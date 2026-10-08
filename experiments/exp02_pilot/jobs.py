# Pilot: learning rate for the residual and mHC, and the two inits of mHC's biases ("hc" vs "sym").
# d384 L8 (27M params), 3000 steps x 16 x 1024 = 49M tokens, seed 0. Longest jobs first.
SLUG = "mhmhc-exp02-pilot"
TRAIN_SHARDS = 10
DEADLINE_SECONDS = 11 * 3600
COMMON = ["--steps", "3000", "--warmup", "200", "--eval-every", "500", "--seed", "0"]
JOBS = [
    {"name": "mhc-h8-local-lr3e-3", "args": [*COMMON, "--hc-heads", "8", "--hc-predictor", "local", "--lr", "3e-3"]},
    {"name": "mhc-hc-lr3e-3", "args": [*COMMON, "--lr", "3e-3"]},
    {"name": "mhc-hc-lr1.5e-3", "args": [*COMMON, "--lr", "1.5e-3"]},
    {"name": "mhc-hc-lr6e-3", "args": [*COMMON, "--lr", "6e-3"]},
    {"name": "mhc-sym-lr3e-3", "args": [*COMMON, "--hc-init", "sym", "--lr", "3e-3"]},
    {"name": "residual-lr3e-3", "args": [*COMMON, "--conn", "residual", "--lr", "3e-3"]},
    {"name": "residual-lr1.5e-3", "args": [*COMMON, "--conn", "residual", "--lr", "1.5e-3"]},
    {"name": "residual-lr6e-3", "args": [*COMMON, "--conn", "residual", "--lr", "6e-3"]},
]
