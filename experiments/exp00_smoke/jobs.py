# Smoke test: environment, compile, throughput and memory of each connection at two widths.
SLUG = "mhmhc-exp00-smoke"
TRAIN_SHARDS = 2
DEADLINE_SECONDS = 3600
COMMON = ["--steps", "150", "--warmup", "20", "--eval-every", "100", "--eval-batches", "5",
          "--final-eval-batches", "10", "--log-every", "10", "--batch", "16", "--micro-batch", "8"]
D384 = ["--dim", "384", "--layers", "8", "--heads", "6", "--mlp-dim", "1024"]
D512 = ["--dim", "512", "--layers", "8", "--heads", "8", "--mlp-dim", "1408"]
JOBS = [
    {"name": "d384-residual", "args": [*COMMON, *D384, "--conn", "residual"]},
    {"name": "d384-mhc-h1", "args": [*COMMON, *D384, "--conn", "mhc", "--hc-heads", "1"]},
    {"name": "d384-mhc-h8-global", "args": [*COMMON, *D384, "--conn", "mhc", "--hc-heads", "8"]},
    {"name": "d384-mhc-h8-local", "args": [*COMMON, *D384, "--conn", "mhc", "--hc-heads", "8", "--hc-predictor", "local"]},
    {"name": "d384-mhc-h1-eager", "args": [*COMMON, *D384, "--conn", "mhc", "--hc-heads", "1", "--compile", "0"]},
    {"name": "d512-residual", "args": [*COMMON, *D512, "--conn", "residual"]},
    {"name": "d512-mhc-h1", "args": [*COMMON, *D512, "--conn", "mhc", "--hc-heads", "1"]},
    {"name": "d384-mhc-h64-local", "args": [*COMMON, *D384, "--conn", "mhc", "--hc-heads", "64", "--hc-predictor", "local"]},
]
