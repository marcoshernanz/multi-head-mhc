# TPU probe: can 8 single-chip torch_xla processes run at once, and how fast / how long to compile is each connection variant?
# 150 real training steps each (d384 L8, batch 16 x 1024), bf16.
SLUG = "mhmhc-exp03-tpu-probe"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 1
DEADLINE_SECONDS = 2 * 3600
COMMON = ["--steps", "150", "--warmup", "20", "--log-every", "10", "--eval-every", "100000", "--final-eval-batches", "10"]
JOBS = [
    {"name": "residual", "args": [*COMMON, "--conn", "residual"]},
    {"name": "mhc-h1", "args": [*COMMON]},
    {"name": "mhc-h8-local", "args": [*COMMON, "--hc-heads", "8", "--hc-predictor", "local"]},
    {"name": "mhc-h4-joint", "args": [*COMMON, "--hc-heads", "4", "--hc-joint", "1"]},
    {"name": "mhc-h16-local", "args": [*COMMON, "--hc-heads", "16", "--hc-predictor", "local"]},
    {"name": "mhc-h1-mb16", "args": [*COMMON, "--micro-batch", "16"]},
    {"name": "residual-mb16", "args": [*COMMON, "--conn", "residual", "--micro-batch", "16"]},
    {"name": "mhc-h8-global", "args": [*COMMON, "--hc-heads", "8"]},
]
