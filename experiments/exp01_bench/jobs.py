# Throughput benchmark of the connection implementations (random tokens, 30 compiled steps each).
SLUG = "mhmhc-exp01-bench"
TRAIN_SHARDS = 1
DEADLINE_SECONDS = 3000
B = {"module": "mhc_lab.bench"}
JOBS = [
    {**B, "name": "residual", "args": ["--conn", "residual"]},
    {**B, "name": "mhc-h1-new", "args": ["--hc-heads", "1"]},
    {**B, "name": "mhc-h1-old-sinkhorn", "args": ["--hc-heads", "1", "--old-sinkhorn", "1"]},
    {**B, "name": "mhc-h8-local-new", "args": ["--hc-heads", "8", "--hc-predictor", "local"]},
    {**B, "name": "mhc-h8-global-new", "args": ["--hc-heads", "8"]},
    {**B, "name": "mhc-h64-local-new", "args": ["--hc-heads", "64", "--hc-predictor", "local"]},
    {**B, "name": "mhc-h1-mb16", "args": ["--hc-heads", "1", "--micro-batch", "16"]},
    {**B, "name": "residual-mb16", "args": ["--conn", "residual", "--micro-batch", "16"]},
    {**B, "name": "mhc-h16-local-new", "args": ["--hc-heads", "16", "--hc-predictor", "local"]},
    {**B, "name": "mhc-h1-eager", "args": ["--hc-heads", "1", "--compile", "0"]},
]
