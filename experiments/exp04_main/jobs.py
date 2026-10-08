# Main sweep: every connection variant at the main scale (d384 L8), paired seeds, 2x the pilot's tokens. Answers H0-H4 and H6 of
# notes/plan.md. The head-count curve is run for both predictors (exp05: global h8 led early and tied at the end; local h8 lost).
# The gain variants were dropped (exp05 answered H8: they hurt).
# LR 3e-3 for every variant (exp02: best for mHC, tied with 1.5e-3 for the residual).
#
# TPU layout: exp03 showed mHC in the token layout ([B, T, n, D], res [.., n, n]) is unusable on a TPU (2.4k tok/s for h1, head
# variants never compiled), so every mHC run here uses --hc-layout streams ([n, D, B*T]; identical math, checked by
# analysis/checks/check_layouts.py). The first round is 8 short benchmarks; bench-mhc-h1 is a gate: below 15k tok/s nothing else starts.
# Then seeds 0-4 in seed-major order; the launcher starts a job only if it can end before the hard stop, so whatever is
# cut is the tail of the last seed.
SLUG = "mhmhc-exp04-main"
ACCELERATOR = "tpu"
TRAIN_SHARDS = 20  # 200M tokens; a run reads 98M
DEADLINE_SECONDS = int(6.5 * 3600)  # no job starts after 6.5 h; hard stop at 7.5 h (Kaggle's TPU limit: 9 h)
STEPS = 6000  # x 16 x 1024 = 98M tokens
LR = "3e-3"
SEEDS = [0, 1, 2, 3, 4]
STREAMS = ["--hc-layout", "streams"]
COMMON = ["--steps", str(STEPS), "--warmup", "200", "--eval-every", "1000", "--log-every", "25"]
VARIANTS = {
    "residual": ["--conn", "residual"],
    "mhc-h1": [],
    "a-global-h2": ["--hc-heads", "2"],
    "a-global-h4": ["--hc-heads", "4"],
    "a-global-h8": ["--hc-heads", "8"],
    "a-global-h16": ["--hc-heads", "16"],
    "a-local-h2": ["--hc-heads", "2", "--hc-predictor", "local"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "a-local-h8": ["--hc-heads", "8", "--hc-predictor", "local"],
    "a-local-h16": ["--hc-heads", "16", "--hc-predictor", "local"],
    "static-h1": ["--hc-predictor", "static"],
    "static-h4": ["--hc-heads", "4", "--hc-predictor", "static"],
    "static-h8": ["--hc-heads", "8", "--hc-predictor", "static"],
    "d-joint-h4-global": ["--hc-heads", "4", "--hc-joint", "1"],
    "d-joint-h4-static": ["--hc-heads", "4", "--hc-joint", "1", "--hc-predictor", "static"],
    "mhc-n8": ["--hc-mult", "8"],
}
BENCH = ["--steps", "150", "--warmup", "20", "--log-every", "10", "--eval-every", "100000", "--final-eval-batches", "10", "--lr", LR]
BENCH_JOBS = [
    {"name": "bench-residual", "args": [*BENCH, "--conn", "residual"]},
    {"name": "bench-mhc-h1", "args": [*BENCH, *STREAMS], "gate_min_tok_s": 15000},
    {"name": "bench-mhc-h1-tokens", "args": [*BENCH], "first_step_timeout": 900},  # the old layout, reduction-form Sinkhorn
    {"name": "bench-mhc-h1-mb16", "args": [*BENCH, *STREAMS, "--micro-batch", "16"]},
    {"name": "bench-a-local-h16", "args": [*BENCH, *STREAMS, *VARIANTS["a-local-h16"]]},
    {"name": "bench-a-global-h16", "args": [*BENCH, *STREAMS, *VARIANTS["a-global-h16"]]},
    {"name": "bench-d-joint-h4-global", "args": [*BENCH, *STREAMS, *VARIANTS["d-joint-h4-global"]]},
    {"name": "bench-mhc-n8", "args": [*BENCH, *STREAMS, *VARIANTS["mhc-n8"]]},
]
# seed-major order: if the batch is cut, the finished seeds are complete sets of paired runs
MAIN_JOBS = [
    {"name": f"{name}-s{seed}", "args": [*COMMON, "--seed", str(seed), "--lr", LR, *STREAMS, *args]}
    for seed in SEEDS
    for name, args in VARIANTS.items()
]
JOBS = BENCH_JOBS + MAIN_JOBS
