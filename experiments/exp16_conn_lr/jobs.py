# Is the heads' high-LR tolerance available without heads? mHC h1 and global h4 with a 4x connection LR, on TPU v6e (d384).
#
# Why (decided 2026-10-04 04:25 UTC, after exp12's read-out (d), before any run of this batch): exp12's reverse check went the
# wrong way for "slower connection dynamics". Global h4 with every connection parameter's LR x4 did not break like mHC h1 at 6e-3;
# it ended 0.20 ± 0.05 better than global h4 (3/3) and 0.27 better than mHC h1 (3/3), with the smallest late gradient norm of any
# 6e-3 run (1.7). mHC h1 with its connection at x0.25 was no better than mHC h1. Note what x4 does: AdamW's decoupled decay uses the
# group's LR, so --conn-lr-mult 4 both speeds up the static coefficients (biases, alpha) and decays the dynamic weights phi 4x
# harder, i.e. pushes the connection toward a static one. If mHC h1 gets the same tolerance from it, the one property the heads
# showed is not a property of heads. This batch is exploratory (the hypothesis came from exp12's data) and says so in the report.
#
# Runs, seeds 0-2, settings as exp12 (v6e-1, micro-batch 16, evals on the same 819k tokens via 10 / 50 batches of 16):
#   mhc-h1-conn4 at 6e-3      against mHC h1 and global h4 conn4 at 6e-3 (exp11 / exp12)
#   mhc-h1-conn4 at 2.1e-3    against mHC h1 at 2.1e-3 (exp12; its best LR): does the faster connection cost or help at the optimum?
#   a-global-h4-conn4 at 2.1e-3 against global h4 at 2.1e-3 (exp12; its best LR), and against mhc-h1-conn4 (heads at equal settings)
# Read-outs in notes/plan.md (2026-10-04 04:25 UTC).
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50", "--conn-lr-mult", "4"]
VARIANTS = {"mhc-h1-conn4": [], "a-global-h4-conn4": ["--hc-heads", "4"]}
LR_TAG = {"2.1e-3": "-lr0.7x", "6e-3": "-lr2x"}  # exp12's tags
RUNS = [("mhc-h1-conn4", "6e-3"), ("mhc-h1-conn4", "2.1e-3"), ("a-global-h4-conn4", "2.1e-3")]
JOBS = [{"name": f"v6e-{name}{LR_TAG[lr]}-s{seed}", "args": [*D384, "--seed", str(seed), "--lr", lr, *VARIANTS[name]],
         "expected_seconds": 1500, "timeout": 3 * 3600}
        for seed in [0, 1, 2] for name, lr in RUNS]
