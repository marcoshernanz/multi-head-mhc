# exp14's four part variants at d384's best LR on v6e (2.1e-3), seeds 0-2.
#
# Why (decided 2026-10-04 05:20 UTC, after two of exp14's three seeds at 3e-3 were in, and run whatever the third shows): exp14 was
# written for 3e-3, "the tuned LR" of the v5e runs, before exp12 found 2.1e-3 to be the best LR on v6e for mHC h1, global h4 and the
# control. At 3e-3 every variant is already 0.02-0.035 off its best, and global h4's lead over mHC h1 there (−0.015, 3/3) vanishes at
# 2.1e-3 (−0.004 ± 0.013): part of a gain at 3e-3 can be LR tolerance. On two seeds, the per-head read alone (pre) is 0.034 ahead of
# mHC h1 at 3e-3, 0.027 below the MLP line. Running the same variants at 2.1e-3 tells structure from tolerance. Unconditional, so the
# decision to run it does not depend on the result it checks.
#
# Read-out: at 2.1e-3, each part variant against mHC h1, global h4 and the MLP line at its own parameter count (mHC h1 to ctrl-mlp1216,
# both at 2.1e-3 from exp12, linear in parameters), with the same test as notes/plan.md's 02:15 rule (excess below zero by more than
# 2 SE of the Δ to mHC h1, all 3 seeds below the line). The 02:15 rule for the d768 follow-up (exp18) is read on 3e-3, as written.
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50", "--lr", "2.1e-3"]
PARTS = {"pre": "pre", "post": "post", "res": "res", "prepost": "pre,post"}
JOBS = [{"name": f"v6e-a-global-h4-{tag}-lr0.7x-s{seed}",
         "args": [*D384, "--seed", str(seed), "--hc-heads", "4", "--hc-head-parts", parts],
         "expected_seconds": 1500, "timeout": 3 * 3600}
        for seed in [0, 1, 2] for tag, parts in PARTS.items()]
