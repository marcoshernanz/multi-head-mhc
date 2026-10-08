# Which part of the head split matters? Global h4 with only some of its coefficient blocks per head, on TPU v6e (d384).
#
# Why: design A gives each channel group its own read (pre), write (post) and doubly stochastic mix (res). At d384 the full split
# (global h4) is worth what its parameters are worth at the tuned LR but degrades far less past it (exp10: 0.2 better than mHC h1
# and the MLP control at 2x LR). Qwen3.8-Next's ablation (arXiv 2608.30320, notes/literature/recheck_2026-10-03.md §3.1) found, for
# a widened stream at 25B-A3B: per-channel read granularity helps, write granularity gives almost nothing, and H_res "adds little"
# once read and write are expressive. Here: --hc-head-parts keeps the listed blocks per head and predicts the others once for all
# heads (model.py; with none per head it is exactly mHC h1). Variants, all global predictor, h = 4:
#   pre        per-head read only                 27.66M parameters (mHC h1 27.34M, global h4 29.13M)
#   post       per-head write only                27.64M
#   res        per-head doubly stochastic mix     28.52M
#   pre,post   per-head read and write, shared mix 27.95M   (design A without its per-group H_res)
# at 3e-3 (the tuned LR) and 6e-3 (2x, where the full split's stability shows), seeds 0-2. mHC h1 and global h4 at both LRs on the
# same chip and settings come from exp11 (3e-3 seeds 0-2, 6e-3 seeds 0-1) and exp12 (6e-3 seed 2).
#
# Read-outs, decided before the runs: (1) at 6e-3, which variants keep most of global h4's advantage over mHC h1 (the stability
# carrier); (2) at 3e-3, each variant against mHC h1 and against the MLP line at its own parameter count (params_frontier.py), i.e.
# whether any single block is worth more than its parameters. Expectation from Qwen: pre carries the gain, post nothing, and
# pre,post matches the full split; from this project's exp04 (global > local at every h, a bigger predictor helps) the res block,
# which holds 2/3 of the extra predictor rows, could carry it instead.
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50"]
PARTS = {"pre": "pre", "post": "post", "res": "res", "prepost": "pre,post"}
LR_TAG = {"3e-3": "", "6e-3": "-lr2x"}
JOBS = [{"name": f"v6e-a-global-h4-{tag}{LR_TAG[lr]}-s{seed}",
         "args": [*D384, "--seed", str(seed), "--lr", lr, "--hc-heads", "4", "--hc-head-parts", parts],
         "expected_seconds": 1500, "timeout": 3 * 3600}
        for seed in [0, 1, 2] for lr in ["3e-3", "6e-3"] for tag, parts in PARTS.items()]
