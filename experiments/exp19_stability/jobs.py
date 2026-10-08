# Is the heads' high-LR tolerance a stability property that would matter in a modern recipe? TPU v6e, d384, exp12's settings.
#
# Why (2026-10-05, prompted by a proposed reframing of the paper around stability): past the optimum the head variants lose less
# than mHC h1 and the parameter-matched control and damp late gradient spikes (exp10, exp12, exp15). Wortsman et al. (2023) use
# exactly this, loss at too-high LRs in small models, as a proxy for large-scale instabilities, and trace the main one to attention
# logits that grow until attention collapses; QK-norm removes it and is standard in current LLMs (Qwen3, Gemma 3, OLMo 2). So
# before the tolerance can be offered as a stability benefit: (1) what fails at high LR here, (2) does the tolerance survive QK-norm,
# (3) is it there over a wide LR range (their LR sensitivity), and (4) are there loss or gradient spikes to count.
#
# Every run logs every step's loss and grad norm (--log-steps 1) and, every 250 steps, diagnostics on 4 fixed validation windows:
# per attention layer the largest logit and the mean entropy, per sublayer the RMS of its input and output, and the output logits'
# z^2 and largest value (--diag-every 250). Neither touches the training computation.
#   A. Replays without QK-norm at 2.1e-3 (the optimum), 4.2e-3 and 6e-3, seeds 0-2, all five variants: diagnostics and per-step
#      logs for runs that exist (their losses must repeat the earlier runs' bit for bit), and the residual at 6e-3 (never run).
#   B. QK-norm (--qk-norm 1): the whole sqrt(2) grid 1.5e-3 ... 2.4e-2, all five variants, seeds 0-2.
#   C. Without QK-norm, the grid continued upward: 8.5e-3 ... 2.4e-2, all five variants, seeds 0-2.
# Read-outs in notes/plan.md (2026-10-05), fixed before any run of this batch.
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50", "--diag-every", "250", "--log-steps", "1"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}
GRID = ["1.5e-3", "2.1e-3", "3e-3", "4.2e-3", "6e-3", "8.5e-3", "1.2e-2", "1.7e-2", "2.4e-2"]
SEEDS = [0, 1, 2]


def job(name: str, seed: int, lr: str, qk: bool) -> dict:
    expected = 500 if name == "residual" else 1600
    return {"name": f"v6e-{name}{'-qk' if qk else '-diag'}-lr{lr}-s{seed}",
            "args": [*D384, "--seed", str(seed), "--lr", lr, *VARIANTS[name], *(["--qk-norm", "1"] if qk else [])],
            "expected_seconds": expected, "timeout": 3 * 3600}


JOBS = (
    [job(name, 0, lr, False) for lr in ["2.1e-3", "6e-3"] for name in VARIANTS]  # first: the bit-identity check
    + [job(name, seed, lr, False) for seed in [1, 2] for lr in ["2.1e-3", "6e-3"] for name in VARIANTS]
    + [job(name, seed, "4.2e-3", False) for seed in SEEDS for name in VARIANTS]
    + [job(name, seed, lr, True) for seed in SEEDS for lr in GRID for name in VARIANTS]
    + [job(name, seed, lr, False) for seed in SEEDS for lr in GRID[5:] for name in VARIANTS]
)
