# Longer training at d384: 4x the tokens (24000 steps x 16 x 1024 = 393M tokens, ~14 tokens per parameter instead of 3.6), on TPU v5e.
#
# Why: the d384 verdict (global h4 = its parameter-matched control, local h4 = mHC h1, both at 98M tokens) could change with
# training length; REPORT §6 lists "much longer training" as one thing that would change it, and exp05 saw a head lead fade
# between 16M and 49M tokens. Same variants, LR 3e-3 (each variant's LR is checked by exp12 at 98M tokens), the same warmup
# (200 steps) and cosine decay to 10% over the longer run, seeds 0-2. The first 6000 steps are not the 98M-token runs (the schedule
# differs), so only the final losses are compared, within this batch.
#
# Chip (changed 2026-10-03 23:50 UTC, before any exp13 run had started): written for v6e, moved to spot v5e. v6e capacity is the
# bottleneck (2 VMs, 12 chips, preempted every 0.5-2 h) and this batch alone was ~19 v6e chip-hours; v5e VMs have run for over
# 2 h without preemption and free up when exp09 ends. The batch carries all its own baselines (mHC h1, control, residual), so
# the chip does not matter for its read-out; it is not compared with exp11-exp15. Settings are exp04's on v5e (micro-batch 8,
# evals on the same 819k tokens via 20 / 100 batches of 8), about 4 h per run (exp04: 3,500-3,640 s for 98M tokens).
#
# Data: the VMs hold the same 20 train shards (200M tokens) as every run, and windows are drawn at random from them, so a 393M-token
# run sees each token about twice. Repeating data up to ~4 times costs little at this scale (Muennighoff et al. 2023), and every
# variant sees exactly the same batches, so the comparison is unaffected; the absolute losses are a little worse than with
# fresh data.
D384 = ["--steps", "24000", "--warmup", "200", "--eval-every", "2000", "--log-every", "50", "--hc-layout", "streams", "--lr", "3e-3"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}
JOBS = [{"name": f"v5e-long-{name}-s{seed}", "args": [*D384, "--seed", str(seed), *args],
         "expected_seconds": 3600 if name == "residual" else 14500, "timeout": 10 * 3600}
        for seed in [0, 1, 2] for name, args in VARIANTS.items()]
