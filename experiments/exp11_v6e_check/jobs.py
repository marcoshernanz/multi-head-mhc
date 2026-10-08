# Hardware check before any result from Google Cloud TPU v6e is mixed with the v5e (Kaggle) ones: d384 runs that already exist on v5e,
# repeated on v6e-1 spot VMs (cloud/controller.py --accel v6e-1). Same software (torch 2.8.0, torch_xla 2.8.0, libtpu 0.0.17),
# same data order, same seeds, same 6000 x 16 x 1024 tokens.
#
# One difference on purpose: micro-batch 16 (the whole 16-sequence batch in one forward/backward) instead of v5e's 8 x 2 accumulated.
# On v6e, 2-step accumulation fused into one XLA graph with the clip and the AdamW step gives wrong gradients (step-0 grad norm 8.43
# for the residual model instead of 4.388 on CPU, v5e and v6e at micro-batch 16; see LOG.md 2026-10-03), and micro-batch 16 also
# fits a v6e chip's 32 GB and is the fastest correct setting (81.9k tok/s for mhc-h1 vs 47k at micro-batch 8 with --micro-sync 1).
# The gradient is mathematically the same, so runs should match v5e's up to floating-point noise, not bit for bit.
#
# Pass criteria (decided before the runs): (1) each variant's per-seed final val loss within the v5e seed-to-seed spread of that
# variant; (2) the paired gaps that the report rests on — global h4 − mhc-h1, global h4 − ctrl-mlp1216, local h4 − mhc-h1,
# mhc-h1 − residual at 3e-3, and global h4 / ctrl-mlp1216 / local h4 − mhc-h1 at 2x LR — reproduce in sign and agree within the v5e
# paired standard error. If they do, later v6e batches (always micro-batch 16, with their own in-batch baselines) can be read on
# the same scale as v5e.
# Also exp07's d768 runs at 1.5e-3 (mhc-h1, global h4, ctrl-mlp2240, seeds 0-1; micro-batch 16 instead of 4 x 4 with --micro-sync):
# d768 runs 4.7x faster on v6e (51.9k tok/s vs ~11k), so any d768 follow-up after exp09 would run there.
# Validation batches are built at the micro-batch size, so at micro-batch 16 the eval counts are halved (10 and 50 batches of 16)
# to score on exactly the same windows as v5e's 20 and 100 batches of 8 (the 819k tokens at the start of validation shard 0).
D384 = ["--steps", "6000", "--warmup", "200", "--eval-every", "1000", "--log-every", "25", "--hc-layout", "streams",
        "--micro-batch", "16", "--eval-batches", "10", "--final-eval-batches", "50"]
VARIANTS = {
    "mhc-h1": [],
    "a-global-h4": ["--hc-heads", "4"],
    "ctrl-mlp1216": ["--mlp-dim", "1216"],
    "a-local-h4": ["--hc-heads", "4", "--hc-predictor", "local"],
    "residual": ["--conn", "residual"],
}


D768 = ["--dim", "768", "--layers", "12", "--heads", "12", "--mlp-dim", "2048", "--eval-every", "500"]  # evals as at d384: the
# same windows as exp07's 40 x 4 and 200 x 4
D768_VARIANTS = {"mhc-h1": [], "a-global-h4": ["--hc-heads", "4"], "ctrl-mlp2240": ["--mlp-dim", "2240"]}


def job(name: str, seed: int, lr: str = "3e-3", tag: str = "") -> dict:
    return {"name": f"v6e-{name}{tag}-s{seed}", "args": [*D384, "--seed", str(seed), "--lr", lr, *VARIANTS[name]],
            "expected_seconds": 1500, "timeout": 3 * 3600}


# The first two jobs ran on v6e-1 VMs; the project allows only 4 external IPs per region, so the rest run on multi-chip VMs where
# capacity allows (v6e-8 / v6e-4, one process per chip, cloud/vm_job.sh), else v6e-1. chk-multi-*: the same two runs repeated on a
# multi-chip VM (min_chips), expected bit-identical to the v6e-1 ones. chk-v6e8-mhc-h1-s0 was meant for that but the controller put
# it on another v6e-1 VM, so it checks that v6e-1 runs repeat exactly across VMs.
JOBS = (
    [{**job("mhc-h1", 0), "name": "chk-v6e8-mhc-h1-s0"}]
    + [{**job(name, 0), "name": f"chk-multi-{name}-s0", "min_chips": 4} for name in ["mhc-h1", "a-global-h4"]]
    + [job(name, seed) for seed in [0, 1, 2] for name in VARIANTS]
    + [job(name, seed, "6e-3", "-lr2x") for seed in [0, 1] for name in ["mhc-h1", "a-global-h4", "ctrl-mlp1216", "a-local-h4"]]
    + [{"name": f"v6e-d768-{name}-s{seed}", "args": [*D384, *D768, "--seed", str(seed), "--lr", "1.5e-3", *args],
        "expected_seconds": 2400, "timeout": 4 * 3600} for seed in [0, 1] for name, args in D768_VARIANTS.items()]
)
