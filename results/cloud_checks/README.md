# Hardware checks run by hand on Google Cloud (2026-10-03)

## `v5e_replay/`: is a Google Cloud v5e chip the same as Kaggle's?

exp04's `mhc-h1-s0` (Kaggle TPU v5e-8, one process per chip) replayed on a GCP `v5litepod-1` spot VM set up by `cloud/setup_vm.sh`
(torch 2.8.0, torch_xla 2.8.0, libtpu 0.0.17, the same 20 train shards), each run cut at 420 s of training (`run_replay.sh`):

| run | setting | against Kaggle's run (`analysis/checks/check_resume.py results/exp04_main/runs/mhc-h1-s0 <run>`) |
|---|---|---|
| `mb8` | micro-batch 8 x 2, fused (exp04's setting) | **bit-identical** at every logged step 0-300 (loss and grad norm) |
| `mb8sync` | micro-batch 8 x 2, graph cut after each micro-step (`--micro-sync 1`) | step-0 grad norm differs by 4e-6, then the runs drift apart (float noise, amplified by training) |
| `mb16` | micro-batch 16 (one 16-sequence step; what all v6e runs use) | same as `mb8sync`: 4e-6 at step 0, then drift |

So GCP v5e reproduces Kaggle v5e exactly at the same setting (exp09 runs there and pairs exactly with exp07; its
`repro-d768-mhc-h1-s0` repeats a full exp07 run as a 6000-step check), and changing how the batch is accumulated changes the run only at
floating-point level. The v6e bug (wrong gradients with fused accumulation; `notes/gcp.md`) was found with the same step-0 grad norm
test: 8.43 on v6e at micro-batch 8 x 2 fused, against the 4.388 above.

## v6e determinism (in `results/exp11_v6e_check/runs/`)

- `chk-v6e8-mhc-h1-s0` repeats `v6e-mhc-h1-s0` on a different v6e-1 VM: bit-identical at all 241 logged steps and 6 evals.
- `chk-multi-{mhc-h1,a-global-h4}-s0` repeat the v6e-1 runs on chips of a v6e-8 VM (one process per chip, `cloud/vm_job.sh`):
  bit-identical to `v6e-{mhc-h1,a-global-h4}-s0` at every logged step and every eval, final loss included
  (`analysis/checks/check_resume.py`). So v6e-1, v6e-4 and v6e-8 VMs give the same runs, and a job may land on any of them.
