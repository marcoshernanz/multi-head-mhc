# Running experiments on Google Cloud TPUs (how-to)

From 2026-10-03 the experiments run on spot Cloud TPU VMs instead of Kaggle (Kaggle's TPU queue took up to 11 h). Budget for this
project: **$300** of Google Cloud credit; every VM's create/delete is in `cloud/ledger.jsonl`.

## Pattern

- One batch = one folder `experiments/<batch>/` with `jobs.py` defining `JOBS` (name, args, `expected_seconds`, optional `timeout`),
  as for Kaggle (SLUG / DEADLINE / TRAIN_SHARDS are only used by the Kaggle launcher and are ignored here).
- `python3 cloud/controller.py experiments/<batch> [experiments/<batch2> ...] --accel v6e-8,v6e-4 [--max-chips 40] [--max-price 0.75]
  [--budget 280] [--drain-other-sizes]` (run it in the background; it prints and appends to `results/<first batch>/controller.log`,
  writes `results/<batch>/status.json`). **Run one controller per chip family**, with all batches of that family in priority
  order: separate controllers make each other fail on the chip quota, and a shared pool keeps a VM busy with the next batch's
  jobs instead of deleting it. `--accel` lists the sizes to create, biggest first (each size is tried in every zone before the
  next); `--drain-other-sizes` stops giving new jobs to adopted VMs of other sizes (they are deleted once idle).
  It creates spot VMs `mhc-<tag>-NNNNNN` (tag = the first batch's `expNN` by default; VMs of every listed batch's tag are
  adopted), skips regions without a free external IP, tries zones cheapest first
  (`cloud/prices.json`), sets each VM up (`cloud/setup_vm.sh`), runs one job per chip (`cloud/vm_job.sh`, detached), polls every
  VM once a minute (`cloud/vm_status.py`), copies each finished run to `results/<batch>/runs/<job>/` (run.json, metrics.jsonl,
  summary.json, log.txt, exit_code), copies running jobs' metrics to `results/<batch>/live/` every ~10 min, and deletes a VM as
  soon as nothing is left for it. Preempted VMs are deleted and their jobs requeued (they restart from step 0 elsewhere: the
  VMs have no Cloud Storage access, see below). A job whose process dies on a live VM is restarted there and resumes from its
  checkpoint (`--ckpt-every 1000`, bit-exact, `analysis/checks/check_resume.py`). Failed jobs are retried once.
- It is stateless: restarted, it adopts the batch's live VMs and skips jobs whose `summary.json` + `exit_code` are already in
  `results/<batch>/runs/`. Stop it with Ctrl-C / kill; **then delete its VMs** (`gcloud compute tpus tpu-vm list --zone=-`).
- `python3 cloud/cost.py` adds up the ledger (per VM: hours × the zone's spot price). It does not see VMs made by hand unless they
  are logged: `python3 cloud/cost.py --log create|delete NAME ZONE TYPE`.

## Software (identical to Kaggle's TPU image)

`cloud/setup_vm.sh`: uv, Python 3.12 venv, `torch==2.8.0` (CPU wheel), `torch_xla==2.8.0`, `libtpu==0.0.17`, `numpy==2.5.0`, and
the 20 train shards + validation shard 0 of `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full` into `~/data`
(exactly 20: `TrainBatches` draws shard ids from the number of shards present, so a different count changes the data order).
VM images: `v2-alpha-tpuv6e` (v6e), `v2-alpha-tpuv5-lite` (v5e). Ubuntu 22.04.

## Comparability with the Kaggle (v5e) runs

- **GCP v5e = Kaggle v5e, bit for bit.** exp04's mhc-h1-s0 replayed on a GCP v5e VM: train loss and grad norm identical at every
  logged step 0-325 (`check_resume.py`). So batches run on GCP v5e pair exactly with the earlier runs (exp09 also repeats one
  full exp07 d768 run to check this over 6000 steps).
- **v6e is a different chip**: results match v5e only up to floating-point noise, and one setting is broken (next point).
  exp11 measures the difference on 23 d384 runs that exist on v5e.
- **v6e bug: never use fused gradient accumulation on v6e.** With micro-batch 8 (2 micro-steps per step) in one XLA graph
  together with `clip_grad_norm_` and the AdamW step, v6e (torch_xla 2.8 / libtpu 0.0.17) computes wrong gradients: step-0 grad
  norm 8.43 for the residual model and 6.98 for global h4, against 4.388 on CPU (bf16), on v5e, and on v6e itself with
  micro-batch 16 or with `--micro-sync 1` (a graph cut after each micro-step). A fused 2-micro-step backward without the clip
  and optimizer in the graph was also correct, so it is a compiler bug triggered by the combination. **On v6e use micro-batch 16
  at d384** (fits the 32 GB HBM, fastest: 81.9k tok/s for mHC h1 vs 47k at micro-batch 8 + micro-sync) or `--micro-sync 1`.
- Runs are deterministic on v6e (the same run twice gives the same log).
- **Validation batches are built at the micro-batch size** (`validation_batches(..., args.micro_batch, ...)`): at micro-batch 16
  pass `--eval-batches 10 --final-eval-batches 50` so the evals score the same 160 / 800 windows (819k tokens) as v5e's 20 / 100
  batches of 8 (d768: 40 / 200 batches of 4). The windows are contiguous from the start of the shard, so these are identical sets.

## Throughput and cost (spot, 2026-10-03)

| chip | config | tok/s | one 98M-token run | $ / run |
|---|---|---|---|---|
| v5e | d384 mHC h1, micro-batch 8 | 32k | ~55 min | ~0.32 |
| v5e | d768 mHC h1, micro-batch 4 + micro-sync | ~11k | ~2.6 h | ~0.89 |
| v6e | d384 mHC h1 / global h4, micro-batch 16 | 81.9k / 81.2k | ~21 min | 0.10-0.23 |
| v6e | d384 residual, micro-batch 16 | 280k | ~6 min | |

Spot $/chip-hour: v6e 0.27 (asia-southeast1), 0.65 (us-central1/east1/west1); v5e 0.34 (us-central1/east1/west1). Each VM also
spends 1-5 min on setup (gcloud's first ssh, the venv, 420 MB of data).

Quotas (checked with `gcloud beta quotas info list --service=tpu.googleapis.com`):
- v6e spot: 16 chips per zone (none in us-east4-a/b, us-east5-c).
- **v5e single-host types (v5litepod-1/4/8) count against the "serving" quota: 4 chips per zone**
  (`TPUV5sPreemptibleLitepodServingPerProjectPerZoneForTPUAPI`), so at most 12 v5e chips at $0.34 (us-central1-a, us-east1-c,
  us-west1-c) and 8 more at $0.52-0.56 (us-west4-b, europe-west4-b).
- **External IPs: 4 per region** (Compute Engine `IN_USE_ADDRESSES`; `INSTANCES` is 8), shared by all batches. Every TPU VM takes
  one, so single-chip VMs cap a region at 4 chips. Use multi-chip VMs (`--accel v6e-8`: one IP, 8 chips, one job per chip) when
  more than a few chips are wanted. Check with `gcloud compute regions describe REGION` (quotas). The controller backs the whole
  region off for 15 min on "IN_USE_ADDRESSES limit" / "INSTANCES limit". (VMs without external IPs would need Cloud NAT for the
  setup downloads and an IAP firewall rule for ssh; not done here.)
- 31 v5e creates sent at once (exp09's first start) mostly failed on that quota (requests in flight count against it). The
  controller now sends at most 4 creation requests a minute, backs a zone off for 15 min after a quota or capacity error, and
  retries the same zone after 30-90 s only on a request-rate error ("per minute" / "quota metric" in the message).

## Practical notes

- First ssh goes through `gcloud compute tpus tpu-vm ssh` (it installs `~/.ssh/google_compute_engine` on the VM); after that the
  controller uses plain ssh/scp to the external IP as user `marcoshernanz` (much faster than gcloud's wrapper).
- In zsh, `$IP:r...` is a history modifier (strips the extension): write `"${IP}:path"` in scp commands.
- The VMs' service account has no Cloud Storage access (403) and this project does not change IAM, so results come back over scp
  and checkpoints stay on the VM's disk.
- Multi-chip VMs (v6e-4/8): `cloud/vm_job.sh` pins one process per chip with the Kaggle `visible_chips` recipe
  (`TPU_VISIBLE_CHIPS`, per-process ports). Used from exp11 on because of the IP limit. The controller records each VM's own
  chip count, so a batch can be restarted with another `--accel` and still adopt its older VMs.
- Killing the controller does not kill the `gcloud ... create` calls it has in flight; their nodes may still appear (the next
  controller run adopts READY ones and deletes failed ones). Nodes stuck CREATING after their client died ended in "creation
  failed" on their own here.
- A spot VM that is preempted stays (state PREEMPTED) until deleted; the controller deletes it, and `cost.py` counts it until then.
- Preemptions on 2026-10-03: v6e VMs lasted ~27-40 min in the worst cases (a v6e-8 with 8 jobs was preempted after 27 min),
  v5e VMs hours. A preempted job restarts from step 0: checkpoints stay on the VM's disk (pulling a 328 MB d384 checkpoint to the
  laptop ran below 3 MB/s, too slow to do for every job). On-demand v6e costs $2.70/chip-h (us-central1 / us-south1), 4x spot;
  it has its own quota (16 chips per zone) but shares the 4 IPs per region.
