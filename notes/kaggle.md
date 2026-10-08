# Running experiments on Kaggle (how-to)

Account: marcoshernanz (Kaggle CLI 2.0.0 at ~/.local/bin/kaggle, credentials in ~/.kaggle).

## Pattern
- One experiment batch = one folder `experiments/<batch>/` with `jobs.py` (SLUG, JOBS, TRAIN_SHARDS, DEADLINE_SECONDS).
- `uv run python experiments/make_kernel.py experiments/<batch>` bundles `src/mhc_lab/*.py` and the job list into a single
  script `experiments/<batch>/kernel/<slug>.py` plus `kernel-metadata.json` (private, GPU, internet on, machine_shape NvidiaTeslaT4 = 2× T4).
- `kaggle kernels push -p experiments/<batch>/kernel` starts a run.
- `kaggle kernels status marcoshernanz/<slug>` polls; `kaggle kernels output marcoshernanz/<slug> -p results/<batch>` pulls
  `/kaggle/working` (runs/<job>/{run.json,metrics.jsonl,summary.json,log.txt}, status.json) and the kernel log.
- In the kernel: the code is written to /tmp/src, the data shards are downloaded from the public HF dataset
  `marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full` (FineWeb-Edu sample-10BT, 16,384-token BPE, 10M tokens per uint16 shard)
  to /tmp/data, and one worker thread per GPU pops jobs and runs `python -m mhc_lab.train` with CUDA_VISIBLE_DEVICES set.

## Limits to remember
- `kaggle quota` (CLI ≥ 2.2, in the scratch venv `kvenv`) prints GPU/TPU hours used and remaining this week.
- GPU quota: 30 h/week; sessions up to 12 h. TPU: 20 h/week, one batch TPU session at a time, and TPU queues can be long.
- T4 = Turing (sm_75): fp16 tensor cores, no fast bf16, so training uses fp16 autocast with GradScaler; the connection math runs in fp32.

## Watching and stopping a running session (`experiments/kaggle_session.py`, run with the kvenv python)
- `log <slug> [seconds]`: the live kernel log (the CLI's `kernels logs` only works after the session ends).
- `files <slug>`: output files and the session id (from the file URLs). Not a live view: for exp03 it listed files while the
  status said RUNNING, but that session had most likely already ended (its quota stopped growing at ~2.6 h and its log had
  ended); for exp04, 26 min into a healthy run, it listed nothing. Only the live log is reliable mid-run.
- `cancel <session id>`: stops the session (status CANCEL_ACKNOWLEDGED). Used once, on the hung exp03 probe, after its data was saved.

## TPU v5e-8 (exp03, exp04)
- Kernel metadata: `enable_tpu: true`, `machine_shape: "TpuV5E8"`, docker image `gcr.io/kaggle-private-byod/python-tpuvm@sha256:a2111cb9…`
  (torch 2.8 + torch_xla 2.8 + JAX, Python 3.12). Host: 96 vCPUs, 377 GB RAM. One TPU session at a time; sessions end at 9 h.
  The queue before a TPU session started was ~7 h once (exp03).
- 8 independent single-chip processes work with, per process: `TPU_VISIBLE_CHIPS=<chip>`, `TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1`,
  `TPU_PROCESS_BOUNDS=1,1,1`, `TPU_PROCESS_ADDRESSES=localhost:<8476+chip>`, `TPU_PROCESS_PORT=<same>`,
  `TPU_RUNTIME_METRICS_PORTS=<8431+chip>`, `CLOUD_TPU_TASK_ID=0` (recipe `visible_chips` in the launcher).
- Per chip: residual d384 L8 at 125k tok/s (micro-batch 8), 185k (micro-batch 16); first step 17 s.
- **Layout matters more than anything else on a TPU.** The last two dims of every array are stored in (8, 128) tiles, so small
  trailing dims ([.., n, n] with n = 4, [.., h, Dh] with Dh = 48) are padded many times over. mHC in the token layout ran at
  2.4k tok/s (h1) or never compiled (heads). Keep the big dim (tokens) last: `--hc-layout streams`.
- Deep elementwise graphs (the entry-by-entry Sinkhorn) compile slowly and run slowly under XLA; tensor reductions are better.
- bf16 autocast; no fp16. The first step includes compilation; every new shape or mode (eval) compiles again.

- The scratch venv `kvenv` (kaggle >= 2.2, needed by `experiments/kaggle_session.py` and `kaggle quota`) lives in a temporary
  directory that can be wiped. Recreate: `uv venv <scratch>/kvenv && uv pip install --python
  <scratch>/kvenv/bin/python "kaggle>=2.2"`.
