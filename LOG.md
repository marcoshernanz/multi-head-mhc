# Research log: multi-head mHC

Chronological. Every entry: what was done, why, what came out, pointers to files.

## 2026-09-26

- **Question**: can mHC be made "multi-head", has anyone proposed or tested it, and is it viable? Full research cycle: literature,
  design, experiments on Kaggle, report.
- Created this folder. Launched three literature searches (citation crawl of HC/mHC via Semantic Scholar/OpenAlex; web + code + social search;
  implementation-grade spec of HC/mHC/Frac-Connections/MUDDFormer/DeepSeek-V4 reference code). Outputs go to `notes/literature/`.
- Wrote the experiment code (`src/mhc_lab/`): a small GPT (RoPE attention + SwiGLU MLP, pre-norm sublayers) whose connection is either the plain
  residual or multi-head mHC with h heads (h = 1 is mHC). Heads split the D channels into h groups; each group has its own pre (1×n), post (1×n)
  and doubly stochastic res (n×n) predicted per token. Predictor options: `global` (each head's coefficients from the whole normalized stream,
  h× the φ parameters), `local` (each head from its own channels only, the same φ parameter count as mHC), `static` (biases only).
- Local checks (`analysis/checks/check_model.py`): Sinkhorn gives doubly stochastic matrices (columns exact, rows within 0.04 after 20
  iterations for very peaked inputs); h heads with identical coefficients reproduce mHC exactly (difference 0.0); each head only mixes its own
  channels (0.0 leak); the column-stochastic mixing preserves the per-channel sum over copies (1e-4); with α = 0 at init the model equals the
  plain residual up to the norm's epsilon (6.7e-4 on logits); every variant trains a few steps.
- Pushed `experiments/exp00_smoke` (kernel `mhmhc-exp00-smoke`): throughput / memory / compile check at d384 and d512.
- Implementation spec landed (`notes/literature/core_papers_spec.md`, raw reference code in `notes/literature/raw/`). My mHC code matches the
  DeepSeek reference exactly: coefficient layout [pre | post | res row-major], `pre = σ+eps`, `post = 2σ`, `comb = softmax(-1)+eps` then
  column-normalize, then 19 × (row, column), applied as `out_j = Σ_i comb[i,j] x_i` (comb = H_res^T). The paper states only α = 0.01; bias and
  φ inits are unspecified anywhere (reference code is inference-only). Decision: default init "hc" = HC's Pre-Norm-equivalent init as far as
  σ/Sinkhorn allow (φ = 0, post bias 0 → exactly 1, pre bias ±2 soft one-hot on copy (sublayer index mod n), res bias 4·I → diagonal 0.95);
  alternative "sym" (pre uniform, res ≈ I, random φ). Reason: with φ = 0 and symmetric biases the n copies stay identical forever and mHC
  degenerates to the residual; some asymmetry is required. Norm eps set to mHC's 1e-20. Weight decay on φ only (HC's rule).
  Collapse before the head: V4-style learned pre-only connection (V4.1 reuses the last pre instead; noted).
- Local 300-step check at tiny scale: both inits break the copy symmetry slowly (copy cosine 0.9991 hc, 0.9997 sym) and learn.
- Citation crawl landed (`notes/literature/citation_crawl.md`): 186 papers citing HC / mHC / MUDDFormer / Frac-Connections; **none** proposes
  channel-grouped multi-head mHC. Closest: Multi-Head Attention Residuals (MHAR, arXiv 2607.27230), heads over channel subspaces for the AttnRes
  depth-read, U-shaped loss in H with optimum H = 4–8, H = 16 gives back part of the gain. I verified MHAR's abstract and 2609.05309's
  ("How Does mHC Use Its Residual Streams?": a site uses ~2 of 4 streams in DeepSeek-V4-Flash; late mixers → identity costs only 1.9% PPL,
  early mixers → +41%) directly on arXiv.
- Web + code + social search landed (`notes/literature/web_and_code_search.md`). New facts beyond the citation crawl:
  - **lucidrains/hyper-connections already contains a "joint" multi-head mHC in code**: `mHCv2.py` with `num_fracs = m > 1` (HC ×
    Frac-Connections) runs Sinkhorn on an (n·m)×(n·m) matrix over the joint (stream, channel-fraction) index, with cross-fraction reads and
    writes. I read the source myself (2026-09-26, `mHCv2.py` lines 230–500): confirmed; its m > 1 init writes every output fraction into
    every fraction of every stream (β = 2σ(1) ≈ 1.46 on all pairs) and starts the 16×16 mix far from identity (diagonal ≈ 0.15). No paper,
    issue or blog reports results for m > 1 with mHC. The v1 mHC file breaks for m > 1 (shape bug). This is design D in `notes/design.md`.
  - VWN "Generalized HC" (arXiv 2511.11238): per-segment reads/writes with one unconstrained n×n mix: segments share the mixing.
  - mHC-lite / BE-HC: H_res as a convex combination of permutations — design E is published in that form.
  - Naming collision: DeepSeek TileKernels calls the n streams "heads". The report will say "per-subspace" / "block-diagonal" heads.
- Implemented D in `src/mhc_lab/model.py` (`hc_joint`): A's per-head pre/post, but one (nh)×(nh) Sinkhorn res over (copy, head) pieces,
  init next to A (−τ logits between heads → 0.4% cross-head mass per row). Theory note: D loses A's per-channel conservation (only the
  sum over all pieces is conserved); gain ≤ 1 and closure still hold. Checks pass (item 9): block-diagonal joint res reproduces A (2e-7),
  sum over all pieces conserved (1e-6), per-channel sum over copies is *not* (changes by 5.3 on a random joint DS matrix).
  Probe now reports `cross_head_mass` for D.
- exp00 smoke (2×T4, d384 L8, 150 steps): residual 60.6k tok/s, mHC h1 30.7k, h8-global 17.2k, h8-local 19.9k, h64-local 4.4k; compile
  ≈ 340 s for mHC (vs ≈ 30 s residual). Reduction-form Sinkhorn + einsum writes were the bottleneck (many small kernels). Rewrote Sinkhorn
  entry-by-entry (elementwise, fusable) and the write as a broadcast-sum; both checked equal to the old forms (1.5e-7 values, 1.8e-7 grads).
  Pushed `experiments/exp01_bench` (kernel `mhmhc-exp01-bench`) to measure the rewrite.
- Verified VWN (Virtual Width Networks, arXiv 2511.11238, ByteDance Seed, Nov 2025) on arXiv HTML myself: "Generalized
  Hyper-Connections" widen the state to D' = (n/m)·D, cut it into n segments of width D/m, and mix segments with **one shared matrix per
  layer** (read A: n×(m+n), write B: m×n, static + dynamic, zero-init dynamic part, cyclic/identity-block static init), **no doubly
  stochastic constraint**. Segments are the mixing unit (an HC × Frac-Connections joint index, like design D, unconstrained); there are no
  per-channel-group independent routers. So VWN is prior art for D's joint index, not for A.
- **Compute constraint found**: the Kaggle account allows 2 concurrent batch GPU sessions, and another kernel on the account
  was running alongside my bench, so pushing the pilot failed ("Maximum batch GPU session count of 2 reached").
  The GPU quota (30 h/week) is shared with other work on the account, so I keep GPU use lean and move the bulk of the runs to the TPU v5e-8
  (separate 20 h/week quota; 8 chips).
- TPU path: `src/mhc_lab/train_xla.py` (same model/data/schedule/outputs as `train.py`; bf16 autocast; no host reads inside a step; one
  `torch_xla.sync()` per step; probe on a CPU copy) and `experiments/launcher_tpu_template.py` (finds an environment recipe that pins one
  process to one chip by starting 8 tiny torch_xla processes at once; then one job per chip; falls back to one process). Dry-run on the CPU
  with a stub `torch_xla` module passed. `make_kernel.py` builds TPU kernels when a batch sets `ACCELERATOR = "tpu"`, on the TPU VM image
  earlier TPU kernels on the account used (torch 2.8 + torch_xla 2.8, 224 CPUs, v5litepod-8).
- Pushed `experiments/exp03_tpu_probe` (kernel `mhmhc-exp03-tpu-probe`): per-chip recipe, compile time and tokens/s for 8 variants.
- Wrote the pilot `experiments/exp02_pilot` (LR × init for the residual and mHC), waiting for a GPU slot or the TPU.
- **Compile-time problem, diagnosed locally.** exp01_bench was still running after 77 min (no cancel in the Kaggle CLI; `kaggle kernels
  logs` from CLI 2.2.2 returns nothing for a running kernel). On the Mac CPU, whole-model `torch.compile` of a 2-layer mHC model takes 100 s
  with the entry-by-entry Sinkhorn vs 17 s with the reduction form: unrolling 16 entries × 20 iterations × 17 sublayers makes one huge
  graph, and Inductor's compile time grows faster than linearly in graph size. Fix: **regional compilation** (`Model.compile_regions()`,
  `--compile 1`, now the default): `torch.compile` the step "connection + sublayer" (`hc_step`) and `residual_step`; every sublayer of a
  type reuses the same compiled graph (recompiles only per sublayer type: 2 graphs per model). Checked on the CPU: identical loss (diff 0.0)
  and gradients (≤ 4.5e-8) vs eager for mHC h1, h8-local, residual and joint h4; first compiled step ≈ 47 s vs ≫ 100 s whole-model.
  The stream is made contiguous after the expansion so that the first sublayer reuses the same graph. `--compile 2` keeps whole-model.
- **exp01_bench finished** (2×T4, d384 L8, micro-batch 8, whole-model `torch.compile`; `results/exp01_bench/`). First step (includes
  compilation) / tokens per second:

  | job | first step | tok/s |
  |---|---|---|
  | residual | 29 s | 75.1k |
  | mHC h1, reduction-form Sinkhorn | 281 s | 43.0k |
  | mHC h1, entry-by-entry Sinkhorn | 2294 s | 35.1k |
  | h8-local, entry-by-entry | 2432 s | 35.3k |
  | h8-global, entry-by-entry | 2522 s | 23.6k |
  | h64-local, entry-by-entry | 2496 s | 14.3k |

  mb16, h16 and eager jobs were skipped by the deadline. Reading: (1) the entry-by-entry rewrite was a mistake on this hardware: 8× the
  compile time and 18% *slower* steps than the reduction form (0.233 vs 0.190 s median step), so the reduction form is back as the default
  (`hc_sinkhorn_form = "reduction"`; the entry-by-entry form stays as an option and the two are still checked equal). (2) h8-local costs the
  same as h1 (35.3k vs 35.1k): channel-group heads are free in wall time when the predictor is local. (3) the global predictor for h8
  costs +49% step time (it has h× the φ matrix). (4) mHC h1 runs at 57% of the residual's throughput in plain PyTorch; the DeepSeek
  fused kernels are what make it cheap in the paper (6.7% overhead), so wall-time comparisons here only rank variants against each other.
- Pushed the pilot `experiments/exp02_pilot` (kernel `mhmhc-exp02-pilot`, GPU, regional compile, entry-by-entry Sinkhorn — pushed before
  the switch back; it also measures the entry-by-entry form under regional compile).
- Probe: added **per-head depth connectivity** (`probe.depth_connectivity`). Unrolling the stream, every sublayer's input is a weighted
  sum of the embedding and all earlier sublayer outputs, u_l = Σ_s c[l, s] y_s, with c[l, s] = pre_l · (R_{l-1}ᵀ … R_sᵀ post_{s-1});
  the plain residual has every c = 1. With channel-group heads each head has its own c matrix, so "do heads specialize?" (H5) becomes
  "do heads route through depth differently?"; `depth_head_spread` = mean over (reader, source) of the std across heads. Checked: at the
  HC init c = 0.88 + 3·0.12 = 1.24 everywhere (the read's total weight), with one-hot reads and identity mixes c = 1 exactly.
- Drafted `experiments/exp04_main/jobs.py` (14 variants × 3 seeds, seed-major order), to be finalized from the pilot and the TPU probe.
- TPU launcher fallback: if none of the environment recipes pins one process per chip, the launcher now tries `torch_xla.launch`
  (`src/mhc_lab/xla_pool.py`): one process per chip, each claims the next job by creating a file with O_EXCL and runs
  `train_xla.main()` in-process, with its fd 1/2 redirected into the job's log. `train_xla` now counts compiles per job (several jobs
  can share a process). Dry-run with a stub `torch_xla` (2 fake chips, 3 tiny jobs): each job ran once, logs and summaries correct.
- Literature gaps closed (method sections read): Multi-Gate Residuals (one scalar gate per stream per layer, broadcast over D; it does
  cite HC/mHC, correcting the abstract-only note), JPmHC (orthogonal O(n) instead of Birkhoff, one n×n matrix ⊗ I_p), mHC-for-PEFT
  (stream-level), Motif 3 (1×n pre/post, post multiplier annealed 2 → 1), Hyperloop Transformers (loop-level HC, sigmoid-diagonal
  H_res). None splits the hidden dimension. Updated `notes/literature/web_and_code_search.md`.
- Theory, written into `notes/design.md` §A: unrolled, HC is a linear state-space recurrence over depth (write = post, transition = res,
  read-out = pre, state size n), so the depth-connectivity matrix c[l, s] (how much sublayer l reads sublayer s) is n-semiseparable:
  every strictly-lower block has rank ≤ n. Channel-group heads give h independent rank-≤ n patterns; raising n raises the rank of the one
  pattern. Checked numerically (static model, random coefficients, 17 × 17 c): the max rank of strictly-lower blocks is exactly 2 for
  n = 2 and 4 for n = 4 (a free lower-triangular matrix reaches 8), and the two heads' patterns differ.
- Probe sanity check on a trained model (CPU, d128 L4, 600 steps, fp32, eager; scratch runs, not results): mHC h1 has one depth pattern
  (spread 0); h4-local's four heads drift apart during training from identical starts (spread 0.14; e.g. one head's final read weights the
  later sublayers ≈ 2.2–2.6 vs ≈ 1.7–1.9 for the others). The data alone breaks the symmetry between heads (each head's coefficients see
  and are graded through different channels), so no symmetry-breaking init is needed.
- **exp02 pilot finished** (2×T4, d384 L8, 3000 steps × 16k tokens = 49M tokens, seed 0, regional compile, entry-by-entry Sinkhorn;
  `results/exp02_pilot/`, plots `figures/exp02_pilot_*.png`). Final validation loss (819k tokens):

  | run | val loss | tok/s |
  |---|---|---|
  | mHC h1, hc init, lr 3e-3 | **4.3796** | 38.2k |
  | mHC h1, sym init, lr 3e-3 | 4.3942 | 38.2k |
  | mHC h1, lr 1.5e-3 | 4.4156 | 38.2k |
  | A-local h8, lr 3e-3 | 4.4328 | 29.3k |
  | residual, lr 3e-3 | 4.4805 | 58.7k |
  | residual, lr 1.5e-3 | 4.4838 | 58.3k |
  | mHC h1, lr 6e-3 | 5.0214 | 38.0k |
  | residual, lr 6e-3 | diverged (NaN at step ~1063) | 60.9k |

  Readings (one seed):
  - H0: mHC beats the residual by 0.10 here, and the lead appears early (train loss, same batches: ≈ +0.11 for the residual from step 250
    on). That is larger than the published gains at scale.
  - LR 3e-3 is best for mHC and ties with 1.5e-3 for the residual. At 6e-3 the residual diverges; mHC survives but degrades slowly
    (+0.69 by the end, no NaN).
  - The hc init beats sym by 0.015 (one seed). Keep hc.
  - **h8-local is 0.053 worse than h1.** The paired train-loss gap grows over training: +0.02 around step 900, +0.06 at the end.
- Probe of the pilot models:
  - h1 uses strongly token-dependent routing early. At sublayer 1, res is far from identity (diagonal mass 0.30, token std 0.25) and
    the read uses about 1 copy. The copies differ (cosine 0.82).
  - h8-local stayed near its init. Res is ≈ identity everywhere but one layer, read weights barely vary with the token (std 0.01–0.1 vs
    0.1–0.3 for h1), writes go to all copies about equally (post eff. copies 3.2–4.0), and the copies are more alike (cosine 0.94). The heads
    did differ in their static parts (head distance of post 0.2–0.9).
- Hypothesis for the deficit: the **local predictor learns ~h× slower**. Each head's logits sum over n·D/h inputs instead of n·D, and
  under Adam the logit change per step scales with that fan-in. Tiny CPU test (d128 L4, 400 steps; average token std of the read weights):

  | variant | read-weight token std |
  |---|---|
  | h1 | 0.057 |
  | h4-local, gain 1 | 0.006 |
  | h4-local, gain 2 | 0.018 |
  | h4-local, gain 4 | 0.035 |
  | h4-global | 0.024 |

  The dynamics grow with the gain, as predicted. Global heads (same fan-in as h1) are also slower than h1: each head's gradient comes from
  1/h of the channels, so it is noisier and Adam's normalized step is smaller. That part is intrinsic to splitting. Added
  `hc_local_gain` (`--hc-local-gain`, default 1; phi starts at 0, so the gain does not change the init).
- The TPU probe had been queued for 3+ hours, so I pushed a focused GPU batch `experiments/exp05_gpu_focus` (~2.5 GPU-hours):
  - seeds 1 and 2 of h1 and h8-local;
  - h8-local with gain 8 on seeds 0–2;
  - static h1, static h8 and global h8 (seed 0);
  - residual seed 1.
  It uses the reduction-form Sinkhorn under regional compile. The seed-0 runs of the pilot pair with it (the two Sinkhorn forms agree to
  1e-7, so they differ only by floating-point noise).
- 17:40 UTC: exp05 (GPU) queued 1.5 h and the TPU probe queued ~5 h. `kaggle quota` (CLI 2.2.2): GPU 8.31 h used / 30 h
  (includes other work on the account), TPU 0 / 20 h, resets 2026-10-03. So the waits are Kaggle capacity, not quota.
- 18:10 UTC: both kernels are still queued; the other kernel is no longer running. Checked the local Mac (Apple M4, 8-core GPU,
  16 GB) as a fallback: d384 L8, micro-batch 8 × 1024, eager, MPS. Residual 6.6k tok/s, mHC h1 3.0k tok/s (fp16 ≈ bf16), so one
  49M-token mHC run would take ~4.5 h. Not viable for the main scale, so Kaggle remains the compute.
- 18:51 UTC: exp05 (GPU) started after 2 h 43 min in the queue. The TPU probe is still queued (~6 h).
- 20:07 UTC: the TPU probe started after ~7 h in the queue.
- **exp05 (GPU focus) finished** (21:38 UTC; `results/exp05_gpu_focus/`). The pilot's seed-0 runs are symlinked under the new names
  in `results/exp02_pilot_aliases/`. Plots: `figures/exp05_focus_*.png`. Val loss, and the paired difference vs mHC h1
  (+ = worse, mean ± std over seeds):

  | variant | seeds | val loss | Δ vs h1 | per seed |
  |---|---|---|---|---|
  | mHC h1 | 3 | 4.3867 ± 0.0099 | – | – |
  | A-global h8 | 1 | 4.3827 | +0.003 | [+0.003] |
  | A-local h8 | 3 | 4.4173 ± 0.0304 | +0.031 ± 0.028 (t = 1.9) | [+0.053, +0.039, −0.000] |
  | A-local h8, gain 8 | 3 | 4.4476 ± 0.0395 | +0.061 ± 0.031 (t = 3.4) | [+0.057, +0.093, +0.032] |
  | static h1 | 1 | 4.4460 | +0.066 | |
  | static h8 | 1 | 4.4531 | +0.074 | |
  | residual | 2 | 4.4884 ± 0.0111 | +0.100 ± 0.002 | [+0.101, +0.098] |

  Throughput (T4, regional compile, reduction-form Sinkhorn):
  - h1 38.6k tok/s (the same as the entry-by-entry form under regional compile: 38.2k);
  - static h1 44.0k;
  - local h8 25.8k, global h8 23.9k, static h8 28.9k;
  - residual 62.6k.

  Readings:
  - **H0 holds** and is very consistent: mHC − residual = −0.100 on both seeds.
  - About a third of mHC's gain is static: static h1 recovers 0.034 of the 0.100. Two thirds needs token-dependent routing.
  - **H8 is refuted.** Making local heads learn faster (gain 8) made them worse, not better (t = 3.4, all seeds worse). The gain did
    speed up the dynamics (res token std 0.08–0.10 vs 0.04, res further from identity: 0.37–0.49 vs 0.17–0.25), so more dynamic
    mixing is not what was missing. One gain-8 seed had a gradient spike at step 500 (norm 3.3).
  - Local h8 is not better than h1: worse on average and on 2 of 3 seeds (tie on seed 2). Its paired spread (0.028) is 15× that of
    residual vs h1: splitting into heads makes the outcome seed-dependent.
  - Global h8 ties h1 on one seed, with 8× the predictor parameters and 1.6× the step time.
  - Static h8 ≈ static h1 (+0.007): per-group static routing adds nothing.
- Probe of the exp05 models:
  - Heads do specialize: depth-connectivity spread across heads is 0.23–0.30 for h8 (global and local), 0.16 for static h8. The
    heads take different routes through depth, but it buys no loss.
  - Copy diversity (mean cosine between the n copies at the end) follows the ranking: h1 0.74–0.84 and global h8 0.86 (best),
    local h8 0.95–0.96, static 0.98–0.99 (worst). mHC's benefit seems to live in keeping the copies different (using the n-dim
    depth state), and local heads keep them more alike.

## 2026-09-27

- **exp03 (TPU probe) read, then cancelled** (`results/exp03_tpu_probe/`, live log in `results/exp03_tpu_probe/live_log.txt`).
  It was still RUNNING 4.6 h after starting (deadline 2 h). Found that the Kaggle API streams a running session's log and
  lists (and lets `kaggle kernels output` download) its files so far; the file URLs give the session id, and the SDK has
  `cancel_kernel_session`. Wrapped as `experiments/kaggle_session.py` (log / files / cancel). Read everything, then cancelled.
  - Per-chip pinning works: recipe `visible_chips` (TPU_VISIBLE_CHIPS + single-chip bounds + own port), all 8 chips at once.
  - Residual: 125k tok/s per chip at micro-batch 8 (2× a T4), 185k at micro-batch 16; first step 17 s.
  - mHC is unusable in the token layout on a TPU:
    | run (150 steps) | first step | tok/s | end |
    |---|---|---|---|
    | mhc-h1 (entry-by-entry Sinkhorn) | 443 s | 2.4k | stuck in the final eval's compile |
    | mhc-h1, micro-batch 16 | 175 s | 4.8k | same |
    | joint h4 (reduction Sinkhorn) | 99 s | 26k | finished |
    | global h8 | never | – | still compiling after 4.6 h |
    | local h8 / local h16 | never | – | killed by signal 9 after 2.3 h / 1.6 h (host OOM, most likely) |
  - Diagnosis: a TPU stores the last two dims of an array in (8, 128) tiles. In the token layout they are small: res
    `[B, T, h, n, n]` pads (4, 4) to (8, 128), 64×; per-head views `[.., h, Dh]` pad Dh = 48 or 24 to 128. With ~40 saved
    Sinkhorn intermediates per sublayer that is ~21 GB for h1 and hundreds of GB for h8/h16, against 16 GB of HBM per chip:
    the compiler rematerializes forever. The entry-by-entry Sinkhorn avoids the padding but is a deep elementwise DAG
    (every entry feeds a row sum and a column sum, 20 times), which XLA's fusion handles badly (443 s compile, 2.4k tok/s).
    The joint variant finished because its res is (16, 16) → 8× padding, not 64×.
- **Fix: the "streams" layout** (`hc_layout = "streams"`, `Model.hidden_streams`, `HyperConnection.*_streams`). The stream is
  `[n, D, N]` with N = B·T tokens last; heads are `[n, h, Dh, N]` (splitting D, a leading dim, moves no data); coefficients are
  `[reads, n, h, N]`, `[n, h, N]`, `[n, n, h, N]`; Sinkhorn reduces over leading dims (`sinkhorn_reference(..., dims=(0, 1))`).
  The sublayer still sees `[B, T, D]` (one transpose in, one out). Same weights, same math:
  `analysis/checks/check_layouts.py` compares logits and every gradient for 10 variants in float64: rel. error ≤ 1e-15 (logits) and
  ≤ 2e-11 (gradients). (In float32 the head connection's tiny scale gradient differs by up to 5e-4 relative from summation
  order alone, hence float64.) The probe (`record`) keeps the token layout; a CPU run of train.py with streams + probe works.
- **Launcher hardening** (`experiments/launcher_tpu_template.py`), all from exp03's failure modes:
  - recipe smoke tests share one 300 s budget (were sequential: up to 40 min per recipe); stale libtpu lock removed first;
  - a single-process smoke test first (`single.json`), as a diagnostic;
  - watchdog: at HARD_STOP = DEADLINE + 1 h, SIGTERM/SIGKILL every descendant process (parent-pid tree) and exit, so
    outputs are saved before Kaggle's 9 h limit;
  - per job: killed if no first step within 30 min (`first_step_timeout` per job), or after 3 h;
  - a job starts only if the longest job so far would still end 10 min before the hard stop;
  - `gate_min_tok_s`: if that job ends slower (or fails), nothing else starts.
  Dry-run locally with fake jobs (hang → killed at its first-step limit; slow gate → queue cleared).
- **Pushed exp04 (main sweep) to the TPU** at 00:56 (`experiments/exp04_main`, kernel `mhmhc-exp04-main`): 8 benchmark runs
  (150 steps; bench-mhc-h1 in the streams layout is the gate at 15k tok/s; bench-mhc-h1-tokens re-measures the token layout
  with the reduction Sinkhorn on the same hardware), then 16 variants × seeds 0-4, seed-major, 6000 steps (98M tokens),
  all mHC runs in the streams layout. No job starts after 6.5 h; hard stop at 7.5 h.
- **Literature re-check while the sweep queues** (`notes/literature/web_and_code_search.md` §5). arXiv API query
  `search_query=all:"hyper-connections" OR abs:"hyper-connection" OR abs:mHC`, sorted by submission date: the newest HC paper is still
  2609.05309. Eight HC-adjacent papers not triaged before (abstracts read): none proposes channel-group / multi-head connection
  coefficients. Verdict unchanged.
- **exp04 started at 12:26** (11.5 h in the TPU queue). From the live log at 26 min:
  - recipe `visible_chips` again; all 8 benchmarks finished (no first-step kills);
  - **gate passed: mHC h1 in the streams layout runs at 29.6k tok/s per chip** (exp03, token layout + entry-by-entry
    Sinkhorn: 2.4k). The token layout with the reduction Sinkhorn also finished its benchmark (350 s vs 320 s wall time);
    its tok/s comes with the final download;
  - residual-s0 (6000 steps) took 900 s, 109k tok/s.
  - Correction to the exp03 entry: listing a session's files works only once it has ended. exp03's listing worked while its
    status said RUNNING, and its TPU quota had stopped growing at ~2.6 h, so that session had most likely already died; the
    cancel probably only cleared a stale status.
- **exp04 at 2.5 h** (live log saved as `results/exp04_main/live_log_2h30.txt`): seed 0 complete at 2.26 h, all 16 runs exit 0, no kills.
  Wall time per 6000-step run on one v5e chip (98M tokens): residual 850-900 s; every mHC variant 3190-3930 s (static 3190-3490,
  h1 3670, global/local h2-h16 3540-3700, n = 8 3700, joint global 3930). In the streams layout the head count costs almost
  nothing; the connection itself costs ~4x the residual's time on a TPU. Expected: seeds 0-2 complete, a few seed-3 runs.
- **exp04 results** (`results/exp04_main/`; `analysis/sweep.py ../results/exp04_main --out ../figures/exp04 --baseline mhc-h1`;
  figures `figures/exp04_hcurve.png`, `exp04_global_curves.png`, `exp04_local_curves.png`). Finished at 7.3 h: seeds 0-2 for all
  16 variants, seed 3 for residual, mhc-h1, global h2-h16 and local h2/h4; the rest was skipped by the deadline. No kills, no
  divergence. Final val loss (819k tokens), paired Δ to mhc-h1 (negative = better), * = exists (|Δ| > 2 SE, all seeds agree):
  | variant | seeds | Δ vs mhc-h1 | t | better seeds |
  |---|---|---|---|---|
  | a-global-h4 | 4 | −0.0312 ± 0.0082 * | −3.8 | 4/4 |
  | a-global-h16 | 4 | −0.0234 ± 0.0032 * | −7.3 | 4/4 |
  | a-global-h8 | 4 | −0.0206 ± 0.0106 | −1.9 | 3/4 |
  | a-global-h2 | 4 | −0.0200 ± 0.0085 | −2.3 | 3/4 |
  | d-joint-h4-global | 3 | −0.0192 ± 0.0135 | −1.4 | 2/3 |
  | a-local-h2 | 4 | −0.0176 ± 0.0102 | −1.7 | 3/4 |
  | a-local-h4 | 4 | −0.0056 ± 0.0054 | −1.1 | 3/4 |
  | mhc-n8 | 3 | −0.0055 ± 0.0106 | −0.5 | 2/3 |
  | a-local-h16 | 3 | +0.0026 ± 0.0103 | +0.2 | 1/3 |
  | a-local-h8 | 3 | +0.0105 ± 0.0175 | +0.6 | 1/3 |
  | d-joint-h4-static | 3 | +0.0133 ± 0.0085 | +1.6 | 1/3 |
  | static-h4 | 3 | +0.0198 ± 0.0082 * | +2.4 | 0/3 |
  | static-h1 | 3 | +0.0228 ± 0.0099 * | +2.3 | 0/3 |
  | static-h8 | 3 | +0.0238 ± 0.0141 | +1.7 | 0/3 |
  | residual | 4 | +0.0450 ± 0.0063 * | +7.1 | 0/4 |
  mhc-h1 itself: 4.1470 (seed spread ± 0.0202 unpaired; pairing cuts the SE of a difference to 0.003-0.010).
  Other pairs (same seeds):
  - H2 structure vs predictor: global − local = −0.002 (h2), −0.026 * (h4, 4/4), −0.027 (h8, 3/3, t −4.0), −0.025 (h16, 3/3,
    t −2.9). Static heads do nothing: static h4 − static h1 = −0.003 ± 0.006, h8 − h1 = +0.001 ± 0.004. So the gain needs a
    per-token predictor that reads the whole stream; the block-diagonal structure alone (static) or with a per-head predictor
    (local, same parameters as h1) gives nothing clear.
  - H4 joint vs block-diagonal: joint global h4 − global h4 = +0.005 ± 0.013 (no difference); joint static h4 − static h4 =
    −0.007 ± 0.003 (3/3, small).
  - H6 heads vs more copies: global h4 − n8 = −0.018 ± 0.012 (3/3, t −1.5); local h4 − n8 = 0.000 ± 0.003.
  - H0: mhc-h1 − residual = −0.045 * (4/4); mhc-h1 − static h1 = −0.023 (3/3): the dynamic part of mHC is half its gain here.
  Curves: the global heads' lead is there by 33M tokens and stays flat; the local heads' deficit (h8, h16) shrinks over training.
  Probe: the copy-cosine story from exp05 (less diverse copies = worse) no longer fits: mhc-h1 has the most diverse copies
  (0.778) but global heads (0.85-0.87) are better. Global heads have the most token-dependent res (token std 0.063-0.071 vs 0.053).
  Throughput: every head count costs the same in the streams layout (28.4-29.3k tok/s over the full runs vs 28.7k for h1);
  benchmarks: residual 118.7k, h1 29.6k (micro-batch 16: 53.0k), token layout with reduction Sinkhorn 31.0k, local h16 30.1k,
  global h16 26.4k, joint h4 24.6k, n8 24.8k.
  **Caveat that decides the next batch:** global heads have more parameters. The predictor φ maps the flattened stream (n·D) to
  h·(2n + n²) coefficients, so it grows with h: at d384 global h4 has 29.13M parameters (+6.5% over h1's 27.34M), h16 36.29M
  (+32.7%). Local heads (h·(n·D/h)·K, same as h1) have none extra. "Global beats h1" may just be "more parameters beat fewer".
- **exp06 (controls, scale check, more seeds) pushed at 19:49** (`experiments/exp06_controls`, kernel `mhmhc-exp06-controls`), queued.
  In queue order:
  1. d768 L12 (112M parameters, micro-batch 4, LR 1.5e-3 = 3e-3 halved for twice the width, not tuned): mhc-h1 vs global h4,
     seeds 0-2 (global h4 is +4.8% parameters at this width). 6 chips; expected ~4 h each (extrapolated: at d384 the mHC cost
     is almost all a fixed cost per micro-step, 0.24 s of 0.55 s, i.e. torch_xla tracing of the many small ops; d768 has
     1.5x the layers and 2x the micro-steps). Final eval over 200 x 4 windows = the same 819k tokens.
  2. H2 matched-parameter controls: mhc-h1 with SwiGLU width 1216 (29.11M, matches global h4) and 1992 (36.26M, matches
     global h16), seeds 0-3, on the 2 remaining chips from the start.
  3. Seed 4 for mhc-h1, global h4, global h16 and both controls; 4. H7: mhc-h1 and global h4 at 4x LR (1.2e-2), seeds 0-1;
  5. `repeat-mhc-h1-s0` (same config and seed as exp04's, to measure cross-session drift since the controls pair across
     sessions); 6. seed 5 for the five variants of 3.
  Launcher: per-job `expected_seconds` (the start guard takes the first queued job that can still end 10 min before the hard
  stop, instead of assuming the longest job so far) and per-job `timeout`. `plots.py` now leaves out the `bench-*` runs.

## 2026-09-28

- **exp06 started at ~05:18** (9.5 h in the queue). Live log at 30 min (`results/exp06_controls/live_log_30min.txt`): recipe
  `visible_chips`, all 8 chips. **All six d768 jobs exited with code 1 after 240-300 s** (global h4 a little sooner than h1);
  the d384 jobs took over all 8 chips. The live log carries only the exit code; the reason is in each run's `log.txt`, which
  comes with the final download. Most likely HBM out of memory: estimated ~9 GB of activations per micro-step at d768 x
  micro-batch 4 (the same per-layer footprint as d384 x 8, 1.5x the layers) plus 1.8 GB of weights and Adam state, and the
  whole step (4 micro-steps) is one XLA graph, so the compiler may keep more than one micro-step alive (16 GB per chip).
  Prepared, not yet used: `--micro-sync 1` in train_xla (a sync after every micro-step, so each is its own graph), and the
  launcher now prints the last 12 lines of a failed job's log into the live log.
- **exp06 ended at 3.1 h** (`results/exp06_controls/`): all 23 d384 runs exit 0 (3390-3760 s each).
  - **d768 failure diagnosed** (each run's `log.txt`): `XLA:TPU compile permanent error. Ran out of memory in memory space
    hbm. Used 22.19G of 15.75G` (mhc-h1; global h4: 20.35G). The largest buffers are f32[48, 1024, 1024] = 4 sequences x 12
    heads of attention scores (192 MB each, 20 of them among the top allocations, already marked `remat`): XLA compiled the
    4 micro-steps as one program and could not fit them even with rematerialization.
  - **H2 answered: the global heads' gain is a parameter effect** (`analysis/exp06.py`, pooled with exp04; 6 seeds each):
    | comparison | Δ | better seeds |
    |---|---|---|
    | global h4 − mhc-h1 | −0.0256 ± 0.0066 * | 6/6 |
    | ctrl-mlp1216 − mhc-h1 (same parameters as global h4) | −0.0311 ± 0.0119 | 5/6 |
    | **global h4 − ctrl-mlp1216** | **+0.0055 ± 0.0127** | 2/6 |
    | global h16 − mhc-h1 | −0.0189 ± 0.0051 | 5/6 |
    | ctrl-mlp1992 − mhc-h1 (same parameters as global h16) | −0.0652 ± 0.0071 * | 6/6 |
    | **global h16 − ctrl-mlp1992** | **+0.0463 ± 0.0043 *** | 0/6 |
    Spending global h4's extra 1.79M parameters on a wider MLP gives the same gain (difference +0.006, detectable limit
    ~0.025); spending global h16's extra 8.9M that way gives 3.5x the gain, so at h16 the heads are a clearly worse use of
    the same parameters. Seeds 4-5 shrank global h4's lead over mhc-h1 a little (−0.031 → −0.026; seed 5: −0.007).
    Curves (`figures/exp06_controls_curves.png`): ctrl-mlp1216 and global h4 track each other from 33M tokens on.
    Throughput is the same for all five (28.5-29.7k tok/s): on this TPU the mHC overhead dominates, the extra FLOPs are free.
  - **Cross-session repeat: bit-identical.** exp06's `repeat-mhc-h1-s0` matches exp04's `mhc-h1-s0` at every evaluation
    to 6 decimals (final 4.175167). Pairs spanning sessions are exact, and runs are deterministic on the TPU.
  - **H7 at 4x LR (1.2e-2): both variants fail** (val loss at 6000 steps: mhc-h1 7.03 / 6.71, global h4 6.14 / 6.03; at the
    tuned LR both are ~4.1). Both stall at ~6.5 within 300 steps, grad norms spike to 10^2-10^3 from step ~800; global h4's
    spikes are smaller (median grad norm 4-9 vs 35) and it starts to recover once the cosine schedule has lowered the LR
    (from ~step 4000), mhc-h1 barely does. Weak evidence of more robustness, but both runs are failures.
- **exp07 pushed at 08:32** (`experiments/exp07_scale`, kernel `mhmhc-exp07-scale`). TPU quota: 13.0 h used, 7.0 h left until
  2026-10-03, so DEADLINE 4.75 h (hard stop 5.75 h).
  - d768 again, with `--micro-sync 1` (micro-batch 4, eval every 500 steps): mhc-h1, global h4, and the matched control
    ctrl-mlp2240 (117.21M vs 117.25M), seeds 0-1, 6 chips.
  - Fallbacks: a new launcher key `only_if_failed`: if a d768 job fails, the same run with micro-batch 2 takes its chip;
    if it succeeds, its fallback is dropped. Dry-run locally with fake jobs (failed job → its fallback ran, the other fallback
    was skipped, a job too long for the deadline was skipped, the failed job's last log lines appeared in the live log).
  - d384 on the other 2 chips (and on all 8 once d768 ends): seeds 6-9 for mhc-h1, global h4 and ctrl-mlp1216, and H7 at
    2x LR (6e-3), seeds 0-1.
- **exp07 still queued at 19:23** (11 h after the push); the watchers had stopped; restarted them.
- **Parameter frontier** (`analysis/params_frontier.py`, `figures/params_frontier.png`): mhc-h1 and the two MLP controls trace
  "what parameters buy in the MLP"; every d384 variant is read against that line at its own parameter count. Excess over the
  line: local h2 −0.018 (no extra parameters), global h2 −0.010, global h4 +0.006, global h8 +0.022, n8 +0.033, joint h4
  global +0.035, global h16 +0.046. Past a few percent, every way of spending parameters on the connection is worse than a
  wider MLP. (Only h4 and h16 have a matched control run; the rest is interpolation along a line that is itself ±0.01.)
- **exp07 started at ~19:47** (11.2 h in the queue). Live log at 25 min (`results/exp07_scale/live_log_25min.txt`): recipe
  `visible_chips`, and no d768 job has exited (in exp06 all six had failed by 5 min): with `--micro-sync 1` the d768 x
  micro-batch 4 programs fit in HBM. (The scratch venv with kaggle 2.2 had been wiped with its temporary directory; recreated with
  `uv venv` + `uv pip install "kaggle>=2.2"`.)

## 2026-09-29

- **exp07 ended at ~00:20** (4.5 h; `results/exp07_scale/`). All 28 jobs that ran exit 0; every fallback was skipped
  (its primary succeeded). d768 x micro-batch 4 with `--micro-sync 1`: 8900-9320 s per run, 11.0-11.2k tok/s.
- **d384, H2 with 10 seeds** (seeds 6-9 added): global h4 − mhc-h1 −0.0188 ± 0.0064 (9/10); ctrl-mlp1216 − mhc-h1
  −0.0238 ± 0.0085 (8/10); **global h4 − ctrl-mlp1216 +0.0050 ± 0.0073** (2/10 better). The detectable difference is now
  ~0.015: at d384, global h4 is worth its parameters and not more.
- **d768 scale check (seeds 0-1, LR 1.5e-3)** — the result goes the other way:
  | comparison | Δ | per seed |
  |---|---|---|
  | global h4 − mhc-h1 | −0.0679 ± 0.0018 | −0.0697, −0.0661 |
  | ctrl-mlp2240 − mhc-h1 | −0.0343 ± 0.0149 | −0.0492, −0.0194 |
  | **global h4 − ctrl-mlp2240** | **−0.0336 ± 0.0131** | −0.0205, −0.0467 |
  Curves (`figures/exp07_scale_curves.png`): the control's gain is set by ~25M tokens and then flat; global h4's keeps
  growing to −0.066/−0.070 (seed 1 gets there by 25M tokens, seed 0 only by 80M). No sign of instability in any d768 run
  (grad norm median 0.39-0.49, max < 2.2, the same few loss jumps in all). Probe: global h4's res is more
  token-dependent (std 0.058 vs 0.038), copies less diverse (cosine 0.81 vs 0.76), as at d384.
- **But the d768 runs look under-tuned**: mhc-h1 at d768 reaches 4.110 vs 4.137 at d384 (10 seeds) after the same 98M
  tokens, and its curve tracks the d384 one step by step (seed 0: 5.25 vs 5.22 at step 1000, 4.55 vs 4.57 at 3000). A model
  with 4x the parameters should learn faster per token. The LR (1.5e-3, 3e-3 halved by the width rule, never tuned) is the
  suspect. That matters here because global heads are the more LR-robust variant:
- **H7 at 2x LR (6e-3, d384)**: both break (mhc-h1 5.17/5.18, global h4 4.94/5.00, vs ~4.1 at 3e-3); global h4 − mhc-h1 at
  2x = −0.208 ± 0.022 (2/2), grad norm median 0.6 vs 1.1-1.4. With 4x (−0.79) this is consistent: the tuned LR 3e-3 sits
  just below mHC's stability edge, and global heads degrade less beyond it.
  So the d768 gap could be (a) a real scale effect (MHAR saw its head effect grow with scale), or (b) an LR effect: at an LR
  that is off for mhc-h1, the heads' robustness shows up as a gain. Needs a d768 LR sweep, which needs the 2026-10-03 reset
  (2.46 TPU h left; a d768 run takes 2.5 h).
- **exp08 pushed at 00:25** (`experiments/exp08_local`): what fits in the remaining quota, one wave of 8 d384 runs with a
  1.5 h hard stop: local h2 seeds 4-9 (the only equal-parameter hint at d384, −0.018 on 4 seeds) and local h4 seeds 4-5.
- **exp09 prepared** (`experiments/exp09_d768_lr`, kernel built, not pushed: needs the quota reset on 2026-10-03 00:00;
  a reminder is set for 00:07). d768 LR sweep: mhc-h1, global h4, ctrl-mlp2240 and **local h4** (exactly
  h1's parameters: the cleanest structural test at scale) at 3e-3, 2.1e-3, 7.5e-4 (1.5e-3 from exp07), seed 0 first, then
  local h4 at 1.5e-3 (seeds 0-1), residual (seeds 0-1), and seed 1 at 2.1e-3 and 3e-3 as time allows. 3 waves x 8 chips,
  hard stop 8.2 h. A replication at each variant's best LR (seeds 2+) would be a second session the same week.
- **exp08 ended** (~1.1 TPU h; `results/exp08_local/`, all 8 jobs exit 0). The local-h2 hint was chance:
  | comparison | seeds | Δ | better seeds |
  |---|---|---|---|
  | local h2 − mhc-h1 | 10 | −0.0030 ± 0.0091 | 5/10 |
  | local h4 − mhc-h1 | 6 | +0.0031 ± 0.0082 | 3/6 |
  | local h4 − global h4 | 6 | +0.0287 ± 0.0072 * | 0/6 |
  | local h2 − ctrl-mlp1216 | 10 | +0.0209 ± 0.0054 * | 0/10 |
  Seeds 4-9 of local h2 average +0.007 (seeds 0-3 had −0.018): regression to the mean, as expected of the best of 15
  variants on 4 seeds. Local h2's per-seed spread is large (std 0.029, seed 7 +0.056). At mHC h1's exact parameter count, heads
  give nothing measurable at d384.
- **Frontier redone with exp08** (`figures/params_frontier.png`, labels repositioned): excess over the MLP line: local h2
  −0.003, local h4 +0.003, global h2 −0.012 (4 seeds; the only variant below the line, within the line's own ±0.01), global h4
  +0.005, global h8 +0.017, n8 +0.028, joint h4 global +0.032, global h16 +0.046.
- Remaining TPU quota ~1.3 h until 2026-10-03 00:00: not enough for a d768 run. exp09 waits for the reset.
- **exp10 pushed at 15:50** (`experiments/exp10_d384_lr`, kernel `mhmhc-exp10-d384-lr`): spends the 1.39 TPU h that would
  expire at the reset. One wave of 8 d384 runs (hard stop 1.25 h), seeds 0-1, testing whether the heads' gain depends on the LR
  and whether it is the heads or their parameters:
  - ctrl-mlp1216 at 2x LR (6e-3): the missing control for H7. If it breaks like mhc-h1, the LR robustness is the heads';
    if it holds like global h4, it is the parameters.
  - mhc-h1, global h4, ctrl-mlp1216 at 0.5x LR (1.5e-3, the same ratio as d768's halved LR). If global h4 beats its control
    only off the tuned LR, exp07's d768 gap is probably an LR effect.
  Numbered after exp09 (already built, waits for the reset) but runs first.
- **`analysis/lr_sweep.py`** written for exp09/exp10 (tested on fake runs in a scratch directory, then the figure regenerated from real
  data only): per width, the loss table variant x LR, paired Δ to mhc-h1 and global h4 − control at each LR, and each variant at its
  own best LR (chosen on the seeds it has at every LR, so exp09's extra seeds at one LR do not bias the choice).
  `figures/lr_sweep.png`: loss against LR per width; means more than 0.15 above the best are drawn as ▲ on the top edge with values.
- **Literature re-check** (`notes/literature/web_and_code_search.md` §7): one new relevant paper, **SiHC** (arXiv 2609.33895,
  2026-09-27): pixel-space DiT with spatially indexed streams (each a subpatch, not copies), identity carry, and **static
  per-channel** read/write maps H_pre, H_post ∈ R^{C×S}, i.e. design A's h = C limit for pre/post, static, no res, no constraint.
  Per-channel beat scalar maps in their ablation (FID 11.50 → 10.10, one run each). Possible reason it helps there and not here: their
  streams hold different content, ours are near-copies (cosine 0.76-0.87). Added to REPORT §2-3.

## 2026-09-30

- **exp10 ran 02:31-03:30** (queue 10.7 h; `results/exp10_d384_lr/`, all 8 jobs exit 0 in 3500-3680 s; live log at 25 min in
  `results/exp10_d384_lr/live_log_25min.txt`). Paired with seeds 0-1 of the earlier runs (`analysis/lr_sweep.py`):
  | LR | global h4 − mhc-h1 | ctrl-mlp1216 − mhc-h1 | global h4 − ctrl |
  |---|---|---|---|
  | 1.5e-3 (0.5x) | +0.0019 ± 0.0070 (1/2) | −0.0114 ± 0.0063 (2/2) | +0.0133 ± 0.0007 (0/2) * |
  | 3e-3 (10 seeds) | −0.0188 ± 0.0064 (9/10) | −0.0238 ± 0.0085 (8/10) | +0.0050 ± 0.0073 (2/10) |
  | 6e-3 (2x) | −0.2083 ± 0.0218 (2/2) * | −0.0138 ± 0.0373 (1/2) | −0.1945 ± 0.0156 (2/2) * |
  - **The LR robustness is the heads', not the parameters'**: ctrl-mlp1216 at 2x ends at 5.115 / 5.205, like mhc-h1 (5.166 /
    5.182), with grad-norm medians 1.7 / 2.3 (h1 1.1 / 1.4, global h4 0.6).
  - **At 0.5x the heads' gain is gone** (+0.002) and global h4 trails the control on both seeds from step 3000 on (+0.009 to
    +0.018 at every eval). The control still gains at 0.5x (−0.011).
  - mhc-h1 is flat between the LRs: 4.1331 / 4.1253 at 1.5e-3 vs 4.1752 / 4.1274 at 3e-3; its seed 0 is the worst of its 10 seeds
    at 3e-3 (next worst 4.145). Each variant at its better LR of the two: global h4 − mhc-h1 −0.007 ± 0.019.
  - Largest grad norm in the last 80% of training (mean over seeds): at 3e-3 mhc-h1 0.90, ctrl 0.77, global h4 0.52, local h4 0.56,
    global h16 0.48; at d768/1.5e-3 mhc-h1 1.29, ctrl 1.09, global h4 0.61. Heads damp gradient spikes with either predictor.
  - Reading: at d384 what global heads buy is tolerance of a high LR, not a better model; where the LR is comfortably stable, the
    same parameters do better in the MLP. The d768 gap (exp07) is probably the same thing: mhc-h1 at d768/1.5e-3 has the largest
    spikes of any setting, i.e. 1.5e-3 is probably near its edge at that width.
  - REPORT §5.6 added, §6 (new H7 row, "what heads buy is LR tolerance") and §7 (learning rate) updated.
  - `figures/lr_sweep.png` redrawn: top row loss, bottom row paired Δ to mhc-h1 at the same LR, both on seeds 0-1 only (the 3e-3
    means over 10 seeds had made the lines join means over different seeds).
- **exp09 revised (not pushed; kernel rebuilt)**: the decisive side is the low LR, so the 3e-3 (2x) runs are dropped (at d384 2x
  broke every variant). New queue: 7.5e-4 seeds 0-1 (8 runs) first, then 1.06e-3 s0, 2.1e-3 s0, local h4 at 1.5e-3 s0-1,
  1.06e-3 s1, **local h4 at 2x LR on d384** (s0-1; local heads have mhc-h1's exact parameters, so this separates head structure
  from predictor size as the source of the LR tolerance), and residual last. 26 jobs; DEADLINE 7.4 h (hard stop 8.4 h), since
  with 8.2 h wave 3 had only ~600 s of slack. A simulation of the launcher's queue rule: all 26 fit even at 9600 s per d768 run.

## 2026-10-03

- **exp09 pushed at 18:47** (`experiments/exp09_d768_lr`, kernel `mhmhc-exp09-d768-lr`, the revised 26-job queue of 09-30). TPU
  quota reset to 20 h (refresh 2026-10-10). The 00:07 reminder never fired, so it went out late; the
  scratch venv with kaggle 2.2 had been wiped again and was recreated. Watcher running (live log 25 min after start, then
  download into `results/exp09_d768_lr/`).
- The project was committed for the first time, one commit per file.
- **Move to Google Cloud**: Kaggle's TPU queues are too slow, so the remaining experiments, starting with exp09, run on Google
  Cloud, with results comparable to the earlier runs and a budget of $300 of GCP credit. Spot TPU VMs, single chip each.
  Everything about running there is in `notes/gcp.md`; code in `cloud/` (controller, VM setup, job wrapper, cost ledger).
  - **Same software as Kaggle**: `cloud/setup_vm.sh` installs torch 2.8.0 (CPU wheel), torch_xla 2.8.0, libtpu 0.0.17,
    numpy 2.5.0 (the versions in Kaggle's TPU image log) and exactly the 20 train shards (the shard count sets the data order).
  - **GCP v5e reproduces Kaggle v5e bit for bit**: exp04's mhc-h1-s0 replayed on a v5litepod-1 VM (us-east1-c): loss and grad norm
    identical at every logged step 0-325. So exp09 runs on GCP v5e and pairs exactly with exp07; it also repeats exp07's
    d768-mhc-h1-s0 in full to check this over 6000 steps.
  - **TPU v6e computes wrong gradients with fused accumulation.** First test on a v6e-1 (exp04 settings, micro-batch 8 = 2
    micro-steps per step): step-0 grad norm 8.43 (residual) / 6.98 (global h4) against 4.388 on v5e and on CPU in bf16 (4.3886,
    scratch script computing the same batch). On v6e itself micro-batch 16 gives 4.3877, and so does micro-batch 8 with
    `--micro-sync 1`. A standalone 2-micro-step backward + sync without clip/optimizer in the graph was correct, so the trigger is
    the combination of accumulation, `clip_grad_norm_` and the AdamW step in one XLA graph (torch_xla 2.8 / libtpu 0.0.17).
    Rule: on v6e never fuse accumulation; use micro-batch 16 at d384 (fits 32 GB HBM) or --micro-sync 1.
  - v6e is deterministic (two processes, same log). Checkpoint/resume added to `train_xla.py` (`--ckpt-every N`, saves model,
    optimizer, data-generator state, pending log records, metrics file offset; atomic replace) and checked bit-exact: a run killed
    at step 230 and resumed from its step-200 checkpoint matches the uninterrupted run at every step
    (`cloud/check_resume.sh`, `analysis/checks/check_resume.py`).
  - Throughput (tok/s, one chip): v6e d384 mhc-h1 81.9k / global h4 81.2k / residual 280k at micro-batch 16 (47k at micro-batch 8 +
    micro-sync); v6e d768 mhc-h1 51.9k at micro-batch 16; v5e (GCP) d384 mhc-h1 32.1k at micro-batch 8 (Kaggle ~29.6k).
    Spot prices (Cloud Billing Catalog, `cloud/prices.json`): v6e $0.27/chip-h in asia-southeast1, $0.65 in us-central1/east1/west1;
    v5e $0.34. A d768 run costs ~$0.89 on v5e (2.6 h) and ~$0.15-0.35 on v6e (~35 min).
  - Plan: exp09 on GCP v5e (exact pairing with exp07; ~$27). exp11 = hardware check on v6e: 23 d384 runs that exist on v5e (5
    variants x seeds 0-2 at 3e-3, 4 variants x seeds 0-1 at 2x LR) plus exp07's six d768 runs at 1.5e-3, all at micro-batch 16, with
    pass criteria written in its jobs.py before the runs. Later batches go to v6e (cheaper), each with in-batch baselines.
  - Caught before launching exp11: validation batches are built at the micro-batch size, so micro-batch 16 with the default
    `--eval-batches 20 --final-eval-batches 100` would score 1.6M tokens instead of the 819k used everywhere. v6e jobs pass 10 / 50
    (the same contiguous windows).
  - The Kaggle exp09 push (above) is still QUEUED; the CLI has no way to cancel a queued session (cancel needs a session id,
    which only exists once a session runs). If it runs it duplicates the GCP runs bit for bit and uses shared Kaggle TPU quota.
  - exp09 started first (31 creates at once): most failed on the v5e quota, which for single-host v5e is the "serving" quota, 4
    chips per zone, with in-flight requests counted. The controller now staggers creates (4 per minute), classifies errors (capacity
    / request rate / quota / regional limit) and backs zones off. It also became restart-safe: a restarted controller adopts the
    batch's VMs, sets up half-configured ones and logs creations missing from the ledger.
  - Next limit, hit by exp11 (16 v6e-1 VMs wanted): **the project allows 4 external IPs per region** (Compute Engine
    IN_USE_ADDRESSES; INSTANCES is 8). Every TPU VM takes one IP, so single-chip VMs cap a region at 4 chips; exp09's 12 v5e
    VMs already hold all IPs in us-east1, us-west1 and europe-west4. Internal-IP VMs would need Cloud NAT and an IAP firewall rule
    (network security changes this project does not make). So exp11 moved to **v6e-8 VMs** (one IP, 8 chips, one process per
    chip via the Kaggle `visible_chips` recipe in `cloud/vm_job.sh`); the controller now keeps the chip count per VM, so the two
    v6e-1 VMs already running were adopted correctly. exp11 gained two jobs, `chk-v6e8-{mhc-h1,a-global-h4}-s0`, which repeat
    the two v6e-1 runs on v6e-8 and should be bit-identical to them.
  - v6e-8 / v6e-4 spot capacity is scarce (capacity errors in asia-southeast1-b, us-central1-a/c, us-west1-c, us-south1-a), and
    four v6e controllers (exp11-exp14) competing for the same zones made each other fail on the chip quota (in-flight requests
    count against it). `chk-v6e8-mhc-h1-s0` had landed on another v6e-1, so it became a cross-VM determinism check, and
    `min_chips` jobs `chk-multi-{mhc-h1,a-global-h4}-s0` now test the multi-chip recipe against the v6e-1 runs.
  - 20:20 UTC: **one shared controller for all v6e batches** (`controller.py exp11 exp13 exp12 exp14 --accel v6e-8,v6e-4
    --max-chips 40`): jobs run in that order, results still go to each batch's folder, the log is exp11's. A VM that finishes one
    batch's jobs takes the next batch's instead of being deleted, so scarce v6e-8s are kept. It tries the biggest size in every
    zone before a smaller one, and skips regions with no external IP left (it reads IN_USE_ADDRESSES before creating). exp09 was
    restarted with `--accel v5litepod-4 --drain-other-sizes`: its 12 single-chip v5e VMs finish their current jobs and are
    deleted; the 19 remaining d768 jobs go to 4-chip v5e VMs (one IP each; the serving quota is 4 v5e chips per zone, so one
    such VM per zone). Kaggle's v5e-8 ran one process per chip the same way, and `repro-d768-mhc-h1-s0` checks it against exp07.
  - 20:30 UTC: the first v6e-8 came up (us-central1-b) and ran 8 jobs at once, one per chip. `chk-multi-mhc-h1-s0` on chip 0 of it
    matched the v6e-1 run bit for bit over the steps compared then; `chk-v6e8-mhc-h1-s0` (another v6e-1) matched `v6e-mhc-h1-s0`
    bit for bit over the whole run (241 logged steps, 6 evals). v6e-8 / v6e-4 / v5litepod-4 spot capacity stayed scarce (capacity
    or "internal error" in every zone tried), and one v6e-1 and one v5e VM were preempted (the v5e job after ~2 h).
  - 20:50 UTC: the d768 follow-up became exp15 (`experiments/exp15_d768_v6e/jobs.py`, `notes/plan.md`): exp09's LR grid on v6e
    with 3 seeds, then seeds 3-4 at each variant's best LR. Pool order exp11, exp15, exp12, exp14, exp13, now with v6e-1 as a last
    resort (regions with free IPs); exp09 now falls back to v5litepod-1 instead of draining its single-chip VMs.
  - 20:56 UTC: the v6e-8 was preempted 27 min after creation, with 8 jobs on it (their processes vanished, then ssh was refused).
    The jobs went back to the queue, but the controller logged it as "process gone" + failed restarts + "deleted (no jobs left)":
    a failed start released the job, the next chip took it again, and an empty VM looked idle. Fixed: a failed start ends the
    filling of that VM for this poll, and a VM with jobs waiting is never deleted as idle (6 polls without a successful start
    delete it as "cannot start jobs"). Preemptions so far: two v6e VMs (after ~40 and ~27 min) and one v5e job twice (exp09's
    d768-ctrl-mlp2240-lr1.06e-3-s0, which restarts from step 0 each time: 2.6 h runs on v5e).
  - Checkpoints cannot follow a job to another VM cheaply: pulling one 328 MB d384 checkpoint to this machine ran below 3 MB/s,
    so preempted jobs restart from step 0 (the VMs have no Cloud Storage access and no credentials are put on them). Fine for the
    20-35 min v6e runs; costly for exp13's 90-min runs if v6e preemptions stay this frequent (on-demand v6e is $2.70/chip-h,
    4x spot).
  - 21:25 UTC: the v6e pool controller can now also keep **on-demand** VMs (`--on-demand-chips 8 --on-demand-budget 90`): at
    most 8 on-demand chips, 4-chip or 8-chip VMs only (a single on-demand chip would cost 4x spot for one of the region's 4 IPs),
    and no new jobs on them once on-demand spend reaches $90. They take exp13's 90-min jobs first (`expected_seconds` >= 4000,
    which a spot VM takes only when no short job is waiting), so the longest runs are the ones not exposed to preemption. Spot
    v6e is still scarce: 5 chips running (one v6e-4, one v6e-1, us-central1-b), every v6e-8 / v6e-4 request failing on capacity
    or "internal error", on-demand too. Spend $16 of $300.
  - 21:51 UTC: exp09's first d768 result, mHC h1 at 7.5e-4 seed 0, ended at 3.938, against 4.146 for the same seed at exp07's
    1.5e-3: the d768 LR was far too high. Decided before any other result (notes/plan.md): the d768 grid goes down to 5.3e-4 for
    every variant (exp15 seeds 0-2, first in its queue; exp09 seeds 0-1, ahead of 2.1e-3), and further down while any variant's
    lowest LR is its best. Both controllers restarted with the new job lists (running jobs adopted).
  - 22:00-22:20 UTC, exp09 at 7.5e-4 (seeds 0-1, v5e): global h4 − mHC h1 +0.008, control − mHC h1 +0.009, local h4 − mHC h1
    +0.023 (both seeds). At 1.06e-3 (seed 0): global h4 3.919 (its best so far), mHC h1 3.975, local h4 3.959. So far the heads'
    LR curve is flatter and its best sits higher; the control at 1.06e-3 and everything at 5.3e-4 are still running.
  - 22:25 UTC: the v6e-8 running 8 jobs had load average ~500 and answered ssh slowly (20 s connect timeouts), which after 6 misses
    would have made the controller delete it with 8 jobs. Fixed: a VM that stops answering is deleted only if gcloud no longer
    says READY; ssh connect timeout 60 s. Both controllers restarted.
  - 22:45 UTC: **exp11 done** (REPORT 5.7). It fails its pre-registered criteria as written: a v6e run is not a noisy copy of its
    v5e twin but behaves like a new seed (per-run v6e − v5e: mean +0.007, std 0.014, the size of the seed-to-seed spread). At the
    tuned LR every gap keeps its sign; at 2× LR the gaps keep their sign but shrink (global h4 − mHC h1: −0.21 on v5e, −0.05 on
    v6e, 1/2 seeds); at d768 global h4 − control goes from −0.034 to −0.012 ± 0.032 (1/2). Consequence (plan): no v6e number in a
    table with a v5e number; every v6e batch read on its own baselines. exp09's v5e local h4 at 2× LR: −0.29 vs mHC h1 (2/2).
  - 23:50 UTC: **exp13 (4x longer d384 training) moved from v6e to v5e**, before any of its runs had started
    (`experiments/exp13_d384_long/jobs.py`, now `v5e-long-*`, exp04's v5e settings). v6e is the bottleneck: 12 chips on two
    us-central1-b VMs, every other v6e request failing on capacity, on-demand included, and ~126 v6e jobs still queued
    (exp15, exp12, exp14). exp13 was ~19 v6e chip-hours of 90-min runs that a preemption restarts from step 0. On v5e the runs take
    ~4 h each, and the US v5e VMs have now run 4.5 h without a preemption (both preemptions were in europe-west4-b). exp13 carries
    its own baselines, so the chip does not touch its read-out; it is never compared with a v6e batch. It queues behind exp09 in
    the v5e controller. Noted in its header: the VMs hold 200M training tokens, so a 393M-token run sees each token about twice,
    for every variant alike. Both controllers restarted (all 24 running jobs adopted). Paper: d384 results and the cross-chip
    appendix drafted.
- 2026-10-04 ~01:00 UTC: **final literature re-check** (`notes/literature/recheck_2026-10-04.md`). arXiv: nothing new. OpenReview's
  ICLR 2027 submissions (public, not on arXiv, missed by every arXiv-centric sweep) hold ~20 HC-family papers. Closest to design
  A: **uHC** (`Qnj7Lf8Bz2`), whose Group-Wise Feature Readout is design A's per-group read (plus a low-rank write-back), 46M-363M,
  mHC 2.947 -> 2.819 at 363M. No per-group doubly stochastic mix, so design A as a whole is still unpublished, but its read half
  is not new. **MHAR**'s newest abstract attributes most of its head-split gain to the flatter per-head softmax, a small residual
  to per-subspace routing. SimpleHC (gating without mixing ~ mHC) and osHC (one marginal suffices for stability) bear on the
  value of the doubly stochastic mix. All verified at abstract level; OpenReview blocks scripted downloads of the full texts. exp14's per-head-read-only variant is the direct test of uHC's read in this project.
- 01:00 UTC: **exp15's low-LR grid complete** (d768, v6e, seeds 0-2). Means: mHC h1 3.9667 / 3.9458 / 3.9501 at 5.3e-4 / 7.5e-4 /
  1.06e-3; global h4 3.9593 / 3.9429 / 3.9422; control 3.9599 / 3.9378 / 3.9492; local h4 3.9844 / 3.9435 / 3.9644. Best LRs (the
  plan's rule): 7.5e-4 for all but global h4 (1.06e-3, by 0.0007). Paired at 7.5e-4: global h4 − mHC h1 −0.003 ± 0.005 (2/3),
  control − mHC h1 −0.008 ± 0.006 (2/3), global h4 − control +0.005 ± 0.001 (0/3), local h4 − mHC h1 −0.002 ± 0.003 (2/3). So at
  a tuned LR the d768 picture so far is the d384 one: heads at most worth their parameters. At 1.5e-3 global h4 still leads mHC h1
  by 0.051 (2/2): the gain lives above the optimum. Seeds 3-4 at the best LRs (plus global h4 at 7.5e-4) queued first; v6e
  controller restarted. Fixed `lr_sweep.best_lr`, which had picked mHC h1's best on seed 0 alone once a half-run LR was in.
- 02:10 UTC: **exp15 decided: the d768 scale condition is not met; d1024 is not run.** Seeds 3-4 at the best LRs: global h4 at
  1.06e-3 − control at 7.5e-4 +0.0055 ± 0.0072 (two-sample, 1/5); local h4 − mHC h1 at 7.5e-4 +0.0130 ± 0.0098 (2/5). Secondary,
  all at 7.5e-4 on 5 paired seeds: global h4 − mHC h1 −0.005 ± 0.003 (4/5), control − mHC h1 −0.010 ± 0.005 (4/5), global h4 −
  control +0.005 ± 0.005 (1/5). Above the optimum the heads degrade less, as at d384: at 1.5e-3 global h4 − mHC h1 −0.063 ± 0.012
  (3/3), local h4 − mHC h1 −0.026 ± 0.008 (3/3); at 2.1e-3 everything breaks (4.67-4.78). Late grad-norm spikes appear only at
  2.1e-3 (mHC h1 48, heads and control 11-16). So exp07's d768 lead over the control came from the untuned LR. Spend $70.
- 04:15 UTC: **exp09 done** (d768 LR grid on v5e, seeds 0-1; REPORT 5.8). Same best LRs as v6e (7.5e-4; 1.06e-3 for global h4).
  At the best LRs: global h4 − control −0.016 ± 0.011 (2/2; v6e +0.006 ± 0.007, 1/5), local h4 − mHC h1 +0.023 ± 0.012 (0/2).
  Pooled over both chips by inverse variance (the chips act as independent seeds): global h4 − control −0.002 ± 0.006, local h4 −
  mHC h1 +0.017 ± 0.008. At 1.5e-3 the heads lead (−0.068, −0.069, 2/2), at 2.1e-3 everything breaks. The exp07 replay on a GCP
  v5e VM matched at all 253 logged losses. Since exp07/exp09 had the residual only at 1.5e-3, added exp15's residual at 5.3e-4 to
  1.5e-3 (12 v6e runs, ~20 min each), so mHC's own gain over the residual is also read at a tuned LR; v6e controller restarted.
  Spend $97.
- 04:23 UTC: **exp12 done** (d384 LR grid on v6e, seeds 0-2; REPORT 5.9). Heads do not move the optimum (2.1e-3 for mHC h1,
  global h4, control and residual; local h4 3e-3 by 0.003), do not widen the usable range (control and residual widest), and at
  the best LR global h4 is 0.018 ± 0.004 behind its control (0/3, effect exists; the pre-registered correction was for the other
  sign). mHC − residual at the best LR −0.072 (3/3). Heads fail less badly only where every variant is ≥ 0.04 off its best. H7b as
  written "partial" (+0.095 ± 0.176), uninformative; its reverse check contradicts the slow-dynamics mechanism: global h4 with the
  connection LR x4 at 6e-3 is 0.20 better than global h4 and 0.27 better than mHC h1 (3/3). x4 also decays phi 4x harder (AdamW
  decay scales with the group LR). Pre-registered **exp16** (exploratory, 9 runs, ~$5): does mHC h1 get that tolerance from the
  same setting, without heads? v6e controller restarted with exp16 queued last. New: `analysis/lr_figure.py` (paper figure).
- 04:45 UTC: **exp15's residual runs done** (d768, v6e, seeds 0-2 at 5.3e-4 to 1.5e-3). Best at 1.06e-3 (4.0019); mHC h1 at its
  best is 0.059 ± 0.008 ahead (two-sample, 3/3), and 0.052-0.061 ahead at every LR. mHC's own gain holds at d768 at a tuned LR;
  the heads add nothing that clears the noise on top of it (pooled ±0.006). REPORT 5.8, paper and `figures/lr_v6e.png` updated.
- 05:50 UTC: **exp14 done** (which blocks of the split matter, d384, v6e; REPORT 5.11). At 3e-3 every split that keeps the mix
  shared beats mHC h1 by 0.025-0.029 (3/3) and lies 0.016-0.020 below the MLP line; the per-head mix alone is above it. The 02:15
  rule read as written: `pre` (uHC's group-wise read) misses (excess −0.019 vs 2 SE 0.022), `pre,post` meets it (−0.016 vs 0.009),
  so **exp18** runs pre,post at d768 (7.5e-4, seeds 0-4). At 6e-3 the tolerance is in read+write per head: pre,post −0.18 vs mHC h1
  and −0.11 vs global h4 (3/3), late grad 5; the per-head mix alone keeps none (late grad 93). Caveat: 3e-3 is past the v6e optimum
  (exp12), where part of a head gain is LR tolerance; **exp17** (decided 05:20, after two seeds, unconditional) runs the four
  parts at 2.1e-3. A VM set up after the head-parts code reproduced exp11's mHC h1 and global h4 seed 0 exactly (488 values), so
  the baselines are the same computation. Spend $115.
- 06:17 UTC: **exp16 done** (exploratory; REPORT 5.10). (1) mHC h1 with the connection LR x4 at 6e-3: +0.030 ± 0.041 vs mHC h1
  (1/3), late spikes 114: the tolerance is not available without heads; at equal settings the split is worth 0.30 (3/3). (2) At
  2.1e-3 the setting does nothing that clears the noise (mHC h1 +0.003, global h4 −0.011). (3) Global h4 x4 − mHC h1 x4 at 2.1e-3
  −0.018 ± 0.013 (2/3). Global h4 x4 is still 0.007 behind the MLP control at 2.1e-3 (0/3).
- 06:45 UTC: **exp18 done** (pre,post heads at d768, v6e, 7.5e-4, seeds 0-4; the 02:15 follow-up). Δ to mHC h1 −0.0056 ± 0.0039
  (3/5); the MLP line at its 113.7M parameters −0.0035; excess −0.0022 against 2 SE 0.0079, 2/5 seeds below the line: the rule is
  not met. Against the control +0.005 ± 0.007 (2/5), against global h4 −0.001 ± 0.005. The d384 parts gain at 3e-3 does not carry
  to d768 at its tuned LR. Per the 02:15 entry: no further follow-up, d1024 stays off.
- 07:07 UTC: **exp17 done** (exp14's parts at 2.1e-3, d384's best LR on v6e; REPORT 5.11). The 3e-3 gains are gone: pre +0.002 ±
  0.015 vs mHC h1, pre,post −0.002, res −0.001, post −0.009 (3/3) but not below the line by 2 SE; every part trails the MLP control
  by 0.013-0.024 on every seed. So no block of the head split is worth more than its parameters at a tuned LR (d384 here, d768 in
  exp18); the per-head read and write carry only the split's high-LR tolerance. All v6e batches are done.
- 08:36 UTC: **exp13 done** (d384 at 4x the tokens, 393M, v5e, LR 3e-3, seeds 0-2; REPORT 5.12). Read-out as pre-registered:
  global h4 − control +0.000 ± 0.019 (2/3), local h4 − mHC h1 −0.009 ± 0.005 (2/3); against the 98M-token runs of the same seeds
  and chip, neither changed by an effect (−0.017 ± 0.011, −0.003 ± 0.005). Established: the control's lead over mHC h1 shrinks by
  0.018 on every seed while global h4's holds (a hint that heads gain on the MLP with length, not an effect for heads − control);
  mHC's gain over the residual grows, 0.041 → 0.048 (3/3). The v5e controller deleted its last VM (54 done, none failed);
  `gcloud compute tpus tpu-vm list --zone=-` lists 0 VMs. Total Google Cloud spend $128.81 of $300. New:
  `analysis/long_training.py`.
- 08:41 UTC: **Done.** No further experiments (notes/plan.md, 08:41). REPORT §1 (summary), §6 (viability), §7, §8 rewritten for
  the final results; paper finished (abstract, introduction, long training, discussion, conclusion; builds with no undefined
  references or overfull boxes, 14 pages). Final answer: as a whole the design is new (its per-group read is not: uHC); at a tuned
  LR it is worth its parameters and no more, at 27M and 112M, on two chips and at 4x the tokens; its one robust property is
  tolerance of a too-high LR, carried by the per-head read and write. Not viable as an architecture improvement at these scales.

## 2026-10-04

- 12:20 UTC: **Paper rewritten** in the format of `paper-template/FORMAT.md` (Moonshot-style technical report; `preamble.tex`
  and `.latexmkrc` copied unchanged): title page with the abstract and the main result as Figure 1, contents, every section on
  its own page, appendix A-C. Five figures in `paper/figures/`, all computed from `results/` by `analysis/paper_figures.py`
  (with `paper_style.py`; `lr_figure.py` and `params_frontier.py` gained `--paper`, their default output is unchanged): Figure 1
  a forest plot of the headline comparisons (±2 SE, filled when established), the method drawing (TikZ, `figures/src/`), the
  parameter frontier, the LR curves at both widths, and the per-head parts against the MLP line and past the optimum. Every
  plotted number was checked against REPORT. Builds with `latexmk` in `paper/`: 20 pages, no undefined references or overfull
  boxes. Found while checking: REPORT 5.8's table gave the paired SE (0.0098) for local h4 − mHC h1 at d768 where the row is
  two-sample; corrected to ±0.0111 (conclusion unchanged, 2/5 seeds).
- 15:11 UTC: **Paper polished** after reading every page as an image. Figure text at 9 pt (the 11 pt body's footnote size);
  Figure 1 fits on the title page with the abstract, its columns no longer touch; Figure 3 shows only the variants that add
  parameters (the rest are in Table 4), with labels clear of the bars and the line; bold caption titles; formulas for the
  per-group blocks (§3.2) and the block-diagonal operator (§3.3); Method ends on its second page. Bibliography: original title
  case, venues and arXiv/OpenReview links, the provenance notes kept out of the printed list; ragged right; References in the
  contents. Builds with `latexmk`: 20 pages, no undefined references or overfull boxes.

## 2026-10-05

- 11:47 UTC: **Reframing question and exp19.** Considered whether the paper should lead with the parameter-matched comparison (no
  gain), mention the extra-parameter comparison briefly, and offer the heads' high-LR tolerance as a stability benefit for larger
  models, citing Wortsman et al. (small-scale proxies) and Qwen3.8-Next's stability stress test. Literature read first
  (`notes/literature/stability_proxies.md`, `qwen_stability.md`, `hc_stability.md`): Wortsman et al. tie small-model LR
  sensitivity to two instabilities (attention-logit growth, output-logit divergence), both removed by QK-norm and z-loss, and
  leave loss spikes out of scope; Qwen test architecture changes on a 25B MoE at 2–4× the optimal LR, counting loss spikes and
  clip crossings; OLMo 2 found 2–4× its LR benign at 7B with QK-norm; Lourie et al. find small-scale sensitivity shrinks with
  scale; the HC family argues stability from grad-norm curves at one LR. The tolerance had been measured only without QK-norm.
  Code: `--qk-norm` (per-head RMSNorm on q and k before RoPE), `--log-steps 1` (per-step loss and grad norm before clipping),
  `--diag-every N` (largest attention logit and attention entropy per layer, sublayer RMS, z², on four fixed validation windows);
  `analysis/checks/check_model.py` covers both. Pre-registered exp19 (`notes/plan.md`, 11:47; S5 and S4 clarified at 11:51 and
  11:57 before any run finished): d384 on v6e, 9 LRs 1.5e-3 to 2.4e-2, five variants, seeds 0–2, with and without QK-norm, 240
  runs, budget cap $220.
- 12:04 UTC: exp19's first start crashed at step 0 of every run in the new diagnostics (`amax()` with no dims reduces nothing on
  XLA). Fixed with `max()` and float32 statistics outside autocast; runs with and without diagnostics repeat every step's loss and
  grad norm exactly on a v6e. Restarted.
- 16:46 UTC: **exp20 designed** (exploratory; `notes/plan.md`, 16:46) after reading most of exp19: d768 with QK-norm, per-step
  logs, five variants, LRs 5.3e-4 to 2.1e-3, seeds 0–2, 75 runs, budget cap $250. It tests the one stability-like difference that
  survives QK-norm at d384, fewer clip crossings for the heads, and the parameter-matched comparison with QK-norm at 112M.
- 16:57 UTC: `v6e-ctrl-mlp1216-diag-lr2.4e-2-s1` failed twice on copying its results back (SSH "connection reset" on its VM;
  the run itself exited 0). Its files were fetched by hand with scp from the same VM; it diverged, as expected at 2.4e-2.
- 17:00–17:16 UTC: a local network outage (DNS failures for Google's auth endpoint); both controllers kept going once it ended.
- 17:20 UTC: committed the code, the literature notes, exp19 and exp20 and the new references, one file each.
- 18:22 UTC: **exp20 stopped, not run.** Spot v6e capacity gave it one VM in 1 h 35 min (preempted before any run finished)
  and two more still being created; exp19's VMs lived 13–22 minutes over the same hour, against about 40 minutes per d768 run,
  which restarts from step 0 after a preemption. Controller stopped, its two VMs deleted and logged in the ledger; spend $204.89.
  What the paper says about d768 rests on exp15 (no QK-norm, every 25th step logged), read descriptively (`notes/plan.md`, 18:22).
- 19:03–19:33 UTC: tried one on-demand v6e-4 VM for exp19's last 12 runs (`--on-demand-chips 4 --on-demand-budget 12`). Every
  create failed with "User does not have permission to submit requests for accelerator type": on-demand v6e is not open to this
  project. Nothing was created or spent; the controller went back to spot only, with its earlier settings.
- 20:06 UTC: the one spot VM that came up since 18:06 (us-east1-d, set up at 19:58) was preempted eight minutes into its runs;
  its four runs were requeued.
- 21:07 UTC: with 228 of 240 runs done and spot capacity in the usual zones gone since about 18:07, the controller was restarted
  with `--max-price 1.7`, which adds us-east5 ($1.62 per chip-hour). Two VMs left in CREATING by the restart were deleted and
  logged; spend $205.74. The missing runs matter for S3: without QK-norm the full-grid S is dominated by whether a run diverges
  at 8.5e-3, and global h4's seed 2 there is one of them (`notes/plan.md`, 21:05).
- 21:23 UTC: v6e-1 added as the last fallback size; through 23:30 every create of every size failed on capacity in every zone
  (quota 0 in us-east5-c).
- 23:31 UTC: **exp19 finalised with 228 of 240 runs.** Controller stopped, no VM left; spend $205.74 of $300 ($76.93 on exp19 and
  exp20). The 12 missing runs are without QK-norm at 8.5e-3 and above, where all 36 finished runs at 1.2e-2 and above diverged.
  Read-outs (`notes/plan.md`, 23:31): S1 42/42 bit-identical; S2 attention-logit growth; S3 **QK-norm removes the tolerance**
  (with QK-norm the heads are more LR-sensitive than mHC h1: +0.015 ± 0.002 and +0.006 ± 0.003, 0/3 each), the same for every
  outcome of the missing runs; S4 the parameter-matched null holds with QK-norm; S5 no loss spikes where training does not fail;
  S6 not triggered. `analysis/stability.py` now prints the full-grid dS on complete seeds and S3's verdict over every outcome of
  missing runs.
- 23:36 UTC: paper, REPORT and figure updated: the stability table (Table 7), the no-QK sensitivity reported on complete seeds and
  over the five LRs where no run diverges, the loss-spike claims qualified to runs that do not fail, run counts (626 runs, 228 in
  the stability test) and costs ($206, $77 of it for the stability test and exp20), and the 12 missing runs in the limitations.
  Committed one file each.

## 2026-10-07

- 13:32 UTC: paper title page. Kept the title, which states the finding; dropped the subtitle ("An empirical study at 27M and 112M
  parameters"), which added a filler phrase and a second "parameters" and would have given a double colon wherever the title is
  one line (arXiv, the PDF metadata); the scale is in the abstract's third sentence. The head note keeps "Technical report" and
  now carries a fixed date, October 2026, instead of `\today`, which changed with every build.
- 13:47 UTC: paper prose rewritten to read like published papers in the area, after reading mHC (2512.24880), HC
  (2409.19606), Narang et al. (2102.11972) and No Train No Gain (2307.06440), whose abstracts are one plain paragraph, whose
  introductions carry no contribution bullets, and whose headings and caption titles are noun phrases. The abstract has no
  semicolons or colons and four numbers; the contributions are prose; Related Work has three subsections; the 35 bold run-in
  heads (Results, Discussion, Limitations, appendix) became prose with topic sentences or noun-phrase subsections; QK-norm got
  its own subsection (5.5); captions lost their bold claim sentences; semicolons went from 68 to none in the prose, and the
  "X, not Y" contrasts are gone. Every number, table, figure and label is unchanged; the preamble (contents page, a page per
  section) is untouched.
- 13:53 UTC: dropped "and we could not find an earlier test of it" from the abstract. A negative result does not rest on novelty,
  the claim reads as defensive and would date badly, and it stays in the introduction, backed by the 186-paper citation crawl
  and the note on uHC. The guarantees moved into the previous sentence.
- 14:01 UTC: Figure 1 moved from between the abstract and the contents (where it filled page 2 alone) to the foot of the first
  introduction page, beside the text that cites it, as in HC. Plural section references no longer print "§§5.4 and 5.5": each
  is a separate `\cref` ("§5.4 and §5.5"), the preamble untouched. The prose rewrite had pushed the contents onto a second
  page; Related Work is back to plain paragraphs and Discussion 6.3/6.4 are merged ("Why the heads do not help"), so the
  contents fit on page 2 again. Build clean, 28 pages.
- 16:55 UTC: removed the literature-search sentences from the introduction ("We could not find a proposal or test of it. A
  citation crawl of the 186 papers ...") and the matching "as far as we can find, had not been proposed or tested" from the
  conclusion. Same reasoning as the abstract clause: the result does not rest on priority, and published papers do not describe
  their search. Related Work still names uHC as the closest concurrent work and says the per-group mix is the one part none of
  the close designs has. The crawl itself stays in `notes/literature/citation_crawl.md`. Layout unchanged, 28 pages.
- 18:48 UTC: removed "and on both chips" from the introduction's summary of the null. The chip split means nothing to a reader
  before Appendix A, and the claim was weak: the 27M best-rate comparison ran on v6e only, and at 112M v5e contributes 2 seeds
  (global − control there is −0.016 ± 0.011, 2/2, not established). The per-chip numbers stay in tab:best-lr, and the pooling
  note stays in the Figure 1 caption. Layout unchanged, 28 pages.
- 20:53 UTC: the paper is now in American spelling throughout. It had mixed British forms (colours, normalised, optimiser,
  Behaviour, initialised, favours, re-parameterises, towards) with American ones (tokenizer). Eleven words were changed, including
  the §5.4 title, which now reads "Behavior above the optimal learning rate". The figures had no British spellings, and the preamble
  has them only in comments.
- 21:35 UTC: §4.3 now says only that every run uses one TPU chip, v5e or v6e, and why the two are kept apart. The Kaggle/Google
  Cloud split and the bit-identical replay moved out of the main text but stay in Appendix A (the reason the v5e runs from both
  platforms count as one chip type) and Appendix C (run counts, and the $206 that covers only the Google Cloud runs). Kaggle stays
  named there because the compute report would otherwise be incomplete and the repository's Kaggle kernels would go unexplained.
  Checked Kaggle's Acceptable Use Policy (version of 2025-06-22): it bars abusing resources for, among others, "activity unrelated
  to ML data science", cryptomining and server farming. Our use was ML training on one account within the weekly TPU quota,
  submitted through the official CLI. Layout unchanged, 28 pages.
- 21:45 UTC: reviewed Figure 4 (the frontier) after the question of what it adds and whether a loss-vs-learning-rate chart is missing.
  That chart already exists as Figure 3 (both widths), and Figure 5 extends it to 24e-3 with and without QK-norm. Figure 4 stays:
  it is the only view showing that the heads' gain over mHC is flat at about 0.02 from h = 2 (0.6M extra parameters) to h = 16 (9M),
  while the MLP line keeps falling to −0.065, which is §5.3's argument and Table 5's data in one picture. Fixed a collision (the
  "mHC, n = 8" label crossed the global h = 8 error bar) in `analysis/params_frontier.py` and regenerated `frontier.pdf`. The
  caption now states the finding. Data unchanged, 28 pages.
- 21:49 UTC: the tolerance claim checked against the per-rate numbers after the question of why the paper says the heads cope better
  with high learning rates when the chart does not seem to show it. Without QK-norm at 27M, global h = 4 minus its control is
  +0.026, +0.018, +0.023, −0.045, −0.173 at 1.5, 2.1, 3, 4.2, 6 (×10⁻³). Only 4.2 (twice the optimum) is a usable regime: at 6
  every variant is 0.8–1.1 above its best, and at 8.5 runs diverge or end more than 1.1 above. At 112M the window is 1.5; at 2.1
  every variant is 0.7–0.9 above its best. Local h = 4 is 0.012 behind mHC at 27M and 4.2. The symlog axis of Figure 3 squeezes a
  0.045 gap between 0.06 and 0.11. §5.4 now points to Figures 1 and 3, says the window is one grid point wide and that the heads
  fail at the same rate as the others, and the Figure 3 caption names where global h = 4 lies below mHC and the control. 28 pages.
- 21:55 UTC: §6.1 "Two inexpensive controls removed this gain" became "Two controls removed this gain". The MLP control is cheap,
  but the learning-rate grid is not (five learning rates for every main variant, most of the compute), so the adjective was weak
  and partly wrong. Layout unchanged, 28 pages.
- 21:59 UTC: filler pass over the whole paper, 32 cuts. Removed: "also" where it linked nothing (abstract, 3.6, 4.1, App. C),
  "simply" (5.4, 6.2), "actually" (6.1), "exactly" (intro, 5.1), "altogether" (RW, 6.3), "then" (3.2, 6.2), "either" (intro),
  "in total" (intro, 4.1), "in any case" (6.3), "in important ways" (6.2), "very" (6.2), "here" (5.4), "anywhere" (5.5), "alone"
  (5.3), "itself" (3.2), "what we call" (intro), "all of" (conclusion), and "own" in "mHC's own gain" (abstract, intro). Kept "own"
  wherever it means each model's own optimum. The first paragraph of 5.1 no longer restates the two controls defined in 4.2.
  Numbers, claims, labels and layout unchanged. Contents still fit on page 2, 28 pages.

## 2026-10-08

- 06:26 UTC: final read-through of the PDF, every page rendered and every figure checked at 200 dpi (labels, overlaps, legends
  all fine). Numbers checked against results/ with lr_sweep.load and params_frontier.py. Fixed: (1) at 112M the residual is best
  at 1.06e-3 (4.0019 against 4.0035 at 0.75e-3), not 0.75e-3 as 5.1 said; (2) Table 11 showed seeds 0–4 means on v6e, which put
  global h = 4's bold 1.06e-3 cell (3.9378) above its 0.75e-3 cell (3.9377); it now shows the seeds 0–2 means the rule uses
  (3.9422 against 3.9429), as Figure 3 does; (3) "3e-3 lies above the optimum of every variant" on v6e was wrong for local h = 4,
  which is best there (5.2, 5.3, 5.6); (4) global h = 2 lies 0.012 below the MLP line in mean (SE 0.009), so 5.3 says none lies
  below it by more than 2 SE; (5) h = 16 ratio is 0.0652/0.0189 = 3.45, written 3.4; (6) the early 112M result at 1.5e-3 was on
  v5e, where that rate costs 0.09–0.18, not 0.09–0.15 (v6e). Prose: intro "conclude that it does not" became "find no evidence that
  it does" (as in the abstract and 6.2), the "first seen ... first observed" repetition in RW is gone, "Qwen credit/count" (British
  collective plural) now names the authors or the model, and 5.6/6.3 name the gradient metric. Build clean, 28 pages, contents on
  page 2. Pages 6, 9 and 11 stay short because each section starts a new page (template).
- 14:06 UTC: **publication-readiness audit** (a re-read of all 28 pages plus three separate checks: every number against results/, all 41
  references against arXiv and venue listings, and a skeptical referee's reading). Fixed in the paper: six citation claims that the sources do
  not support (Zhao et al. on collapse, Depth-Attention's baseline, Kramer's matching and 590M, Qwen's gate and 125B production
  model, Wortsman's 540B and "the one quantity"), xHC and mHC-SSM moved out of the mix re-parameterizations, the anonymous MHAR
  submission no longer called a version of the arXiv paper; Figure 1 vs 5.1 SE (0.011), run counts (603 at 98M tokens and 117
  on Kaggle; the 8 Kaggle bench runs are 150 steps), Appendix A ranges and convention, 5.5 gradient-norm bounds (heads below 0.9,
  residual 1.53), Table 3 seeds, static mHC row in Table 6 (+0.023 ± 0.010, 0/3), write-only named as the closest call in 5.6 and
  the promotion rule's gap stated, abstract/intro/conclusion scoped to 27M where the evidence is. Alimaskina et al. is an ICML 2026
  workshop paper (its PDF), not CIKM. Still open: repository URL (repo is private), arXiv
  endorsement, and the four OpenReview entries, which scripted requests cannot open. REPORT.md still says 611/125.
- 21:07 UTC: **public release** at https://github.com/marcoshernanz/multi-head-mhc, with a fresh history; the development history
  (1083 commits) stays in the private repository `multi-head-mhc-dev`. Left out: the compute account notes
  (account and billing details) and the paper template. Changed for the release: the Google Cloud project ID in three
  controller logs is now `PROJECT_ID`; other work on the Kaggle account is no longer named; the five live logs at the top of
  `results/` moved into their batch folders; the literature scripts use relative paths; `notes/literature/raw/` says where the
  DeepSeek files come from and keeps their MIT license; REPORT.md has the 603/117 counts and a note that the paper supersedes it;
  `stability_figure.py` documents `--partial`, which draws the paper's figure. New README (layout, setup, the command behind
  every table and figure, results format, batches), MIT license, CITATION.cff. Paper: Appendix C gives the URL and the
  introduction points to it (28 pages, no overfull boxes). Checked before publishing: no tokens, keys or account details in any
  file; all five data figures regenerate identical to the committed PDFs except for the creation date; every README command runs;
  a short CPU training run works.

## 2026-10-09

- 11:06 UTC: **shorter abstract**, 273 → 191 words (11 → 10 sentences). For comparison, the 49 cited papers whose abstracts
  Semantic Scholar has: median 185 words (quartiles 146 and 226); mHC 149, Hyper-Connections 112, Frac-Connections 120,
  MUDDFormer 143, the 2026 mHC variants 175–264, Narang et al. 2021 (a negative result on transformer changes) 114. Only Multi-Head
  Attention Residuals (291) was longer than ours. Kept every main claim: the design, the runs and controls, no gain at equal
  parameters (both cases), the 27M checks, the shared-learning-rate illusion, the failure above the optimum and QK-norm, and the
  stability verdict. Dropped: the 40% figure, "which the heads do not prevent", and the gradient-norm difference that survives
  QK-norm (all in the introduction and 5.5). Still 28 pages, no overfull boxes, contents on page 2.
- 19:08 UTC: **all 41 references checked again** against their primary sources: metadata, and every sentence of the paper that
  cites each one. The four ICLR 2027 submissions are now read in full on OpenReview (`notes/literature/reference_check_2026-10-09.md`).
  uHC's group-wise read is our per-head read with a local predictor, its write-back adds unmatched parameters, it keeps the mix of
  mHC, and it gains 0.006 and 0.008 over mHC at 46M and 128M; the 0.128 at 363M is against an mHC baseline 0.093 worse than the
  residual. Fixed in the paper: the ViT-22B logit instability appeared at about 8B, with Zhai et al. cited for the name and
  Henry et al. for QK-norm itself; OLMo 2 normalizes the whole query and key projections, so our per-head QK-norm follows Gemma 3
  and Qwen3 (also in `model.py`); Qwen3.8-Next's stress test is of its new architecture and optimizer on a 25B MoE (3B active)
  with a 201-step median; mHC uses three linear maps; MUDDFormer predicts its depth weights; VWN widens the stream before
  splitting it; SiHC (954M) is not a large design; Wortsman et al. do not rule out the architecture for loss spikes; the
  temperature control of the anonymous MHAR submission explains the split over a single query; osHC bounds the product of the
  mixes; MHAR's gain was larger at 350M and 1B than at 100M; the uHC paragraph, its row in Table 1 and the limitation now follow
  the full text. Bib: three titles as on the source, Fleuret's ç, Pawar's full name, ViT-22B at ICML 2023, OLMo 2's COLM paper as
  the shorter version, and updated version and venue notes. Still 28 pages, no overfull boxes.
- 20:40 UTC: **two citations added** to Related Work, both checked at the source: Narang et al. (EMNLP 2021), where most
  transformer modifications reimplemented in one code base did not meaningfully improve performance, and Melis et al. (ICLR
  2018), where standard LSTMs beat newer recurrent architectures once all were tuned by the same hyperparameter search. Still
  28 pages, no overfull boxes.
- 21:35 UTC: **every number in the paper recomputed from `results/`** with the repo's scripts (about 680 values, tables and
  figures included), and the cited sentences read once more against their sources. No number was wrong in sign or count.
  Fixed: the chip table used four v5e seeds of local h4 where six exist (`hardware_check.py` now reads `exp08_local`), so that
  gap is +0.003 ± 0.008 (3/6) over all v5e seeds, the opposite sign to v6e, and the misses are 0.014–0.021; the run count is 629,
  since the 3 finished compile-time probes of exp03 were not counted; Table 6 now lists all variants at 3e-3, adding static h8,
  the static joint mix and local h8 and h16, none of them established; 3e-3 is called the pilot rate of mHC, not its tuned v5e
  rate; Appendix A and Table 3 say which runs used v5e; small roundings (0.0008, −0.003 for the 112M line, 0.037–0.048 for
  QK-norm, below 0.5%, logits and entropies as seed means); the loss-spike claims are limited to runs with per-step logs; the
  112M residual crosses the clip most often at three of four rates. Sources: TEMPER redesigns the dense predictor maps, not the
  mix; xHC updates a few of its streams per sublayer; SiHC and identityHC fix the mix to the identity; Kramer et al. see only a
  trend at 1B; Melis et al.'s LSTMs are regularized; Wortsman et al. cap at the loss at initialization; the Qwen median is
  rolling; Gemma 3 and Qwen3 state QK-norm but not its form; gated attention did not tune each design; multi-head attention
  described as in Vaswani et al. Added the per-head ablation of Attention Residuals (1.752 against 1.746 for 16 heads) to Related
  Work and 6.3. Now 29 pages, no overfull boxes.
