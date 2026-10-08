# Qwen's stability / high-LR evidence, and whether our exp10 claim can lean on it

Date: 2026-10-05. Question this file answers: the report wants to cite "Qwen 3.8 flash" as evidence
that *tolerating a high learning rate and showing fewer loss spikes* is a valuable property, and that
multi-head mHC's high-LR tolerance at 27M/112M "might reduce training instabilities in bigger models".

Everything below was read **at the source today** unless the line says otherwise. Levels:
**FULL-TEXT** = the relevant sections of the paper/PDF/blog were read verbatim; **SECONDARY** = read
only through a third-party page or a search snippet (treat as unverified).
Direct quotes are kept under 15 words; everything else is paraphrase.

Prior context: `recheck_2026-10-03.md` §3.1 (Qwen3.8-Next / GR) and §3.2 (GatedNorm, 2601.22966).
This file supersedes §3.1 on the stability material: §3.1 missed the production-run verification
(§3.3 *Verification at the Production Run*), which is the single most important item for our argument.

---

## 0. Bottom line

1. **"Qwen 3.8 flash" is a model, not a document.** `Qwen3.8-Flash` is the hosted production version
   on QwenCloud of the open-weights model **Qwen3.8-Flash-Next** (released 2026-08-26). It has **no
   technical document of its own**. All of the architecture and stability evidence is in one report —
   the one we already cite as Qiu et al. 2026, arXiv **2608.30320** — whose title says *Qwen3.8-Next*
   while the model it describes is called *Qwen3.8-Flash-Next*. Cite the report, and when naming the
   model in prose write "Qwen3.8-Flash-Next (served as Qwen3.8-Flash)".
2. **The Qwen evidence for "stability margin is worth having" is real, quantified, and stronger than
   our earlier notes recorded.** The report does not stop at a stress test: it runs a three-arm
   comparison over the first **276B tokens of the production recipe at the shipped learning rate**,
   and states that this reproduces the stress-test finding at **8x the model scale** of the stress
   test. The full 125B run then completed with no loss spike and no qk-clip/SwiGLU-clip.
3. **But in every single Qwen instance the stable variant also wins at its own tuned learning rate.**
   GR lowers loss by 0.026 at 276B tokens at the shipped LR; Table 5 has GR at 1.590 vs pre-norm
   1.617; gated attention beats its baseline at the baseline's *own* tuned LR at every scale. Qwen
   never argues that a component with no gain at the tuned LR is worth keeping — §2.2 says they kept
   added expressiveness only where it paid for itself, and they *dropped* a change that was free in
   pre-training loss. That is the exact shape of our exp10 result, and it is where the citation
   cannot carry us.
4. **The mechanism Qwen credits is not coefficient granularity, and on `H_res` their design moves the
   opposite way from ours.** They attribute the margin to a bounded multiplicative gate supplying
   rescaling (instead of the network growing activation outliers) and, separately, to **dropping
   `H_res`** because it "requires separate constraints" and is a potential instability source. Our
   multi-head mHC keeps `H_res` and adds one Sinkhorn-projected mix *per group*.
5. **Their cleanest single-variable stability isolation is a GatedNorm toggle, not GR**, and the
   4x-LR "zero spikes" run is Muon + GR against AdamW and Muon baselines — Muon carries most of the
   categorical gap. Their GR ablations are also **not parameter-matched** (unlike their gating papers,
   which trim the FFN to match).
6. **Usable framing:** cite Qwen for the *methodology* (deliberate over-LR stress testing as a
   recognised proxy for production instability, with concrete metrics) and for the *mechanism*
   (bounded multiplicative rescaling reduces spike frequency and magnitude), and state our own
   high-LR result as consistent-with / hypothesis-generating. Do **not** write that Qwen's results
   show a component whose only benefit is LR tolerance is valuable at scale. See §6.

---

## 1. Document inventory: what "Qwen 3.8" actually published

| # | item | date | type | level |
|---|---|---|---|---|
| 1 | **arXiv 2608.30320**, "On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability" — `https://arxiv.org/abs/2608.30320`, HTML `https://arxiv.org/html/2608.30320v1` | v1 2026-08-31 06:35 UTC, cs.CL, 1 version, no comments field | the technical report; the only Qwen3.8 document with stability content | FULL-TEXT |
| 2 | `tech_report.pdf` in `github.com/QwenLM/Qwen3.8-Flash-Next` — `https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf` | cover date **2026-08-26**, 28 pages | **the same document as #1** (identical title and abstract; §3.3 text matches word-for-word). The repo's own BibTeX is a `@techreport` with no arXiv id, so the PDF is the version of record and arXiv is the mirror. | FULL-TEXT (pdftotext) |
| 3 | Release blog, "Qwen3.8-Flash-Next: A New Architecture, Towards Ultimate Cost-Efficiency" — `https://qwen.ai/blog?id=qwen3.8-flash-next` | 2026-08-26 | release notes; qualitative on stability, no numbers | FULL-TEXT (read in a browser; the page is client-rendered, so curl returns an empty shell) |
| 4 | `github.com/QwenLM/Qwen3.8-Flash-Next` README | release line dated 2026-08-26 | pointer only | FULL-TEXT (via `gh api`) |
| 5 | `huggingface.co/Qwen/Qwen3.8-Flash-Next` model card | Aug 2026 | pointer only; gives "125B with 6B activated, plus 51B n-gram embedding and 4B MTP" | FULL-TEXT |

**There is no separate Qwen3.8-Flash paper, blog or card.** The release blog says the production
version, with 1M context and built-in tools by default, is "served as Qwen3.8-Flash on QwenCloud"
(0.15 / 0.47 USD per M in/out tokens). Open weights are Flash-Next; Flash is the hosted variant.

I enumerated `https://qwen.ai/research` back past 2026-08-26 (three pages). The other Qwen3.8-era
posts in that window are product releases with no architecture/stability content: Qwen3.8-Max
(2026-08-03), Qwen3.8-Omni-Flash and Qwen3.8-LiveTranslate (both 2026-09-18), Qwen-Image-2.1
(2026-09-20). **No fourth Qwen3.8 architecture document exists as of 2026-10-05.**
arXiv API phrase queries for `Qwen3.8` / `Qwen3.8-Next` / `Qwen3.8-Flash-Next` (ti/abs/all) return
**zero** entries — the API tokenizer appears to choke on the string, so do not read that as absence;
the paper is reachable by id.

**Authorship for the BibTeX.** The arXiv metadata lists 36 named authors, first author **Zihan Qiu**,
last two Bo Zheng and Dayiheng Liu. The PDF byline and the official BibTeX both say "Qwen Team".
Our existing `Qiu et al. 2026` attribution is defensible (and matches 2505.06708 / 2601.22966, also
Qiu-first), but Qwen's own citation string is `@techreport{qwen2026design, author = {{Qwen Team}}}`.

**Model, for the record.** 125B total / 6B activated per token / +51B n-gram embedding tables held
off-accelerator (+4B MTP per the HF card); 262,144 native context, 1M with YaRN; 3 GDN layers : 1
QSA layer; GR with `n_r = 4` branches, a separate GR module for the attention and the MLP block of
every layer. Training tokens are **never stated in absolute terms** — only "1/3 the training tokens"
and ~1/9 the training FLOPs of the 397B-A17B predecessor (Qwen3.7-Plus). The "48 layers, hidden 2560"
figure that search results quote is **SECONDARY** (respan.ai / intuitionlabs); I did not find it in
the report.

---

## 2. arXiv 2608.30320 — all stability content, with scales

### 2.1 Which experiment is at which scale (this matters more than any single number)

| experiment | model | tokens / steps | §, Tab/Fig |
|---|---|---|---|
| widening alone (static AltUp-style, `n_r` scalars per block) | 25B-A3B MoE | 400B tokens | §2.2 |
| read/write design ablation (pre-norm, mHC static, mHC dynamic, GR) | 25B-A3B MoE, `n_r = 4` | **560B tokens** | §2.2, Tab. 5 |
| GR vs Full/Block AttnRes, +/- GatedNorm | 28 layers (L = 56 sublayers); also 48 layers | token count **not stated** | §2.2, Tab. 6 |
| **stability stress test** (2x and 4x optimal LR, constant LR) | **28-layer 25B-A3B MoE** | 19,932 steps | §3.3, Figs 10-11 |
| gate isolation (GatedNorm off/on, AdamW, 3x optimal LR) | the same 28-layer model | not stated | §3.3, Fig. 12 |
| batch-size validation | 20-layer 10.8B-A0.89B MoE | 4T tokens | §3.2, Fig. 8 |
| LR validation (optimum, /sqrt2, xsqrt2, +25% batch, old recipe) | **48-layer 156B-A7B MoE** | 419B tokens | §3.2, Fig. 9, Tab. 10 |
| **production verification** (Qwen3.5+Muon vs +GR vs full recipe) | the 125B-A6B production model | **first 276B tokens** | §3.3, Fig. 13 |
| production run | 125B-A6B | full budget (unstated) | §3.3 |

Note the ladder: their *smallest* stability experiment is a 28-layer 25B-A3B MoE. Our exp10 is at
27M and 112M dense — three orders of magnitude below the point where Qwen considers the phenomenon
observable at all. Their own motivation says stability challenges at trillions of parameters and tens
of trillions of tokens are "entirely absent in smaller-scale experiments", which is *why* they amplify
them by raising the LR. So Qwen's position simultaneously licenses the over-LR method and warns that
small-scale behaviour at the tuned LR is uninformative about scale.

### 2.2 The stress test (§3.3), verbatim numbers

Design: hold the LR constant at a multiple of its optimum, bypassing the decay schedule, "to simulate
the prolonged peak learning rate of a production run"; 28-layer MoE at 2x and 4x optimal LR. They
credit the method to Wortsman et al. 2023. All runs share batch size and a **gradient-norm clipping
threshold of 0.5**. Acceptance criterion: the new recipe must be at least as stable as the
already-scaled Qwen3.5 + AdamW structure under equal stress.

Three metrics (worth copying into our paper):
- **loss spikes** = steps exceeding a **201-step rolling median by more than 0.1**;
- **p99.9 of the pre-clip gradient norm**, and the **number of clip-threshold crossings**;
- **per-block maximum activation**.

Results (Fig. 10; the three arms are Qwen3.5 structure + AdamW, same structure + Muon, Muon + GR):

| stress | AdamW baseline | Muon | Muon + GR |
|---|---|---|---|
| 2x optimal LR | 4.3 spikes / 10k steps | 0.2 / 10k | 0.2 / 10k (both Muon configs) |
| 4x optimal LR | **183 spikes / 10k**, crosses clip threshold on **213 of 19,932 steps** | never crosses | never crosses, **zero loss spikes** |

They add that GR reduces both the **frequency and the magnitude** of gradient-norm spikes and the
magnitude of activation outliers (Fig. 11), and that at 2x LR the Muon runs have a *higher* median
gradient norm and larger maximum activations than AdamW yet far fewer spikes.

### 2.3 The clean single-variable isolation is a **GatedNorm** toggle, not GR (Fig. 12)

Same 28-layer model, **AdamW**, structure and data order fixed, **3x** the optimal LR, GatedNorm off
vs on: spike rate **32.0 -> 3.2 per 10k steps**, clip-threshold crossings **256 -> 20**. An LR ladder
on the ungated baseline (1x, 2x, 3x) shows activation outliers growing roughly proportionally with
the LR while the spike rate grows much faster; with the gate on at the highest LR the outlier level
falls below the ungated baseline at the lowest LR. Their mechanism: at high LR the network needs a
rescaling mechanism, and without an explicit gate it manufactures one by growing activation outliers,
which leaves it fragile; a multiplicative gate supplies the rescaling directly.

So the *attributable-to-a-gate* stability number is about GatedNorm (which GR carries inside its read),
not about the widened stream, the branch count, or coefficient granularity.

### 2.4 Production verification — the transfer evidence (§3.3, Fig. 13)

Three runs sharing data order, LR schedule and optimizer, first **276B tokens**, at the **shipped**
learning rate: (a) Qwen3.5 structure + Muon, (b) the same + GR, (c) the full Flash-Next recipe
(further refined GR + n-gram embedding layer).

- **Loss:** GR lowers the loss at 276B tokens by **0.026**; the full recipe by a further **0.032**;
  total **0.058** over the Muon baseline. They say this translates into significant benchmark gains.
- **Gradient norm:** Muon alone has roughly **2x** the median and **4.2x** the p99.9 of either gated
  run — **0.097 / 0.298** vs **0.053 / 0.071** and **0.043 / 0.066** — and is the only arm to cross
  the clipping threshold. The gated runs are **4.3-4.7x** steadier in the standard deviation inside a
  1000-step rolling window, which they describe as reproducing the stress-test finding at **8x the
  model scale** and at the production LR.
- **Activations:** GR markedly reduces the residual maximum at every probed depth, which they say
  permits stable training without qk-clip or SwiGLU-clip.
- Intro claim about the shipped run: full-scale training proceeded "without a single loss spike or
  anomalous fluctuation in gradient norms".

This is the strongest published instance of *stress-test-at-moderate-scale -> stability at production
scale* for a residual-stream component. Caveats: one run per arm, no seeds, one LR, 276B of the
budget (not the whole run), and (b) vs (c) mixes GR refinements with the n-gram layer.

### 2.5 Learning-rate and batch-size scaling (§3.2) — the "margin gets cashed in" step

They refit the hyperparameter scaling law for the new architecture + Muon and state that the pair
shifts the near-optimal **batch size and learning rate substantially upward**, with slower LR decay as
model size grows. The stated *reason* for looking is §3.3: the new architecture and optimizer train
noticeably more stably, "suggesting room for a more aggressive hyperparameter setting".

Validations:
- Batch size, 20-layer 10.8B-A0.89B, 4T tokens: old recipe B = 12.6M -> predicted B = 25.2M is worth
  7.2e-3 loss; 1.5x further (37.7M) costs 4.3e-4 (insignificant). Loss rises steeply below the
  prediction, nearly flat above. Batch-size *warmup* is unnecessary: both ramp variants end 2.5e-4 /
  3.5e-4 worse and need **18.8%** more optimizer steps. No step in that sweep exceeded its local
  median by 0.1; p99.9 pre-clip grad norm 0.088-0.190 against the 0.5 threshold.
- LR, 48-layer 156B-A7B, 419B tokens, five runs: predicted optimum B = 8.4M, eta = 1.76e-3; the old
  Qwen3.5 recipe (B = 4.2M, eta = 6.8e-4) ends **7.8e-3 above** it; the four settings near the optimum
  end within 7e-4 of each other — a bowl flat over at least sqrt(2) in LR either way and +25% in batch.
  Downstream (Tab. 10): predicted optimum avg 60.55, eta x sqrt2 60.10, B x 1.25 60.06, eta / sqrt2
  59.14, Qwen3.5 recipe 56.41. Clipping never engages after warmup in any of the five; max pre-clip
  grad norm is 28% of the threshold at the optimum vs 51% under the old recipe; no loss spike anywhere.

**Confound to state if we cite this:** the upward LR/batch shift is attributed to the architecture
**and** the optimizer **together**. Nothing here isolates GR's contribution to the optimal LR.
Counterweight already in our notes: DepthBench (2609.32534) measured mHC's optimal LR as equal to
Pre-LN's (2e-3) at 400M on a coarse grid — so "mHC-family changes move the usable LR up" is not
established for mHC itself.

### 2.6 GR and its ablations (§2.2) — for completeness; §3.1 of `recheck_2026-10-03.md` has the rest

Confirmed again verbatim today: the five conclusions (sigmoid beats tanh in loss *and* stability;
data dependence costs little loss but gains benchmarks; **read granularity matters more than write
granularity**; read all branches with a group RMSNorm; **`H_res` adds little**), Table 5
(pre-norm 1.617 / 50.91, mHC static 1.596 / 52.49, mHC dynamic 1.594 / 54.47, GR 1.590 / 54.66) and
Table 6 (pre-norm 1.789 -> 1.787 with GN; Block AttnRes S=4 1.773 -> 1.768; S=2 1.770 -> 1.766;
Full AttnRes 1.762 -> 1.758; GR with GN 1.762, and GR has no "without GN" entry because the gate is
intrinsic to its read; at 48 layers Block AttnRes S=4 1.711 vs GR 1.707).

New details relevant to stability and to our framing:
- They say GR and mHC-dynamic "at this scale the two perform comparably", differing mainly in the
  elementwise `H_mix` and the removal of `H_res`. GR's advantages are stated as **efficiency** (one
  fewer full read of the residual state per block) and **stability**, the latter twofold: GatedNorm
  improves stability, and dropping `H_res`, which needs separate constraints, removes a potential
  instability source.
- Static terms bring no improvement for GR; standard random init suffices.
- Because the read already normalizes and gates, GR **replaces** pre-normalization rather than
  preceding it, so widening adds no normalization layer.
- Inference-efficiency section, worth knowing: a **sparse read** (each block reading only the two
  highest-gated branches) left pre-training loss and benchmarks almost unaffected but "degraded
  clearly after post-training", so it was rejected — explicitly flagged as a case where pre-training
  metrics alone would have given the wrong decision. The residual state is kept in **FP8**, which the
  gates make safe by bounding what enters the stream. They cite xHC (Zhang et al. 2026) for larger
  `n_r` and did not pursue it.

**Parameter matching, checked explicitly:** the report gives **no** parameter count for GR and makes
**no** parameter-matched claim for Table 5 or Table 6. GR's bottleneck is rank `r = d/8` with
`W_d` in `R^{r x n_r d}` and `W_u` in `R^{n_r d x r}` plus `W_w` in `R^{n_r x n_r d}` and per-branch
gains, two GR modules per layer — of order `d^2` added parameters per layer, i.e. far more than mHC's
dynamic projections (order `n_r^2 d`). The only matched statement in §2.2 concerns the *static*
widening variant ("adds `n_r` parameters per block"). Their n-gram section does use fixed-budget
comparisons and 300 tokens-per-active-parameter; the residual section does not.

---

## 3. arXiv 2505.06708 — Gated Attention (Qiu et al., NeurIPS 2025). FULL-TEXT

`https://arxiv.org/abs/2505.06708`, v1 2025-05-10, cs.CL; 13 authors, first author Zihan Qiu.
This is the closest precedent for "a component's value is that it tolerates larger LRs and reduces
spikes", and it is the paper Qwen cites for the attention output gate in both Qwen3-Next and
Qwen3.8-Flash-Next.

Abstract claim, verbatim fragment: the modification "also enhances training stability, tolerates
larger learning rates, and improves scaling properties". Intro restates it as nearly eliminating loss
spikes and enabling larger learning rates.

Scales: 30 variants of **15B-A2.54B MoE** models and **1.7B dense** models, subsets of a **3.5T-token**
corpus, sequence length 4096, AdamW, GQA, 128 experts top-8 with global-batch LBL and z-loss. Gating
costs <2% wall-clock latency, and **parameter parity is restored by narrowing the FFN**.

The evidence (Table 2, dense models; "Avg PPL", lower is better; `-` = diverged):

| setting | method | max LR | Avg PPL |
|---|---|---|---|
| 28L, 1.7B, 400B tok, bsz 1024 | baseline | 4.0e-3 | 7.499 |
| | SDPA elementwise gate | 4.0e-3 | **7.404** |
| 28L, 1.7B, **3.5T tok**, bsz 2048 | baseline | 4.5e-3 | 6.180 |
| | SDPA elementwise | 4.5e-3 | **6.130** |
| 48L, 1.7B, 400B tok, bsz 1024 | baseline | 4.0e-3 | 7.421 |
| | baseline | **8.0e-3** | **9.195** (badly degraded) |
| | baseline + sandwich norm | 8.0e-3 | 7.407 (convergence restored, no gain) |
| | SDPA elementwise | 4.0e-3 | **7.288** |
| | SDPA headwise | 4.0e-3 | 7.370 |
| | SDPA elementwise | 8.0e-3 | 7.325 |
| 48L, 1.7B, **1T tok**, bsz 4096 | baseline | 5.3e-3 | 7.363 |
| | baseline | 8.0e-3 | **diverged** |
| | SDPA elementwise | 5.3e-3 | 7.101 |
| | SDPA elementwise | **8.0e-3** | **7.078** (best) |

Read this carefully, because it is the crux of §6:
- **The gate wins at the baseline's own tuned LR in every block** (7.404 vs 7.499; 6.130 vs 6.180;
  7.288 vs 7.421; 7.101 vs 7.363). LR tolerance is an *additional* property, not the whole case.
- **LR tolerance only cashed out at the larger budget.** At 400B tokens the gated model at 2x LR
  (7.325) is slightly *worse* than the gated model at 1x (7.288) — the tolerance prevented a
  catastrophe but bought nothing. At 1T tokens / bsz 4096, where the baseline diverges at 8e-3, the
  gated model at 8e-3 is its best run (7.078 vs 7.101). So "tolerance converts into a gain" is
  demonstrated once, at 1.7B / 48 layers / 1T tokens, with a single run per cell.
- Fig. 1 (right) is the spike claim: smoothed training loss over 3.5T tokens, baseline vs SDPA-gated
  1.7B dense models under identical hyperparameters; the caption says gating gives lower final loss
  and "substantially enhanced training stability, mitigating loss spikes". **It is a figure, and no
  spike count is reported anywhere in the paper** — unlike 2608.30320, there is no spike metric, no
  threshold, no p99.9 gradient norm. Treat "fewer loss spikes" here as a qualitative, single-run claim.
- Mechanism, §4.3-4.4: head-specific input-dependent sigmoid gating of the SDPA output introduces
  sparsity, which removes the attention sink and reduces massive activations; smaller activations are
  then less exposed to BF16 numerical error. They locate the massive activations in early layers'
  FFN outputs, which pre-norm then propagates — hence sandwich norm also helping.
- **Useful negative result (App. A.5):** clipping attention/FFN outputs into (-clip, clip) at
  clip = 300 or 100 did **not** fix convergence at 8e-3. Their conclusion is that pre-norm
  instability is not solely due to large activations in the residual stream, and that any layer with
  large outputs can cause it. This is a caution against a tidy "our variant shrinks spikes, therefore
  it will scale" story: in their hands, a direct intervention on the symptom did not buy stability.
- Granularity, for our (b): head-shared gating scores are consistently worse than head-specific ones
  (rows 12/13 vs 10/11), and headwise gating adds <2M parameters on the 15B-A2B model. But they also
  say that as long as heads get distinct scores, "the granularity of gating ... ha[s] relatively minor
  impacts". So head-specific > shared is clear; elementwise > headwise is weak.

---

## 4. Qwen3-Next release notes (2025-09-10). FULL-TEXT (browser)

`https://qwen.ai/blog?id=qwen3-next`, "Qwen3-Next: Towards Ultimate Training & Inference Efficiency",
2025/09/10, QwenTeam. Model: Qwen3-Next-80B-A3B, ~3B activated, trained on a uniformly sampled
**15T-token** subset of Qwen3's 36T corpus, <80% of the GPU hours of Qwen3-30B-A3B and 9.3% of the
compute of Qwen3-32B. Gated DeltaNet : standard attention mixed **3:1**.

**What the release notes actually say about stability** (section "Training-Stability-Friendly
Designs"), and it is less than one might assume:
- the attention **output gating** mechanism helps eliminate attention sink and massive activation,
  ensuring numerical stability (citing 2505.06708);
- Qwen3's QK-Norm left some layer-norm weights abnormally large, so Qwen3-Next adopts **zero-centered
  RMSNorm** with weight decay on norm weights to prevent unbounded growth;
- MoE **router parameters are normalized at initialization** so every expert is unbiasedly selected
  early;
- these designs "make small-scale experiments more reliable" and help large-scale training run
  smoothly.

**There is no loss-spike claim, no spike count, no learning-rate claim and no stress test anywhere in
the Qwen3-Next release notes.** Any "fewer loss spikes / tolerates larger LR" sentence attributed to
Qwen3-Next must be sourced to **2505.06708** (the gated-attention paper it cites) or to
**2608.30320** (Qwen3.8-Next), not to the Qwen3-Next blog. The blog's only architecture-level
transfer claim is the sentence about small-scale experiments being more reliable.

The Qwen3.8-Flash-Next blog (2026-08-26) adds, qualitatively: GR combines HC-style widening with
GatedNorm's elementwise dynamic gating in the read; once read and write are expressive enough,
additional branch mixing "yields no significant benefits" and is removed, reducing memory traffic
*and* "sources of instability"; the gate suppresses activation outliers and improves training
stability; the refitted scaling law shows the model can stably use larger learning rates and batch
sizes. Same content as the report, no numbers. It also records that the GDN + Gated Attention design
from Qwen3-Next has been used across Qwen3.5, 3.6, 3.7 and 3.8 — i.e. **four generations of adoption**
of a component whose published case rests substantially on stability and LR tolerance. That adoption
fact is probably the most quotable thing in the blog for our purposes.

---

## 5. arXiv 2601.22966 (GatedNorm) — the "tolerance vs optimum" distinction, re-verified at source

"A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer
Training", Qiu et al., 2026-01-30, 19 authors. Scales: standard softmax, linear, and linear-full
hybrid architectures at **1B, 7B and 24B parameters, 120B to 1T tokens**.

Verified verbatim today (§3.5): equipping DyT with gated attention "enables the model to train stably
at the baseline's learning rate (4.3e-3), though its optimal learning is 2e-3." This is the cleanest
statement in the Qwen corpus of the distinction our exp10 turns on — **stable-at-a-higher-LR is not
the same as optimal-at-a-higher-LR** — and it is made by Qwen themselves, about their own component.
Also verified: under *tensorwise* gating all non-sigmoid activations (tanh, SiLU, none) cause
divergence, while elementwise sigmoid gating is both the best performer and the stable one — the
closest thing in the literature to "finer gating granularity is more robust", though it is an
accident of their ablation grid rather than a controlled granularity-vs-stability experiment.
The elementwise-vs-tensorwise performance ablation and the +3.7M-parameter / FFN-trimmed parameter
parity are as recorded in `recheck_2026-10-03.md` §3.2 (not re-read today).

---

## 6. Assessment: can our paper use this?

### 6.1 What the Qwen results genuinely support

1. **That "stability margin under a deliberately raised LR" is a first-class design axis at frontier
   scale, with a defined protocol and metrics.** 2608.30320 §3.3 evaluates every candidate change on
   stability alongside loss and cost, and gates acceptance on a stress test that holds the LR at 2x
   and 4x the optimum with the decay schedule bypassed. We can cite this to justify *why* we ran an
   over-LR comparison at all, and to borrow their metrics.
2. **That a gate-induced stability margin measured at moderate scale can show up at much larger
   scale.** Their 28-layer stress-test finding reappeared in the 276B-token production comparison at
   8x the model scale, and the 125B run reported no loss spikes without qk-clip or SwiGLU-clip. This
   is the one published small-to-large transfer for a residual-stream component.
3. **That the margin has a concrete payoff route:** more stability -> a more aggressive refitted
   scaling law -> higher optimal LR and batch -> better loss (7.8e-3 over the old recipe at 156B-A7B).
   And once, in 2505.06708, the gated model's best run was the one at the LR where the baseline
   diverged (7.078 at 8e-3).
4. **A mechanism compatible with ours:** bounded multiplicative gating supplies the rescaling that an
   ungated network obtains by growing activation outliers; gates reduce the frequency *and* the
   magnitude of gradient-norm spikes and the size of activation outliers. Since mHC's coefficients are
   sigmoid-bounded and our heads multiply the number of independently bounded rescaling knobs per
   sublayer, "the same family of mechanism" is a fair thing to say.
5. **Adoption as evidence of value:** a component whose published case leans heavily on stability and
   LR tolerance (gated attention, 2505.06708) has shipped in four consecutive Qwen generations.

### 6.2 Where the analogy is a stretch — state these, or a referee will

1. **In every Qwen case the stable variant also won at the tuned LR. Ours does not.** This is the
   decisive difference. GR: -0.026 at 276B tokens at the shipped LR, and 1.590 vs 1.617 pre-norm in
   Table 5. Gated attention: better than the baseline at the baseline's own tuned LR at all four
   settings. Qwen's own methodology is explicitly "keep expressiveness only where it pays for itself",
   and they rejected a sparse read that was *free* in pre-training loss. **Nothing in the Qwen corpus
   supports keeping a component that is neutral at the tuned LR because it degrades less at 2-3x.**
   Our sentence must therefore be a hypothesis about scale, flagged as untested, not an inference
   licensed by Qwen.
2. **Different mechanism, and opposite direction on `H_res`.** Qwen credit (i) a multiplicative
   elementwise read gate that replaces pre-norm and bounds what enters the stream, and (ii) *deleting*
   `H_res`, whose separate constraints they call a potential instability source. Multi-head mHC keeps
   `H_res` and multiplies the number of Sinkhorn-projected doubly stochastic mixes by `h`. Citing GR
   as support for our stability story while our novelty is per-group `H_res` invites the obvious
   counter-reading: Qwen's data say that component is the one to remove. Nowhere do they attribute
   stability to coefficient granularity — their granularity ablation is reported on loss and
   benchmarks only.
3. **Their clean isolation is GatedNorm, not GR, and the headline stress number is confounded with
   Muon.** The 4x-LR comparison is AdamW vs Muon vs Muon + GR; both Muon arms never cross the clip
   threshold, so most of the categorical gap belongs to the optimizer. GR's isolated increment at 4x
   is "zero spikes" versus an unstated small number, plus smaller spike magnitude (a figure). The
   32.0 -> 3.2 spikes/10k and 256 -> 20 crossings are a **GatedNorm on/off** toggle under AdamW with
   the structure fixed — a gate on the sublayer output, not a widened stream and not a connection
   coefficient.
4. **Scale gap, by their own argument.** Their smallest stability experiment is 28 layers / 25B-A3B;
   they say the relevant instabilities are absent at smaller scale, which is why they amplify via LR.
   We are at 27M/112M dense. Their framework licenses the *method* at small scale while denying that
   the *phenomenon* is observable there — and nothing tells us the 27M -> 112M -> ? trend in LR
   tolerance is monotone. Compare Review Residuals (2606.31859): a per-channel input-dependent gate
   that is null-to-negative below 320M and only significant at 590M-1B.
5. **Their GR ablations are not parameter-matched; our claim is.** No GR parameter count appears in
   the report, and GR's `d/8` bottleneck adds of order `d^2` per layer against mHC's order `n_r^2 d`.
   Their *gating* papers do match parameters (FFN trimming in 2505.06708 and 2601.22966), so cite
   those for parameter-matched granularity claims and not Table 5.
6. **Single runs.** Stress test: one run per arm. Production verification: one run per arm, one LR,
   first 276B tokens only. Table 2 of 2505.06708: one run per cell. Fig. 1's spike claim has no
   quantification. Our own exp10 should not be presented as more certain than theirs.
7. **The "mHC moves the usable LR up" claim is contradicted elsewhere.** DepthBench (2609.32534)
   found mHC's optimal LR equal to Pre-LN's (2e-3) at 400M. Qwen's upward shift is attributed to the
   architecture *and* Muon jointly and never decomposed. So we cannot borrow Qwen's LR shift as
   evidence about mHC-family components specifically.

### 6.3 Concretely, what I would and would not write

Defensible (paraphrase, as a methods/discussion note):

> Frontier training reports now evaluate architectural changes on stability as a separate axis from
> loss, using deliberately raised learning rates at moderate scale as a proxy for the instabilities
> that only appear in long production runs [2608.30320 §3.3, after Wortsman et al. 2023], and in one
> documented case a gate-induced margin measured on a 28-layer MoE reproduced at eight times that
> scale at the shipped learning rate and permitted a refitted, more aggressive hyperparameter recipe
> [2608.30320 §3.2-3.3]. Qwen attribute the margin to bounded multiplicative rescaling, which an
> ungated network otherwise obtains by growing activation outliers. Multi-head mHC's behaviour at
> 2-3x the optimal learning rate is consistent with that mechanism; we note it as a hypothesis about
> larger models rather than as a demonstrated benefit, since at our scales and at the tuned learning
> rate the heads are neutral, whereas in every Qwen result the more stable variant was also the
> better one at its own tuned learning rate.

Not defensible: "Qwen3.8-Flash shows that tolerance of a high learning rate is valuable, therefore
multi-head mHC will reduce instabilities at scale." Also not defensible: attributing any spike or LR
claim to the Qwen3-Next release notes (§4), or implying Qwen attribute stability to finer-grained
connection coefficients (§6.2.2).

### 6.4 Cheap experiments that would convert this from analogy to evidence

1. **Adopt their metrics** so the claim is comparable and citable: spikes per 10k steps against a
   201-step rolling median with a 0.1 threshold, p99.9 of the pre-clip gradient norm, and crossings
   of a fixed clipping threshold — instead of "smaller late gradient-norm spikes".
2. **Test whether the margin has cash value.** Qwen's whole chain is margin -> higher *optimal* LR ->
   lower loss at that LR. Sweep the LR per variant and ask whether multi-head mHC's optimum is
   actually higher than mHC's. If the optimum does not move (as DepthBench found for mHC vs Pre-LN),
   then by Qwen's own logic the margin buys nothing, and the honest claim shrinks to "fails more
   gracefully when mis-tuned" — which is still worth one sentence, but is a robustness claim, not a
   scaling claim.
3. **Separate the gate from the granularity.** Their evidence says the *gate* buys stability. If our
   heads' margin survives a control that keeps one gate per stream but adds the heads only in the
   read (GR-like), we can claim the granularity contributes; otherwise we are re-measuring the gate.
4. **Report the 2-3x LR runs at more than one seed**, given that every Qwen stability number is
   single-run and a referee will apply that criticism symmetrically.

---

## 7. Not verified / open

- **Figures only, not read as numbers:** 2608.30320 Figs 10-13 (the loss and gradient-norm traces,
  the activation panels, the outlier-vs-LR ladder) and 2505.06708 Fig. 1 (the spike claim). I used
  the numbers the text states and did not estimate values off plots.
- **Token counts not stated in the source:** Qwen3.8-Flash-Next's absolute pre-training budget
  (only "1/3 the training tokens" of the 397B-A17B predecessor); the token count behind Table 6
  (28-layer and 48-layer residual comparison); the gate-isolation run (Fig. 12).
- **GR's parameter count** is not reported; my order-of-magnitude estimate from `r = d/8` is mine, not
  theirs.
- **48 layers / hidden 2560 for Flash-Next**: SECONDARY (respan.ai, intuitionlabs), not in the report.
- **`tech_report.pdf` vs arXiv**: I verified the title, abstract and §3.3 text match; I did not diff
  the whole 28-page PDF against the HTML.
- **2601.22966**: only §3.5's LR sentence and the tensorwise-divergence sentence were re-verified
  today; the rest stands on `recheck_2026-10-03.md` §3.2.
- **Qwen3.8-27B / Qwen3.8-Max / Qwen3.5 reports**: not read. The Flash-Next report cites
  "Qwen Team, 2026" for the Qwen3.5 structure and recipe, which is the baseline in every stress-test
  comparison; I did not locate or read that document, so the *baseline* side of the stress test is
  characterised only as Flash-Next describes it.
- **arXiv API cannot be queried for these strings** (`Qwen3.8*` returns zero for ti/abs/all), so I
  cannot rule out a further Qwen3.8 preprint by phrase search; the `qwen.ai/research` enumeration back
  to 2026-08-03 and the GitHub/HF pointers are the basis for the "one document only" conclusion.
- Whether the production run's "no loss spike" claim covers the whole budget or only the window shown
  in Fig. 13 is not fully explicit; the intro states it unconditionally for the full-scale training.
