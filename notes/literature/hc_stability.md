# Stability evidence in the hyper-connection family, and whether per-channel / per-group
# coefficients buy stability or LR tolerance

Written 2026-10-05. Scope set by three questions: (1) how mHC and the HC family actually *show*
stability; (2) whether per-channel / per-group connection or residual coefficients are known to help
stability or LR tolerance; (3) a concurrency re-check since `recheck_2026-10-04.md`.

**Companion files, read them first and do not duplicate:**
- `stability_proxies.md` — Wortsman et al. 2309.14322, the LR-sensitivity metric, spike-counting
  definitions, and the assessment of our own "tolerates 2-3x optimal LR" argument.
- `qwen_stability.md` — all of Qwen's stress-test numbers (GR, GatedNorm, Gated Attention) with scales.
- `recheck_2026-10-03.md`, `recheck_2026-10-04.md`, `citation_crawl.md`, `core_papers_spec.md`,
  `web_and_code_search.md` — the HC-variant census and the mHC spec.

Access markers used below: **FULL-TEXT** (read the body), **ABSTRACT**, **CODE**, **SUMMARY**.

---

## 0. Bottom line

1. **mHC's stability case is: stability is the stated reason the manifold exists, and the evidence is
   three figures from one 27B run at one learning rate.** The abstract names "severe training
   instability" as what the manifold fixes. Fig. 2 (27B) pairs HC's loss surge near step 12k with its
   gradient norm; Fig. 3 (27B) gives the Amax Gain Magnitude, HC's composite peaking near 3000 against
   mHC's ~1.6; Fig. 5 (27B) is the loss-gap + gradient-norm comparison of baseline / HC / mHC.
   **There is no LR sweep, no high-LR stress test and no spike count anywhere in the mHC paper**, and
   no stability ablation of the three mappings (the only ablation table is HC's, on loss). Verified.
2. **Nobody in the HC family has published an LR-robustness experiment.** Across 15 family papers read
   at source, "stability" is shown as: gradient-norm curves at a single LR (mHC-lite, KromHC, TBP-mHC,
   sHC), spectral/Lipschitz bounds (osHC, oHC, BE-HC, JPmHC, ITO-HC), deviation-from-double-stochasticity
   measurements (mHC-lite), or extreme-depth convergence (BE-HC). The only two LR-aware papers adjacent
   to the family are **DepthBench** (per-architecture LR sweep; mHC's optimum equals Pre-LN's) and
   **Qwen's GR / GatedNorm** work (spike counts at multiples of the optimal LR) — neither is an HC
   variant paper, and neither varies connection granularity against LR.
3. **Yes, two family papers claim "more stable" from small-scale single-LR runs**: **mHC-lite** (6-24
   layers, d512-1024) says mHC itself "can still exhibit instability" and that mHC-lite removes it;
   **TBP-mHC** (6-12 layers) claims greater stability from lower gradient norms — and excluded KromHC
   runs that had gradient instability from its tables. Both are gradient-norm-curve arguments, not LR
   sweeps. Our paper's claim type (small-scale, *above* the optimum, spike counts) is **stronger
   evidence than anything the family has published**, which is a reason to state it carefully rather
   than loudly.
4. **The polarity of the family's stability claims is not consistent, and that is citable.** The
   original HC paper reports **no spikes in any DHC run** at OLMo-1B and no spikes at OLMo-7B where the
   plain-residual baseline "exhibits frequent spikes"; mHC reports HC as the unstable one at 27B. Same
   method, opposite claim, different scale. sHC adds a third position: the doubly stochastic
   constraint buys gradient stability by collapsing the mixer toward identity.
5. **On per-group coefficients and stability, the strongest precedents are all outside the HC family,
   and none is a clean granularity-vs-stability ablation.** LayerScale is per-channel diagonal residual
   scaling explicitly motivated by deep-model optimization stability, and its authors call the
   per-channel degrees of freedom decisive over a single scalar — but the comparison rows differ in
   normalization and warmup too, and their baseline converges once its dropout is retuned. Admin's
   `omega` is a genuine per-channel (D-dimensional) residual scale and it is the only precedent with a
   **learning-rate grid**: on a 6-layer model over 15 LR x beta2 settings, Post-LN diverges in 7, Admin
   does not — but Admin never ablates per-channel against scalar. Qwen's elementwise read gate is the
   only precedent that pairs *channel-wise* coefficients with *measured* spike counts at multiples of
   the optimal LR (see `qwen_stability.md`), and it is a gate-vs-no-gate result, not a
   granularity-vs-granularity one.
6. **New and directly useful for our decomposition: SiHC's Table 4 isolates per-channel read/write with
   the mix set to identity, at near-constant parameters.** FID 11.50 (static scalar maps + identity
   carry) -> 10.10 (channel-specific read and write), +147,264 parameters. That is our paper's
   "tolerance sits in the read and write, not the mix" shape — but measured as FID on a diffusion
   transformer, with **no** stability or LR claim attached.
7. **Counter-evidence our paper should cite against itself**: DepthBench finds mHC does not move the
   optimal LR at 400M; Review Residuals finds a per-channel input-dependent residual gate worth nothing
   or less than nothing below ~320M under parameter matching; MHAR's own newest abstract reattributes
   most of its multi-head gain to a softmax-temperature artifact; and two 2026 papers give
   architecture-independent accounts of spikes (LM-head logit growth; AdamW moment mismatch).
8. **Concurrency: nothing new.** No arXiv, OpenReview or GitHub item proposing multi-head / grouped /
   per-head / channel-wise mHC. arXiv's index reaches 2026-10-02 as of today, so 10-03..10-05 is
   unobservable, not empty. Three items are new *to this project*: the **DeepSeek-V4.1-Flash report's
   Single-Pass mHC section** (2609.19969 §2.4.1), the **looped-residual-scaling workshop paper**
   (`bj8l2FYjSd`), and **SpanNorm** (`9bLiqb6Vec`).

---

## 1. mHC's own stability evidence, and the DeepSeek reports

### 1.1 mHC — arXiv 2512.24880v2 (2025-12-31, v2 2026-01-05). FULL-TEXT (re-verified at source today)

`arxiv.org/abs/2512.24880`, `arxiv.org/html/2512.24880v2`. ICML 2026 spotlight (OpenReview
`mDhyxu8WRb`).

**Is stability the main motivation for the manifold constraint? Yes, explicitly, in the abstract.**
The abstract says HC's diversification compromises the identity mapping property, "which causes severe
training instability and restricted scalability", and that mHC projects onto a manifold to restore that
property. §1 adds that HC's unconstrained form leads to "unbounded signal amplification or
attenuation"; §3.1 ("Numerical Instability") that signal magnitude "is prone to explosion or vanishing"
in both passes. Memory-access overhead is the second stated motivation; performance is framed as a
consequence, not the premise.

**How stability is shown — all of it, and all at 27B:**

| evidence | where | what it is | scale |
|---|---|---|---|
| loss surge + gradient norm correlation | **Fig. 2**, caption: HC shows a loss surge "around the 12k step", "highly correlated with the instability in the gradient norm" | two curves from one run pair | **27B** |
| Amax Gain Magnitude | **Fig. 3**: (a) single-layer `H^res`, (b) composite mapping "within the 27B model". Max absolute row sum = forward, max absolute column sum = backward, averaged over tokens of one selected sequence | a forward/backward gain diagnostic, not a training outcome | **27B** |
| gradient-norm profile | **Fig. 5(b)**: baseline / HC / mHC gradient norms; mHC "maintaining a stable profile comparable to the baseline" | curve comparison | **27B** |
| final-loss claim | §5.2: mHC "effectively mitigates the training instability observed in HC", final loss reduction **0.021** vs baseline | one number | **27B** |

Amax numbers: HC's composite peaks near **3000**; mHC's composite reaches a maximum of approximately
**1.6**; the paper states the reduction as three orders of magnitude. mHC's *single-layer* backward gain
deviates slightly from 1 because 20 Sinkhorn iterations are only approximate.

**What is absent.** No LR sweep, no multi-LR runs, no spike counts, no seeds. The four configurations
(3B / 9B / 27B / 3B-1T) each use **one** base LR (8.6e-4 / 5.9e-4 / 4.0e-4 / 9.0e-4), so every
stability figure is a single-LR comparison. The only ablation table is HC's component ablation on loss
(`res` -0.022, `+pre` -0.025, `+pre+post` -0.027), and it reports no stability column. No mHC-side
ablation of `n`, of static vs dynamic, or of the Sinkhorn iteration count. And — carried over from
`core_papers_spec.md` §1.12 — the paper never mentions heads, groups or channel splits; its only
future-work direction is other manifolds.

### 1.2 DeepSeek-V4 — arXiv 2606.19348. FULL-TEXT (unchanged from `core_papers_spec.md` §6)

mHC is presented as already-solved stability: the restatement in §2.2 repeats the doubly stochastic
manifold, Sinkhorn with `t_max = 20`, and sigmoid-bounded `A_l`/`C_l` "to avoid the risk of signal
cancellation". The report adds **no new stability measurement** for mHC. Its §4.2.3 instability
section is about **MoE routing**, not mHC — Anticipatory Routing and SwiGLU clamping to `[-10, 10]` —
which is worth saying plainly: at 1.6T total / 49B active, mHC did not remove all instability.
Optimizer: Muon for most parameters, AdamW for embeddings, prediction head and all RMSNorm weights;
the report does not say which optimizer the `hc_*` tensors use.

### 1.3 DeepSeek-V4.1-Flash — arXiv 2609.19969 (2026-09-17). FULL-TEXT. **New detail for this project**

`arxiv.org/abs/2609.19969`. 552B backbone parameters, 16B active per token at decode and 8B at prefill,
45T multimodal pretraining tokens, up to 1M context. `web_and_code_search.md` §2 records this report at
SUMMARY level and says mHC "is not re-described there". **That is slightly wrong and should be
corrected**: §2.4.1 is titled Single-Pass mHC and describes the V4.1 change that
`core_papers_spec.md` §2.6 reverse-engineered from the HF inference code:

- "Single-Pass mHC, which shifts the input-mixing coefficients by one block", i.e. each block consumes
  the previous block's coefficients;
- the deployment kernel is **Mega-mHC**, fusing residual update, input mixing and coefficient
  prediction; it implements mHC with `(3n+2)d` activation reads/writes and Single-Pass mHC with
  `(2n+2)d`;
- the justification is efficiency only: the shift "incurs negligible performance degradation";
- §4.2.1 repeats expansion factor 4 and 20 Sinkhorn iterations.

**No stability, loss-spike, gradient-norm or learning-rate claim is attached to mHC anywhere in the
V4.1-Flash report.** So the family's entire frontier-scale stability case is still the three 27B
figures of §1.1.

### 1.4 Hyper-Connections — arXiv 2409.19606v3 (ICLR 2025). FULL-TEXT. The opposite polarity

The original HC paper claims HC is the *stabilizing* change, with spike observations at two scales:

- OLMo-1B: DHC's training-loss decline is steeper than baseline's and DHC "demonstrates greater
  stability, with no spikes observed in any DHC experiments" (§4.2). Best variant OLMo-1B-DHCx8
  without tanh: V2 eval loss -0.034, V3 -0.029 vs baseline.
- OLMo-7B-DHCx4 (Fig. 6, past 400B tokens): the baseline "exhibits frequent spikes, while our model
  with DHCs shows no spikes throughout the training".
- HC's framing motivation is the Pre-Norm/Post-Norm "seesaw effect between gradient vanishing and
  representation collapse" (abstract), not high-LR stability. It cites Wortsman et al. 2309.14322.

Method: eyeballed spike presence on loss curves, one LR per setting, no spike counts, no sweep.

**The tension is a usable framing for our paper**: within one family, HC says it removes spikes a
plain residual has (1B/7B); mHC says HC has a loss surge a plain residual does not (27B); sHC says the
constraint that fixes it does so by collapsing the mixer to identity. Nobody has measured any of this
against the learning rate.

---

## 2. HC-family follow-ups: what each one does and does not evaluate

### 2.1 Summary table (all rows verified at the source named)

| work | id / forum | date | access | stability evidence | LR treatment | scale |
|---|---|---|---|---|---|---|
| HC | 2409.19606v3 | 2024-09 | FULL-TEXT | spike presence on loss curves; "no spikes" in DHC runs | one LR per setting | OLMo 1B, 7B, OLMoE |
| mHC | 2512.24880v2 | 2025-12 | FULL-TEXT | Fig. 2 loss surge + grad norm, Fig. 3 Amax gain, Fig. 5 grad norm | one LR per size | 3B/9B/**27B** |
| Frac-Connections | 2503.14125 | 2025-03 | FULL-TEXT | none (motivation is HC's memory cost) | — | OLMo2 / OLMoE |
| mHC-lite | 2601.05732 | 2026-01 | FULL-TEXT | **grad-norm curves (Fig. 2)** + column-sum deviation from double stochasticity (Fig. 3); claims mHC still unstable | single LR per size (Tables 2-3) | 6/12/24 layers, d512/768/1024 |
| KromHC | 2601.21579v2 | 2026-01 | FULL-TEXT | "lowest gradient norms" of the mHC variants (App. I) | separate LRs per param group (Muon 0.02 main; AdamW 0.005 for the connection branch); no sweep | d384 (2.5k steps), d768 (7k steps) |
| sHC / s2HC | 2603.20896v2 | 2026-03 (v2 09-25) | FULL-TEXT | grad-norm curves (Fig. 6, Fig. 11) + layerwise/propagation stability of residual matrices | single LR per backbone | Qwen3 15B tok, Gemma3 20B, Qwen2.5 30B |
| go-mHC | 2604.02309 | 2026-04 | notes (FULL-TEXT) | exact Birkhoff parameterization; spectral argument | — | — |
| TBP-mHC / RTBP | 2605.21724 | 2026-05 | FULL-TEXT | **grad-norm curves (Fig. 1)**, "consistently achieve lower gradient norms"; a Stability column in its comparison table | single LR per size; **excluded KromHC runs with gradient instability** | 6L/d512, 12L/d768 |
| mHC-SSM | 2605.08300 | 2026-05 | ABSTRACT | calls mHC "a stability-motivated variant" | — | — |
| Birkhoff projection | 2606.07574 | 2026-05 | ABSTRACT | argues slow Sinkhorn convergence undermines mHC's norm-control guarantees | — | — |
| stream collapse | 2606.03483 | 2026-06 | FULL-TEXT (grep) | collapse diagnosis; no stability/LR experiment | single LR (6e-4 at 12L, 3e-4 at 24L) | 12L/d768, 24L/d1024 |
| xHC | 2607.14530 | 2026-07 | FULL-TEXT | **real instability finding**: at 18B, conv-branch/main cosine can exceed 0.7 and removing Gram-Schmidt "leads to training instability"; also row-sum clamping after Sinkhorn | WSD, LR scaled with size; no sweep | up to 18B MoE |
| TEMPER | 2608.07851 | 2026-08 | FULL-TEXT | cites mHC's stabilization; no own stability experiment | — | — |
| oHC | 2609.02672 | 2026-09 | FULL-TEXT | SO(4) norm preservation; observes a redundant DoF that "dominate[s] the global gradient norm" and consumes the clipping budget | — | — |
| How Does mHC Use Its Residual Streams | 2609.05309 | 2026-09 | FULL-TEXT (grep) | diagnostic: mixers "settle close to identity" after an attention-side spike at layer 22 | — | — |
| SiHC | 2609.33895 | 2026-09 | FULL-TEXT | **none** — FID ablation only (see §3.4) | constant LR, no warmup | DiT B/L/H |
| DepthBench | 2609.32534 | 2026-09 | FULL-TEXT (per 10-03 note) | **the only per-architecture LR sweep**: {5e-4, 1e-3, 2e-3, 5e-3}; mHC's optimum **2e-3**, same as Pre-LN; effective rank 1.44-1.65 (mHC) vs 2.53-2.81 (HC) | **sweep, each architecture at its own best** | 400M (8B tok), 1.6B (32B tok) |
| JPmHC | 2602.18308v2 | 2026-02 | ABSTRACT | dynamical isometry; operator-norm-bounded manifolds "prevent gradient pathologies" | — | — |
| BE-HC | `jpIjkN1B1Q` (Sci4DL 2026) | 2026-02-02 | ABSTRACT | **extreme-depth convergence**: stable at **1000 layers**, 35.71% accuracy "where ReZero and other baselines fail to converge"; 8K tokens on one V100 at 22.56% val acc; 1.47x throughput at 4K; 4x accuracy retention under INT8 | — | 1000 layers (toy-scale accuracies) |
| SimpleHC | `i2WyUVJUJ2` | 2026-09-06 | ABSTRACT | "smoother gradient dynamics", "remains stable under stress" — **the stress test is not described in the abstract** | unknown | 15B -> 30B MoE |
| osHC | `1BC0eYN2uS` | 2026-09-17 | ABSTRACT | **theory**: one marginal alone bounds the depth-composed spectral norm by `sqrt(n)` at any depth; second marginal only tightens to 1 | — | 3 scales, 2 corpora |
| SHC-PPO | `dzXT0zmzQn` | 2026-09-17 | ABSTRACT | gradient-norm anisotropy preserved "two orders of magnitude better"; returns within confidence intervals | — | 6-layer RL policies |
| ITO-HC | `nKRfYmPtP2` | 2026-09-18 | ABSTRACT | conditional perturbation + spectral stability bounds; concedes "the need for further scale" | — | GPT-2 / OpenWebText |
| uHC | `Qnj7Lf8Bz2` | 2026-09-19 | ABSTRACT | asserts "more stable optimization"; no method stated | unknown | 46M-363M |
| `wdlctc/hyper-connection-factory` | GitHub | commits to 2026-10-04 | CODE | **plain MUDDFormer diverged at 24 layers**, gradient norms to 7e4 from ~step 3000, loss collapsed by step 7000; PrePostDANorm fixes it (max grad norm 14); mHC and HC did not diverge | one recipe, `connection_lr_mult` exists but **no LR sweep** | 38.5M - 1.21B |

### 2.2 The three sub-questions, answered

**(a) Do HC follow-ups evaluate stability or LR robustness, and how?** Stability yes, almost always as a
**gradient-norm curve at one learning rate**, sometimes supplemented by a spectral or Lipschitz bound,
by a measurement of how far the Sinkhorn output is from doubly stochastic, or by extreme-depth
convergence. **LR robustness: none of them.** No HC-variant paper sweeps the learning rate, trains at
multiples of the optimum, or counts loss spikes. The two LR-aware neighbours are DepthBench (sweep,
per-architecture optimum) and Qwen's GR/GatedNorm line (spike counts at 2x/3x/4x the optimum) —
see `qwen_stability.md`.

**(b) Does anyone claim "more stable" from small-scale high-LR runs?** Claims of "more stable" from
**small-scale** runs: yes — mHC-lite (6-24 layers), TBP-mHC (6-12 layers), KromHC (d384/d768), ITO-HC
(GPT-2 scale). But **none of them from high-LR runs**; all are single-LR gradient-norm comparisons, and
TBP-mHC additionally drops the unstable KromHC runs from its tables rather than reporting them. So the
family precedent for "a variant is more stable" is *weaker* than what our paper has, and the honest
framing is that we are supplying the measurement type the family has been asserting without.

**(c) Does anyone show per-channel / per-group coefficients help or hurt stability?** Inside the HC
family, **no one tests it**. The nearest things:
- **uHC** pairs its Group-Wise Feature Readout with a claim of "more stable optimization" — abstract
  only, mechanism and measurement unknown, and the full text is unreadable from here (§4.3).
- **SiHC** has per-channel read/write maps but reports FID, not stability (§3.4).
- **sHC** is the only *hurt* direction, and it is about the constraint rather than the granularity:
  mHC, mHC-lite and KromHC buy gradient stability "at the cost of collapsing into the identity
  mapping", which DepthBench's effective-rank numbers (1.44-1.65 vs HC's 2.53-2.81) measure
  independently.
- Everything that *does* test granularity against stability is outside the family: §3.

---

## 3. Per-channel / per-group residual coefficients and stability — the precedents

### 3.1 LayerScale — CaiT, Touvron et al., arXiv 2103.17239, ICCV 2021. FULL-TEXT (ar5iv)

The canonical per-channel residual scaling, and the one I set out to check. Verified today.

- **Mechanism**: `x + diag(lambda) f(x)`, a learned diagonal matrix, described as a per-channel
  multiplication of each residual block's output rather than a single scalar.
- **Motivation is stability with depth**, stated as such: the goal is to increase optimization
  stability when training image transformers, especially as depth grows; deeper ViT/DeiT do not train
  effectively.
- **Initialization**: `eps = 0.1` up to depth 18, `1e-5` at depth 24, `1e-6` deeper.
- **Table 1** (ImageNet top-1, DeiT-S-style, by depth 12 / 18 / 24 / 36):

  | variant | 12 | 18 | 24 | 36 |
  |---|---|---|---|---|
  | baseline, dr=0.05 | 79.9 | 80.1 | 78.9† | 78.9† |
  | baseline, dropout-rate retuned | 79.9 [.05] | 80.7 [.10] | 81.0 [.20] | 81.9 [.25] |
  | ReZero (per-layer scalar) | 78.3 | 80.1 | 80.8 | 81.6 |
  | T-Fixup | 79.4 | 81.7 | 81.5 | 82.1 |
  | Fixup | 80.7 | 82.0 | 82.3 | 82.4 |
  | LayerScale, alpha = eps | 80.4 | 81.6 | 81.1 | 81.6 |
  | **LayerScale** | 80.5 | 81.7 | **82.4** | **82.9** |

  († = did not converge.)
- **The per-channel-vs-scalar sentence** (the one worth citing): LayerScale offers more optimization
  diversity than adjusting a whole layer by one learnable scalar as in ReZero/SkipInit, Fixup and
  T-Fixup, and the paper calls per-channel degrees of freedom "a decisive advantage".
- **Two caveats a referee will raise, both verified in the text.** First, the rows are *different
  methods*, differing in warmup and pre-normalization as well as granularity: the authors state that
  removing warmup and layer normalization is what destabilizes Fixup and T-Fixup, and they put both
  back so those methods converge. So this is not a clean granularity-only ablation. Second, the
  baseline's depth-24/36 failure is partly a regularization-tuning artifact — with dropout retuned it
  reaches 81.0 and 81.9. LayerScale still wins, by 1.4 and 1.0 points, but "the baseline diverges
  without LayerScale" is not what the table says.
- **No learning-rate sweep.** LayerScale is a depth-stability result, not an LR-tolerance result.

### 3.2 Admin — Liu et al., "Understanding the Difficulty of Training Transformers", arXiv 2004.08249v3, EMNLP 2020. FULL-TEXT (ar5iv). **The strongest LR-grid precedent, and new to this project**

- **Mechanism**: `b_i = x_{i-1} * omega_i + f_i(x_{i-1})`, then layer norm — and the paper states
  directly that `omega_i` is a **D-dimension vector** with elementwise product, i.e. **a genuine
  per-channel residual scale**, profiled from an initialization run.
- **Motivation is stability**: the diagnosis is an amplification effect by which heavy dependency on
  the residual branch makes training unstable, while a light dependency caps the model's potential.
  The paper rejects unbalanced gradients as the root cause.
- **Learning-rate evidence**: a grid search on IWSLT'14 De-En with a 6-layer, 37M model, whose axes are
  (x) Adam's `beta2` and (y) the learning rate, 15 settings. **Pre-LN converges in all 15; Post-LN
  diverges in 7 of 15**; when Post-LN converges it beats Pre-LN in 7 of 8. Admin stabilizes Post-LN
  and outperforms Pre-LN across this space. Separately: extending Post-LN's warmup from 8k to 16k, 24k
  and 32k updates still fails at 18 layers — i.e. warmup alone is not the fix.
- **Depths**: 6L-6L, 12L-12L, 18L-18L; deepest 60-layer encoder with 12-layer decoder (WMT En-Fr).
- **The caveat that matters**: Admin **never ablates the per-channel `omega` against a scalar**. It is
  the best available example of "per-channel residual scaling + stability + an LR grid", but the LR
  grid compares architectures (Post-LN / Pre-LN / Admin), not granularities.

### 3.3 Per-layer scalar baselines: ReZero and DeepNet

- **ReZero** (arXiv 2003.04887, 2020): one zero-initialized scalar per residual connection, motivated
  by initial dynamical isometry. Trains 120-layer transformers; 56% faster convergence than baseline at
  12 layers on enwiki8. **Shows a per-layer scalar is already enough for depth**, which is why
  LayerScale's margin over it is the interesting number, not the fact that it works.
- **DeepNet / DeepNorm** (arXiv 2203.00555, 2022): a normalization function plus theoretically derived
  initialization, with bounded model updates; scales transformers to **1,000 layers**; the 200-layer
  3.2B model beats a 48-layer 12B model by 5 BLEU on 7,482 translation directions. The scaling
  constants are **per-layer, not per-channel**. DepthBench measures DeepNorm's optimal LR at **1e-3**,
  the only architecture in its ten whose optimum sits below Pre-LN's 2e-3.

### 3.4 SiHC — arXiv 2609.33895 (2026-09-27). FULL-TEXT. **A per-channel read/write ablation at matched parameters — new detail**

Note the terminology trap: **SiHC is "Spatially Indexed Hyper-Connections"**, not a channel-group
method. Its streams are spatial cells of a patch. But its interface ablation is exactly our
decomposition, and nothing in the notes carries it.

- **Mechanism**: `H_mix = I_S` (the stream mix is **dropped**, identity carry), and the read/write are
  static, input-independent matrices `H_pre, H_post` in `R^{C x S}`, shared across spatial cells, so
  each channel reads its own combination of states and writes with its own weights:
  `h_in[c] = sum_s H_pre[c,s] h[c]^(s)` and `h'[c]^(s) = h[c]^(s) + H_post[c,s] delta h[c]`.
- **Table 4** (B-size backbone, 16x16 computational patches, direct v-prediction; parameters and GFLOPs
  "remain nearly unchanged across the progression"):

  | design | state region | streams | FID |
  |---|---|---|---|
  | plain DiT | 16x16 | 1 | 139.83 |
  | mHC, copied input | 16x16 | 4 | 25.36 |
  | mHC, spatial input | 8x8 | 4 | 13.25 |
  | + identity carry, **scalar maps** | 8x8 | 4 | 11.50 |
  | + **feature-wise maps (SiHC)** = channel-specific read and write | 8x8 | 4 | **10.10** |
  | + 4x4 subpatches | 4x4 | 16 | 8.07 |

  Feature-wise access costs **147,264 parameters** on a ~131M model; scalar maps use `4LS` interface
  parameters against `4LCS` for feature-wise, with the same leading arithmetic.
- **What this is and is not.** It is the cleanest published isolation of *per-channel read + write* from
  *everything else*, at fixed parameters, with the mix already set to identity — i.e. the
  `h = D` limit of our read/write half, with our mix half deleted. It is **not** stability evidence:
  the paper reports FID and representation diagnostics, trains at a constant LR without warmup, and
  makes **no** claim about spikes, gradient norms or LR tolerance. Domain is pixel-space diffusion, not
  LM pretraining.

### 3.5 Qwen: elementwise read gating with measured spike counts — see `qwen_stability.md`

Cross-reference rather than duplicate. The two load-bearing facts for this file:

- **Gated Residual, arXiv 2608.30320** (Qwen3.8-Next design report, FULL-TEXT per
  `recheck_2026-10-03.md` §3.1 and `qwen_stability.md`): the read is refined "from one scalar per
  branch to one weight per branch and channel" and that helps, while the same refinement of the write
  "gives almost nothing"; `H_res` adds little once read and write are expressive. The stress test holds
  the LR at multiples of the optimum on a 28-layer MoE and counts spikes (>0.1 above a 201-step rolling
  median): at 4x optimal the AdamW baseline spikes 183 per 10k steps and crosses the clipping threshold
  on 213 of 19,932 steps, while **the gated-residual configuration records zero loss spikes**. A
  GatedNorm on/off toggle at 3x optimal takes spikes from 32.0 to 3.2 per 10k. Their mechanism: without
  an explicit gate the network rescales by growing activation outliers, leaving it fragile.
- **GatedNorm, arXiv 2601.22966** (FULL-TEXT per `recheck_2026-10-03.md` §3.2): the one
  **parameter-matched granularity ablation** in this space — elementwise (score shape `d`) versus
  tensorwise (shape 1) gating, with parity restored by trimming the FFN. Elementwise is consistently
  better; under *tensorwise* gating all non-sigmoid activations diverge. Also the "tolerance vs
  optimum" precedent: DyT + gated attention trains stably at the baseline's 4.3e-3 although its own
  optimum is 2e-3.

**Why this is the strongest support our paper has, and its limit.** It is the only line that pairs
channel-wise coefficients with spike counts above the optimal LR — but GR's zero-spike result is
*gate vs no gate*, and the elementwise-vs-tensorwise comparison in 2601.22966 is scored on loss and
outlier magnitude, not on spike counts at high LR. Nobody has published
"per-group coefficients vs shared coefficients, same parameters, spike counts across a LR grid".
That is the gap our exp10 occupies.

### 3.6 uHC — ICLR 2027 submission `Qnj7Lf8Bz2` (2026-09-19). ABSTRACT only

The closest published item to our read/write half, and the only family paper to attach a stability
claim to group-wise coefficients. Verbatim from the abstract: existing HC share "readout patterns
across feature dimensions"; uHC introduces **Group-Wise Feature Readout** for feature-dependent stream
aggregation plus Low-Rank Directional Write-Back, and "achieves more stable optimization". Reported:
46M-363M, validation loss **2.9470 (mHC) -> 2.8188 (uHC)** at 363M.

**Unverifiable from here.** No measurement, no metric, no scale is given for the stability claim; the
abstract never mentions `H_res`, Sinkhorn or the Birkhoff polytope. Re-checked today:
`openreview.net/pdf?id=Qnj7Lf8Bz2` returns **HTTP 403** and the forum page **HTTP 307** into the same
browser check, which blocks scripted access. **Treat "uHC reports more stable optimization"
as an unverified claim of unknown evidence type** — and note that if a referee reads that abstract, our
"the tolerance sits in the read and write" sentence will be read against it.

### 3.7 Residual scaling and stability, two items new to this project

- **`bj8l2FYjSd`, "On the Residual Scaling of Looped Transformers: Stability and Transferability"**
  (LIT Workshop @ ICLR 2026, 2026-02-09). ABSTRACT, via the OpenReview search API. Asks which residual
  scaling gives stable training and transferable hyperparameters across loop counts `L`; a tied-weight
  residual-MLP analysis says looped models need `1/L`, not the common `1/sqrt(L)`; experiments on
  looped LLMs **across loop counts and learning rates** find `1/L` gives "significantly better
  stability and hyperparameter transfer". Per-layer scalar, not per-channel — but it is a residual-scale
  paper whose evidence is an LR x depth grid, which is the methodology our claim needs, and it is a
  precedent for "a residual-coefficient choice shifts hyperparameter transfer, not just the optimum".
- **`9bLiqb6Vec`, SpanNorm** (ICML 2026 regular, 2026-01-23). ABSTRACT. Normalization placement, not
  coefficients: a Pre-Norm residual path with a Post-Norm-style normalization of the residual sum,
  claimed to keep signal variance bounded and to avoid both Post-LN's gradient issues and Pre-LN's
  representation collapse, in dense and MoE settings. Relevant only as the current statement of the
  Pre/Post-Norm seesaw that HC's own abstract invokes.

### 3.8 Counter-evidence, collected

1. **DepthBench 2609.32534**: with parameters and recipe fixed and a per-architecture LR sweep, mHC's
   optimal LR is **2e-3 — identical to Pre-LN's** at 400M, and "the optimal learning rate is
   consistent across width-depth shapes within each architecture". So the family's flagship stabilizer
   does **not** move the usable LR. (Caveat from the 10-03 note: the grid is coarse, 2.5x steps, and
   the per-LR losses are figure-only, so how gracefully mHC degrades above its optimum is unread.)
2. **Review Residuals 2606.31859** (FULL-TEXT per `recheck_2026-10-03.md` §3.4): a per-channel,
   input-dependent write gate, with baselines **parameter-matched by widening** and 2-3 seeds. Through
   320M the differences are within noise and at 60M a parameter-matched plain residual is slightly
   better; significance appears only at 590M (t = 3.3 / 3.7) and 1B. Their convex Highway variant
   reintroduces vanishing gradients and stalls beyond ~20 layers. **The closest published precedent
   for our own null result at matched parameters.**
3. **MHAR's attribution flip** (`recheck_2026-10-04.md` §2.2): the ICLR 2027 abstract (2026-09-08,
   newer than arXiv v2) attributes most of the multi-head gain to the flatter per-head softmax the
   reshape provides, with only a small residual consistent with per-subspace routing. mHC's operator is
   sigmoid-gated and Sinkhorn-normalized with no softmax over a growing width, so the mechanism does
   not transfer.
4. **Architecture-independent spike accounts** (`recheck_2026-10-04.md` §2.3.3): LM-head logit growth
   under finite precision (`ptNmbmCdj3`) and AdamW first/second-moment time-scale mismatch
   (`pAjrfibDB9`, spike episodes 17 -> 0 at GPT-2 774M). Plus Qwen's gate-as-rescaling account. Our
   spike-damping claim competes with three mechanisms that have nothing to do with connection
   granularity, and `stability_proxies.md` §4.2 should govern how we phrase it.
5. **sHC 2603.20896v2**: all the constrained variants suppress HC's gradient spikes, but mHC, mHC-lite
   and KromHC do so "at the cost of collapsing into the identity mapping" — stability bought by
   degeneration, which is the same object as DepthBench's effective-rank collapse and oHC's /
   osHC's / GDHC's mean-preservation arguments.
6. **`wdlctc/hyper-connection-factory`**: the one divergence in that controlled 11-variant comparison
   is **MUDDFormer** at 24 layers (grad norms to 7e4), fixed by a **normalization** change
   (PrePostDANorm), not by connection granularity. mHC and HC did not diverge. Also from that repo,
   against small-scale extrapolation generally: mHC was worse than pre-norm in their laptop runs and
   static HC beat dynamic HC there; neither held on GPU.

---

## 4. Concurrency re-check (since 2026-10-04)

### 4.1 Queries run today, all at the source

- **arXiv API** (`export.arxiv.org/api/query`): `all:"hyper-connections"` (40 hits),
  `all:"hyper-connection"` (40), `abs:mHC` (40) — newest family item is still **2609.33895
  (2026-09-27)**, unchanged from the 10-03 and 10-04 checks. Exact-phrase queries
  `all:"multi-head hyper-connections"`, `all:"grouped hyper-connections"`,
  `all:"group-wise hyper-connections"`, `all:"channel-wise hyper-connections"`,
  `all:"per-head residual mixing"`, `all:"multi-head mHC"` — **0 hits each**.
  `all:"DeepSeek-V4.1"` and `ti:"DeepSeek"` (to find the V4.1 report).
- **arXiv index coverage**: `cat:cs.CL AND submittedDate:[202610020000 TO 202610060000]` returns 90
  entries, **all dated 2026-10-02**; the 09-30..10-02 window returns 295. So the index reaches
  **2026-10-02** and **2026-10-03..10-05 remains unobservable from here**, as it was on 10-04 for
  10-02..10-04. Also scanned 400 cs.LG/cs.CL titles from 09-28..10-03 for residual/stream/skip
  keywords: 12 hits, all irrelevant (the only architectural one is **2610.02907 "Do ResNets Route?"**,
  a Mobius-inversion interaction analysis of ImageNet ResNets — no HC, no per-group coefficients).
- **OpenReview search API** (`api2.openreview.net/notes/search`, 7 terms, 226 distinct notes
  collected): the HC cohort is unchanged. Latest `cdate` among HC-family submissions is
  **2026-09-19** (the ICLR 2027 deadline) and the latest `mdate` is **2026-10-03**; **no HC-family
  note created after 10-04**. Queried `grouped hyper-connections`, `multi-head hyper-connections`,
  `channel-group residual mixing language model`, `per-head residual stream mixing` — the top results
  are the same ~25 known items plus unrelated time-series/vision work.
- **GitHub** (`gh api search/code`, `search/repositories`): identifiers `num_hc_heads`, `n_hc_heads`,
  `num_mhc_groups`, `per_group_sinkhorn`, `group_sinkhorn` return **only this project's own repo**
  (`marcoshernanz/multi-head-mhc`) plus noise. `hc_heads` / `hc_groups` / `mhc_heads` return
  unrelated projects (hypercube LSH, Hoshino bots, multi-head causal decoding).
  Repo search `hyper-connections pushed:>2026-09-25`: `wdlctc/hyper-connection-factory` (10-04),
  `aHapBean/xHC` (09-30), `6zHAOyi/s2HC` (09-28), plus Windows-Server spam — all three already in the
  notes. `lucidrains/hyper-connections` newest commit is still **2026-05-13**.
- **Web**: two web searches on grouped/multi-head HC and on per-channel residual scaling
  returned only known arXiv items, the factory repo, and secondary explainers.

### 4.2 Verdict

**Design A (per-channel-group `H_pre` + `H_post` + per-group `n x n` doubly stochastic `H_res`) is
still unpublished as of 2026-10-05**, on everything observable from here. The per-group *read* remains
claimed by uHC (abstract); the per-group *mix* remains unclaimed. Nothing new was proposed between
10-04 and 10-05, and the 10-03..10-05 arXiv slice is not yet indexed, so the next check should re-run
the family listing against that window.

### 4.3 New *to this project* (three items, all verified)

1. **DeepSeek-V4.1-Flash report, arXiv 2609.19969 §2.4.1 Single-Pass mHC** — see §1.3. Corrects
   `web_and_code_search.md` §2's claim that mHC is not re-described there. 552B backbone, 45T tokens,
   Mega-mHC kernel, `(3n+2)d` -> `(2n+2)d` activation traffic, no stability claim.
2. **`bj8l2FYjSd`** looped-transformer residual scaling (§3.7) — residual scaling with an LR x loop-count
   grid.
3. **`9bLiqb6Vec`** SpanNorm (§3.7).

Also newly read at source rather than newly discovered: **BE-HC** (`jpIjkN1B1Q`, §2.1 — present in the
notes only as a parameterization name), **SiHC's Table 4** (§3.4), **HC's own spike claims** (§1.4),
**Admin's per-channel `omega` and LR grid** (§3.2), **LayerScale's Table 1** (§3.1).

---

## 5. Assessment: what our paper can cite for "per-group coefficients give LR tolerance / stability"

### 5.1 Defensible as written

- **"Stability is the stated motivation for mHC's manifold constraint."** Cite 2512.24880 abstract and
  §3.1. Solid, verbatim.
- **"mHC's stability evidence is a 27B single-LR comparison of loss and gradient-norm curves plus a
  forward/backward gain diagnostic, with no LR sweep."** Cite Figs. 2, 3, 5 and §5.2. Solid, and worth
  stating because it sets the bar our measurement clears.
- **"No HC-family paper has measured LR robustness."** Solid against the 20+ items in §2.1. Phrase as
  "we are not aware of", since OpenReview full texts are unread.
- **"Within the family, stability claims point in both directions."** HC: no spikes at 1B/7B where the
  baseline spikes. mHC: HC surges at 27B near step 12k. sHC: the constraint stabilizes by collapsing
  the mixer. DepthBench: mHC's optimal LR equals Pre-LN's. All verified.
- **"Per-channel residual scaling is long-established and motivated by stability."** LayerScale
  (per-channel diagonal, depth 24/36) and Admin (D-dimensional `omega`, 15-setting LR x beta2 grid,
  Post-LN diverges in 7) are the two to cite, with ReZero and DeepNet as the per-layer-scalar
  reference points.
- **"Channel-wise read gating has been shown to eliminate loss spikes above the optimal LR at scale."**
  Cite 2608.30320's stress test (zero spikes at 4x optimal; baseline 183 per 10k) and 2601.22966's
  parameter-matched elementwise-vs-tensorwise ablation. Use `qwen_stability.md` §6.2 for the caveats —
  this is a gate-vs-no-gate result on a 25B-A3B MoE, not a granularity result on connection
  coefficients.
- **"The read and write are where granularity has paid off; the mix is the component the field is
  dropping."** Cite Qwen GR ("read granularity matters more than write granularity"; `H_res` adds
  little), SimpleHC (gating without mixing matches mHC at 15B-30B), osHC (one marginal suffices),
  SiHC (mix = identity, per-channel read/write worth 11.50 -> 10.10 FID at fixed parameters), uHC
  (group-wise readout). This is the single best-supported sentence in our motivation.

### 5.2 Must be qualified or dropped

- **Do not write "per-group coefficients are known to improve stability."** No published experiment
  varies connection-coefficient granularity and measures stability. The nearest claims are uHC's
  unverified abstract sentence and Qwen's gate-vs-no-gate stress test. The correct claim is that
  per-channel residual *scaling* and per-channel *output gating* have stability precedents, and that
  extending that to per-group *connection* coefficients is what we test.
- **Do not lean on LayerScale as "per-channel beats scalar for stability."** Its table varies
  normalization and warmup alongside granularity, and its baseline converges once dropout is retuned.
  The citable form is the authors' own sentence about per-channel degrees of freedom being decisive,
  plus the 82.4 / 82.9 vs ReZero's 80.8 / 81.6 margin at depth 24 / 36.
- **Do not cite Admin as a granularity result.** It never compares `omega` as a vector against a
  scalar. Cite it for "a per-channel residual scale with an explicit LR grid, where the unstable
  baseline diverges in 7 of 15 LR x beta2 settings at 37M".
- **Do not cite SiHC as stability evidence.** It is an FID ablation on a diffusion transformer with no
  LR or spike measurement. Cite it as the parameter-matched isolation of per-channel read/write with
  the mix set to identity.
- **Do not cite uHC's "more stable optimization" without marking it unverified.** Abstract only;
  the OpenReview PDF returns 403.
- **Keep the three competing spike mechanisms in the text.** LM-head logit growth, AdamW moment
  mismatch, gate-as-rescaling. Per `stability_proxies.md` §4.2, a spike-damping result at d384/d768
  should rule these out or explicitly decline to.
- **State the DepthBench counter-result in our own stability section, not in a footnote.** "mHC does
  not shift the optimal LR at 400M under a per-architecture sweep" is the most direct published
  challenge to any LR claim in this family, including ours.

### 5.3 The honest one-paragraph framing

Our result is the first measurement of connection-coefficient *granularity* against the learning rate,
in a family whose stability claims have so far been single-LR gradient-norm curves (mHC-lite, KromHC,
TBP-mHC, sHC) or spectral bounds (osHC, oHC, BE-HC), and whose flagship stability evidence is three
figures from one 27B run. The precedent that per-channel coefficients can matter for optimization is
real but lives elsewhere — LayerScale and Admin on the residual scale, Qwen's elementwise gate on the
sublayer output — and it is in every case *not* a granularity-versus-granularity stability ablation.
The direction of our finding (read and write, not mix) matches where the field is independently
moving, and the magnitude (no gain at matched parameters and tuned LR) matches the one published
parameter-matched per-channel-gate study at our scale.

---

## 6. Not verified / limits of this file

- **All ICLR 2027 full texts remain unread.** Re-confirmed today: `openreview.net/pdf?id=Qnj7Lf8Bz2`
  -> **403**, `openreview.net/forum?id=Qnj7Lf8Bz2` -> **307** into a browser check.
  uHC's stability claim and SimpleHC's "under stress" test are therefore both unresolved; these are
  the two highest-value unread documents for this file, and both forum pages need to be read in a
  browser.
- **ICML 2026 review threads for mHC (`mDhyxu8WRb`) and KromHC (`TI7Q2o6EIa`)**: same wall, still
  unread. Open since 2026-09-26.
- **arXiv 2026-10-03..10-05** is not indexed yet (§4.1), so "nothing new" covers everything arXiv has
  indexed, not everything submitted.
- **Figure-only numbers I did not estimate**: mHC Figs. 2/3/5 curve values beyond the stated peaks,
  DepthBench Fig. 12 per-LR losses, mHC-lite Figs. 2/3, TBP-mHC Fig. 1, sHC Figs. 6/11, HC Fig. 6,
  KromHC App. I.
- **BE-HC (`jpIjkN1B1Q`)**: abstract only. Its 1000-layer result is reported at 35.71% accuracy on an
  unnamed task and 22.56% validation accuracy at 8K context on one V100 — toy-scale numbers whose task
  I could not identify from the abstract. Treat the "where ReZero fails" comparison as unverified.
- **mHC-lite / TBP-mHC / sHC / KromHC peak learning-rate values** are rendered as MathML in the arXiv
  HTML and were stripped by my text extraction, so I report "single LR per size" from the table
  structure (one LR cell per configuration) rather than the numbers themselves.
- **SimpleHC, osHC, GDHC, ITO-HC, MV-HC, DAHC, cmHC, Structured mHC, TaskHC, DSR, RadFree, EER,
  Internal Attention, MSAR, Delta AttnRes, ResLorB** — all still ABSTRACT level, as recorded in
  `recheck_2026-10-04.md` §2. Nothing in this file upgrades them.
- **X/Twitter**: no API access; web search is the only proxy.

---

## 7. Sources checked today (2026-10-05)

**Read at source, full text or body sections:** 2512.24880v2 (mHC, re-verified), 2609.19969
(DeepSeek-V4.1-Flash, §2.4.1 / §4.2.1), 2409.19606v3 (HC), 2601.05732v1 (mHC-lite), 2603.20896v2
(sHC), 2605.21724v1 (TBP-mHC), 2601.21579v2 (KromHC), 2609.02672v1 (oHC), 2607.14530v1 (xHC),
2606.03483v1 (stream collapse), 2602.18308v2 (JPmHC), 2609.05309v1 (mHC stream usage), 2609.33895v1
(SiHC, Table 4 and the interface definition), 2103.17239 (LayerScale/CaiT, abstract + LayerScale
section + Table 1, via ar5iv), 2004.08249v3 (Admin, via ar5iv), 2601.08131 (ExoFormer, stability
grep), `wdlctc/hyper-connection-factory` README at commit 2026-10-04.

**Abstracts verified at source:** 2003.04887 (ReZero), 2203.00555 (DeepNet), 2309.14322 (Wortsman,
abstract re-verified; body already in `stability_proxies.md`), 2511.11238v2 (VWN), 2610.02907,
and via the OpenReview search API: `jpIjkN1B1Q` (BE-HC), `i2WyUVJUJ2` (SimpleHC), `1BC0eYN2uS` (osHC),
`nKRfYmPtP2` (ITO-HC), `Qnj7Lf8Bz2` (uHC), `ZaJDPN5xJY` (MV-HC), `dzXT0zmzQn` (SHC-PPO),
`bj8l2FYjSd` (looped residual scaling), `9bLiqb6Vec` (SpanNorm), `sy983SmQYr` (hyperbolic residual
connections, screened out: hyperbolic geometry, not coefficient granularity), `GeD5rC29iZ` / `9FqARW7dwB`
(HC).

**APIs / endpoints used:** `export.arxiv.org/api/query` (11 queries), `api2.openreview.net/notes/search`
(10 queries), `api.github.com` via `gh` (8 code searches, 1 repo search, 4 commit listings),
`arxiv.org/html/<id>`, `ar5iv.labs.arxiv.org/html/<id>`, 2 web searches.

**Blocked:** `openreview.net/pdf?id=...` (403), `openreview.net/forum?id=...` (307 -> browser check).
