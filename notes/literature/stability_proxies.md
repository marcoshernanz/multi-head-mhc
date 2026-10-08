# Stability proxies: can "tolerates 2-3x the optimal LR at 27M-112M" support "may reduce
# instabilities at scale"?

Compiled 2026-10-05.

**Short verdict (full argument in §4).** The *protocol* is legitimate and has a strong, very recent
industrial precedent that does almost exactly this and cites Wortsman et al. for it. The *inference
as currently phrased* is not supportable, for four independent reasons: (i) Wortsman et al.
explicitly exclude loss spikes from their scope; (ii) they explicitly say reduced LR sensitivity does
not matter if the optimal LR does not move, which is exactly this paper's situation; (iii) everyone
who has actually validated an architectural stability intervention ran the proxy at 0.8B-25B, two to
three orders of magnitude above 27M-112M, and the one paper with spike data at 546B found the
dominant mechanism **absent** at 7B and 30B; (iv) the quantity this paper measured (how far loss
falls *past* the cliff) is not the quantity with a known scale dependence (where the cliff *is*).
Recommended reframing and the cheap experiments that would fix it: §4.3 and §5.

### Companion files

Two sibling notes were written earlier today by a parallel pass and name this file as their
companion; read all three together and keep them from drifting:

- `hc_stability.md` — how mHC and the HC family actually *show* stability, and whether per-channel /
  per-group coefficients are known to buy stability or LR tolerance.
- `qwen_stability.md` — Qwen's stress-test numbers (Gated Residual, GatedNorm, Gated Attention) with
  scales, read at the source.

This file deliberately restates the Qwen3.8-Next protocol and metric in §2.1 and §2.7 so that it
stands alone; `qwen_stability.md` is the fuller treatment and should be treated as authoritative if
the two ever disagree.

### Provenance of each fact

Read in full by me at the primary source: Wortsman 2309.14322 (ar5iv HTML), Zhai 2303.06296 (ar5iv),
Lourie 2608.11859 (arXiv HTML), Step Law 2503.04715 (arXiv HTML), Qwen3.8-Next 2608.30320 §3.3
(arXiv HTML). Read via a single-page fetch: ExoFormer 2601.08131, SPAM 2501.06842, Wang 2512.24503.
Everything in §2 comes from a first pass over the primary sources that flagged what it could not
verify: **[FIRST-PASS]** (the primary source was read in that pass) or **[UNVERIFIED]** (it could
not be, or only the abstract was read). The **[FIRST-PASS]** items were not re-read, so treat exact
numerals in §2 as one-source-removed until used in a submission.

**[UNVERIFIED]** The ICLR 2024 reviews of 2309.14322 could not be read: `openreview.net/forum?id=...`
and both `api.openreview.net` and `api2.openreview.net` return "Challenge verification required"
to scripted requests. So I cannot report what the actual reviewers objected to.

---

## 1. Wortsman et al., "Small-scale proxies for large-scale Transformer training instabilities"

- arXiv **2309.14322**, v1 25 Sep 2023, v2 16 Oct 2023.
- **ICLR 2024 (oral)**; OpenReview forum id `d8w0pmvXbZ`; `iclr.cc/virtual/2024/oral/19743`;
  proceedings PDF `proceedings.iclr.cc/paper_files/paper/2024/file/d848cb2c84f0bba7f1f73cf232734c40-Paper-Conference.pdf`.
  Note the venue does **not** appear on the arXiv abs page.
- 17 authors, Google DeepMind. Senior authors **Jaehoon Lee and Justin Gilmer**; Gilmer is the
  "Gilmer et al. 2023" that ViT-22B credits for qk-layernorm, so this paper and the qk-norm lineage
  are the same group.

### 1.1 LR sensitivity — exact definition

With `θ = A(η)` the weights from training at max LR `η`, `ℓ(θ)` the validation loss, for a range
`[a,b]`:

```
ℓ*  = min_{η ∈ [a,b]} ℓ(A(η))        # best loss in the range
ℓ_0 = loss at initialisation
LR-sensitivity = E_{η ∈ [a,b]} [ min( ℓ(A(η)), ℓ_0 ) − ℓ* ]
```

- Default range **[3e-4, 3e-1]** (three decades), AdamW.
- `η` is the **peak of a cosine-decay schedule with warm-up** (min LR 1e-5).
- Expectation and minimum are over the **7-point grid {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1}** —
  a uniform average over log-spaced points, not an integral.
- The `min(·, ℓ_0)` clip stops a diverged run from dominating: a run worse than init contributes
  exactly `ℓ_0 − ℓ*`.
- Larger = worse. A model that fails at high LR scores high.

### 1.2 Default setup (needed to replicate the metric comparably)

AdamW β=(0.9, 0.95), ε=1e-8, **gradient clipping at global norm 1**; warm-up 5e3 linear then cosine;
**1e5 steps, batch 256 x 512 tokens ≈ 13B tokens**; independent weight decay 1e-4; **z-loss on by
default (1e-4)**; **qk-layernorm on by default** (per-head with shared params; they ablate whole-dim
and find per-head better); pre-LN, no biases, RoPE, no weight tying, LN ε 1e-6; MLP hidden 4d, gelu;
1/sqrt(d) attention scaling; C4; bf16 on TPU; codebase **NanoDO** (Flax/JAX). Parameter counts
**exclude embeddings and head**. Scaling is **joint** over embedding size, depth and heads.

Sizes named in the text: **2.4M, 10M, 1.2B, 4.8B**; the sweep spans ~three decades.
**[UNVERIFIED]** the exact grid of model sizes — it appears only in figure axes.

### 1.3 The two instabilities, mechanisms, mitigations

**Scope statement (footnote 1 on page 1) — read this before citing them for spikes:** they study
instabilities that cause **slow divergence, not loss spikes**. Fast spikes appear only as a related-work
survey (§4, and §1.7 below).

**(a) Attention logit growth.** `z_ij = <q_i, k_j>/sqrt(d_h)`. q and k grow in norm, `z` blows up,
softmax collapses to one-hot ("attention entropy collapse", Zhai et al.). First hit at 22B in
ViT-22B. Mitigation **qk-layernorm** (LayerNorm on q and k before the dot product), equally effective
at small scale.
- Appendix E.1: growth is in the **norms** of q and k, not their **alignment**. Hypothesis
  (Appendix C): attention logits are the only feature whose magnitude is **quadratic** in parameter
  RMS (`<XW_1, XW_2>`), so parameter-norm growth hits them first.
- **Without qk-layernorm, the LR at which models diverge falls as model size rises.** This is the
  single sentence that makes the small→large proxy coherent at all.
- With qk-layernorm they train **1.2B at LR 0.3**.
- **Both with and without qk-layernorm, LR sensitivity increases with scale.**
- Not the softmax's fault: it still occurs with a pointwise attention variant (Fig. E.11).

**(b) Output logit divergence.** `p_i = e^{y_i}/Z`, `Z = Σ_j e^{y_j}`; logits diverge very negative
**late in training** (shown at 2.4M, LR 0.1). From PaLM. Mitigation **z-loss**: auxiliary `log^2 Z`
with coefficient **1e-4**. Occurs with **no weight decay regardless of scale**; z-loss resolves it;
weight decay also mitigates it for their larger models.

### 1.4 What they actually show about small→large transfer

Three claims of increasing strength — the gap between them is where the paper is usually over-read.

1. **Reproduction.** Two instabilities first seen at 22B (ViT) and 540B (PaLM) appear in small models
   at high LR and the same fixes work. Direction of inference: **large → small**. They already knew
   the instability and the fix; they found a cheap testbed for it.
2. **Extrapolating a model characteristic — their only validated forward prediction (§3.3).** Track
   max attention logit (block 0, step 2e3), fit a **quadratic in parameter count per LR**, note that
   **every run above max logit 1e4 diverged**, predict a **4.8B** model at **LR 1e-2** without
   qk-layernorm will cross it, train it, it diverges — and the fitted logit value extrapolates
   closely too. Separately they transplant a fixed max logit `κ` into a 10M model via
   `g(z) = sqrt(κ)·z/sqrt(E_i[z_i^2])`: loss degrades around `κ=1e3`, and at `κ=1e4` is worse than a
   zero-layer bigram model.
3. **Finding a *new* issue by extrapolating characteristics (§3.4).** `RMS(g) = sqrt(E_i[g_i^2])`
   **decreases with both model size and LR**, approaching the default AdamW ε=1e-8, at which point
   `Δ = v/(sqrt(u)+ε)` collapses. Fix: **ε 1e-8 → 1e-15** improves loss at 4.8B/LR 0.3;
   **ε → 1e-6 diverges**. Mechanism (Appendix C): pre-LN output RMS grows with LR and depth, the
   final LayerNorm scales gradients by 1/input-RMS, so the gradient reaching the trunk shrinks.
   Note this is a hyperparameter whose correct direction is **invisible below ~1B** — and it points
   the *opposite* way from OLMo 2's 1e-5 → 1e-8 change (§2.3), which was tuned in another regime.

**What they never do:** compare two competing *architectures* at matched parameters on LR sensitivity
and conclude the lower-sensitivity one will be more stable at scale. Their interventions are a known
fix (qk-layernorm, z-loss), an optimiser/schedule choice (warm-up, weight decay, ε), a
parameterisation (μParam), or a shape choice (width vs depth).

**Does the best loss change?** Loss and sensitivity are reported separately and must be read together
(Appendix B). Longer warm-up reduces sensitivity **and** loss; independent weight decay reduces
sensitivity; **depth scaling increases sensitivity but gives the lowest loss at their largest
scale**; μParam changes neither. So lower LR sensitivity does **not** imply better loss, and in their
own paper the more LR-sensitive option was the better-performing one.

### 1.5 Every other intervention and its effect on LR sensitivity

| Intervention | Effect on LR sensitivity | Notes |
|---|---|---|
| qk-layernorm | **Reduces** strongly; raises max trainable LR | still rises with scale |
| z-loss (1e-4) | **Reduces**; fixes output logit divergence | |
| Longer warm-up | **Reduces**; also lowers loss | clearest for larger models; total steps fixed at 1e5 |
| Independent weight decay `θ ← θ − s_t(ηΔ − λθ)` | **Reduces** vs the PyTorch/Optax default `θ ← θ − s_tη(Δ − λθ)` | λ=1e-4 independent, λ=0.1 coupled; for the coupled case they report the **min** sensitivity over [1e-4,1e-1] and [3e-4,3e-1] |
| Weight decay magnitude | increasing it slightly **shifts the optimal LR right** | Fig. E.10 |
| Scaling **depth** vs width | depth **increases** sensitivity faster than width | but depth gives the lowest loss at the largest scale; joint scaling best, and extrapolates most reliably from <1e8 params |
| μParam (simple and full) | **stabilises the optimal LR**; does **not** improve loss or reduce sensitivity; does **not** remove the need for qk-layernorm | |
| Total steps 5e4 / 1e5 / 2e5 | **no meaningful change** | Fig. E.7 |
| Batch size 256 / 512 / 1024 | **no meaningful change** | Fig. E.9 |
| qk-layernorm whole-dim vs per-head | per-head better | Fig. E.8 |
| AdamW ε 1e-8 → 1e-15 | improves loss at 4.8B/LR 0.3; ε=1e-6 diverges | §3.4 |
| LR scaled by parameter RMS | **sensitivity not reported** — changes the meaning of LR | Fig. E.14 |

### 1.6 Their stated limitations (Appendix B)

1. **Interventions that change the meaning of the LR** invalidate the comparison (e.g. passing
   sqrt(LR) to the optimiser). Their validity test is empirical: for qk-layernorm, z-loss and warm-up
   "the LR vs. loss curves are indistinguishable up to some critical learning rate", so the meaning
   of the LR is unchanged. For μParam they accept it because the per-layer rescaling is linear and
   constant along the curve. For LR-scaled-by-parameter-RMS they refuse to report it.
2. **The definition ignores the optimal LR shifting.** They recommend sliding the three-decade window
   with the optimum, and note their own main experiments were not large enough to need the shift.
3. **It is invariant to the scale of the loss**: a model at chance across all LRs scores zero. No fix
   offered; they instruct that it always be read with the LR-vs-loss curves, and call it a summary of
   those curves rather than something to optimise on its own.

### 1.7 What they say about *loss spikes* (§4)

They place fast spikes in the **adaptive edge of stability** frame, i.e. as an optimiser/curvature
phenomenon, not an architectural one:

- Classical bound: instability above `LR = 2/λ_max(H)`; edge-of-stability work complicates this with
  progressive sharpening vs self-stabilisation (Cohen et al.; Damian et al.; Gilmer et al. 2021 tie
  warm-up's benefit to self-stabilisation shrinking `λ_max(H)`).
- For adaptive optimisers, preconditioned sharpness `λ_max(P^{-1}H)` oscillates around an
  optimiser-specific threshold — **38/LR for Adam with β1 = 0.9** (Cohen et al. 2022).
- **Large β2 → `H` shrinks and fast spikes appear; small β2 → `P^{-1}` shrinks instead and no spikes
  are observed.** They use this to explain the folklore that lowering β2, or the "stale optimizer
  state" account of Shazeer and Stern, removes spikes, and the periodic spikes of large runs.

So the paper's own account of fast spikes makes **β2 and the LR** the governing knobs — not
architecture.

---

## 2. Follow-up and adjacent work

### 2.1 ★ The one paper that runs almost exactly this argument — and the template to copy

**"On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability"**,
Qwen Team (first author Zihan Qiu), arXiv **2608.30320**, 31 Aug 2026. §3.3 read in full by me at
`arxiv.org/html/2608.30320`; every number and the metric below is verified verbatim there. No venue
on the arXiv page.

**Protocol.** Citing Wortsman et al. by name for the premise that large-scale instabilities can be
reproduced in small models by raising the LR, they add a **second amplification: hold the LR
*constant* at a multiple of its optimum, bypassing the decay schedule**, to imitate the prolonged
peak LR of a production run. Applied at **2x and 4x** optimal (and 3x for a single-variable gate
toggle) to a **28-layer 25B-total / 3B-activated MoE**, ~19,932 steps, shared batch size and
**gradient-norm clip 0.5**.

**Metric — three quantities, and this is the cleanest published operationalisation:**
1. **loss spikes = steps whose loss exceeds a 201-step rolling median by more than 0.1**, reported as
   a **rate per 10k steps**;
2. **p99.9 of the pre-clip gradient norm**, plus the **count of clip-threshold crossings**;
3. **per-block maximum activation**.

**Results:**

| Config | Spikes / 10k steps | Clip crossings |
|---|---|---|
| 2x optimal LR, Qwen3.5 structure + AdamW | 4.3 | — |
| 2x optimal LR, both Muon configs | 0.2 | — |
| 4x optimal LR, Qwen3.5 + AdamW | **183** | **213 of 19,932** |
| 4x optimal LR, Muon (both) | — | **never crosses** |
| 4x optimal LR, Muon + Gated Residual | **0** | 0 |
| 3x optimal LR, gate **off** (AdamW, structure and data order fixed) | 32.0 | 256 |
| 3x optimal LR, gate **on** | 3.2 | 20 |

**Mechanism — the most transferable idea here.** An LR ladder on the ungated baseline shows
**activation outliers grow roughly proportionally with the LR while the spike rate grows much
faster**. Their reading: high-LR training needs a rescaling mechanism; without an explicit gate the
network supplies it by growing activation outliers, which leaves it fragile, whereas a multiplicative
gate supplies the rescaling directly. Also: at 2x optimal, the Muon runs had a *higher* median
gradient norm and *larger* max activations than AdamW yet far fewer loss spikes — i.e. raw gradient
magnitude and spike rate dissociate.

**Two framing details to copy.** (a) The acceptance criterion is **relative and anchored**: the new
recipe must be at least as stable as the previous Qwen3.5 structure with AdamW, which has already
been scaled successfully. They never claim an absolute spike-rate prediction. (b) They then
**confirmed against the production run** (125B-total / 6B-activated), which trained with no loss
spike and no anomalous gradient-norm fluctuation, and without qk-clip or SwiGLU-clip.

**Why this both helps and hurts this paper.** It is proof that "degrades less at 2-4x the optimal LR,
measured by spike rate" is accepted practice, published, and in this instance predictive. It is
equally proof that the people who rely on it (i) run the proxy at **25B-A3B**, two to three orders of
magnitude above 27M-112M, (ii) add the constant-LR amplification because raising the LR alone was not
treated as sufficient, (iii) anchor to a recipe already proven at the target scale, and (iv) confirm
against production before claiming the benefit. Their own motivation for §3.3 says that at trillions
of parameters and tens of trillions of tokens the model enters a regime where stability challenges
are entirely absent in smaller-scale experiments.

### 2.2 The strongest counterexample: a 546B mechanism absent at 7B and 30B

**Molybog et al., "A Theory on Adam Instability in Large-Scale Machine Learning"**, arXiv
**2304.09871**, 19 Apr 2023 (v2 25 Apr 2023), Meta AI / OPT team. No venue. **[FIRST-PASS]**

Mechanism, in terms of `r[i,t] = m[i,t]/sqrt(v[i,t])` (the Adam update with ε=0): gradients of a
group of parameters (typically **early layers**) decay far below ε; `m` and `v` follow; the spatial
distribution of the actual update `u` collapses to a spike at 0; the gradients become
**time-correlated** (because the parameters have stopped moving, and because a **large batch** gives
small time-domain gradient variance); `r` flips from unimodal to **bimodal**; a rare batch then pushes
the gradient back above ε and `u` snaps onto `r`'s bimodal shape, which is the divergence. Note
`d/dx [x/(|x|+ε)]` at 0 is `1/ε`, so tiny gradient changes produce disproportionate update changes.
Analytically, with exactly correlated gradients and ε=0, `u_t = (β1/sqrt(β2))·sign(∇f)`, i.e. modes
at **±0.92** for β=(0.9,0.95) — and this does not depend on the gradient scale. The descent condition
yields an LR requirement scaling as **1/n** in the parameter count.

**Scales: 7B, 30B, 65B, 546B** (batch 2048 at 7B vs **65536** at 546B; ε=1e-8, β=(0.9,0.95)
throughout). The decisive observations: 65B showed moderate instability vs 546B's, while **30B and 7B
showed no explosive divergence at all**, and **no bimodal-update layers could be detected in 7B or
30B**. Their own framing notes that reproducing spikes at smaller scale is not necessarily caused by
the same factors. Proposed mitigations are correspondingly structural: lower LR, lower ε, set ε=0 and
map `u→0` when `v=0`, **reduce batch size**, lower β1/β2, and monitor unimodality of `r` via
**Hartigan's dip statistic**.

**Verdict: undercuts decisively.** The dominant spike mechanism at 546B is invisible four orders of
magnitude below it, and its drivers are depth, batch size, and a fixed ε against shrinking gradients
— none exercised by a 27M-112M sweep.

### 2.3 OLMo 2 — the spike score, and an inconvenient LR result

**"2 OLMo 2 Furious"**, Team OLMo et al. (Ai2), arXiv **2501.00656**, v1 31 Dec 2024, v3 8 Oct 2025;
shorter version at **COLM 2025**. §3. **[FIRST-PASS]**

**Spike score (exact):** the percentage of values in a time series that are at least **seven standard
deviations** from a **rolling average of the last 1,000 values**. Applied primarily to **training loss
and the L2 norm of the gradient**. Credits Karpathy (2024) for the related idea of skipping an update
on a 7σ loss/grad-norm outlier. **[UNVERIFIED]** the window used for the **standard deviation** is
never specified (only the rolling *average* window is), and "percentage" is never pinned down with a
worked example, though the reported values (0.40, 0.03, 0.108, 0.069, 0.16, 0.092) are consistent
with percent-of-steps.

Reported deltas (gradient-L2 unless noted):

| Intervention | Spike score before → after |
|---|---|
| Init: scaled init → N(0, 0.02²) | 0.40 → 0.03 (also no loss spikes) |
| Reordered norm + QK-norm (together) | 0.108 → 0.069 |
| Weight decay on embeddings → off | 0.16 → 0.092 |

Full intervention set: repeated-n-gram document filter (drop documents with **≥32 repeated n-grams**,
n = any span of 1-13 tokens) plus loss-masking of such spans; `N(0, 0.02²)` init; RMSNorm in place of
non-parametric LayerNorm (ablations showed **no difference** — switched for safety); reordered norm
`h := x + RMSNorm(Attn(x))`, `h_out := h + RMSNorm(MLP(h))`; QK-norm; z-loss; **AdamW ε 1e-5 → 1e-8**;
weight decay excluded from embeddings. They also define a **growth exponent**
`λ = (1/n_layers)·log(‖v'‖/‖v‖)` on activations and gradients between first and last layer over 50
Pile documents — an *initialisation* diagnostic, no training.

**[UNVERIFIED] and important: the paper never states the parameter count of the stability-ablation
runs.** §3.2 says only that they built a baseline that reproduces spikes quickly, mainly by reducing
warm-up, and that the effect persists across model scales and token counts. The 7B/13B/32B figures
are the *production* runs. Do not cite a scale for those ablation figures.

**Cites Wortsman explicitly** (twice for z-loss; substantively in §4.1, saying it expands on
Wortsman's observation that small models' performance is largely LR-invariant and that QK-norm and
z-loss enhance that, and that the results hold at much larger token and parameter scales).

**The inconvenient part (§4.1).** Baseline peak LR 3e-4; they also ran 6e-4, 9e-4, 12e-4, 30e-4.
**Only 30e-4 (10x) was unstable**; 2x and 3x trained normally, and all four converged to the **same
loss after annealing**. They also warn that a shorter hyperparameter experiment can reach the wrong
conclusion (the ordering crosses over past 200B tokens). So in a 7B-class setting, the exact window
this paper is probing — 2-3x the optimum — produced **nothing to measure**.

### 2.4 Rybakov et al. — the cleanest instance of this inference pattern, which declines to extrapolate

**"Methods of improving LLM training stability"**, Rybakov, Chrzanowski, Dykas, Xue, Lanir (NVIDIA),
arXiv **2410.16682**, 22 Oct 2024, single version, no venue. **[FIRST-PASS]**

Metric: binary converge/diverge judged on validation loss, tabulated across LR ∈ {6e-3, 8e-3, 20e-3,
40e-3, 60e-3, 80e-3} at fixed seed; they state outright that, as in Wortsman et al., a model is more
stable if it trains at higher LR without diverging. Diagnostic quantity: per-step L2 norms of W, X, Y
for QKV/Proj/FC1/FC2 — divergent runs show >2x the converged run's output norm by step 1000. They
extend Wortsman's logit-growth observation from attention logits to **all linear-layer outputs**,
finding QKV, Proj and FC2 grow fastest.

| Method | Diverges at |
|---|---|
| bf16 baseline | 8e-3 |
| soft_temp (β=0.5), soft_clip (ζ=1.03, γ=−0.03) | 20e-3 |
| σReparam, LayerScale | 40e-3 |
| soft_cap (tanh(logit/c)·c, c=50), QK_norm, QK_FC_norm | 60e-3 |
| QKV_norm (LayerNorm after QKV, pre-norm removed) | 80e-3 |
| QK_norm_cap (QK-norm + softmax cap) | no divergence in the sweep |

**Scale: one model, 830M** (24 blocks, hidden 1024, 16 heads, batch 512, seq 4096, grad clip 1.0,
bf16 linear / fp32 optimiser, 32 H100s); separate perplexity runs at LR 3e-4 on 0.2T tokens.
**[UNVERIFIED]** the divergence runs' step/token counts.

**Two reasons this undercuts.** It is the closest existing analogue to the proposed argument and it
**declines the extrapolation**, naming large-model validation as future work. And internally,
**QK_FC_norm has the same divergence ceiling as QK_norm (both 60e-3) despite normalising the very
layers observed to be exploding**, while still beating baseline on perplexity (10.87 vs 11.19). So
"relieves the diagnosed instability signal" and "raises the stable-LR ceiling" come apart within one
paper at one scale.

### 2.5 Takase et al., "Spike No More" — the one genuinely supporting paper, and the right template

arXiv **2312.16903**, v1 28 Dec 2023, v4 25 Jul 2025; **COLM 2025**. **[FIRST-PASS]**

From the pre-LN chain rule `‖∂L/∂x_1‖ ≤ ‖∂L/∂y_N‖ · Π_n ‖∂y_n/∂x'_n‖ · ‖∂x'_n/∂x_n‖` (Eq. 7), and
`‖∂LN(x')/∂x'‖ = 1/σ_x'` for `d ≫ 1` (Eq. 14), they bound

- FFN: `‖∂y/∂x'‖ ≤ 1 + (σ_1σ_2/σ_x')·C_ffn` with **`C_ffn = (sqrt(d) + sqrt(d_ffn))^2`** (Eq. 15)
- Attention: `‖∂x'/∂x‖ ≤ 1 + (σ_O/σ_x)·C_Attn` with **`C_Attn = 2·sqrt(d)·‖J^Z‖`** (Eq. 20)

giving two sufficient conditions: **small sub-layers** (`σ_1σ_2 ≪ σ_x'`) and **large shortcut**
(`σ_O ≪ σ_x`). Satisfied by the standard init (`σ = sqrt(2/5d)`, with `W_2`, `W_O` scaled by
`sqrt(1/2N)`) plus either **Scaled Embed** (multiply embeddings by `sqrt(d)`) or **Embed LN**.

Scales **350M / 1.7B / 13B** (~38B / ~38B / ~105B tokens, C4, float16; β2 = 0.999 at 350M-1.7B and
0.95 at 13B). Main LR 5e-4, with a 1e-3 / 5e-4 / 1e-4 sweep; 13B at 3e-4 and 1e-4. Findings: the
larger the LR the more frequent the spikes in the vanilla model; at **13B / LR 3e-4** vanilla's loss
rose from ~10,000 steps and gradients grew too large to continue, while Scaled Embed trained through;
at 13B / LR 1e-4 both train comparably. **No quantitative spike metric** — spikes are read off
curves, with grad-norm growth as the stated leading indicator.

**Why this is the template.** It is the only paper here whose mechanism is **provably monotone in
width** (the bound grows with `d` because `σ_x`, `σ_x'` ≈ the embedding std), validated across three
scales, with the intervention's advantage appearing **only** at higher LR and **widening** with scale.
That is the shape of argument that actually licenses an extrapolation. Three mismatches with this
paper: the criterion is binary divergence at fixed LR, not graceful degradation above the optimum; the
mechanism is specific to embedding/shortcut scale, so it says nothing about an architecture that does
not change `σ_x`; and its smallest scale is 350M, 3-13x above 27M-112M. Their own conclusion concedes
it is hard to claim the conditions completely solve instability.

### 2.6 Parameterisation work: small-scale robustness optima demonstrably move

**Everett et al., "Scaling Exponents Across Parameterizations and Optimizers"**, arXiv **2407.05872**,
8 Jul 2024, Google DeepMind + MIT, **ICML 2024**. **[FIRST-PASS]**
Scale correction to the premise: **fourteen widths D=128→16,384, i.e. 9.9M→26.8B params, tens of
thousands of models** — and **depth fixed at 8 blocks**, context 512, batch 256, 50,000 steps
(~6.5B tokens) at every size. Prescription (standard parameterization + Adam + full alignment):
embedding LR ∝ n⁰, hidden and readout LR ∝ n⁻¹. **Adam-atan2** replaces `x/(sqrt(v)+ε)` with
`atan2(x, sqrt(v))`, removing ε. They define a **log alignment ratio**
`A_l^t = log_fan-in( ‖W_l z_{l-1}‖_RMS / (‖W_l‖_RMS·‖z_{l-1}‖_RMS) )` and find alignment is
intermediate and highly dynamic during training. No spike metric; they use "LR sensitivity"
informally and never compute Wortsman's expectation.
**Undercuts, from an unexpected direction:** their **best-performing** prescription is the **least
LR-robust** — full alignment has very high LR sensitivity with the optimum close to the maximum stable
LR, which they describe as risky for stability, and they recommend it anyway. So good loss and low LR
sensitivity can be *anti*-correlated. Their one transfer failure (mean-field above 2B, from Adam-ε
underflow) was found only by running at 2B+. §6 notes production models co-scale width, depth, batch,
horizon, weight decay and LR, and that optimal co-scaling is open.

**Blake et al., "Unit Scaling"**, arXiv **2303.11257**, 20 Mar 2023, **ICML 2023**, and **"u-μP: The
Unit-Scaled Maximal Update Parametrization"**, arXiv **2407.17465**, 24 Jul 2024 (v3 10 Jan 2025).
**[FIRST-PASS]** Unit Scaling targets **numerics**, eliminating the *loss scale* rather than the
LR; evidence up to BERT-LARGE (~340M) in FP16/FP8, no billion-scale run, and it explicitly does not
address adapting scales *during* training. u-μP runs **1B / 3B / 7B on 300B SlimPajama tokens** in FP8.
**u-μP is the closest thing to a direct negative result on small-scale HP proxies:** §5.5 reports that
the large-scale optima **differ non-trivially** from the small-scale ones; with 8 layers and width 512
the HP-loss landscape was too noisy and they had to double the width to resolve the optima; and §3.1
reports that μTransfer holds in the Tensor-Programs-V setup but **breaks down under a standard Llama
training setup** (confirming Lingle, 2404.05728), recovered only by two stability fixes
(non-parametric norms, independent weight decay — the latter from Wortsman). §7 concedes that neither
μP nor Unit Scaling guarantees well-behaved quantities *over the course of* training. §6: they
deliberately omit QK-norm and z-loss to avoid confounds, so they offer no evidence on those.

### 2.7 Architecture + stability at 1.7B-1T: gated attention, QK-Clip, and the outlier-rescaling twist

**Qiu et al., "Gated Attention for Large Language Models"** (Qwen), arXiv **2505.06708**, 10 May 2025,
**v1 only**. **[FIRST-PASS]** Head-specific sigmoid gate on the SDPA output, 30 variants, FFN
width reduced to match parameters. Scales: **MoE 15B total / 2.54B activated** and **dense 1.7B**
(28- and 48-layer), up to **3.5T tokens**. Stability evidence: a 48-layer 1.7B at 1T tokens,
batch 4096, **diverges at max LR 8e-3 without the gate** and trains fine (and slightly better) with
it; nearly eliminated loss spikes on a 1.7B / 3.5T-token run at max LR 4.5e-3.
**Correction to the premise: this paper has no gradient-spike score.** Its two quantitative
metrics (Table 4) are activation proxies: **M-Act** = mean over layers of the max hidden-state
activation (baseline **1053** → elementwise SDPA gate **94**; head-shared 286; input-independent 471)
and **F-Attn** = attention mass on the first token (**0.467 → 0.048**). The spike score the premise
refers to is in Qwen3.8-Next (§2.1), same first author.

**Kimi K2 technical report**, arXiv **2507.20534**, 28 Jul 2025 (v2 3 Feb 2026), Moonshot AI.
**[FIRST-PASS]** **1.04T total / 32.6B activated** MoE, 61 layers, MLA, **15.5T tokens**, WSD with
10T at constant 2e-4. **QK-Clip**: with per-head
`S_max^h = (1/sqrt(d))·max_{X∈B} max_{i,j} Q_i^h K_j^{h⊤}`, apply **`γ_h = min(1, τ/S_max^h)`**
*post-update*; for MLA, scale `q^C` and `k^C` by `sqrt(γ_h)`, scale the head-specific rotary `q^R` by
`γ_h`, and leave the **shared** rotary `k^R` untouched. **τ = 100**. **12.7% of heads** triggered it at
least once in the first **70,000 steps**, after which it self-deactivated; **zero loss spikes** over
the 15.5T-token run. Appendix E: `|q·k| ≤ ‖x_i‖‖x_j‖‖W_q‖‖W_k‖`, so with RMSNorm the driver is the
**spectral norm** of `W_q`/`W_k`; Muon's full-effective-rank updates inflate it more than Adam's.
Motivation: soft-capping acts too late, and **QK-Norm is inapplicable to MLA** because keys are not
fully materialised. **The evidence chain runs the opposite way to this paper's:** the *need* was
established at **9B-activated / 53B-total**, where max logits quickly exceed ~1000 under vanilla
Muon; the **0.5B-activated / 3B-total** run with an aggressive τ=30 was only a **do-no-harm check**
(negligible loss effect). Small scale established absence of cost, never presence of benefit. Also
note QK-Clip is a **hard constraint** that provably bounds the causal quantity — categorically
stronger than "degrades less at 2-3x optimal LR".

**"A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for
Transformer Training"** (Qwen), arXiv **2601.22966**, 30 Jan 2026. **[FIRST-PASS]** Main ablations
at **2B / 120B tokens** (FFN width adjusted to hold parameters constant), scaling at
**MoE-7.4B-A1.7B / 1.2T tokens** and **MoE-24.6B-A2.7B / 500B tokens**. Outliers *and their
normalisations* perform rescaling that is load-bearing: removing normalisation (DyT) kills the
outliers but diverges at the baseline LR and only converges at 5e-4 with +0.259 loss; clipping
residual activations at ≤100 diverges early and at 1000 gives frequent spikes; once Gated Attention
reintroduces explicit rescaling, even clipping at 10 converges. Defines
`GatedNorm(u) = RMSNorm(u) ⊙ σ(W_2 SiLU(W_1 RMSNorm(u)))`. **Consequence for metric choice: the
activation-outlier proxy is not monotone in stability** — suppressing it without supplying a
replacement rescaling mechanism makes things worse.

**Gemma 2** (arXiv **2408.00118**, 31 Jul 2024) and **Gemma 3** (arXiv **2503.19786**, 25 Mar 2025).
**[FIRST-PASS]** Gemma 2 uses **logit soft-capping** `logits ← soft_cap·tanh(logits/soft_cap)`
with **soft_cap 50.0 in each self-attention layer and 30.0 at the final layer**, plus pre- *and*
post-norm RMSNorm, stated as being for stability. Gemma 3 **replaces soft-capping with QK-norm**,
justified purely by citing Dehghani et al., **Wortsman et al.** and Chameleon. Neither report defines
any quantitative stability metric, neither computes LR sensitivity, neither reports an LR sweep, and
**Gemma 3 never ablates QK-norm**. So Gemma 3 is a frontier lab adopting an architectural
intervention *on the authority of the small-scale proxy study* — a citation of the inference, not
evidence for it. Worth flagging that soft-capping was adopted and abandoned one generation later with
no published stability comparison either way.

### 2.8 What actually happened in real large runs

All **[FIRST-PASS]** unless noted.

- **PaLM**, arXiv **2204.02311**, Apr 2022, §5.1. Loss spiked **roughly 20 times**, **only in the
  540B model** — not observed in 8B or 62B with the same recipe — at highly irregular intervals,
  sometimes late in training, **despite gradient clipping being enabled**. Mitigation: restart from a
  checkpoint ~100 steps earlier and skip ~200-500 data batches. **The crucial negative control:** they
  retrained on *those same batches* from a different, earlier checkpoint and **saw no spike** —
  concluding the spikes require the combination of specific batches *with* a particular parameter
  state, explicitly not "bad data" alone, and that they found no principled mitigation strategy. PaLM
  is also the origin of z-loss (`1e-4·log^2 Z`).
- **OPT-175B**, arXiv **2205.01068**, May 2022, §2.5 + the `facebookresearch/metaseq` OPT chronicles.
  **At least 35 manual restarts** and **70+ automatic** over 2 months, >100 hosts cycled; ~90 restarts
  across the lineage, 148 pages of notes. Loss divergences correlated with the **dynamic loss scalar
  crashing to 0** and the **L2 norm of final-layer activations spiking**; gradient clipping lowered
  **1.0 → 0.3**; Pile subsets dropped for causing grad-norm spikes **at the 1.3B scale**; they briefly
  tried vanilla SGD. **The single best documented counterexample:** a "kitchen-sink" configuration
  validated for stability at **1.3B** (tensor parallelism, LPE, Normformer, GeLU→ReLU) **exploded at
  175B** with grad-norm/loss explosions and NaNs a few hundred updates after each restart; after
  lowering LR, clipping 2.5→1.5→1.0, weight decay 0.05→0.1, **β2 0.98→0.95**, and swapping ReLU for
  GeLU, the lineage was abandoned and restarted from scratch. The logbook states outright that small
  (<~13B) results do not necessarily hold when scaled up.
- **BLOOM-176B**, arXiv **2211.05100**, Nov 2022. Remarkably clean: **one loss spike**, swiftly
  recovered. The chronicles pin it to **2022-04-12, iterations 31214→31219: lm loss 2.199 → 5.098,
  grad norm 0.242 → 960.4**, recovered to 2.216 by iteration 31250 (~30 iterations). The decisive
  fixes were **numerics, discovered at 104B in fp16**: embedding LayerNorm (bitsandbytes
  StableEmbedding) improved stability but penalises zero-shot generalisation; final training used
  **bf16**, which the paper says proved to solve the instability problem, hedging that bf16 may
  alleviate the need for the embedding LayerNorm. The 104B precursor had **irreversible divergences**
  attributed to fp16's dynamic range; on the bad-data hypothesis the chronicles record that shuffling
  the data changed little.
- **GLM-130B**, arXiv **2210.02414**, Oct 2022. Frequent fp16 loss spikes that became **more frequent
  as training went on**; some self-recover, others arrive with a soaring gradient norm then a spike or
  NaN. Two scale-triggered mechanisms: with pre-LN, the main branch's value scale becomes extremely
  large in deeper layers (fixed by **DeepNorm**, `LayerNorm(α·x + Network(x))`, `α = (2N)^{1/2}`); and
  **attention scores grow beyond FP16's range as the model scales up**. **Embedding Layer Gradient
  Shrink (EGS, α=0.1)**: a collapse usually **lags a gradient-norm spike by a few steps**, and those
  spikes are usually caused by embedding-layer gradients, whose norm is several orders larger than
  other layers early on. Final run: only **three late-stage divergences**. CogView's PB-Relax,
  Sandwich-LN and BLOOM's embedding norm were each tried and rejected.
- **ViT-22B**, Dehghani et al., arXiv **2302.05442**, Feb 2023, §2.2 + Appendix B. Divergent loss
  after a few thousand steps, observed **around 8B parameters**; at the default Adam LR 1e-3 the loss
  steadily rose within **2000 steps**, and for the 8B model **attention logits quickly exceed 50000**
  in magnitude, giving near-one-hot attention with near-zero entropy. Fix: QK-norm, credited to
  **Gilmer et al. 2023** (the unpublished work behind Wortsman et al.). **Most useful single data
  point against the proposed inference:** ViT's LR had to be *reduced* with scale under the default
  recipe (1e-3 down to 4e-4 for ViT-H); with QK-norm, 1e-3 remains stable, with increasing benefits
  at scale.
- **Llama 3**, arXiv **2407.21783**. Contrast case at 405B: few loss spikes and no interventions
  needed to correct divergence (helped by batch-size warm-up 4M → 8M → 16M tokens) — but **466 job
  interruptions in 54 days**, 78% hardware, including **6 silent-data-corruption events** and
  stragglers where one slow GPU drags thousands.
- **DeepSeek-V3**, arXiv **2412.19437**, Dec 2024. States twice that they experienced no
  **irrecoverable** loss spikes and performed no rollbacks — note the qualifier. Appendix B.2:
  **block-wise FP8 quantisation of activation gradients (Dgrad) caused divergence** on a ~16B-total
  MoE at ~300B tokens. §2.1.2: unbalanced expert load leads to **routing collapse**.
- **DeepSeek-V4**, arXiv **2606.19348**, 26 Apr 2026. §4.2.3: rollbacks temporarily restore the state
  but do not prevent recurrence; spikes are consistently tied to **outliers in the MoE layers**, with
  the routing mechanism appearing to exacerbate them. Fixes: **Anticipatory Routing** (~20% wall-clock
  overhead, auto-triggered by a spike detector plus a short rollback, then reverted) and **SwiGLU
  clamping** (linear component to [-10,10], gate capped at 10). They concede the underlying principles
  remain insufficiently understood. **Directly relevant to this project:** §3 says plain
  hyper-connections frequently exhibit numerical instability **when stacking multiple layers**, which
  hinders scaling HC — i.e. the published failure mode in this architecture family is a function of
  **depth**, not of LR.
- **"Scaling FP8 training to trillion-token LLMs"**, arXiv **2409.12517**, Sep 2024, **ICLR 2025**.
  At 7B / 2T tokens they uncover instabilities not observable in shorter runs, traced to **outlier
  amplification by SwiGLU** that happens only over prolonged training, via a weight-alignment process.
  A **token-budget-dependent** mechanism: invisible to any short run at any size.
- Hardware: **"Understanding Silent Data Corruption in LLM Training"**, arXiv **2502.12340**, Feb 2025
  (AWS) — SDC can push models to different optima and cause loss spikes; **LLM-PRISM**, arXiv
  **2604.10390**, Apr 2026 (SC'26) — RTL-level fault injection over 7,664 runs shows catastrophic
  divergence at moderate fault rates for some datapaths and precisions.

### 2.9 Other metric and methodology papers worth citing

- **SPAM, "Spike-Aware Adam with Momentum Reset"**, Huang et al., arXiv **2501.06842**, 12 Jan 2025;
  an ICLR 2025 proceedings PDF exists (**[UNVERIFIED]** venue is not on the arXiv page). Definition
  2.1: **`GSS(g_i) = |g_i| / [ (1/(T+1)) Σ_{j=0..T} |g_j| ]`** per coordinate, a spike when
  `GSS > θ`; **θ = 50** for offline analysis over the first 1,000 steps, and an **online
  approximation `GSS ≈ g_i²/V_i`** (Adam's second moment) with **θ = 5000** for the clipper
  `g_i ← sign(g_i)·sqrt(θ V_i)`; momentum reset every ΔT=500 steps with 150 cosine warm-up steps.
  Scales **60M / 130M / 350M / 1B** (1.3-11.6B tokens, C4 and SlimPajama), seq 256, batch 512.
  **Directly useful: spikes reach ~1000x typical gradient magnitude, and gradient spikes coincide with
  loss bumps, at exactly this project's parameter range.** Also: **zeroing** spiked gradients
  *improves* final perplexity (so spikes are net-harmful, not informative), and **LayerNorm layers**
  spike most often despite having the fewest parameters.
- **ZClip**, Kumar et al., arXiv **2504.02507**, 3 Apr 2025, v1 only, no venue. **[FIRST-PASS]**
  The cleanest EMA z-score spike definition: with `g_t = ‖g_t‖_2`,
  `μ_t = α μ_{t-1} + (1-α) g_t`, `σ_t = sqrt(α σ_{t-1}² + (1-α)(g_t − μ_t)²)`,
  **`z_t = (g_t − μ_t)/σ_t`, spike iff `z_t > z_thres`**, defaults **α = 0.97, z_thres = 2.5**.
  Single scale (LLaMA 1B, 50B tokens), so no cross-scale evidence.
- **"Adaptive Preconditioners Trigger Loss Spikes in Adam"**, Bai et al., arXiv **2506.04805**,
  5 Jun 2025 (v2 25 May 2026), **ICML 2026**. **[FIRST-PASS]** A spike *predictor*:
  **`λ_grad(Ĥ_t) := ∇L(θ_t)ᵀ Ĥ_t ∇L(θ_t) / ‖∇L(θ_t)‖²`** (curvature along the gradient), with a spike
  when it exceeds **2/η**, plus a noise-robust sustained variant
  `min(λ_grad(Ĥ_{t-1}), λ_grad(Ĥ_t), λ_grad(Ĥ_{t+1}))`. Headline: `λ_max(Ĥ_t)` alone is
  **insufficient** — in their Adam run it crossed 2/η at 10 steps but only 7 spikes occurred, each
  coinciding with a `λ_grad` crossing. Mechanism: `v_t` decouples from `g_t²` and decays at ≈β2 while
  gradients rise. Scales up to a **187M LLaMA on 100B SlimPajama tokens**; reducing β2 consistently
  reduces spike frequency; stated limitation that Hessian-vector-product eigenvalue analysis beyond
  ~200M params is computationally demanding. **Consequence: two architectures with identical high-LR
  loss curves can differ in `λ_grad` dynamics, and β2 is a confound.**
- **Zhai et al., "Stabilizing Transformer Training by Preventing Attention Entropy Collapse"**, arXiv
  **2303.06296**, **ICML 2023**. Read by me via ar5iv. `Ent(A_i) = −Σ_{j=1..T} A_ij log A_ij`, and
  `Ent(A) = (1/T) Σ_i Ent(A_i)`, averaged over query positions, heads and examples. Methodologically
  important: they **elicit instability by doubling the learning rate** (ViT, 5e-4 → 1e-3) and show
  entropy collapsing to near zero with divergence. So high-LR stress-testing at modest scale is a
  citable methodology independent of Wortsman. σReparam = spectral normalisation plus a learned
  scalar `γ`, which they argue gives robustness to LR because the spectral norm of each layer is
  controlled by one parameter.
- **ExoFormer, "Attention Projection Mixing with Exogenous Anchors"**, Jonathan Su, arXiv
  **2601.08131** v4, 27 May 2026. Read by me via a full-page fetch. Appendix F spike score: the
  percentage of **gradient L2-norm values exceeding seven standard deviations from a rolling mean over
  the last 1,000 steps** — i.e. the OLMo 2 definition applied to an architecture comparison. Values at
  450M/1B on 10B/20B tokens: ResFormer (unnormalised) 0.0458, naive combination 0.0108, gated
  attention 0.0085, E-NuResFormer 0.0081, E-ExoFormer 0.0054. This project already cites the paper,
  so it is the cheapest route to a comparable number.
- **Lourie, Cho, Ullrich, Lotfi, "Small-Scale Experiments: Are We There Yet?"**, arXiv **2608.11859**,
  12 Aug 2026. Read by me in full. **Hyperparameter sensitivity fades with scale**, through
  parameters *and* data jointly (neither alone suffices), because the **hyperparameter loss surface
  becomes lower-dimensional**; scaling laws at small scale exist but only emerge on the fully tuned
  frontier, which requires a far larger search than most run. Case study: pre-norm vs post-norm at
  **4M / 34M / 134M** with **511 / 512 / 128** random configurations for post-norm and 128 at each
  scale for pre-norm, fitting on the two smaller and validating on the largest; they recover the
  large-scale pre-norm result. Two useful secondary points: post-norm's **own hyperparameter
  sensitivity is itself reported as a finding** ("post-norm was harder to tune at every turn"), which
  legitimises HP sensitivity as a reportable architectural property; and they still warn that
  extrapolation hits statistical limits.
- **Wang et al., "Can Small Training Runs Reliably Guide Data Curation? Rethinking Proxy-Model
  Practice"**, arXiv **2512.24503**, 30 Dec 2025 (v2 12 Apr 2026), **ICLR 2026**. Read by me at the
  abs page. Different target (data recipes) but the same methodological failure: holding the training
  configuration fixed across arms in the name of fairness is unsound because the optimum is
  arm-dependent, and conclusions can flip under minor hyperparameter changes. **Their prescription is
  a *reduced* proxy LR**, which they prove preserves dataset ordering for random-feature models and
  validate across 23 recipes. **[UNVERIFIED]** model scales.
- **Step Law, "Predictable Scale: Part I"**, Li et al., arXiv **2503.04715**, v7 19 Aug 2025. Read by
  me in the HTML. **`η_opt = 1.79·N^{-0.713}·D^{0.307}`** and **`B_opt = 0.58·D^{0.571}`**, from
  >3,700 models and ~1M H800-hours; they also report the hyperparameter landscape is **convex with a
  broad optimum**, and that estimated optima deviate from exhaustive search by 0.094% on test.
- **[UNVERIFIED], abstract-level only — do not cite numbers from these without reading them:**
  "Taming Curvature: Architecture Warm-Up for Stable Transformer Training" (**2606.16768**, curvature
  surges coincide with instabilities and **curvature grows with depth**); "Mechanism-Driven Monitors
  for Preemptive Detection of LLM Training Instability" (**2606.28116**, after a fault training may
  continue thousands of steps while loss and gradient norms still look normal); "Spectral Alignment as
  Predictor of Loss Explosion" (**2510.04202**, weight/grad norms are lagging and ambiguous predictors
  whose values are not comparable across models or even layers); "Scaling with Collapse" (**2509.25087**,
  loss curves collapse onto a universal trajectory precisely when HPs are optimal, so
  deviation-from-collapse is an early diagnostic); "Weight Decay may matter more than muP for Learning
  Rate Transfer in Practice" (**2510.19093**); "Dense Local Dependencies Induce Attention-Logit
  Explosion..." (**2505.15548**, logit growth increases with **sequence length**); "On the Surprising
  Effectiveness of Large Learning Rates under Standard Width Scaling" (**2505.22491**); "Controlling
  changes to attention logits" (**2511.21377**, parameter-dependent LRs for `W_q`/`W_k`, works in the
  MLA setting where QK-norm does not).

### 2.10 The gap

**No paper found measures high-LR robustness for an architecture at 27M-112M (or anything near it) and
then confirms reduced instabilities at multi-billion scale.** The closest is Qwen3.8-Next (§2.1), and
its proxy is 25B-A3B with a known-good anchor and a production confirmation. Published proxy scales
for architectural stability claims: Wortsman 2.4M-1.2B plus a 4.8B confirmation (but for *known*
mechanisms with *known* fixes), Rybakov 830M, Takase 350M-13B, Gated Attention 1.7B / 15B-A2.5B,
Kimi K2's do-no-harm check 0.5B-A/3B with the *need* established at 9B-A/53B, u-μP 1B-7B,
Qwen3.8-Next 25B-A3B. Also: no paper was found that sets out to *audit* the Wortsman proxy
methodology as such — it is cited approvingly or sidestepped, never tested.

---

## 3. Citable metrics, with exact definitions

Roughly in order of how cheaply they can be added to the existing sweep.

### 3.1 LR sensitivity (Wortsman et al. 2309.14322 §2.2)

`E_{η∈[a,b]}[ min(ℓ(A(η)), ℓ_0) − ℓ* ]`, 7 log-spaced points over three decades, `η` = cosine peak,
clipped at the init loss. Details in §1.1. Costs a wide LR sweep plus the init loss.

### 3.2 Loss-spike rate against a rolling median (Qwen3.8-Next 2608.30320 §3.3) — **recommended**

**Steps whose loss exceeds a 201-step rolling median by more than 0.1, reported per 10k steps.**
Companions: **p99.9 of the pre-clip gradient norm** and the **number of clip-threshold crossings**
(their threshold 0.5), plus **per-block maximum activation**. Verified verbatim. This is the only
published operationalisation designed for exactly the experiment this paper ran (variants at a
multiple of the optimal LR), it uses a median rather than a mean so a spike does not inflate its own
baseline, and the clip-crossing count solves the gradient-clipping objection in §4.2.4.

### 3.3 Spike score against a rolling sigma (OLMo 2 2501.00656 §3.2; ExoFormer 2601.08131 App. F)

**Percentage of values in a time series at least seven standard deviations from a rolling average of
the last 1,000 values**, applied to the training loss and to the gradient L2 norm. ExoFormer applies
the same definition to an architecture comparison and reports 0.0458 / 0.0108 / 0.0085 / 0.0081 /
0.0054 across variants at 450M-1B. **[UNVERIFIED]** the window for the standard deviation is not
specified in OLMo 2, so we have to state our own choice. Scale-free, hence comparable across runs, unlike
"largest late grad norm = 16 vs 51 vs 397".

### 3.4 EMA z-score spike detection (ZClip 2504.02507)

`μ_t = α μ_{t-1} + (1-α) g_t`; `σ_t = sqrt(α σ_{t-1}² + (1-α)(g_t − μ_t)²)`;
**`z_t = (g_t − μ_t)/σ_t`, spike iff `z_t > z_thres`**; defaults **α = 0.97, z_thres = 2.5**.
Assumes gradient norms are locally approximately normal.

### 3.5 Per-coordinate gradient spike score (SPAM 2501.06842 Def. 2.1)

`GSS(g_i) = |g_i| / [ (1/(T+1)) Σ_{j=0..T} |g_j| ]`; spike iff `GSS > θ`; **θ = 50** offline,
**θ = 5000** with the online form `GSS ≈ g_i²/V_i`. Validated at 60M-1B, i.e. this project's range.

### 3.6 Max attention logit (Wortsman §3.1.1, §3.3; Kimi K2 §Appendix E)

`z_ij = <q_i, k_j>/sqrt(d_h)`; report the **max over the batch for block 0** (empirically largest) at
a **fixed early step (2e3)**. Thresholds from the literature: **>1e4 ⇒ every Wortsman run diverged**;
transplanting `κ = 1e3` already degrades loss; ViT-22B's 8B model reached **>50000**; Kimi K2 clips at
**τ = 100** per head via `γ_h = min(1, τ/S_max^h)` with
`S_max^h = (1/sqrt(d))·max_{X∈B} max_{i,j} Q_i^h K_j^{h⊤}`. The only characteristic in this
literature with a *validated* extrapolation to a larger model.

### 3.7 Activation-outlier metrics (Gated Attention 2505.06708 Table 4; Qwen3.8-Next)

**M-Act** = mean over layers of the max hidden-state activation (baseline 1053 → gated 94);
**F-Attn** = attention mass on the first token (0.467 → 0.048); Qwen3.8-Next's **per-block max
activation**. **Caveat from 2601.22966: this proxy is not monotone in stability** — suppressing
outliers without supplying a replacement rescaling mechanism makes training worse. So report it as a
mechanism *diagnostic*, never as a stability score on its own.

### 3.8 Output-logit / z-loss diagnostics (Wortsman §3.1.2, after PaLM)

Track `log Z` with `Z = Σ_j e^{y_j}`; the penalty is `log^2 Z` (coefficient 1e-4; **[UNVERIFIED]**
OLMo 2 is internally inconsistent here, §3.3.3 saying 1e-4 and Table 1 saying 1e-5). Failure
signature: `log Z` drifting very negative **late in training** — which is when this project's
grad-norm spikes occur, making it a cheap mechanism check.

### 3.9 Gradient RMS vs AdamW ε (Wortsman §3.4)

`RMS(g) = sqrt(E_i[g_i^2])` per layer (they use MLP layer 1 of block 0), averaged over the last 500
steps, plotted against model size and LR with ε drawn as a horizontal line. Failure mode: `RMS(g) → ε`
collapses the update. Note DeepSeek-V4's own recipe already uses **AdamW ε = 1e-20**.

### 3.10 Mechanism-specific predictors, for completeness

Preconditioned curvature along the gradient `λ_grad(Ĥ_t) > 2/η` (Bai et al. 2506.04805, §2.9);
bimodality of `r = m/sqrt(v)` via Hartigan's dip statistic (Molybog et al. 2304.09871); attention
entropy `Ent(A) = (1/T) Σ_i Ent(A_i)` (Zhai et al. 2303.06296); spectral norm of `W_q`, `W_k`
(Kimi K2 Appendix E).

### 3.11 What *not* to report

Raw "largest late gradient norm" as the headline number. 2510.04202 (**[UNVERIFIED]**, abstract only)
argues norms are lagging and ambiguous predictors whose values are not comparable across models or
even across layers of one model; Qwen3.8-Next found Muon runs with a *higher* median gradient norm and
*larger* max activations than AdamW yet **far fewer** spikes; and with clipping at global norm 1 the
difference between 51 and 397 never reaches the weights.

---

## 4. Assessment

The claim to be judged: *multi-head mHC loses less loss at 2-3x the tuned LR and has smaller late
grad-norm spikes at 27M-112M, therefore (citing Wortsman et al.) it might reduce training
instabilities in bigger models.*

### 4.1 What is solid

- **The protocol has a strong, recent precedent.** Qwen3.8-Next (§2.1) runs essentially this
  experiment — variants at 2x/3x/4x the optimal LR, scored by spike rate — cites Wortsman for the
  premise, and its conclusion did hold at production scale. "Stress-test architectural variants above
  the tuned LR" is not a methodological novelty that needs defending; it is current practice at Qwen,
  and Zhai et al. (2303.06296) were doing the LR-doubling version in 2023.
- **The result passes Wortsman's own validity test for the metric.** Appendix B invalidates
  LR-sensitivity comparisons when an intervention changes the meaning of the LR, and their empirical
  check is that the curves coincide below some critical LR. This project reports an **identical
  optimum and identical near-optimal band** with divergence only above it — exactly the pattern they
  accept. Say so explicitly; it pre-empts the obvious "you changed the effective step size" objection.
- **The middle link is citable at this exact scale.** SPAM (§2.9) shows gradient spikes coincide with
  loss bumps at **60M-1B**, with spikes reaching ~1000x typical magnitude, and that zeroing spiked
  gradients *improves* perplexity. So "fewer grad-norm spikes" ↔ "fewer loss bumps" is not invented.
- **There is a published mechanism this result could plausibly instantiate — and it is testable
  cheaply.** Qwen3.8-Next's account is that high-LR training *requires* a rescaling mechanism, and
  that without an explicit gate the network supplies it by growing activation outliers, which leaves
  it fragile. 2601.22966 sharpens this: outlier-driven rescaling is load-bearing, and removing
  outliers without replacing the rescaling mechanism makes things worse. **Per-group connection
  coefficients on the residual stream are exactly such a rescaling mechanism.** That is a specific,
  named, falsifiable hypothesis for why multi-head mHC would tolerate a higher LR, and testing it
  costs one logging change (§5.1.2).
- **HP sensitivity is a legitimate reportable architectural property.** Lourie et al. (§2.9) treat
  post-norm's tuning difficulty as a finding in its own right, and note that an architecture better in
  theory is worse in practice if good hyperparameters cannot be found.

### 4.2 What is not solid, worst first

1. **Wortsman et al. explicitly exclude loss spikes.** Footnote 1 on page 1: slow divergence, **not**
   loss spikes. Their §4 attributes fast spikes to the adaptive edge of stability, where the governing
   knob is **β2** (large β2 → spikes, small β2 → none). So citing 2309.14322 as the authority for
   "fewer loss spikes ⇒ fewer large-scale instabilities" misattributes the paper, and a referee who
   knows it will say so in one line. Cite 2309.14322 for the *methodology* and cite
   SPAM / OLMo 2 / Takase / Qwen3.8-Next for spikes.
2. **Wortsman pre-emptively deflates exactly this result.** §3.2.4, on μParam: reducing LR sensitivity
   is not important if the optimal LR does not change. This project reports no shift in the optimum
   and no widening of the within-0.02 band. By the proxy paper's own standard that is the low-value
   case. We should quote it against ourselves.
3. **We measured the depth of the cliff, not its location.** Wortsman's one scale-dependent empirical
   law is that the **lowest divergent LR falls as N rises**. Nothing connects the *loss penalty beyond*
   that threshold, or the grad-norm magnitude at 3x the optimum, to large-scale stability. Our own
   result says the threshold did not move — so the quantity that scales is the one that did not change,
   and the quantity that changed has no documented scaling story. Rybakov's internal dissociation
   (QK_FC_norm and QK_norm share a divergence ceiling despite different effects on the diagnosed
   signal) is the same point from the other direction.
4. **Gradient clipping probably neutralises the headline numbers.** Clipping at global norm 1 is the
   default in Wortsman, PaLM, Rybakov and essentially every large run (Qwen3.8-Next uses 0.5; OPT
   lowered 1.0 → 0.3). With clipping on, pre-clip norms of 16, 51 and 397 are all clipped to 1 and give
   the same update direction up to scale — the 397 is not 25x worse in any sense that reaches the
   weights. State whether clipping was on; if it was, report **clip-threshold crossings** and
   **p99.9 pre-clip norm** (Qwen3.8-Next's companions) or the number loses its force. Note also that
   PaLM's ~20 spikes at 540B happened **despite** clipping being enabled.
5. **Standard mitigations may already remove the headroom.** qk-layernorm and z-loss are *defaults* in
   Wortsman's own setup, in OLMo 2, and in Gemma 3; DeepSeek-V4's recipe is already hardened (AdamW
   ε = 1e-20, SwiGLU clamping, loss-free load balancing). If the multi-head advantage at 3x LR
   disappears once qk-norm and z-loss are on, the result is about an unprotected baseline. **This is
   the single most important missing experiment.**
6. **The effect may shrink with scale rather than grow.** Lourie et al. (2608.11859) find
   hyperparameter sensitivity **fades** with scale and the HP loss surface becomes lower-dimensional,
   so an HP-tolerance advantage at 27M-112M should be *smaller* at 7B. Note the apparent tension with
   Wortsman's "LR sensitivity increases with scale" rather than papering over it: the definitions
   differ (Wortsman = expected excess loss over a *fixed* 3-decade LR range at fixed other HPs;
   Lourie = spread of a random search over *many* HPs with per-budget tuning). A referee can cite
   either against us, so we should cite both. Corroborating from the other end: **OLMo 2 found 2x and 3x
   the optimal LR entirely benign at 7B class** — only 10x spiked, and all four LRs annealed to the
   same loss. The window this paper probes was empty in their setting.
7. **No one has shown this direction of transfer for an architecture, and two groups documented the
   opposite.** Every success story runs **large → small**: a failure seen at 8B (ViT), 130B (GLM),
   540B (PaLM) or 53B (Kimi) was then reproduced cheaply and fixed. The OPT logbook states that
   <~13B results do not necessarily hold at scale and documents a **1.3B-validated configuration that
   exploded at 175B**; ViT-22B's default recipe required the LR to be *lowered* as scale grew; u-μP
   found its large-scale HP optima differ non-trivially from its small-scale ones, and its μTransfer
   result broke under a standard Llama recipe. Wortsman's only validated small→large prediction
   extrapolated a *mechanistic characteristic with a known failure threshold* (max attention logit vs
   1e4), not a loss-robustness ranking — which is why the claim as written is unfalsifiable.
8. **Molybog et al. is a direct counterexample at the scale the claim is about.** The dominant spike
   mechanism at **546B** (bimodal `m/sqrt(v)` in early layers after gradients fall below ε, driven by
   depth and a 65536 batch) **was not present at 30B or 7B**, and no bimodal-update layers could even
   be detected there. Four orders of magnitude above 112M, the mechanism that mattered was invisible.
9. **Scale- and budget-dependent mechanisms our sweep cannot see.** At 27M-112M dense on ~98M tokens
   we cannot exhibit: MoE routing collapse or routing-amplified outliers (DeepSeek-V3/V4);
   fp16/bf16/fp8 dynamic-range failures (BLOOM 104B, GLM-130B, DeepSeek-V3's Dgrad divergence);
   **token-budget-dependent** SwiGLU outlier amplification (2409.12517, visible only after prolonged
   training at any size); size-dependent susceptibility to a specific batch (OLMo 2 reports the same
   repeated-n-gram sequence spiking a larger model and not a smaller one); PaLM's state×data
   interaction (replaying the same batches from an earlier checkpoint produced **no** spike);
   embedding-gradient dominance (GLM-130B's EGS); pre-LN deep-branch value-scale growth (DeepNorm);
   long-context logit growth (2505.15548); or hardware SDC and stragglers (Llama 3: 466 interruptions
   in 54 days). Per this project's own notes (`core_papers_spec.md` §6.4, from DeepSeek-V4 §4.2.3),
   V4's stability fixes are attributed to **MoE routing outliers, not to mHC** — mHC did not remove all
   instability at 1.6T.
10. **The incumbent's own stability claim is a different failure mode, and it is about depth.** mHC
    motivates itself (per `core_papers_spec.md` §1.1, quoting mHC §1) by HC's composite residual map
    failing to preserve the global feature mean, giving unbounded amplification or attenuation — a
    *signal-propagation* argument, with the doubly-stochastic constraint as the fix and gradient norm
    among the reported metrics. DeepSeek-V4 §3 sharpens it: plain HC is numerically unstable **when
    stacking multiple layers**, which hinders scaling it. So the published failure mode in this family
    is a function of **depth**, which a fixed-depth LR sweep does not vary. Our multi-head split does
    not change the doubly-stochastic constraint, so a *further* stability benefit is a second,
    different mechanism. Name it or do not claim it. It also means **plain HC is the natural positive
    control** (§5.1.4), not just mHC and a wider MLP.
11. **Optimal LR falls with scale, so the absolute region we probe is unreachable at scale.** Step
    Law gives `η_opt = 1.79 N^{-0.713} D^{0.307}`; under compute-optimal `D ∝ N` that is
    `η_opt ∝ N^{-0.41}`, so from 112M to 7B the optimum falls ~5x (~19x at fixed D). Only the
    *relative* framing (3x the local optimum) is defensible, and it should be stated as such. Step Law
    also reports a **broad, convex** optimum, which weakens "tolerance above the optimum" as a
    practically scarce property.
12. **Small budget, tail statistics, probably one seed.** 27M and 112M on ~98M tokens, with **no gain
    at matched parameters and tuned LR**. The compared quantities (loss at 3x the optimum, largest late
    grad norm) live in the noisiest part of the sweep. A reviewer will ask for seed variance.
13. **Possible confounds not yet excluded.** β2 (Wortsman §4; Bai et al.: reducing β2
    consistently reduces spike frequency), warm-up length (OLMo 2 built its spiking baseline mainly by
    *shortening* warm-up), and weight-decay coupling (Wortsman §3.2.2). Any of these could produce the
    observed difference without it being architectural.

### 4.3 Recommended framing

- **(A) Honest minimal, no new runs.** Report "reduced LR sensitivity above the optimum at matched
  parameters", with the LR range stated, scored with the Qwen3.8-Next spike rate (§3.2) or the
  OLMo 2 / ExoFormer spike score (§3.3) rather than a raw max grad norm. In the same breath: the
  optimal LR does not move, the near-optimal band does not widen, and by Wortsman's own §3.2.4 remark
  this limits the practical value. Cite 2309.14322 and 2608.30320 for the *method*, 2303.06296 for the
  high-LR stress-test precedent, 2501.06842 for the grad-spike/loss-bump link at 60M-1B. **Do not
  write "may reduce training instabilities in bigger models";** write that the mechanism is
  unidentified and name the experiment that would identify it.
- **(B) Defensible claim, cheap runs (§5.1-5.2).** Turn on qk-norm and z-loss and see whether the
  tolerance survives; log max attention logit, `log Z` and per-block max activation to name the
  mechanism; add plain HC as a positive control; report clip-crossings. Then the claim becomes "the
  advantage is / is not redundant with the standard mitigations, and here is the mechanism" — a
  reviewable claim either way, where a negative result is still a real contribution.
- **(C) The claim as written.** Would need a monotone trend across ≥3 sizes plus an identified
  characteristic with an extrapolable threshold, and even then the precedent (Qwen3.8-Next) adds a
  known-good anchor and a production confirmation. Out of reach at this budget. Do not make it.

Wording that would survive review:

> At matched parameters the multi-head variants are less sensitive to learning rate above the tuned
> optimum, measured as [metric]. The optimal learning rate and the near-optimal range are unchanged,
> so by the standard of Wortsman et al. (2024) this is a robustness result rather than a stability
> result. We do not claim it transfers: Molybog et al. (2023) show the dominant 546B spike mechanism is
> absent at 7B and 30B, Wortsman et al. report LR sensitivity continuing to rise with scale even under
> their mitigations, and the industrial practice this test descends from (Qwen Team, 2026) runs the
> same 2-4x stress test at 25B total / 3B active parameters and calibrates it against a recipe already
> known to scale.

---

## 5. Experiments ranked by value per cost

Cost in units of one 112M run (R), assuming the existing sweep infrastructure.

### 5.1 Highest value per cost

1. **Turn on qk-layernorm and z-loss, repeat the high-LR comparison.** ≈4-8R (two variants plus the
   wider-MLP baseline, at the optimum and 2-3x above). *Why first:* it decides whether the paper has a
   result at all. If the advantage survives the standard mitigations, we have something new and
   framing 4.3(B) is earned; if it vanishes, we have a clean negative ("the tolerance is redundant
   with qk-norm/z-loss") that is still worth a paragraph and saves us from a referee finding it.
   Either outcome is publishable; not knowing is not.
2. **Log max attention logit (block 0, fixed early step), `log Z`, and per-block max activation.**
   ≈0-3R (instrumentation only; free if checkpoints can be post-processed). *Why:* it converts
   "degrades less" into a **mechanism**, and it tests a *named published hypothesis* — Qwen3.8-Next's
   claim that high-LR training needs a rescaling mechanism and that without one the network grows
   activation outliers. If the wider-MLP baseline's 397 grad norm coincides with max attention logit
   crossing ~1e3-1e4 while multi-head's does not, we can name the instability and cite the validated
   1e4 threshold. If multi-head shows *lower* per-block max activation at high LR while matching loss,
   we have the Qwen mechanism in our architecture, which is a far stronger paper than a loss
   comparison. If neither crosses, we have ruled the Wortsman mechanisms out — also decisive, and it
   tells us to stop citing them.
3. **Re-score existing logs with a published spike metric, and report clipping.** ≈0R, pure
   post-processing. Use the Qwen3.8-Next rate (steps >0.1 above a 201-step rolling median, per 10k
   steps) plus p99.9 pre-clip grad norm and clip-threshold crossings; optionally the OLMo 2 /
   ExoFormer 7σ-over-1000-step score for comparability with 2601.08131. *Why:* "16 vs 51 vs 397" is
   not comparable to anything published and is probably neutralised by clipping (§4.2.4).
4. **Add plain HC as a positive control at the high LR.** ≈3-6R. *Why:* HC is the one architecture in
   this lineage with a *published* instability claim (mHC §1; DeepSeek-V4 §3 on depth-stacking). If
   our protocol does not separate HC from mHC in the direction mHC claims, the protocol is not
   measuring what we are citing it for. Cheap, and it validates the instrument.
5. **State and, if absent, add gradient clipping at global norm 1.** ≈0-4R. Without it the baseline is
   non-standard and every gradient number is contestable.

### 5.2 High value, moderate cost

6. **Extend the LR sweep upward and measure the *stability margin* and its trend across our two
   sizes.** ≈14-21R (7 log-spaced LRs x 2-3 variants at 27M, which is cheap; then only the decisive
   points at 112M). Define `margin(variant, N) = (lowest divergent LR) / (optimal LR)` and report both
   the Wortsman LR-sensitivity scalar and this ratio at **both** sizes.
   *Why this is the best single experiment for the claim we want:* the one scale-dependent empirical
   law in this literature is that the lowest divergent LR **falls** as N rises, i.e. the margin shrinks
   with scale — that is the whole reason high LR at small scale proxies large scale. Our current
   result says the margin did not change, but we measured it as the band within 0.02 of the best loss,
   which is a near-optimality band, not a divergence threshold, and our sweep may not reach the cliff
   at all. If the margin is genuinely larger for multi-head **and** shrinks more slowly from 27M to
   112M, we have a trend on the right quantity. If it is identical, we have falsified the strong
   claim cheaply and should adopt framing 4.3(A). Report the LR-vs-loss curves next to the scalar and
   clip at the init loss (Wortsman Appendix B insists on both).
7. **Adopt the constant-LR amplification.** ≈6-10R. Hold the LR constant at a multiple of the optimum
   instead of decaying it, exactly as Qwen3.8-Next does, to imitate a prolonged peak LR. *Why:* they
   added this precisely because raising the LR alone was not enough to surface the production failure
   mode, and it makes our protocol directly comparable to theirs. It also lengthens the window in
   which late-training spikes can occur, which is where our effect reportedly lives.
8. **High-resolution per-step loss and grad-norm logging on the high-LR runs.** ≈0-6R. *Why:* it
   distinguishes **slow divergence** (Wortsman's actual subject) from **fast spikes with recovery**
   (edge of stability, β2). These need different citations and support different claims; cheap, and it
   settles which literature applies.
9. **β2 ablation at the high LR (0.9 / 0.95 / 0.99).** ≈6-9R. *Why:* Wortsman §4 and Bai et al. both
   make β2 the governing knob for fast spikes. If our grad-norm-spike difference moves with β2, the
   effect is optimiser-mediated rather than architectural and the whole framing changes. Also check
   warm-up length, since OLMo 2 induced spikes mainly by shortening it. This is the cheapest way to
   falsify our own claim, which is the most valuable thing we can offer a skeptical reviewer.

### 5.3 Needed for the strong claim, expensive

10. **A third model size, and the *trend* of the effect.** ≈20-60R. *Why:* a claim about bigger models
    needs at least a sign on the derivative, and Wortsman's and OLMo 2's point is that levels shift
    while trends persist. Calibrate expectations: Lourie et al. used three well-separated scales
    (4M / 34M / 134M) with 128-512 configurations each and still warn that extrapolation hits
    statistical limits. Three points let us say the effect grows, shrinks or is flat — not that it
    helps at 7B.
11. **Depth sweep at fixed parameters.** ≈10-20R. *Why:* this is better motivated than it first looks.
    Wortsman found depth scaling raises LR sensitivity faster than width; DeepSeek-V4 §3 says plain HC
    is unstable specifically **when stacking layers**; 2606.16768 (**[UNVERIFIED]**) reports curvature
    growing with depth. If multi-head mHC's tolerance is really about the residual stream, depth is
    where it should show up — and a depth trend is a mechanism-flavoured result rather than a
    single-point comparison. Arguably worth promoting above item 10.
12. **Seed replication of the decisive high-LR points.** ≈6-12R (3 seeds). *Why:* the comparison lives
    in the tail of the sweep, the noisiest place to measure anything. Without this, §4.2.12 stands.

### 5.4 Not worth it at this budget

- Trying to produce a genuine instability at the *tuned* LR at these scales: at 27M-112M on ~98M
  tokens, properly tuned, there is nothing to prevent.
- Any attempt at MoE, precision, long-context or distributed instabilities (§4.2.9).
- μParam, unless the goal becomes moving the optimal LR — and Wortsman found it stabilises the optimum
  *without* reducing sensitivity, which is the complementary and more useful property.
- Hessian-based predictors (`λ_grad`): Bai et al. flag that Hessian-vector-product eigenvalue analysis
  beyond ~200M parameters is already computationally demanding.
