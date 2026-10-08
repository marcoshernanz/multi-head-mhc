# Prior-art search: "multi-head" / grouped / channel-wise mHC

Search date: 2026-09-26.

## How to read the confidence labels

| label | meaning |
|---|---|
| **READ** | I fetched the actual text/code and read the relevant passage myself (verbatim quotes below). |
| **SUMMARY** | I fetched the page, but what I got back was a small-model summary of it. Verbatim quotes inside these entries came through that summary; treat exact wording as second-hand. |
| **SNIPPET** | I only ever saw a search-result snippet. Existence not independently verified. Do not cite without checking. |

Anything labelled SNIPPET below could in principle be a search-engine artefact. The arXiv IDs I
successfully fetched an abs or HTML page for are marked as such.

---

## Bottom line (details in "Verdict")

- **No paper, preprint, workshop paper, OpenReview submission or repo proposes "multi-head mHC"**
  in the sense of splitting D into h channel groups that each get their own n×n doubly stochastic
  H_res. The name is unclaimed; the mechanism is *partly* claimed by three different lines of work.
- The **closest implementation** is `lucidrains/hyper-connections` with `num_fracs > 1`, which
  Sinkhorns a single **(n·m)×(n·m)** matrix over the joint (stream, channel-group) index. Not
  block-diagonal, not per-head — the *superset* of the multi-head variant.
- The **closest published architecture** is ByteDance's **Generalized Hyper-Connections** inside
  Virtual Width Networks (arXiv 2511.11238): the read and write matrices are indexed by channel
  segment, so each channel group of the branch input gets its own pre-mix. No manifold constraint.
- The **closest published "multi-head routing" paper** is **MHAR** (arXiv 2607.27230), which does
  exactly the per-subspace-head trick — but for depth attention residuals, not for HC stream mixing.
- **Warning on naming:** "multi-head hyper-connections" is already in use in the wild to mean
  *plain mHC with n streams*. See the naming-collision section.

---

## 1. Queries run

### 1.1 Web search

| # | query | what it turned up |
|---|---|---|
| 1 | `"multi-head hyper-connections"` | No paper of that name. Surfaced mHC, xHC, oHC, sHC, an MRI-segmentation HC paper. |
| 2 | `"multi-head" mHC manifold-constrained hyper-connections` | mHC paper + explainers only. Also Motif 3, a semantic-coding mHC paper. |
| 3 | `"grouped hyper-connections" OR "channel-wise hyper-connections"` | Nothing. Search engine explicitly reported these are not established terms. |
| 4 | `mHC ... DeepSeek Sinkhorn doubly stochastic residual` | Background/explainer material; go-mHC, mHC-PEFT. |
| 5 | `"hyper-connections" "heads" residual stream expansion rate variant ICLR 2026 OpenReview` | **Found `wdlctc/hyper-connection-factory`** (11-variant comparison incl. MHAR) and the Sci4DL causal-analysis paper. |
| 6 | `"frac-connections" hyper-connections splitting hidden dimension fractions` | Frac-Connections (2503.14125) + **Virtual Width Networks (2511.11238)**. |
| 7 | `lucidrains hyper-connections num_fracs frac connections github` | Confirmed `num_fracs` "also allows you to mix streams and fractions of feature dimension". |
| 8 | `"head-wise" OR "per-head" hyper-connections ... each head own` | **MHAR**, Multi-Gate Residuals, Dual-Stream Transformer, stream-collapse paper. |
| 9 | `mHC frac-connections combined channel groups doubly stochastic per group` | go-mHC; no per-group mHC. |
| 10 | `"block-diagonal" residual mixing matrix hyper-connections channel groups` | **TEMPER (2608.07851)**, DeepCrossAttention, SCHEME. |
| 11 | `多头 超连接 mHC 流形约束 分组 通道` (Chinese) | Zhihu/CSDN/博客园 explainers of mHC only. No multi-head proposal in Chinese-language material. |
| 12 | `"Sinkhorn" multi-head doubly stochastic residual stream mixing per channel group` | mHC-SSM, JPmHC, Sinkhorn-attention rank-decay paper. |
| 13 | `Virtual Width Networks ByteDance 2511.11238 ...` | Confirmed VWN + **Generalized Hyper-Connections (GHC)** "unifies HC and Frac-Connections". |
| 14 | `知乎 mHC 改进 多头 分组 残差流 超连接 变体 想法` | Explainers only; no community proposal of a multi-head variant. |
| 15 | `"mHC" variants survey list oHC sHC xHC KromHC TBP-mHC go-mHC mHC-lite` | **TBP-mHC (2605.21724)**, KromHC comparison text. |
| 16 | `mHC-lite FFTYYY mixture of permutation matrices` | **mHC-lite (2601.05732)** = Birkhoff–von Neumann convex combination of permutations. |
| 17 | `"TEMPER" tensorized manifold-constrained residual routing github` | TEMPER + a list of small mHC reimplementations. |
| 18 | `MUDDFormer multiway dynamic dense connections per-head depth-wise multi-head Q K V` | MUDDFormer (2502.12170): per-stream (Q/K/V/R) dense weights. |
| 19 | `OpenReview mHC ... reviewer suggestion` | **Birkhoff-Exact Hyper-Connections** (Sci4DL 2026). |
| 20 | `DeepSeek V4 mHC DeepGEMM "Mega-mHC" kernel hc_pre hc_post` | vLLM PR #56962, SGLang/TileLang mHC; no head/group dimension mentioned. |
| 21 | `modded-nanogpt speedrun hyper-connections mHC record attempt` | No speedrun record built on HC/mHC found; nothing multi-head. |
| 22 | `"multi-head" "hyper-connections" ... each own doubly stochastic matrix` | Restated KromHC / mHC-lite; no multi-head. |
| 23 | `reddit/HN hyper-connections mHC "multi-head" discussion` | No Reddit or HN thread surfaced at all; only blogs and arXiv. |
| 24 | `arxiv "hyper-connections" "per-channel" OR "channel-group" ... future work` | **ExoFormer (2601.08131)** — scalar vs headwise vs elementwise granularity ablation. |
| 25 | `Motif 3 technical report hyper-connections variant residual streams heads grouped` | Motif 3 uses "a modified form of mHC", n=4, doubly stochastic, post-multiplier annealed 2→1. |
| 26 | `"value residual learning" ResFormer SVFormer per-head learnable lambda` | ResFormer/SVFormer (2410.17897) — per-layer λ, not per-head by default. |

### 1.2 OpenReview API (`api2.openreview.net/notes/search`)

Terms run: `hyper-connections`, `multi-head hyper-connections`, `grouped hyper-connections`,
`head-wise residual mixing`, `channel-wise hyper-connections`, `Birkhoff-Exact Hyper-Connections`.

Complete set of HC/mHC-family records returned (this is a useful census):

| title | venue |
|---|---|
| Hyper-Connections | ICLR 2025 Poster (forum `9FqARW7dwB`) |
| Frac-Connections | CoRR 2025 (`Ne9hqbjbbF`) |
| mHC: Manifold-Constrained Hyper-Connections | **ICML 2026 spotlight** (`mDhyxu8WRb`) |
| KromHC | ICML 2026 (`TI7Q2o6EIa`) |
| mHC-lite: You Don't Need 20 Sinkhorn-Knopp Iterations | ICLR 2026 Workshop GRaM (`5IJX6kvOif`) |
| xHC: Expanded Hyper-Connections | OpenReview Archive direct upload (`o2y2TvlSSv`) |
| Analyzing Stream Collapse in Hyper-Connections | ICML 2026 Workshop WSS (`Xyrii4O2nw`) |
| Ablate and Rescue: A Causal Analysis of Residual Stream Hyper-Connections | ICLR 2026 Workshop Sci4DL (`Zu3B2TKOPV`) |
| Birkhoff-Exact Hyper-Connections | ICLR 2026 Workshop Sci4DL (`jpIjkN1B1Q`) |

**No record in OpenReview contains "multi-head", "grouped", "head-wise" or "channel-wise"
hyper-connections in its title.** Review bodies for the ICML 2026 mHC and KromHC forums were not
retrievable through the public API (`notes?forum=...` returned 0 notes), so I could **not** check
whether a reviewer suggested a per-head variant. That is a genuine gap in this search.

### 1.3 GitHub (`gh search repos`, `gh api search/code`, direct clone)

- `gh search repos "hyper-connections"` → 30 results, full list triaged in §3.
- Code searches for the exact strings `"multi-head hyper-connections"`, `"multi-head mHC"`,
  `"per-head hyper"`, `"num_hc_groups"`, `"hc_num_heads"`:
  - `"multi-head mHC"` → **0 hits on all of GitHub.**
  - `"num_hc_groups"` → **0 hits.**
  - `"multi-head hyper-connections"` → 20 hits, all either (i) prose in digests/notes referring to
    ordinary mHC, or (ii) `waefrebeorn/wubuwizard`, which uses the phrase to mean plain mHC
    (see naming collision, §5).
- Repos cloned and read locally: `lucidrains/hyper-connections`, `wdlctc/hyper-connection-factory`,
  `wz1119/KromHC`, `tokenbender/mHC-manifold-constrained-hyper-connections`.
- Files read directly from raw.githubusercontent / GitHub contents API:
  `FFTYYY/mhc-lite/hyper_conn/mhc_lite.py`, `deepseek-ai/TileKernels/tile_kernels/mhc/*`,
  `waefrebeorn/wubuwizard/include/wubu_mhc_mh.h`.

---

## 2. Relevant sources

### 2.1 The foundations

**Hyper-Connections** — arXiv [2409.19606](https://arxiv.org/abs/2409.19606), Zhu et al.
(ByteDance), Sep 2024, ICLR 2025 poster. SUMMARY/known.
n parallel streams, H_pre (1×n), H_post (1×n), H_res (n×n), optionally token-dynamic.
Relevance: baseline, none of (a)–(e).

**mHC: Manifold-Constrained Hyper-Connections** — arXiv
[2512.24880](https://arxiv.org/abs/2512.24880), Xie et al. (DeepSeek), 2025-12-31 (v2 Jan 2026),
**ICML 2026 spotlight**. **READ** (I fetched the v2 HTML and grepped the whole 63k-char body).

This is the important negative result of the whole search. Full-text term counts:

```
'multi-head'    : 2 hits  -> both are MQA/GQA/MLA references in related work, and a bib entry
'multi head'    : 0
'head-wise'     : 0
'per-head'      : 0
'per-channel'   : 0
'channel-wise'  : 0
'grouped'       : 2 hits  -> "grouped convolutions (Xie et al. 2017)" and "Grouped-Query Attention"
'group-wise'    : 0
'block-diagonal': 0
'fraction' / 'Frac-Connections' : 0
```

So mHC **never** discusses per-channel or per-head mixing, and — notably — **never cites
Frac-Connections at all**, even though it is by the same lab lineage. H_pre, H_post and H_res are
scalar-per-(stream, stream) and shared across all D channels. The conclusion only gestures at other
manifolds: *"the framework accommodates the exploration of diverse manifold constraints tailored to
specific learning objectives"* (SUMMARY-level quote, from the conclusion).
Relevance: **establishes that (a)–(e) are all open w.r.t. the source paper.** Confidence: high.

**Frac-Connections** — arXiv [2503.14125](https://arxiv.org/abs/2503.14125), Zhu et al.
(ByteDance), Mar 2025. **READ** (HTML fetched and grepped).
Splits hidden state into m parts instead of widening; expansion rate generalized to fractional
n = 1/m. Verbatim from the body: *"introducing frac-connections is to address the seesaw problem in
residual connections while retaining the flexibility of constructing connection strengths, without
incurring the additional memory overhead of splitting hidden states into n parts as in
hyper-connections... When n = 1, frac-connections are equivalent to..."*.
It never uses the words "head" or "group" for the fractions and it does **not** define a joint
n-streams × m-fractions mixing. Relevance to (a): this is the channel-splitting half of the idea,
without the "each group gets its own mixing" half and without any manifold constraint.
Confidence: high.

### 2.2 The single closest published architecture — VWN / GHC

**Virtual Width Networks** — arXiv [2511.11238](https://arxiv.org/abs/2511.11238), ByteDance Seed
(corresponding author Defa Zhu, the HC/Frac author), 2025-11-14. **READ** (fetched HTML, extracted
§3.3 verbatim).

§3.3 "Generalized Hyper-Connections (GHC)". Verbatim (LaTeX-ified by the HTML):

```
GHC^l = ( 0    B^l )   with   B^l ∈ R^{m×n},  A^l = (Å^l  Â^l) ∈ R^{n×(m+n)}
        ( A^l      )   ,      GHC^l ∈ R^{2m×(m+n)}

H'^l = B^l⊺ T^l( Å^{l⊺} H'^{l-1} ) + Â^{l⊺} H'^{l-1}          (Eq. 8)

with H'^l = Reshape(h'^l, (n, D'/n)) the Over-Width Hidden States,
and d_b := D'/n = D/m the per-block width.
```

Read this carefully, because it is the crux:

- The residual state is an over-width embedding D' = r·D reshaped into **n segments of width d_b**.
- The backbone input has width D = **m segments of the same width d_b**.
- `Å ∈ R^{n×m}` therefore supplies **m separate read vectors** — channel block i of the branch input
  is its own weighted combination of the n segments. Likewise `B ∈ R^{m×n}` gives **m separate
  write vectors**.
- `Â ∈ R^{n×n}` mixes the segments on the skip path; within a segment it is shared across the d_b
  channels.
- Dynamic version (DGHC, Eqs. 10–12) predicts both from the token: `S_β ∘ tanh(H̄'W_β/τ)⊺ + B` and
  `S_α ∘ tanh(H̄'W_α/τ) + A`, with `τ = √(D/m)`, `W_β ∈ R^{d_b×m}`, `W_α ∈ R^{d_b×(m+n)}`, zero-init.
- The paper states GHC *"integrates the advantages of both [HC and Frac-Connections] — expanding the
  hidden dimension while further subdividing it into structured sub-states"* (SUMMARY-level quote).

Relevance: **(a) partially — yes for the read and write paths (per-channel-block coefficients), no
for the residual mixing** (single n×n over segments). **(b) yes** — m distinct pre-mixes is exactly
the "multiple read heads" idea. **(c) no** (no Kronecker). **(d) no** (no manifold constraint at
all; tanh-based, can be negative). Tested at MoE-A3.3B scale with r=8. Confidence: high for the
structure, medium for my reading of which index the m read vectors range over (I read Eq. 8 and the
shape definitions, not an accompanying figure).

### 2.3 The single closest published "multi-head routing" paper — MHAR

**Multi-Head Attention Residuals** — arXiv [2607.27230](https://arxiv.org/abs/2607.27230),
Luo et al., Jul 2026. **SUMMARY** (abs page fetched; the quote below came back verbatim inside the
summary).

> "the routing query is reshaped into H per-subspace heads, each with its own softmax over the
> depth history. The read becomes block-diagonal."

This is precisely mechanism (a) — h channel groups, each with its own routing distribution,
block-diagonal read — applied to **Attention Residuals** (Kimi/Moonshot, arXiv 2603.15031: replace
the residual sum with a softmax attention over previous sublayer outputs), **not** to HC/mHC stream
mixing. The paper does not mention hyper-connections or mHC.

Relevance: **(a) yes, in a different mechanism; (e) yes by name ("multi-head ... residuals")**.
This is the best available evidence that the per-channel-group routing idea works, and also the
clearest prior claim on the *name*. Confidence: medium-high (abstract-level only; I have not read
the method section).

### 2.4 Granularity ablation — ExoFormer

**Attention Projection Mixing with Exogenous Anchors (ExoFormer)** — arXiv
[2601.08131](https://arxiv.org/abs/2601.08131) v3, Jonathan Su, 2026-01-28. **SUMMARY** (HTML
fetched; quotes below returned verbatim inside the summary).

Systematically ablates three coefficient granularities for residual/attention-projection mixing:

> - **Scalar (S):** "a single coefficient shared across all channels"
> - **Headwise (H):** "one coefficient per head, broadcast across d_k"
> - **Elementwise (E):** "one coefficient per channel"

Findings: for the internal-anchor variant scalar won on downstream accuracy (49.83%) while
elementwise won on perplexity (14.15); for the external-anchor variant *"elementwise mixing
(E-ExoFormer) achieves the overall best performance, attaining the highest average accuracy
(49.85%)"*. The authors call the reversal *"architecture-dependent"*. And the observation most
relevant to a multi-head design: *"For the value pathway (V), the heatmaps exhibit sharp boundaries
that align precisely with head blocks"* — i.e. given full per-channel freedom, the learned
coefficients spontaneously organise into head blocks.

Relevance: **(a) strongly adjacent** — it is the direct scalar-vs-headwise-vs-elementwise experiment,
just on attention projections rather than on H_res, and it gives an empirical argument that the
head-block granularity is the natural one. Does not compare against mHC. Confidence: medium-high.

### 2.5 The mHC-variant zoo (none of them multi-head)

All of these constrain or reparameterize **H_res over the stream index only**. None indexes by
channel group.

| paper | ID / venue | what it does | (a) | (b) | (c) | (d) | (e) | conf. |
|---|---|---|---|---|---|---|---|---|
| **mHC-lite** | [2601.05732](https://arxiv.org/abs/2601.05732), Yongyi Yang & Jianyang Gao, 2026-01-09, ICLR 2026 GRaM workshop | H_res = softmax-weighted convex combination of **all n! permutation matrices** (Birkhoff–von Neumann), exact double stochasticity, no Sinkhorn iterations. Abstract confirmed: no mention of multi-head/per-head/grouped/channel-wise | no | no | no | **yes** | no | high (read the code + abs) |
| **Birkhoff-Exact HC (BE-HC)** | OpenReview `jpIjkN1B1Q`, Sci4DL 2026, Hyunjun Kim | same BvN convex-combination construction; claims stable training at 1000 layers | no | no | no | **yes** | no | high (read abstract via API) |
| **KromHC** | [2601.21579](https://www.arxiv.org/abs/2601.21579), **ICML 2026** | H_res = Kronecker product U_1 ⊗ … ⊗ U_K of small (2×2) doubly stochastic factors; each U_k is itself a convex combination of i_k! permutations. Kronecker is **"along each mode of the tensorized residual stream"** — the n index is factorized (n = 2^K), **not** n × channel groups | no | no | **partial** | **yes** | no | high (read README + code listing + abs) |
| **TBP-mHC** | [2605.21724](https://arxiv.org/abs/2605.21724) | transportation-polytope parameterization, (n−1)² d.o.f., exact and fully expressive; RTBP recursive variant | no | no | no | no | no | **SNIPPET only** |
| **go-mHC** | [2604.02309](https://arxiv.org/abs/2604.02309) | generalized orthostochastic matrices, direct parameterization | no | no | no | no | no | **SNIPPET only** |
| **sHC (Spectral-Sphere)** | [2603.20896](https://arxiv.org/abs/2603.20896), Liu/Zhang/Li, 2026-03-21 | replaces the Birkhoff polytope with a spectral-norm sphere to allow **negative entries** / "subtractive interactions" | no | no | no | no | no | medium (abs fetched) |
| **oHC** | [2609.02672](https://arxiv.org/abs/2609.02672), Guo et al., 2026-09-02 | H_res ∈ SO(4) via a pair of unit quaternions; zero extra params | no | no | no | no | no | medium (abs fetched) |
| **xHC** | [2607.14530](https://arxiv.org/abs/2607.14530), Zhang et al., 2026-07-16 | scales past the n=4 plateau: N=16 streams but only **k=4 updated per layer** (sparse stream update) + temporal feature augmentation | no | no | no | no | no | medium (abs fetched) |
| **TEMPER** | [2608.07851](https://arxiv.org/abs/2608.07851) | tensorizes the *generator* `W_res ∈ R^{n×d×n×n}` over (input-stream, **feature**, output-stream) modes with CP/Tucker. **Important:** the feature mode is *contracted away* (`Z_res = W_res ×_{1,2} Ĥ`), yielding one n×n matrix per token — **not** per-feature-group matrices | no | no | no | no | no | medium-high (HTML fetched, contraction confirmed) |
| **mHC-SSM** | [2605.08300](https://arxiv.org/abs/2605.08300) | mHC for SSMs; "stream-specialized adapters" = per-stream scaling through a shared bottleneck; simplex-constrained pre/post mixing | no | no | no | no | no | medium (abs fetched) |
| **Stream collapse analysis** | [2606.03483](https://arxiv.org/abs/2606.03483), ICML 2026 WSS | diagnoses dominant-stream collapse; mitigation is **symmetry breaking at stream init**, not multi-head | no | no | no | no | no | medium (abs fetched) |
| **How Does mHC Use Its Residual Streams?** | [2609.05309](https://arxiv.org/abs/2609.05309) | measures effective stream count / cross-stream weights / cosine sim in DeepSeek-V4-Flash; finds selective routing and near-identity mixing. Does **not** discuss finer granularity | no | no | no | no | no | medium (abs fetched) |
| **Ablate and Rescue** | OpenReview `Zu3B2TKOPV`, ICLR 2026 Sci4DL | causal stream-ablation framework, first open-source mHC LM | no | no | no | no | no | low (title + snippet) |
| **Multi-Gate Residuals** | [2605.23259](https://arxiv.org/abs/2605.23259), Zheng et al., 2026-05-22 | n = 4 or 8 streams; each stream gets **one scalar gate β per layer, broadcast over all D features** (s'_i = (1−β_i) s_i + β_i F(h)); softmax-normalized scoring, no Sinkhorn; cites HC, mHC and AttnRes (the abstract-only note "does not cite HC/mHC" was wrong). 0.12B–0.77B; beats Full AttnRes | no | no | no | no | no | high (HTML method read 2026-09-26) |
| **JPmHC** | [2602.18308](https://arxiv.org/abs/2602.18308) | replaces the Birkhoff (doubly stochastic) constraint with the orthogonal group O(n) via a Cayley transform; mixing is (H ⊗ I_p), i.e. **one n×n matrix broadcast over all p channels**; no grouped/per-head wording | no | no | no | no | no | high (HTML method read 2026-09-26) |
| **mHC for PEFT** | [2607.18130](https://arxiv.org/abs/2607.18130) | wraps frozen Transformer layers with mHC routing (h_pre, h_post ∈ R^n, H_res ∈ R^{n×n}, Sinkhorn) as a fine-tuning method; stream-level coefficients broadcast over channels; no grouped/per-head wording | no | no | no | no | no | high (HTML method read 2026-09-26) |
| **Motif 3** | [2608.09119](https://arxiv.org/abs/2608.09119) | 314B MoE, "modified mHC": n = 4, doubly stochastic kept, H_pre / H_post are 1×n (scalar per stream, shared over the hidden dimension); main change: post multiplier annealed 2 → 1 during pretraining against activation outliers | no | no | no | no | no | high (HTML read 2026-09-26) |
| **Hyperloop Transformers** | [2604.21254](https://arxiv.org/abs/2604.21254) | looped middle blocks with hyper-connections applied only at the loop level; H_res is a **sigmoid-diagonal** matrix instead of Sinkhorn (beats identity and Sinkhorn in their Table 6); per-stream scalars | no | no | no | no | no | high (HTML read 2026-09-26) |

### 2.6 DeepSeek's own production code (relevant, and a terminology trap)

**deepseek-ai/TileKernels**, `tile_kernels/modeling/mhc/functional.py` and
`tile_kernels/mhc/head_compute_mix_kernel.py`. **READ** (fetched raw files).

The file list alone (`pre_split_mixes`, `pre_apply_mix`, `sinkhorn_kernel`, `post_kernel`,
`head_compute_mix_kernel`, `multilayer_recompute_kernel`, `expand_kernel`) shows the production
decomposition. Two things matter here:

1. **DeepSeek calls the streams "heads".** Verbatim docstrings:

   ```python
   def expand_from_embedding(x: torch.Tensor, mhc_mult: int = 4) -> torch.Tensor:
       """Expand embedding from (..., H) to (..., mhc_mult, H).

       This is the entry point that converts a standard transformer embedding
       into the multi-head residual format required by MHC.
       ...
           mhc_mult: number of hyper-connection heads (currently only 4 is guaranteed to work)
       """
   ```

2. **The coefficients are confirmed scalar-per-stream-pair, shared across channels.** In `mhc_pre`:

   ```python
       fn: weight matrix of shape [mhc_mult * (mhc_mult + 2), mhc_mult * hidden_size]
       scale: sigmoid scaling of shape [3]
       base: mix biases of shape [mhc_mult * (mhc_mult + 2)]
   ```

   i.e. the coefficient generator reads all n·D channels and emits exactly n(n+2) scalars — the
   n×n H_res plus the two n-vectors H_pre and H_post. There is **no group axis anywhere**. And
   `head_compute_mix_kernel` is just the elementwise gate
   `output_mix[i,j] = sigmoid(input_mix[i,j] * mhc_scale[0] + mhc_base[j]) + pre_eps`
   over a `(num_tokens, mhc_mult)` tensor — "head" there means stream, again.

Also found: `deepseek-ai/DeepGEMM` has `csrc/apis/mega_mhc.hpp`,
`deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`, `tests/test_mega_mhc.py` (the fused
"Mega-mHC" inference kernel referenced in the V4.1-Flash report), and vLLM PR
[#56962](https://github.com/vllm-project/vllm/pull/56962) integrates it. I did not read the CUDA;
the Python-level API in TileKernels is the authoritative statement of the shapes.

**DeepSeek-V4.1-Flash report** — arXiv [2609.19969](https://arxiv.org/abs/2609.19969),
2026-09-17. SUMMARY. The abstract is about CSA2 + FP4 KV caching; mHC is inherited from V4 and is
not re-described there, so there is no V4.1-Flash-specific multi-head mHC.

---

## 3. Code repositories

### 3.1 `lucidrains/hyper-connections` — **the closest existing implementation**

Cloned and read. 188 stars. Files: `hyper_connections.py` (plain HC),
`manifold_constrained_hyper_connections.py` (mHC v1), `mHCv2.py` (current mHC),
`triton_sinkhorn.py`, `residuals.py`, `vit.py`, plus multi-branch / multi-input-stream variants.

#### Every option, and what it computes

| option | default | what it does |
|---|---|---|
| `num_residual_streams` (n) | — | the HC streams. |
| **`num_fracs` (m)** | 1 | **splits D into m channel groups and folds them into the mixing index.** See below. |
| **`num_input_views` (v)** | 1 | branch receives **v different pre-mixes** of the streams; output stacked on a new leading axis. Literally "multiple read heads" at the stream level. |
| **`num_dynamic_alpha_proposals` (p)** | 1 | computes p independent dynamic alphas, **Sinkhorns each one, then averages them**. A uniform mixture of p doubly stochastic matrices. |
| `residual_mix_constraint_fn` | Sinkhorn | pluggable manifold constraint (added "to allow researchers to explore more Hres constraint functions"). |
| `sinkhorn_iters` / `log_domain_sinkhorn` / `use_triton_sinkhorn` | 20 / False / False | Sinkhorn variants; `iters=0` short-circuits. |
| `mix_streams_before_norm` | False | a `nn.Conv2d(n, n, 1, bias=False)` across streams before the norm, `dirac_` init. Commented as *"equivalent to separable depthwise convs from yesteryears (with a norm in between)"*. |
| `add_stream_embed`, `add_attn_pool_reduce_stream`, `layer_index`, `channel_first`, `residual_transform`, `depth_residual_fn`, `add_branch_out_to_residual`, `disable`, `dropout`, `forward_method_names` | — | plumbing, not relevant to multi-head. |

#### What `num_fracs` actually computes (the key finding)

```python
num_residual_streams_fracs = num_residual_streams * num_fracs   # n*m
num_input_views_fracs      = num_input_views * num_fracs        # v*m

init_alpha0 = torch.zeros((num_residual_streams_fracs, num_input_views_fracs))
init_alpha0[init_residual_index, :] = 1.
self.static_alpha = nn.Parameter(cat((init_alpha0, torch.eye(num_residual_streams_fracs)), dim = 1))
```

so `static_alpha` is `(n·m) × (v·m + n·m)`. Then in `mHCv2.width_connection`:

```python
wc_weight = rearrange(wc_weight, '... s1 f2 mix -> ... (s1 f2) mix')
alpha = dynamic_alpha + self.static_alpha.float()
alpha_pre, alpha_residual = alpha[..., :self.num_input_views * self.num_fracs], \
                            alpha[...,  self.num_input_views * self.num_fracs:]
alpha_pre      = alpha_pre.sigmoid()
alpha_residual = self.residual_mix_constraint_fn(alpha_residual)     # <- Sinkhorn
...
alpha  = rearrange(alpha, '... (s f) t -> ... s f t', s = streams)
mix_h  = einsum(alpha, residuals.float(), '... s f tf, ... s f d -> ... tf d')
```

and `sinkhorn_knopps` begins with `assert log_alpha.shape[-2] == log_alpha.shape[-1]`.

I replicated the shape algebra numerically (pure numpy, n=4, m=4, v=1, d_frac=8):

```
static_alpha (16, 20) = (n*m, v*m + n*m) = (16, 20)
alpha_pre (1, 2, 3, 16, 4) | alpha_residual (1, 2, 3, 16, 16) | square (sinkhorn assert): True
branch_input (2, 3, 32) | residual out (2, 3, 4, 32)
```

**Conclusion: with `num_streams=4, num_fracs=4`, the Sinkhorn projection is applied to a 16×16
matrix that is doubly stochastic over the joint (stream, channel-group) index, and H_pre becomes
16×4, i.e. four distinct read vectors — one per channel group.** The `einsum` contracts over both
`s` and `f`, so channel groups genuinely exchange information.

This is a **superset** of the multi-head variant, not the multi-head variant: the block-diagonal
restriction (h independent n×n doubly stochastic blocks) is *not* implemented, and there is no
option for it. Note the difference in parameter/compute cost — one (nm)² Sinkhorn vs h independent
n² Sinkhorns.

H_post is also channel-group-aware but *not* manifold-constrained:

```python
self.static_beta    = nn.Parameter(torch.ones(num_residual_streams, num_fracs, 1))
self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim, num_fracs))
beta = (dynamic_beta + static_beta).sigmoid() * 2     # "for H_post manifold constraint"
output = einsum(branch_output, beta, 'b ... f1 d, b ... s f1 f2 -> b ... s f2 d')
```

so beta is `(streams, frac_in, frac_out)` — an m×m per-stream write map, sigmoid-gated, no
row/column normalization.

#### A real bug worth knowing about

`manifold_constrained_hyper_connections.py` (the v1 mHC file) splits pre/residual as

```python
alpha_pre, alpha_residual = alpha[..., :self.num_input_views], alpha[..., self.num_input_views:]
```

**missing the `* self.num_fracs`**. For m > 1 that leaves `alpha_residual` with last dim
`m(v+n) − v`, which is not square, so the Sinkhorn `assert` fires. `mHCv2.py` (commit `0204111`,
2026-01-16) has the corrected `:self.num_input_views * self.num_fracs`. **Only the mHCv2 path
supports fracs + mHC.**

#### Dated history (from `git log`)

| commit | date | message |
|---|---|---|
| `3e95af8` | 2025-06-17 | "add the frac-connections... **also generalize to allow for multiple streams and fractions**" ← the joint (n, m) index predates mHC by six months |
| `fb8fbb0` | 2026-01-01 | "incorporate findings from deepseek for stabilizing hyper connections... (manifold constrained matrices), sans all the efficiency stuff" |
| `4721f15` | **2026-01-10** | **"improvise multiple alpha proposals, and average the sinkhorn constrained Hres"** ← mechanism (d) |
| `f304a6b` | 2026-01-14 | "allow for researchers to explore more Hres constraint functions for mHC" |
| `0204111` | 2026-01-16 | "add a mHCv2 where the stream dimension is placed next to the feature dimension" |
| `7e681f9` | 2026-01-16 | log-space Sinkhorn (credited to @hjc18) |
| `15f9fe1` | 2026-01-17 | Triton Sinkhorn |
| `4775103` | 2026-02-04 | "allow for mixing the streams before the norm for mHCv2" |
| `d389b55` | 2025-01-30 | "allow a branch to receive **multiple combinations of the set of residual streams**" ← `num_input_views`, mechanism (b) |

There is **no commit, issue title or README line mentioning heads, groups or per-head mixing.**
The README's only note is: `get_init_and_expand_reduce_stream_functions(1, num_fracs = 4)
# also allows you to mix streams and fractions of feature dimension`.

### 3.2 `wdlctc/hyper-connection-factory` — the one repo that implements a multi-head routing variant

Cloned and read. Created 2026-09-25 (very recent; 5 commits). Eleven residual wirings behind one
interface, trained under an identical recipe. README variant table (verbatim excerpts):

| key | README description |
|---|---|
| `hc` | "n parallel residual streams; learned read (A_m), write (B) and stream mixing (A_r)" |
| `mhc` | "HC with σ-gated read/write and a **doubly-stochastic** (Sinkhorn) stream-mixing matrix; learned read-out head" |
| `frac` | "HC without widening: split d into m fractions" |
| `muddformer` | "**dynamic, per-token** dense weights, separately for Q, K, V and residual streams" |
| `attnres` | "**replace** the residual sum with softmax attention over previous sublayer outputs (depth attention)" |
| **`mhar`** | **"AttnRes with H independent depth softmaxes (one per channel group)"** |
| `dar` | "keep the residual stream; **add** depth-routed deltas; optional zero-init gate" |

`hcfactory/connections/depth_attention.py`, the multi-head routing core (verbatim):

```python
class DepthRouter(nn.Module):
    """One routing site: H-head softmax over sources, zero-init query."""

    def __init__(self, d_model: int, heads: int = 1):
        super().__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.query = nn.Parameter(torch.zeros(d_model))          # zero init -> uniform mix
        self.key_norm_weight = nn.Parameter(torch.ones(d_model)) # affine of RMSNorm(keys)

    def weights(self, keys):
        """keys: list of RMS-normalised sources [B,T,D] -> alpha [N,B,T,H]."""
        q = (self.query * self.key_norm_weight).view(self.heads, -1)
        logits = torch.stack(
            [torch.einsum("bthe,he->bth", k.unflatten(-1, (self.heads, -1)), q) for k in keys]
        )
        return logits.float().softmax(0)

    def forward(self, sources, keys):
        alpha = self.weights(keys).to(sources[0].dtype)
        H = self.heads
        out = 0
        for a, s in zip(alpha, sources):
            out = out + (a.unsqueeze(-1) * s.unflatten(-1, (H, -1))).flatten(-2)
        return out, alpha
```

This is the h-channel-group routing mechanism, in ~25 lines — but note it routes over the **depth
history** (softmax over previous sublayer outputs), not over n parallel streams, and the weights are
a softmax (row-stochastic), not doubly stochastic. **The repo's `hc` and `mhc` implementations have
no heads/groups option.** Its `mhc` does have a "HyperHead" — but that is a read-out
(`self.head_phi = nn.Parameter(torch.randn(n, n * d) * phi_std)`, a learned sigmoid read-out of the
streams before the final norm), not per-head mixing.

Results caveat from its own README: *"laptop-scale smoke comparison only (12 layers, d=256, 9.6M
non-embedding params, 12.3M FineWeb-Edu tokens, 1 seed, Apple M3 Pro). Seed-to-seed noise has not
been measured yet."* Do not treat its numbers as evidence for anything.

### 3.3 `wz1119/KromHC` (ICML 2026)

Cloned. `hyper_conn/{Kromhc,mhc,mhc_lite,hyper_connections,mhc_analysis}.py`. Docstring verbatim:

```python
class KromHC(Module):
    """
    Kronecker Low-Rank Hyper-Connections (KromHC)
    ...
    the Hres matrix is represented as a Kronecker product of small doubly stochastic
    ...
    - Instead of n! permutation combinations (factorial complexity)
    - Each U_k is a convex combination of i_k! permutation matrices
    - Hres = U_1 ⊗ U_2 ⊗ ... ⊗ U_K (Kronecker product)

    For n = 2^K (power of 2), each factor is 2x2 with only 2 permutations
    """
```

and helper functions `get_2x2_perm_matrices`, `factorize_into_twos(n)`, `get_all_permutations(n)`.
The code carries `num_fracs` through the constructor but comments `#### We assume num_fracs = 1,
num_input_views = 1 ###`.

**The Kronecker factorization is over the stream index n only (n = 2^K), not over channel groups.**
A Kronecker structure across (copies ⊗ channel groups) — mechanism (c) as defined in the question — is
**not** what KromHC does, despite the name match. Confidence: high (read the docstring and helpers).

### 3.4 `FFTYYY/mhc-lite`

Read `hyper_conn/mhc_lite.py` directly. The H_res construction (verbatim):

```python
        # ------
        # XXX MHC Lite impl.
        # H_res is from nC to n!
        # ------
        ...
        res_coeff = self.residual_scale * dynamic_residual + static_residual
        res_coeff = torch.softmax(res_coeff, dim = -1)
        alpha_residual = einsum(res_coeff, perms, '... r, r i j-> ... i j') # (..., s, s)
```

So H_res is a **per-token softmax mixture over all n! permutation matrices** — exactly mechanism
(d), for the stream index. `num_fracs` is plumbed through the constructor
(`init_alpha1 = torch.ones(len(perms) * num_fracs) * -8`) but the einsum `'... r, r i j -> ... i j'`
requires `r = n!`, so the frac path looks inconsistent for m > 1; I did **not** run it, and all the
shipped configs (`config/with_mhc_lite.py`) are the m=1 case.

### 3.5 `tokenbender/mHC-manifold-constrained-hyper-connections` (377 stars)

Cloned. A fork of lucidrains' code plus a nanoGPT harness, sweep infra, and value-residual configs.
Decisive line in `hyper_connections/hyper_connections_mhc.py`:

```python
        assert num_fracs == 1, "`num_fracs` must be 1 for mHC"
```

and in `hyper_connections/hyper_connections.py`:

```python
            assert num_fracs == 1, "mhc currently requires num_fracs = 1"
```

So this repo explicitly **disables** the frac path for mHC — the opposite of lucidrains' mHCv2.
Grepped the whole tree for `multi-head|multihead|per-head|head-wise|channel group|grouped`: the only
hits are `num_fracs` plumbing. `infra_scripts/open_challenges.md` is purely about sweep
orchestration (timeouts, retries, W&B gating) — no architectural ideas.

### 3.6 Other repos triaged (from `gh search repos "hyper-connections"`, 30 results)

None has a heads/groups option. Checked via targeted code search for `heads` / `num_fracs`:
`hjc18/mHC-transformers` (Qwen3/Llama + HC/mHC for HF Trainer; `heads` hits are attention heads,
0 `num_fracs`), `dhcode-cpp/mHC-pytorch` (0/0), `aHapBean/xHC` (0/0),
`yixuan/mHC-proj` (accelerated Birkhoff projection), `WithNucleusAI/mHC-triton`,
`ParadoxZW/mHC_Ascend`, `svdrecbd/mhc-mlx`, `enochyearn/mhc-vs-resnet-mlx`,
`smlab-niser/mhc-gnn`, `bassrehab/mhc-visualizer`, `rahvis/shc` ("Sparse Selective
Hyper-Connections"), `6zHAOyi/s2HC`, `brain-lab-research/hc-stream-collapse`,
`Meiyim/oHC-...`, `KennyStryker/manifold-constrained-hyper-connections`,
`Kareem404/hyper-connections`, `MarcoDotIO/mhc-deepseek-implementation`, `richardhahahaha/mHC`,
`Aaryyan777/mHC-Implementation`, `Lazarus-931/manifold-model`,
`aamir-gmail/MC-hyper-connections-and-Engrams`, `autumn-DL/HyperConnectionsModelWrapper`.

`Caiyun-AI/MUDDFormer` is the reference MUDD implementation (see §4).

### 3.7 modded-nanogpt

Searched the speedrun record history. I found **no** record or public attempt built on
hyper-connections or mHC; the architectural wins in that line are elsewhere (e.g. a per-(layer,head)
`tanh(α)` gate on exclusive self-attention around the 81.2 s record, May 2026). Confidence: medium
— this is a negative from search, not from reading the full record log.

---

## 4. Adjacent prior art for per-channel / grouped residual weighting

Granularity is the axis that matters here. From coarsest to finest:

| work | what is learned | granularity | relation to a multi-head mHC |
|---|---|---|---|
| **ReZero / SkipInit** | `x + α·f(x)` | **one scalar per layer** | the degenerate case. |
| **LayerScale** (CaiT) | `x + diag(λ)·f(x)` | **one scalar per channel** (diagonal) | already "fully elementwise" on the *write* path, but no cross-stream and no cross-depth mixing. Shows per-channel residual weights are standard practice and train fine. |
| **Highway Networks** | `T(x)⊙f(x) + (1−T(x))⊙x` | **per-channel, input-dependent gate** | the 1995/2015 ancestor of a dynamic per-channel H_post. Doubly stochastic in a trivial 2-source sense (weights sum to 1 per channel) — worth citing as the origin of "convex combination per channel". |
| **DenseFormer (DWA)** | static weighted average of all previous block outputs | **one scalar per (layer, source)** | depth mixing, no channel structure. |
| **MUDDFormer** ([2502.12170](https://arxiv.org/abs/2502.12170), ICML 2025) | dynamic per-token dense weights, **decoupled per input stream (Q, K, V, R)** | per-(token, source, stream), shared across channels within a stream | **this is mechanism (b)**: multiple read heads with different pre-mixes. It is the canonical citation for "different reads for different consumers". Not per-channel-group. |
| **LAuReL** | `x + αf(x) + xAB` (learned residual weights + low-rank skip) | scalar + low-rank matrix | a low-rank *channel-mixing* skip; different axis from head-grouping. |
| **Value residual learning** (ResFormer/SVFormer, [2410.17897](https://arxiv.org/abs/2410.17897)) | `V_n = λ_{n,1}V_1 + λ_{n,2}H_{n−1}W^V_n` | λ scalar per layer (Learnable-ResFormer variant) | a residual on the value path; the per-head version is what ExoFormer then ablates. |
| **ExoFormer** ([2601.08131](https://arxiv.org/abs/2601.08131)) | mixing coefficients for attention projections | **explicitly ablates scalar / headwise / elementwise** | the closest granularity study; finds elementwise coefficients spontaneously organise into head blocks. |
| **MHAR** ([2607.27230](https://arxiv.org/abs/2607.27230)) | depth-routing softmax | **H per-subspace heads, block-diagonal read** | mechanism (a), on depth attention. |
| **Frac-Connections** ([2503.14125](https://arxiv.org/abs/2503.14125)) | m×2m mixing over m fractions of d | **per channel group**, but the groups *are* the streams | supplies the channel-splitting half. |
| **VWN / GHC** ([2511.11238](https://arxiv.org/abs/2511.11238)) | `Å ∈ R^{n×m}`, `B ∈ R^{m×n}`, `Â ∈ R^{n×n}` | **per (segment, segment)**; m distinct reads/writes | the closest published unification of streams and channel groups. |
| **Sinkformers** (Sander, Ablin, Blondel, Peyré, [2110.11773](https://arxiv.org/abs/2110.11773), AISTATS 2022, PMLR 151:3515–3530) | doubly stochastic **attention** via Sinkhorn | attention matrices are per-head, so the Sinkhorn runs independently per head | the precedent that Sinkhorn normalization is applied *independently per head* — a direct analogy argument for a per-head H_res. Also relevant: LOTFormer ([2509.23436](https://arxiv.org/abs/2509.23436)), doubly-stochastic linear attention via low-rank OT. |
| **Sinkhorn networks / Gumbel-Sinkhorn** (Mena et al., ICLR 2018) | latent permutations on the Birkhoff polytope | — | the parameterization ancestor of mHC, mHC-lite, BE-HC and KromHC. |
| **DeepCrossAttention** ([2502.06785](https://arxiv.org/abs/2502.06785)) | learned depth-wise combination of previous layer outputs feeding Q/K/V | per-token, per-stream | adjacent to MUDD. SNIPPET-level for me. |

The generalisable observation: **per-channel residual scaling (LayerScale, Highway) and per-head
routing (Sinkformers, MHAR, ExoFormer) are both well established; what is missing is applying that
granularity to the doubly stochastic H_res of mHC.**

---

## 5. Naming collision — important

"Multi-head hyper-connections" already denotes **plain mHC with n streams** in at least two places:

1. **DeepSeek's own TileKernels**: *"converts a standard transformer embedding into the **multi-head
   residual format** required by MHC"*, and `mhc_mult: number of hyper-connection **heads**`.
2. `waefrebeorn/wubuwizard`, `include/wubu_mhc_mh.h` (READ, via the contents API):

   ```c
   /*
    * wubu_mhc_mh.h -- Multi-head Hyper-Connections (the 2512.24880 form).
    * ...
    * This module is the PAPER-form MULTI-HEAD variant: a GROUP of nh hidden
    * states h[0..nh-1] each of dim d, a learned nh x nh mixing matrix M
    * whose rows are softmax-constrained (the manifold constraint -- convex
    * combination), a gated write, ...
    */
   ```

   (Note in passing: this implementation only row-softmaxes M, so it is row-stochastic, not doubly
   stochastic — it is not a faithful mHC.)

Calling per-channel-group mixing "multi-head mHC" invites confusion. Something like
**"grouped-channel mHC"**, **"block-diagonal mHC"** or **"per-subspace mHC"** disambiguates, and
"per-subspace" matches MHAR's vocabulary.

---

## 6. Verdict

**Has multi-head mHC been proposed or tested? No — not under that name, and not as the specific
mechanism of h channel groups each carrying its own n×n doubly stochastic H_res.**

Evidence for the negative:
- The mHC paper's full text contains zero instances of per-head / per-channel / channel-wise /
  head-wise / block-diagonal, and never mentions Frac-Connections (verified by grep on the v2 HTML).
- No arXiv or OpenReview title in the HC/mHC family uses multi-head / grouped / channel-wise.
- `"multi-head mHC"` and `"num_hc_groups"` return **0 hits across all of GitHub**.
- Every published mHC variant I could verify (mHC-lite, BE-HC, KromHC, TBP-mHC, go-mHC, sHC, oHC,
  xHC, TEMPER, mHC-SSM) reparameterizes or constrains H_res **over the stream index only**.

Breaking it down by the categories of the question:

- **(a) h channel groups each with own mixing coefficients** — *not done for mHC's H_res.*
  Closest: (i) `lucidrains/hyper-connections` `num_fracs=m`, which Sinkhorns the **full joint
  (n·m)×(n·m)** matrix — a superset, not the block-diagonal restriction, and never described as
  multi-head; (ii) **VWN/GHC**, which gives per-channel-segment read and write vectors but keeps a
  single n×n skip mixing and imposes no manifold constraint; (iii) **MHAR**, which does exactly the
  block-diagonal per-subspace read but for depth attention rather than stream mixing.
- **(b) multiple read heads / different pre-mixes per input** — **done.** MUDDFormer (Q/K/V/R
  decoupled dense reads, ICML 2025), VWN/GHC (m read segments), and lucidrains'
  `num_input_views > 1`. This one is well covered by prior art.
- **(c) Kronecker across copies *and* channel groups** — *not done.* KromHC is Kronecker across
  **modes of the tensorized stream index only** (n = 2^K). TEMPER tensorizes the generator over a
  feature mode but **contracts it away**, yielding one n×n matrix per token. The copies ⊗ groups
  Kronecker is genuinely unclaimed.
- **(d) mixtures or products of several doubly stochastic matrices** — **done, and published.**
  mHC-lite (softmax mixture of all n! permutations, ICLR 2026 GRaM workshop) and BE-HC (Sci4DL 2026)
  are mixtures; KromHC is a Kronecker *product* of doubly stochastic factors that are themselves
  mixtures. And lucidrains' `num_dynamic_alpha_proposals = p` (commit `4721f15`, 2026-01-10) averages
  p Sinkhorned proposals — implemented, apparently never benchmarked publicly.
- **(e) anything named multi-head / grouped / channel-wise HC or mHC** — **no paper exists.** The
  name "multi-head hyper-connections" is, however, already used informally to mean ordinary mHC.

**The genuinely unclaimed combination** is: a doubly stochastic constraint applied **per channel
group** (h independent n×n Sinkhorns, block-diagonal in the joint index), as opposed to either the
one shared n×n of mHC or the one big (n·m)×(n·m) of lucidrains' frac path. Note that the
block-diagonal form is a strict *restriction* of what lucidrains already computes, so the interesting
question is not "can it be done" but "is the restriction better than the full joint matrix, and by
how much per unit of Sinkhorn cost" — that specific comparison is, as far as I can tell, nowhere in
the literature.

### Gaps in this search (be honest about these)

1. **ICML 2026 review bodies for mHC (`mDhyxu8WRb`) and KromHC (`TI7Q2o6EIa`) were not retrievable**
   through the public OpenReview API. A reviewer may well have suggested a per-head variant.
2. Several papers are **SNIPPET-only**: TBP-mHC (2605.21724), go-mHC (2604.02309), JPmHC (2602.18308),
   Motif 3 (2608.09119), mHC-PEFT (2607.18130). Verify before citing.
3. I read abstracts, not method sections, for xHC, oHC, sHC, MHAR, TEMPER, Multi-Gate Residuals,
   mHC-SSM and the stream-collapse paper. A grouped variant hiding in an appendix would have been
   missed.
4. **X/Twitter and Zhihu were searched only through the general web index**, not natively. The KromHC
   README links an X thread (`x.com/wuyang_zhou/status/2027727691634311646`) that I did not open.
5. No dedicated search of Discord/Slack research communities, where informal variant proposals live.

## 5. Re-check, 2026-09-27 (before the main sweep's results)

- arXiv API, `all:"hyper-connections" OR abs:"hyper-connection" OR abs:mHC`, newest first (40 results, raw query in LOG.md): the newest
  hyper-connection paper is still 2609.05309 (Sep 4). Nothing newer on HC.
- Three web searches ("multi-head" hyper-connections mHC channel groups; grouped / per-channel routing; head-wise / group-wise /
  channel-wise H_res) returned only papers already listed above.
- Eight HC-adjacent arXiv papers not triaged before, abstracts read (SUMMARY level: abstract only). None splits the channels into
  groups with their own connection coefficients:

| paper | what it does with HC |
|---|---|
| [2608.05549](https://arxiv.org/abs/2608.05549) mHC for speaker representation (Aug 2026) | plain mHC in ECAPA-TDNN / ResNet speaker models |
| [2608.13253](https://arxiv.org/abs/2608.13253) mHC semantic coding (Aug 2026) | mHC in a wireless semantic-communication transceiver ("channel" = radio channel) |
| [2606.26744](https://arxiv.org/abs/2606.26744) HyperDFlash (Jun 2026) | speculative decoding aligned to DeepSeek-V4's HC stream |
| [2606.07980](https://arxiv.org/abs/2606.07980) DeRes (Jun 2026) | CTR models: identity path + an adaptive path (AttnRes-like), not multi-stream mixing |
| [2605.15741](https://arxiv.org/abs/2605.15741) HyperDiT (May 2026) | "hyper-connected" cross-scale attention in pixel diffusion; not the HC residual |
| [2605.04421](https://arxiv.org/abs/2605.04421) FLUID (May 2026) | continuous-time "hyperconnected" sparse transformer; not the HC residual |
| [2605.06729](https://arxiv.org/abs/2605.06729) EΔ-MHC-Geo (May 2026) | mHC + Cayley/Householder orthogonal residual operators; stream-level |
| [2606.01495](https://arxiv.org/abs/2606.01495) CART (May 2026) | looped transformer; "multi-head" is its latent attention |

The verdict of the search does not change.

## 7. Re-check, 2026-09-29 (after the controls and the d768 check)

- arXiv API, the same query as in §5 (`all:"hyper-connections" OR abs:"hyper-connection" OR abs:mHC OR abs:"hyper connections"`,
  newest first, 25 results): two papers newer than the last check, one relevant.
- **[2609.33895](https://arxiv.org/abs/2609.33895) Residual-Stream Burden Shapes Representation Learning in Diffusion Transformers**
  (Liang, Kou, Xi, Singh, Zhou, Deng; submitted 2026-09-27). Read at the source (arXiv HTML, §5 and appendix E.3). FULL-TEXT level.
  - Pixel-space DiTs. Plain mHC (4 copied streams) already helps v-prediction a lot (FID 139.8 → 25.4). Their design, **SiHC**
    (Spatially Indexed Hyper-Connections), gives each of S streams one subpatch of the input and of the prediction (streams are not
    copies), keeps the carry as identity (H_mix = I, no mixing) and uses **static, feature-wise** read and write maps
    H_pre, H_post ∈ R^{C×S}: every channel c reads its own combination of the streams and writes its update with its own weights.
  - That is the **h = C limit of our design A**, for pre and post only, with a static predictor, no H_res and no manifold constraint.
  - Their ablation (Table 4 / 21, B-size, 200 epochs, one run per row, FID, v-prediction, 4 streams of 8×8 subpatches):
    static scalar maps + identity carry 11.50 → **feature-wise (per-channel) maps 10.10**. One run per row, no seeds reported:
    weak evidence, other domain, but it is the first published instance of per-channel HC read/write maps helping.
  - Why it may differ from our result (static h4/h8 ≈ static h1, and every structural head effect ≈ 0 at d384): their streams hold
    **different content** (different pixels), so a per-channel choice among streams has something to choose between. Ours are copies of
    one stream that stay fairly similar (probe: cosine 0.76–0.87 between copies).
  - Cost remark relevant to §6 of the report: static maps with identity carry compose across a stage, which lets them fuse the reads and
    writes and materialize the wide state only at stage boundaries (peak memory 30.2 → 19.8 GiB). Dynamic per-head maps, like ours,
    rule that out.
- [2609.32534](https://arxiv.org/abs/2609.32534) DepthBench (Sep 26): width-depth ratio sweep over 10 architectures; HC and Full AttnRes
  keep improving at deep-narrow shapes where Pre-LN variants do not. No head or channel-group variant. SUMMARY level (abstract).
- [2607.28097](https://arxiv.org/abs/2607.28097) (Jul 30): numerical divergence of MoE reduction orders in DeepSeek-V4-Flash; mentions
  post-mHC states only as a checkpoint. Not relevant.

The verdict changes in one detail: design A as such (h channel groups, each with its own pre, post and doubly stochastic res) is still
unpublished, but its h = C, static, pre/post-only limit now exists (SiHC), with a single-run gain in pixel diffusion.
