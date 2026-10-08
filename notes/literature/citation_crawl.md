# Prior-art check: has anyone proposed "multi-head" / grouped / channel-wise mHC?

Citation crawl performed **2026-09-26**. Raw JSON in `notes/literature/raw/`.

---

## 1. Method

### APIs and queries

**Semantic Scholar Graph API** (primary). One call per seed, paginated by `offset`,
`limit=1000`, with 429-backoff (sleep 12s × attempt, up to 8 attempts):

```
https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations
  ?fields=title,abstract,year,externalIds,publicationDate,url,venue,authors,citationCount
  &limit=1000&offset=<n>
```

Saved as `raw/s2_cites_<arxivid>_off<offset>.json`. Crawl driver: `scripts/crawl.sh`.

**Seeds (round 1)** — the four papers the question starts from:

| seed | arXiv | citations retrieved |
|---|---|---|
| Hyper-Connections (HC) | 2409.19606 | 87 |
| mHC | 2512.24880 | 82 |
| MUDDFormer | 2502.12170 | 33 |
| Frac-Connections | 2503.14125 | 7 |

**Round 2 (snowball).** Round 1 turned up a dense cluster of mHC-parameterization
papers. Crawling *their* citations found two mechanism papers that round 1 had
**missed** — `2608.07851` (TEMPER) and `2607.18130` — because Semantic Scholar's
citation edges for these preprints record only KromHC/mHC-lite, not mHC itself.
This is a real coverage gap in S2, so the crawl was extended to the whole HC family:

| seed | arXiv | citations retrieved |
|---|---|---|
| Attention Residuals (Kimi) | 2603.15031 | 55 |
| mHC-lite | 2601.05732 | 12 |
| KromHC | 2601.21579 | 9 |
| sHC | 2603.20896 | 3 |
| JPmHC | 2602.18308 | 3 |
| go-mHC | 2604.02309 | 2 |
| TBP-mHC | 2605.21724 | 2 |
| Stream Collapse | 2606.03483 | 2 |
| xHC | 2607.14530 | 1 |
| Multi-Head Attention Residuals | 2607.27230 | 1 |
| TEMPER | 2608.07851 | 0 |
| oHC | 2609.02672 | 0 |
| mHC-as-PEFT | 2607.18130 | 0 |
| Dual Attention Residuals | 2607.18730 | 0 |
| How Does mHC Use Its Streams | 2609.05309 | 0 |

**Totals: 19 seeds crawled, 299 citation rows, 186 unique citing papers, 7 with no
abstract (checked by title and venue by hand).**

### OpenAlex cross-check — attempted, unusable

All four primary seeds resolve in OpenAlex
(`HC` = W4403813222, `mHC` = W7117976675, `Frac` = W6892106537, `MUDD` = W4407759118),
but `https://api.openalex.org/works?filter=cites:<id>` returns
**`count: 0` for all four**. OpenAlex has the records but no inbound citation edges for
this (mostly arXiv-preprint) literature. **OpenAlex could not be used as a cross-check.**
Raw responses kept in `raw/openalex_seed_*.json` and `raw/openalex_cites_*.json` so the
negative result is reproducible.

### Independent arXiv API check

`https://export.arxiv.org/api/query` (note: `http://` 301-redirects and silently returns
0 bytes; `https://` required). Exact-phrase searches over the arXiv metadata index:

| query | total results |
|---|---|
| `all:"multi-head hyper-connections"` | **0** |
| `all:"grouped hyper-connections"` | **0** |
| `all:"channel-wise hyper-connections"` | **0** |
| `all:"multi-head mHC"` | **0** |
| `all:"per-head hyper-connections"` | **0** |
| `all:"head-wise hyper-connections"` | **0** |
| `abs:"hyper-connections" AND abs:"heads"` | 4 (all irrelevant: federated learning, Motif 3 report, CART, HyperDFlash) |

Caveat: arXiv's `all:` field searches title/abstract/authors/comments, **not** full
body text. So this establishes that no paper *announces* multi-head hyper-connections,
not that no paper mentions the phrase in a body paragraph.

### Screening rule

Two-stage, scripted then manual.

1. **Keyword screen** over title + abstract of all 186 unique papers
   (`scripts/screen.py`, `scripts/final_screen.py`). Two tiers:
   - *general*: head, group, channel, split, frac, partition, block, Kronecker,
     multiway, stream, Sinkhorn, doubly stochastic, Birkhoff, manifold, residual, mHC,
     hyper-connection, width, expansion, depth, skip
   - *strong* (weight ×3): hyper-connection, mHC, Sinkhorn, doubly stochastic, Birkhoff,
     frac-connection, multiway, residual stream, dense connection, Kronecker
2. **Targeted multi-head screen** with 13 regexes specifically for the axis in question
   (`multi-?head`, `per-head`, `head-?wise`, `channel-?wise`, `channel group`,
   `per-channel`, `block-?diagonal`, `kronecker`, `grouped`, `partition`, `multi-?way`,
   `per-stream|stream-specific`, `split`). **25 papers matched.**
3. Every paper scoring ≥ 6, every one of the 25 targeted matches, and every paper whose
   *title* names a connection mechanism was read at abstract level by hand.
4. Everything surviving that was fetched as full text (`arxiv.org/html/<id>`, converted
   with `scripts/h2t.py`, which preserves LaTeX from MathML `alttext`) and the method
   and experiment sections were read.

**Inclusion rule for the table in §2:** the paper must *modify the connection
mechanism*. Papers that merely apply mHC in a new domain, cite it as related work, or
analyse a model that happens to use it are excluded unless they change the
parameterization. Two analysis papers are included because they bear directly on the
verdict.

---

## 2. HC / mHC follow-ups that modify the connection mechanism

Relevance column: **direct** = splits channels/heads or instantiates several
read/write heads in the connection itself; **partial** = adjacent structure (channel
groups used as streams, Kronecker/tensor factorization, multiple write components,
per-stream capacity); **none** = modifies the mechanism on a different axis.

| Title | arXiv | Date | One-line summary | Multi-head relevance |
|---|---|---|---|---|
| Hyper-Connections | 2409.19606 | 2024-09-29 | Seed: widens residual stream to n copies, learns H_pre/H_post/H_res. | none (baseline) |
| MUDDFormer | 2502.12170 | 2025-02-13 | Dense depth connections with **four separate read heads** for Q/K/V/R; calls it "depth-wise multi(4)-head attention". | **direct** (read heads, by role, not channels) |
| Frac-Connections | 2503.14125 | 2025-03-18 | **Splits** D into m chunks instead of replicating; m×m mixing over the chunks. | **partial** (channel groups as streams) |
| Virtual Width Networks | 2511.11238 | 2025-11-14 | Decouples representational width from backbone width; 8× embedding expansion. | none |
| mHC | 2512.24880 | 2025-12-31 | Seed: projects H_res onto Birkhoff polytope via Sinkhorn-Knopp; sigmoid H_pre/H_post; n=4 in DeepSeek-V4. | none (baseline) |
| mHC-GNN | 2601.02451 | 2026-01-05 | Ports mHC to GNNs; over-smoothing bounds; 2–128 layers. | none |
| mHC-lite | 2601.05732 | 2026-01-09 | Exact double stochasticity as convex combination of the n! permutation matrices; drops Sinkhorn. | none |
| White-Box mHC (ES-mHC) | 2601.15757 | 2026-01-22 | Uses **spectral band groups** as the mHC streams; interprets H_res spatially. | **partial** (input channel groups as streams) |
| KromHC | 2601.21579 | 2026-01-29 | H_res = **Kronecker product** of small doubly stochastic factors, one per mode of the tensorized *stream* axis; channel mode is the identity. | **partial** (Kronecker over n, not D) |
| JPmHC | 2602.18308 | 2026-02-20 | Constrains the mixer on operator-norm-bounded manifolds (bistochastic/Stiefel/Grassmann); free-probability Jacobian analysis. | none |
| Sparse Selective HC (SHC) | — (IEEE SoutheastCon, DOI 10.1109/SoutheastCon63549.2026.11476522) | 2026-02-20 | Birkhoff–von Neumann sparse convex combination + Cayley routing + rank-r JL factorization + SSM distillation. | none (abstract only) |
| mHC-HSI | 2603.03418 | 2026-03-03 | **Divides spectral bands into physical groups used as the mHC streams**; H_res as per-pixel soft cluster maps. | **partial** (input channel groups as streams) |
| Attention Residuals (AttnRes) | 2603.15031 | 2026-03-16 | Replaces additive residual with softmax attention over past sublayer outputs; **one query shared across the full width**; Block variant partitions *layers*. | none (parent of MHAR) |
| Beyond the Birkhoff Polytope (sHC) | 2603.20896 | 2026-03-21 | Moves H_res from the polytope to a spectral-norm sphere in the affine subspace; allows negative entries. | none |
| go-mHC | 2604.02309 | 2026-04-02 | Exact Birkhoff parameterization via generalized orthostochastic matrices, O(d³); proves Kronecker factorizations are spectrally limited. | **partial** (composes with Kronecker over n) |
| Hyperloop Transformers | 2604.21254 | 2026-04-23 | Looped middle block augmented with HC applied once per loop; ~50% fewer parameters. | none |
| Can an MLP Absorb Its Own Skip Connection? | 2604.23705 | 2026-04-26 | Theory: invertible-linear skip branches (incl. HC/mHC) reduce to the identity-skip case; absorption generically impossible. | none |
| mHC-SSM | 2605.08300 | 2026-05-08 | Static mHC around an SSM block plus **stream-specialized adapters** (shared bottleneck, per-stream scaling). | **partial** (per-stream capacity) |
| Delta Attention Residuals | 2605.18855 | 2026-05-13 | Routes over per-sublayer deltas instead of cumulative states; fixes routing collapse. | none |
| TBP-mHC / RTBP | 2605.21724 | 2026-05-20 | Exactly doubly stochastic H_res via transportation polytopes with (n−1)² d.o.f.; full Birkhoff expressivity. | **partial** (discusses Kronecker submanifold) |
| Multi-Gate Residuals (MGR) | 2605.23259 | 2026-05-22 | Scoring + gating to keep multi-stream context without AttnRes's communication cost; attention pooling to read. | none |
| Accelerating Birkhoff Projection | 2606.07574 | 2026-05-26 | Newton dual solver + implicit differentiation + warp-level CUDA kernel for 4×4 Birkhoff projection; >20× speedup. | none |
| Analyzing Stream Collapse in HC | 2606.03483 | 2026-06-02 | Diagnoses dominant-stream collapse and near-identity mixing; mitigates by breaking stream symmetry at init. | none (**motivating**) |
| HAARES | 2606.06564 | 2026-06-04 | Adds a **half-split** detail basis (first-half minus second-half sublayer updates) to block residual routing — a split over *depth*, not channels. | none |
| Variable-Width Transformers | 2606.18246 | 2026-06-16 | Non-uniform width across depth with parameter-free residual resizing. | none |
| Low-Rank Attention Residuals | 2607.09694 | 2026-06-19 | Uses the **last r of d channels** of each source as the routing key; one shared route. | **partial** (channel slice for routing) |
| xHC | 2607.14530 | 2026-07-16 | Sparse k-of-N streams (N=16, k=4) + temporal write-back augmentation; **H_post becomes k×K_r**, i.e. several write components. | **partial** (multiple write heads, temporal) |
| mHC for PEFT | 2607.18130 | 2026-07-20 | mHC as a finetuning axis around frozen OLMo-2; identity H_res often best in finetuning. | none |
| Dual Attention Residuals (DAR) | 2607.18730 | 2026-07-21 | Two streams, each scoring its history from the **other** stream's states; 2×2 exactly doubly stochastic gated writes. | **partial** (per-stream read heads) |
| Multi-Head Attention Residuals (MHAR) | 2607.27230 | 2026-07-22 | **Splits the routing query into H per-subspace heads with H independent softmaxes**; block-diagonal depth read; zero added parameters. | **direct** (channel-group read heads) |
| Role-Decoupled AttnRes | 2608.01075 | 2026-08-02 | **Two read routes**: one shared by Q and K, an independent one for V. | **direct** (read heads, by role) |
| TEMPER | 2608.07851 | 2026-08-08 | CP/Tucker/TT factorization of the routing *generators* over the **input-stream, feature, and output-stream modes**; routing interface unchanged. | **partial** (feature mode in generator only) |
| WhiteMatter | 2608.18486 | 2026-08-19 | All-to-all cross-layer connections by mixing L layer states into k cached KV channels. | none |
| oHC | 2609.02672 | 2026-09-02 | Constrains H_res to SO(n), closed-form via two unit quaternions at n=4; proves mHC contracts inter-stream differences. | none |
| How Does mHC Use Its Residual Streams? | 2609.05309 | 2026-09-04 | Measures DeepSeek-V4-Flash: each site effectively uses ~2 of 4 streams; late mixing removable at 1.9% ppl cost. | none (**motivating**) |

---

## 3. Detailed notes on the partially and directly relevant papers

Ordered by closeness to the question.

### 3.1 Multi-Head Attention Residuals (MHAR) — **the closest work**

- **Citation:** *Multi-Head Attention Residuals*, arXiv:2607.27230, 2026-07-22.
  <https://arxiv.org/abs/2607.27230>
- **Confidence: read full text** (method §2, experiments §3, related work App. A).

**Parameterization.** Builds on AttnRes, which replaces the additive residual with a
softmax read over the ordered list of sources
$\mathcal{S}=(\mathbf{s}_0,\dots,\mathbf{s}_{N-1})$, $N=2L+1$ (token embedding plus every
sublayer's raw output). AttnRes uses one learned pseudo-query $\mathbf{q}\in\mathbb{R}^d$
per site:

$$\alpha_i = \frac{\exp(\mathbf{q}^\top \mathrm{RMSNorm}(\mathbf{s}_i))}{\sum_j \exp(\mathbf{q}^\top \mathrm{RMSNorm}(\mathbf{s}_j))}, \qquad \tilde{\mathbf{h}} = \sum_i \alpha_i \mathbf{s}_i .$$

MHAR partitions the $d$ coordinates into $H$ **contiguous heads of width $d/H$**, gives
each its own query $\mathbf{q}_h\in\mathbb{R}^{d/H}$, and routes each independently:

$$\alpha^{(h)}_i = \mathrm{softmax}_i\!\left(\mathbf{q}_h^\top \mathrm{RMSNorm}(\mathbf{s}_i)_{[h]}\right), \qquad \tilde{\mathbf{h}}_{[h]} = \sum_i \alpha^{(h)}_i \mathbf{s}_{i,[h]}$$

and concatenates the $H$ slices back to width $d$. Each head scores **only its own slice**
of each source and mixes only that slice, so the $H$ softmaxes are fully independent and
the depth read is **block-diagonal** over channel groups.

**Channel/head split: yes — this is exactly the channel-group split.** The $H$ queries form
an $(H, d/H)$ tensor with exactly $d$ parameters, so $H$ is a *reshape*: zero added
parameters, negligible FLOPs, and $H=1$ recovers AttnRes exactly.

**But it is not applied to hyper-connections.** There is no $n$-copy residual stream, no
$H_{\text{res}}$ and no $H_{\text{post}}$. The paper is explicit in App. A:
"these enrich the connection topology between layers, whereas we keep a single stream and
enrich the routing that reads the depth history — our heads act on routing distributions,
not on separate streams."

**Experiments.** From-scratch decoder-only Transformers, Qwen3-style, on a deduplicated
Nemotron-based anneal corpus (`anneal_pt_v3`), 20K steps, AdamW, cosine schedule, bf16.
Three scales: 100M ($d$=512, $L$=12), 350M ($d$=1024, $L$=24), 1B ($d$=1280, $L$=36).
Validation loss (tail-mean over last 11 evals, $\Delta$ vs baseline):

| scale | baseline | HC (n=4) | AttnRes (H=1) | MHAR (H=8) | MHAR (H=16) |
|---|---|---|---|---|---|
| 100M | 3.031 | 2.999 (−0.032) | 2.970 (−0.060) | **2.969 (−0.061)** | 2.996 (−0.035) |
| 350M | 2.997 | 2.931 (−0.066) | 2.876 (−0.121) | **2.848 (−0.149)** | 2.895 (−0.102) |
| 1B | 2.894 | 2.807 (−0.086) | 2.759 (−0.134) | **2.754 (−0.140)** | 2.825 (−0.069) |

Key findings directly relevant to us:

- **Validation loss is U-shaped in $H$**, with a flat optimum at $H=4$–$8$ across scales.
  $H$ is a real design axis, not a free knob.
- **Over-splitting hurts, and the penalty grows with scale**: $H$=16 gives back a third of
  the gain at 350M and half at 1B (+0.026 / +0.047 / +0.071 over the $H$=4–8 optimum).
  Their explanation: each head then routes a subspace narrower than what one KV group
  consumes, so coherent features are split across independently-routed slices.
- **The stated mechanism is learned subspace disagreement**: different feature subspaces
  want to read different layers, and disagreement grows with model width. A direct probe of
  the trained queries is offered as confirmation.
- MHAR beats HC (n=4) at every scale (by 0.029 / 0.083 / 0.053), which they read as
  "enriching the depth routing of a single stream is consistently stronger than enriching
  the connection topology across streams."
- Fused Triton routing kernels: routing throughput 0.55–0.88× of baseline (from 0.2–0.5×).
- 8B mid-training via an identity-preserving conversion with delta attention residuals:
  +3.2 GSM8K, +3.1 GPQA.

**Code:** pseudocode in Fig. 3; fused Triton kernels described in App. E. No repository URL
found in the HTML.

### 3.2 MUDDFormer — "depth-wise multi(4)-head attention"

- **Citation:** *MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway
  Dynamic Dense Connections*, arXiv:2502.12170, 2025-02-13.
  <https://arxiv.org/abs/2502.12170>
- **Confidence: read full text** (method §2.3 and experiments).

**Parameterization.** Decouples a Transformer block's single input into four streams and
instantiates **four separate depth-aggregate (DA) modules**, one per stream:
$\mathrm{B}(X)\to\mathrm{B}'(X^Q,X^K,X^V,X^R)$, each with its own dynamic, position-dependent
dense connection weights over all preceding layer outputs. Connection weights are generated
per sequence position and per stream, unlike DenseFormer's static shared weights.

The paper itself frames this as the multi-head idea: "MUDD connections can be seen as
**depth-wise multi(4)-head attention**" and "the cross-layer communication bandwidth is
expanded far beyond the restriction of the residual stream."

**Channel/head split: no — the heads are *roles*, not channel groups.** Each of the four DA
modules reads the full width; the four reads differ in which of Q/K/V/R they feed, not in
which channels they cover. This is category 2 of the research question (multiple read
heads / different pre-mixes for different layer inputs), realised on dense depth
connections rather than on HC's $n$ copies.

**Experiments.** Decoder-only, Pile. MUDDFormer matches Transformers trained with
~1.8–2.4× compute. MUDDPythia-2.8B matches Pythia-6.9B on pretraining perplexity and
downstream tasks and rivals Pythia-12B five-shot, adding **0.23% parameters and 0.4%
computation**. Gains shown from 405M to 1.4B.

**Code:** <https://github.com/Caiyun-AI/MUDDFormer> (JAX and PyTorch, plus pretrained models).

### 3.3 Role-Decoupled Attention Residuals (RD-AttnRes)

- **Citation:** *Role-Decoupled Attention Residuals: Separating Matching and Content
  Retrieval Across Depth*, arXiv:2608.01075, 2026-08-02.
  <https://arxiv.org/abs/2608.01075>
- **Confidence: abstract only** (full text not read).

Minimal extension of Block AttnRes: **shares one depth route between queries and keys, and
learns an independent value route** over the same residual sources. The motivation is that
Q/K decide *where* attention matches while V decides *what* is retrieved, so they need not
read from the same depth. Tying the two routing queries recovers the parent architecture;
decoupling adds one model-width vector per layer and no extra token-to-token attention.

**Channel/head split: no** — two read heads by role, same as MUDD's multiway idea, now on
AttnRes. Directly relevant to category 2, not to category 1.

**Experiments.** Frozen paired pretraining on FineWeb-Edu, 5 matched seeds, 120M and 343M
parameters, 2.0B-token budget. Improves validation NLL in **all 10** matched comparisons;
mean reductions 0.0301 and 0.0247 (perplexity −2.97% and −2.43%). Cites MHAR.

### 3.4 Frac-Connections

- **Citation:** *Frac-Connections: Fractional Extension of Hyper-Connections*,
  arXiv:2503.14125, 2025-03-18. <https://arxiv.org/abs/2503.14125>
- **Confidence: read full text** (method §4).

**Parameterization.** Instead of replicating $\mathbf{h}\in\mathbb{R}^d$ into $n$ copies,
Frac-Connections **splits** it:

$$\mathbf{H} = (\mathbf{h}_1\ \mathbf{h}_2\ \cdots\ \mathbf{h}_m)^\intercal = \texttt{Reshape}(\mathbf{h}, (m, d/m)), \qquad \mathbf{h}_i \in \mathbb{R}^{d/m},$$

with $m = 1/n$ the frac-rate (number of partitions), and then runs the HC machinery over
those $m$ chunks:

$$\mathcal{FC} = \begin{pmatrix}\mathbf{0}_{1\times m} & \mathbf{B}\\ \mathbf{Y} & \mathbf{A}\end{pmatrix}, \qquad \mathbf{H}^k = \mathbf{B}^{k\intercal}\mathcal{T}^k(\mathbf{Y}^{k\intercal}\mathbf{H}^{k-1}) + \mathbf{A}^{k\intercal}\mathbf{H}^{k-1}.$$

Dynamic variant predicts $\mathcal{B},\mathcal{Y},\mathcal{A}$ per token with zero-initialized
projections, `tanh`, and a small learnable scale (following Dynamic HC).

**Channel/head split: yes, but with one shared mixing matrix, not $h$ independent ones.**
The $m$ channel chunks *replace* the $n$ stream copies. Because $\mathbf{Y}$ is $m\times m$,
chunk $i$ of the layer input is a distinct mix $\sum_j \gamma_{j,i}\mathbf{h}_j$ of the
chunks, so the read is channel-group-structured — but there is exactly one $\mathcal{FC}$
per site governing all groups, which is the *opposite* of giving each group its own
$H_{\text{pre}}/H_{\text{post}}/H_{\text{res}}$. No manifold constraint (predates mHC).

**Experiments.** OLMo2-style dense models and OLMoE MoE models; ablations on frac-rate;
full tables in appendices §8–§10 of that paper. Motivation is avoiding HC's $n\times$
memory overhead.

**Code:** PyTorch implementation inlined in §7 of the paper.

### 3.5 KromHC

- **Citation:** *KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product
  Residual Matrices*, arXiv:2601.21579, 2026-01-29. <https://arxiv.org/abs/2601.21579>
- **Confidence: read full text** (method §4, experiments §5, App. K).

**Parameterization.** Keeps $H^{\text{pre}}_l$ and $H^{\text{post}}_l$ exactly as in mHC and
replaces only the residual mixing. With $n=\prod_{k=1}^K i_k$, tensorize the stream into
$\mathcal{X}_l \in \mathbb{R}^{i_1\times\cdots\times i_K\times C}$ and mix along each of the
first $K$ modes with a learned doubly stochastic $\mathbf{U}^k_l\in\mathbb{R}^{i_k\times i_k}$:

$$H^{\text{res}}_l\mathbf{X}_l = \mathrm{mat}\big(\mathcal{X}_l \times_1 \mathbf{U}^1_l \times_2 \cdots \times_K \mathbf{U}^K_l \times_{K+1} \mathbf{I}_{C\times C}\big) = \Big(\bigotimes_{k=K}^{1}\mathbf{U}^k_l\Big)\mathbf{X}_l .$$

Each factor is a convex combination of the $i_k!$ permutation matrices of its size, with
**its own token-dependent coefficient predictor**:

$$\mathbf{a}^k_l = \mathrm{Softmax}\!\left(\alpha^{\text{res}}_l \mathbf{x}'_l \mathbf{W}^{\text{res},k}_l + \mathbf{b}^{\text{res},k}_l\right), \qquad \mathbf{U}^k_l = \sum_{m=1}^{i_k!} \mathbf{a}^k_l(m)\mathbf{P}_m .$$

Kronecker closure of the Birkhoff polytope (their Thm 4.2) makes $H^{\text{res}}$ exactly
doubly stochastic, so norm preservation and compositional closure are retained. Parameter
complexity drops from mHC's $O(n^3C)$ and mHC-lite's $O(nC\cdot n!)$ to $O(n^2C)$.

**Channel/head split: no — and decisively so.** The Kronecker factorization is over the
**stream index $n$ only**; the channel mode is multiplied by the explicit identity
$\mathbf{I}_{C\times C}$. $\mathbf{W}^{\text{pre}}_l, \mathbf{W}^{\text{post}}_l \in
\mathbb{R}^{nC\times n}$ produce $n$-vectors shared across all $C$ channels.

**Partial relevance** on category 4 instead: KromHC does combine **$K$ separate coefficient
predictors via a product of doubly stochastic matrices** — the "several predictors
combined" idea, applied across the stream axis.

**Experiments.** Nanochat backbone, FineWeb-Edu, token:parameter ≈ 20, 4–8 RTX PRO 6000.
Two scales: ~60M ($D$=6 blocks) and ~186M ($D$=12), $n=4$.

| method | ΔParams (K), D=12 | train loss | val BPB | CORE |
|---|---|---|---|---|
| Residual | — | 2.971 | 0.864 | 14.774 |
| mHC | 1844 | 2.964 | 0.861 | 16.023 |
| mHC-lite | 2433 | 2.972 | 0.864 | 13.217 |
| KromHC | **959** | 2.966 | 0.862 | **16.872** |

Commonsense-reasoning averages 41.1% ($D$=6) and 47.7% ($D$=12), best in both. Scaling
$n\in\{4,8,16\}$ improves loss and BPB monotonically (ΔParams 0.96M / 2.67M / 11.37M).
Lowest gradient norms of the three mHC variants. App. K: the more restricted $2\times2\times2$
factorization at $n$=8 slightly *beats* $4\times2$ (−0.002 train loss, −0.003 BPB).
Ablation: sharing $\alpha^{\text{res}}_l$ across factors beats per-factor $\alpha^{\text{res},k}_l$.

**Code:** <https://github.com/wz1119/KromHC>

### 3.6 TEMPER

- **Citation:** *TEMPER: Tensorized Efficient Manifold-constrained Parameterization for
  Expressive Residual Routing*, arXiv:2608.07851, 2026-08-08.
  <https://arxiv.org/abs/2608.07851>
- **Confidence: read full text** (method §4, experiments §5).

**Parameterization.** Keeps $\hat{\mathbf{H}}=\mathrm{RMSNorm}(\mathbf{H})\in\mathbb{R}^{n\times d}$
un-flattened and views mHC's dense coefficient predictors as tensors
$\mathcal{W}^{\text{pre}},\mathcal{W}^{\text{post}}\in\mathbb{R}^{n\times d\times n}$ and
$\mathcal{W}^{\text{res}}\in\mathbb{R}^{n\times d\times n\times n}$ over the
**input-stream, feature, and output-stream modes**, then replaces each with a CP, Tucker,
or Tensor-Train factorization. Logits contract the stream and feature modes:

$$\mathcal{Z}^{\text{res}}_{i,j} = \sum_{s=1}^{n}\sum_{c=1}^{d} \hat{H}_{s,c}\,\mathcal{W}^{\text{res}}_{s,c,i,j}, \qquad \mathbf{A}^{\text{res}} = \mathrm{SK}\big(\exp(\alpha^{\text{res}}\mathcal{Z}^{\text{res}} + \mathrm{mat}(\mathbf{b}^{\text{res}}))\big).$$

**Channel/head split: no.** This is the subtlest near-miss in the literature and worth
stating precisely: the feature mode $d$ appears only on the **input side of the coefficient
predictor**. The *outputs* are still $\mathbf{a}^{\text{pre}}\in\mathbb{R}^{n}$,
$\mathbf{A}^{\text{res}}\in\mathbb{R}^{n\times n}$, $\mathbf{a}^{\text{post}}\in\mathbb{R}^{n}$
— one scalar per stream, shared across all $d$ channels. The paper says so explicitly: the
factorization "preserves token-dependent manifold-constrained routing interface", and
Remark 1 notes matricizing the tensors recovers the dense mHC generators exactly. So TEMPER
*low-rank-compresses* the predictor over the feature axis rather than *splitting* the
routing over it.

**Experiments.** nanochat backbone, $D$=12, $d$=768, 6 attention heads, context 2048,
24 residual modules; ClimbMix-400B, 32,768-token BPE, 4×A100-80GB. KromHC-style init.
At $n$=8, TEMPER-Tucker raises CORE from 0.195 (KromHC) to **0.206** using 1.93M extra
parameters vs KromHC's 3.39M and mHC's 11.95M (**~84% fewer than mHC**). mHC-lite OOMs at
$n$=8. Commonsense averages 50.0% ($n$=4) and 51.3% ($n$=8), +1.5% over the best baseline
at $n$=8. TEMPER-CP best on the LM/BBH suite (27.1% / 27.3%). Ablations: frozen Tucker core
is worse; TT worse than Tucker; tensorizing all three generators beats tensorizing
$\mathbf{W}^{\text{res}}$ alone (CORE 0.176→0.192 at $n$=4); CP rank barely matters
(0.184–0.189 for $r\in\{2,...,32\}$, best at $r$=2), suggesting the dense generator is
over-parameterized.

**Code:** none found.

### 3.7 xHC — multiple write-back components

- **Citation:** *xHC: Expanded Hyper-Connections*, arXiv:2607.14530, 2026-07-16.
  <https://arxiv.org/abs/2607.14530>
- **Confidence: read full text** (method §3, experiments §4).

**Parameterization.** Two changes to push $N$ past 4. (i) **Temporal feature augmentation**:
the sublayer output is expanded into $K_r=r+1$ components via causal per-channel depthwise
convolutions (kernels {4,8,12}, $r$=3), orthogonalized against each other by modified
Gram-Schmidt, applied only after MLP/MoE sublayers. (ii) **Sparse residual streams**: a
sigmoid router selects $k$=4 active of $N$=16 streams; $H^{\text{res}}$ and $H^{\text{post}}$
are generated from the active state only.

$$\mathcal{H}^{\text{pre}}_l = \sigma(\cdot),\ W^{\text{pre}}\in\mathbb{R}^{NC\times N}; \quad \mathcal{H}^{\text{res}}_l = \mathrm{SK}(\exp(\cdot)),\ W^{\text{res}}\in\mathbb{R}^{kC\times k^2}; \quad \mathcal{H}^{\text{post}}_l = 2\sigma(\cdot),\ W^{\text{post}}\in\mathbb{R}^{kC\times kK_r}.$$

**Channel/head split: no.** The interesting part is that $\mathcal{H}^{\text{post}}_l$ becomes
a $k\times K_r$ **matrix** rather than a $k$-vector — several write heads, each writing a
different signal into the streams. But the $K_r$ components are *temporal* variants
(different conv kernel widths over the time axis), not channel groups. The per-channel
language in the paper refers only to depthwise convolution.

**Experiments.** DeepSeekMoE-style MoE with GQA, 144 experts top-8, context 8192, multilingual
+ code + math mix. 18B-total/1.7B-active and 28B-total/2.7B-active headline models; 10B for
ablations, 2.5B for $N$-sweeps. On the 18B MoE, xHC ($N$=16, $k$=4) improves the average
downstream score by **4.0 points over mHC** ($N$=4). Scaling laws: vanilla needs 1.50× and
mHC 1.19× the compute of xHC for equal loss. xHC-Flash cuts per-sublayer memory traffic from
73.5C to 40C (mHC at $N$=4 is 34C). Compatible with Muon.

**Code:** none found.

### 3.8 Dual Attention Residuals (DAR) — per-stream read heads

- **Citation:** *Dual Attention Residuals*, arXiv:2607.18730, 2026-07-21.
  <https://arxiv.org/abs/2607.18730>
- **Confidence: read full text partially** (abstract, intro, method outline, ablation
  descriptions; did not read all result tables).

Combines multi-stream residual pathways with depth-wise retrieval. For each target stream,
DAR computes depth weights from the normalized states of the **opposite** stream and applies
them to values from the target stream's own history ("reciprocal cross-stream addressing").
Writes are constrained and gated; for two streams their parameterization "directly guarantees
a doubly stochastic matrix: each row and column sums to one, so no iterative normalization
is required" — i.e. an exact $2\times2$ Birkhoff point.

**Why it matters here:** the ablation **DAR-SelfKV** is precisely "each stream retrieves
independently from its own history" — *independent per-stream read heads*, which is the
per-stream (not per-channel) version of the multi-head idea, and it is reported as *worse*
than reciprocal cross-stream addressing. The intro states the motivation: "assigning an
independent retriever to each stream still prevents one trajectory from influencing depth
selection in another."

**Channel/head split: no** — heads are per *stream*, not per channel group.

**Experiments.** Dense models 0.1B–1B plus a 7B sparse MoE; consistent validation-loss
improvements over standard residual Transformers and AttnRes. Routing ablations rule out
"an extra stream or value projection alone". Cross-stream CKA shows the two streams stay
distinct.

### 3.9 mHC-HSI and White-Box mHC (ES-mHC) — channel groups as streams

- **Citations:** *mHC-HSI: Clustering-Guided Hyper-Connection Mamba for Hyperspectral Image
  Classification*, arXiv:2603.03418, 2026-03-03, <https://arxiv.org/abs/2603.03418>;
  *White-Box mHC: Electromagnetic Spectrum-Aware and Interpretable Stream Interactions for
  Hyperspectral Image Classification*, arXiv:2601.15757, 2026-01-22,
  <https://arxiv.org/abs/2601.15757>. Same group (GSIL, University of Calgary).
- **Confidence: read full text** (method sections of both).

**Parameterization.** Both keep mHC's equation and Sinkhorn-projected $H^{\text{res}}$ but
**replace the replicate-the-input step with a channel split**: "Instead of duplicating the
input feature to build multi-stream representations HC and mHC did, ... splitting the
original HSI cube into non-overlapping sub-cubes", namely VIS (400–700nm), NIR (700–1000nm),
SWIR1 (1000–1800nm), SWIR2 (1800–2500nm), plus the full-band cube, giving $n=5$ streams.
mHC-HSI additionally reinterprets $H^{\text{res}}_{l\mathcal{M}}\in\mathbb{R}^{L\times n\times n}$
(with $L=H\times W$, so one matrix **per pixel**) as $n^2$ soft cluster-membership maps that
drive token selection in $n^2$ parallel spatial Mamba blocks, and $\mathcal{F}$ is a
clustering-guided Mamba.

**Channel/head split: yes, of the *input* channels, with one shared mixing matrix.**
This is the closest anyone comes to "channel-wise mHC", and it is the Frac-Connections
structure (channel groups standing in for stream copies) transplanted onto mHC. It is not
the multi-head structure: there is still a single $H^{\text{pre}}/H^{\text{post}}/H^{\text{res}}$
governing all groups, the groups are physical spectral bands of the input rather than
hidden-dimension heads, and the setting is image classification, not language modelling.

**Experiments.** HSI classification benchmarks against SOTA HSI methods; accuracy and
explainability gains reported. ES-mHC additionally reports that increasing the expansion
rate accelerates the emergence of structured interaction patterns in $H^{\text{res}}$.
No language-model or scaling evidence.

**Code:** <https://github.com/GSIL-UCalgary/mHC_HyperSpectral>

### 3.10 go-mHC and TBP-mHC — why the Kronecker line stays on the stream axis

- **Citations:** *go-mHC: Direct Parameterization of Manifold-Constrained Hyper-Connections
  via Generalized Orthostochastic Matrices*, arXiv:2604.02309, 2026-04-02;
  *TBP-mHC: full expressivity for manifold-constrained hyper connections through
  transportation polytopes*, arXiv:2605.21724, 2026-05-20.
- **Confidence: read full text** of the relevant method/expressivity sections of both.

Both attack KromHC's expressivity loss, and both stay entirely on the $n\times n$ matrix.

**go-mHC** parameterizes the Birkhoff polytope exactly via generalized orthostochastic
matrices at $O(d^3)$, with one hyperparameter $s$ interpolating between a cheap boundary and
full expressivity. Its §4.1.2 proves the limitation that matters for us: **the spectral
space of a $k$-fold Kronecker product of doubly stochastic matrices is invariant to $k$**, so
with $2\times2$ factors the eigenvalues stay in $\{1,\lambda_A,\lambda_B,\lambda_A\lambda_B\}
\subset[-1,1]$ — real only. Kronecker-factored mixing therefore "can only model pure
diffusion ... without advection", and cannot represent cycles of order $>i_k$. It notes
go-mHC and KromHC are complementary and could be composed to learn "tensor products of
generalized orthostochastic matrices" — still products over the stream axis.
Validation: synthetic stream-mixing (reaches the minimum theoretical loss, up to 10× faster
convergence) plus a 30M-parameter GPT-style model (6 layers, $d_{\text{model}}$=384, 6
attention heads, TinyStories).

**TBP-mHC** builds exactly doubly stochastic matrices from transportation polytopes with
$(n-1)^2$ degrees of freedom — full Birkhoff expressivity, no iterative normalization, no
factorial blow-up — plus a recursive variant (RTBP). It describes KromHC as restricted to
"a structured submanifold of the Birkhoff polytope". LM pre-training results reported as
competitive with improved stability; gradient-norm analysis included.

**Channel/head split: neither.**

### 3.11 mHC-SSM — per-stream capacity

- **Citation:** *mHC-SSM: Manifold-Constrained Hyper-Connections for State Space Language
  Models with Stream-Specialized Adapters*, arXiv:2605.08300, 2026-05-08.
  <https://arxiv.org/abs/2605.08300>
- **Confidence: abstract only.** `arxiv.org/html/2605.08300` returns 404 (no HTML version);
  abstract read from the abs page. Comments field says 28 pages, 3 figures, code available.

Static mHC around an SSM block: expand to parallel streams, simplex-constrained pre-mixing
to a single SSM input, simplex-constrained post-mixing back out, Sinkhorn-projected
$H^{\text{res}}$ per layer. Adds **stream-specialized adapters** — lightweight
stream-specific capacity through a *shared bottleneck with per-stream scaling* — applied
both before aggregation and after the SSM output.

**Channel/head split: no.** "Stream-specialized" means extra per-stream *capacity in the
adapter*, not per-channel-group connection coefficients. Closest to category 5 by name only.

**Experiments.** WikiText-2, single small scale, checkpoint-based evaluation:
validation loss 6.3507 (baseline SSM) → 6.2448 (static mHC) → 6.1353 (+ adapters);
perplexity 572.91 → 515.35 → 461.88; throughput 1025.52 → 964.81 → 938.90 tok/s;
peak memory 2365 → 2568 → 3092 MB. These are very high perplexities — a tiny-scale study.

### 3.12 Low-Rank Attention Residuals — a channel slice for routing

- **Citation:** *Low-Rank Attention Residuals*, arXiv:2607.09694, 2026-06-19.
- **Confidence: abstract only.**

Keeps full-dimensional residual values but uses **the last $r$ of $d$ dimensions of each
value as the routing key** ($r<d$), decoupling routing from representation and making
routing FLOPs independent of full width. At 1B and 4B with $r=d/4$: lower final validation
loss, higher average downstream accuracy, and higher measured throughput than AttnRes.
Releases code, a fused kernel, and trained models.

**Channel/head split: partial — one shared route computed from a channel slice**, which is
the low-rank cousin of MHAR's split rather than a per-group split.

### 3.13 The two analysis papers that motivate the question

Neither proposes multi-head anything, but both are the strongest available evidence that
the capacity multi-head mHC would target is currently unused.

**How Does mHC Use Its Residual Streams?** (arXiv:2609.05309, 2026-09-04; read full
abstract, not full text). Measures the four-stream pathway of **DeepSeek-V4-Flash**:
a typical attention or FFN site effectively uses **about two of the four streams**;
the dominant stream changes across depth; representations stay directionally distinct.
Residual mixing is modest and concentrated in early layers — in layers 22–42 the pathway
mostly carries streams forward separately. Interventions: replacing **late** mixers by
identity costs only **1.9%** C4 perplexity and preserves the six-task average, while
replacing **early** mixers costs **41%**. Fixing each early mixer to its diagnostic mean
costs 0.2% perplexity and 0.25pp. Keeping only the top-3 routing weights per token costs
≤2.7% perplexity. Conclusion: "the studied model realizes only part of the flexibility
afforded by four-stream mHC".

**Analyzing Stream Collapse in Hyper-Connections** (arXiv:2606.03483, 2026-06-02; read full
text of the relevant sections). HC's permutation symmetry over stream indices is resolved
badly in practice: after an early seeding stage residual mixing stays near identity, and
signal and interpretable features concentrate in a **dominant stream**, so the nominally
multi-stream pathway behaves closer to single-stream. Because HC expansion replicates the
same representation, $\mathbf{E}(\mathbf{x})=[\mathbf{x};\dots;\mathbf{x}]$ is in the fixed
subspace of the symmetry ($\mathbf{P}\mathbf{E}(\mathbf{x})=\mathbf{E}(\mathbf{x})$), so role
separation must emerge from training dynamics alone. Breaking symmetry at stream
initialization reduces dominance and improves performance across mHC variants. Code public.

Related: **oHC** (arXiv:2609.02672) proves that within the doubly stochastic set the mixing
step can reduce stream norm *only* by shrinking inter-stream differences, so streams grow
more alike with depth — measured at $1.9\times10^{-14}$ composed over 24 sublayers on a
trained mHC model. Its §2 is also the most complete survey of the HC family available
(HC, Frac, mHC, mHC-lite, KromHC, go-mHC, TBP, sHC, xHC, iHC, oHC) and **it lists no
multi-head or channel-grouped variant**.

---

## 4. Verdict

**No one has proposed or tested a multi-head, grouped, channel-wise, per-head or
block-diagonal mHC — or a multi-head hyper-connection of any kind.** Across 186 unique
papers citing HC, mHC, Frac-Connections, MUDDFormer and twelve further HC-family papers,
not one splits the hidden dimension into $h$ channel groups and gives each group its own
$H_{\text{pre}}/H_{\text{post}}/H_{\text{res}}$. The arXiv metadata index contains zero
occurrences of "multi-head hyper-connections", "grouped hyper-connections",
"channel-wise hyper-connections", "per-head hyper-connections", "head-wise
hyper-connections" or "multi-head mHC". The negative is sharp and structural, not merely
terminological: **every** mHC follow-up I read preserves mHC's routing interface —
$H_{\text{pre}}, H_{\text{post}} \in \mathbb{R}^{1\times n}$ and
$H_{\text{res}} \in \mathbb{R}^{n\times n}$, one scalar per stream shared across all $C$
channels — and competes only on how the $n\times n$ matrix is constrained or parameterized
(Sinkhorn, permutation combinations, Kronecker, orthostochastic, transportation polytopes,
spectral spheres, $SO(n)$, tensor networks).

**Closest work, and it is close on the mechanism but not on the architecture: Multi-Head
Attention Residuals** (arXiv:2607.27230, 2026-07-22). It is the only paper that does the
exact operation in question — reshape the read coefficients into $H$ contiguous per-subspace
heads, run $H$ independent softmaxes, make the read block-diagonal over channel groups, at
zero parameter cost with $H=1$ recovering the parent. And it validates the axis: the
optimum is $H$=4–8, loss is U-shaped in $H$, over-splitting at $H$=16 gives back a third to
a half of the gain and the penalty *grows* with scale, and the stated cause is learned
subspace disagreement that grows with width. But it heads the *depth-history read of
Attention Residuals*, not hyper-connections. It has no $n$ parallel copies, no
$H_{\text{post}}$, no $H_{\text{res}}$ and no manifold constraint — and it draws the
contrast itself: "our heads act on routing distributions, not on separate streams."

**The other near-misses, and how each falls short.**
- *Multiple read heads* exists, but keyed by **role, not channels**: MUDDFormer's four DA
  modules for Q/K/V/R — which it calls "depth-wise multi(4)-head attention" — and
  RD-AttnRes's shared QK route plus independent V route. Both on dense depth connections.
- *Channel groups as streams* exists, but with **one shared mixing matrix**:
  Frac-Connections reshapes $\mathbf{h}$ into $m$ chunks and mixes them with a single $m\times m$
  matrix; mHC-HSI and ES-mHC do the same on mHC using physical spectral band groups as the
  $n=5$ streams. None of these is $h$ *independent* mixers.
- *Kronecker-structured mixing* exists, but factorizes **$n$, never $n\cdot h$**: KromHC
  tensorizes the stream axis and multiplies the channel mode by the explicit identity
  $\mathbf{I}_{C\times C}$; go-mHC and TBP-mHC then argue against its expressivity, all on
  the same axis.
- *Several coefficient predictors combined into a product of doubly stochastic matrices*
  exists — KromHC's $K$ per-factor predictors — but again over streams.
- *The feature axis inside the coefficient predictor* exists — TEMPER's CP/Tucker
  factorization over the input-stream, **feature**, and output-stream modes — but it
  explicitly "preserves the routing interface": the feature mode is compressed on the
  predictor's input side and the outputs remain per-stream scalars. This is the subtlest
  near-miss in the literature.
- *Multiple write heads* exists in xHC ($H_{\text{post}}\in\mathbb{R}^{k\times K_r}$), but the
  $K_r$ components are temporal convolution branches, not channel groups.
- *Per-stream read heads* were tested and rejected: DAR's DAR-SelfKV ablation gives each
  stream its own independent retriever and loses to reciprocal cross-stream addressing.

**The gap is not only unfilled but motivated.** Two 2026 analyses argue that mHC's existing
stream capacity is underused in exactly the way a per-head split would address: in
DeepSeek-V4-Flash each site effectively uses ~2 of 4 streams and replacing all late mixers
with the identity costs 1.9% perplexity; and HC models collapse onto a dominant stream with
near-identity mixing after early training. MHAR's probe supplies the complementary positive
evidence that different feature subspaces genuinely want to read differently, and that the
disagreement grows with width. So a multi-head mHC — $h$ channel groups, each with its own
Sinkhorn-projected $H_{\text{res}}$ — appears to be genuinely unclaimed prior art, sitting
precisely at the intersection of MHAR's head-splitting and mHC's manifold constraint. The
two caveats a designer should carry over from the existing literature: MHAR's $H$=16 result
says over-splitting is a real and scale-growing failure mode, and go-mHC's spectral proof
says structured factorizations of the mixer can silently lose the ability to represent
rotation/advection rather than only diffusion.

**Confidence and caveats.**
- Full text read: MHAR, MUDDFormer, Frac-Connections, KromHC, TEMPER, xHC, go-mHC,
  TBP-mHC, mHC, oHC (survey + analysis), sHC, mHC-lite, stream collapse, mHC-HSI, ES-mHC,
  DAR (partial).
- Abstract only: RD-AttnRes, mHC-SSM (no HTML on arXiv — 404), LR-AttnRes, SHC, mHC-GNN,
  JPmHC, Accelerating Birkhoff Projection, and the domain applications.
- **Sparse Selective Hyper-Connections** is an IEEE SoutheastCon 2026 paper
  (DOI 10.1109/SoutheastCon63549.2026.11476522) with **no arXiv version and no open-access
  PDF**. I could not read beyond its abstract; its "rank-$r$ factorization inspired by
  Johnson-Lindenstrauss" and "tensor decomposition" are the one place a channel-group split
  could be hiding that I could not verify. Treat that single paper as unchecked.
- **OpenAlex returned zero citation edges for all four primary seeds**, so the required
  cross-check could not be performed; the crawl rests on Semantic Scholar alone. S2 was
  demonstrably incomplete — TEMPER and two other mechanism papers were reachable only by
  snowballing through KromHC and mHC-lite — so despite 19 seeds I cannot claim the
  enumeration is exhaustive. Three of the crawled seeds are recent enough to have few or no
  citations yet (oHC, TEMPER, mHC-as-PEFT, DAR, mHC-usage all returned 0).
- 7 of 186 records had no abstract in S2 and were screened by title and venue only.
