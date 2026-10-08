# Core Papers Spec: mHC and its lineage

Extracted 2026-09-26 from primary sources. Every equation below is quoted from the source named in
its heading; equation numbers are the source's own. Where the LaTeX is reproduced it was taken from
the MathML `alttext` of the arXiv HTML render, so it is verbatim.

## 0. Source access log

| # | Source | URL used | Status |
|---|---|---|---|
| 1 | mHC, "Manifold-Constrained Hyper-Connections", DeepSeek-AI | `arxiv.org/html/2512.24880` (v2, 05 Jan 2026) | Read in full |
| 2 | Hyper-Connections, Zhu et al. | `arxiv.org/html/2409.19606` | Read in full |
| 3 | Frac-Connections | `arxiv.org/html/2503.14125` | Read in full |
| 4 | MUDDFormer, Xiao et al. | `arxiv.org/html/2502.12170` | Read (scoped to weight generation) |
| 5 | DeepSeek-V4 report | `arxiv.org/html/2606.19348` | Read §2.2, §3.4.2, §4.2.x |
| 6 | Reference inference code | `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/{model,kernel,generate,convert}.py` + `config.json` | Downloaded |
| 6b | Also fetched for comparison | same paths under `DeepSeek-V4-Flash` and `DeepSeek-V4-Pro` | Downloaded; the two V4 `model.py` are byte-identical to each other and **differ from V4.1-Flash** (see §2.6) |

`deepseek-ai/DeepSeek-V4` (no suffix) returns HTTP 401 — not public. Raw files saved to
`notes/literature/raw/`:

```
DeepSeek-V4.1-Flash_model.py      DeepSeek-V4.1-Flash_kernel.py     DeepSeek-V4.1-Flash_config.json
DeepSeek-V4.1-Flash_generate.py   DeepSeek-V4.1-Flash_convert.py    DeepSeek-V4.1-Flash_requirements.txt
DeepSeek-V4-Flash_model.py        DeepSeek-V4-Flash_config.json
DeepSeek-V4-Pro_model.py          DeepSeek-V4-Pro_config.json
```

### Notation map across the four sources

The same three mappings are named differently everywhere. This table is the Rosetta stone for the
rest of the document.

| Role | mHC paper | HC paper (Zhu) | DeepSeek-V4 report | Reference code |
|---|---|---|---|---|
| read-out: stream → layer input | `H^pre_l ∈ R^{1×n}` | `A_m ∈ R^{n×1}` | `A_l ∈ R^{1×n_hc}` | `pre` `[b,s,hc]` |
| write-in: layer output → stream | `H^post_l ∈ R^{1×n}` | `B ∈ R^{1×n}` | `C_l ∈ R^{n_hc×1}` | `post` `[b,s,hc]` |
| stream mixing | `H^res_l ∈ R^{n×n}` | `A_r ∈ R^{n×n}` | `B_l ∈ R^{n_hc×n_hc}` | `comb` `[b,s,hc,hc]` |
| expansion rate | `n` | `n` (a.k.a. `rate`) | `n_hc` | `hc_mult` |
| width | `C` | `d` / `d_model` | `d` | `dim` |

Note the V4 report reuses `B` for the *residual* matrix while the HC paper uses `B` for the
*output* mapping. Below I use the mHC paper's `pre` / `post` / `res` names throughout.

---

## 1. mHC — arXiv 2512.24880

Authors: Zhenda Xie*†, Yixuan Wei*, Huanqi Cao*, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai
Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang,
Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang (DeepSeek-AI).
Report number 001. `*` core contributors, `†` corresponding (xie.zhenda@deepseek.com).

### 1.1 Baseline and HC recap (§1, §3)

Plain residual, Eq. (1) and its recursive unrolling Eq. (2):

```latex
\mathbf{x}_{l+1}=\mathbf{x}_{l}+\mathcal{F}(\mathbf{x}_{l},\mathcal{W}_{l}),                    (1)
\mathbf{x}_{L}=\mathbf{x}_{l}+\sum_{i=l}^{L-1}\mathcal{F}(\mathbf{x}_{i},\mathcal{W}_{i}),      (2)
```

> "The term identity mapping refers to the component $\mathbf{x}_{l}$ itself, which emphasizes the
> property that the signal from the shallower layer maps directly to the deeper layer without any
> modification." (§1)

Single-layer HC, Eq. (3):

```latex
\mathbf{x}_{l+1}=\mathcal{H}_{l}^{\mathrm{res}}\mathbf{x}_{l}
                +\mathcal{H}_{l}^{\mathrm{post}\,\top}\mathcal{F}(\mathcal{H}_{l}^{\mathrm{pre}}\mathbf{x}_{l},\mathcal{W}_{l}),   (3)
```

with, from §1 and §3: `x_l ∈ R^{n×C}`, `H^res_l ∈ R^{n×n}`, `H^pre_l, H^post_l ∈ R^{1×n}`, and

> "the input to the $l$-th layer, $\textbf{x}_{l}\in\mathbb{R}^{1\times C}$, is expanded by a factor
> of $n$ to construct a hidden matrix
> $\textbf{x}_{l}=(\textbf{x}^{\top}_{l,0},\ldots,\textbf{x}^{\top}_{l,n-1})^{\top}\in\mathbb{R}^{n\times C}$
> which can be viewed as $n$-stream residual." (§3)

Multi-layer unrolling of HC, Eq. (4) — this is the instability argument:

```latex
\mathbf{x}_{L}=\left(\prod_{i=1}^{L-l}\mathcal{H}_{L-i}^{\mathrm{res}}\right)\mathbf{x}_{l}
 +\sum_{i=l}^{L-1}\left(\prod_{j=1}^{L-1-i}\mathcal{H}_{L-j}^{\mathrm{res}}\right)\mathcal{H}_{i}^{\mathrm{post}\,\top}\mathcal{F}(\mathcal{H}_{i}^{\mathrm{pre}}\mathbf{x}_{i},\mathcal{W}_{i}),  (4)
```

> "the composite mapping $\prod_{i=1}^{L-l}\mathcal{H}_{L-i}^{\mathrm{res}}$ in HC fails to preserve
> the global mean of the features. This discrepancy leads to unbounded signal amplification or
> attenuation, resulting in instability during large-scale training." (§1)

### 1.2 HC's own parameterization as mHC restates it (§3, Eq. 5)

```latex
\begin{cases}\tilde{\mathbf{x}}_{l}=\text{RMSNorm}(\mathbf{x}_{l})\\
\mathcal{H}^{\mathrm{pre}}_{l}=\alpha_{l}^{\mathrm{pre}}\cdot\tanh(\theta^{\mathrm{pre}}_{l}\tilde{\mathbf{x}}^{\top}_{l})+\mathbf{b}_{l}^{\mathrm{pre}}\\
\mathcal{H}^{\mathrm{post}}_{l}=\alpha_{l}^{\mathrm{post}}\cdot\tanh(\theta^{\mathrm{post}}_{l}\tilde{\mathbf{x}}^{\top}_{l})+\mathbf{b}_{l}^{\mathrm{post}}\\
\mathcal{H}^{\mathrm{res}}_{l}=\alpha_{l}^{\mathrm{res}}\cdot\tanh(\theta^{\mathrm{res}}_{l}\tilde{\mathbf{x}}^{\top}_{l})+\mathbf{b}_{l}^{\mathrm{res}},\\
\end{cases}   (5)
```

with `RMSNorm(·)` "applied to the last dimension", `θ^pre_l, θ^post_l ∈ R^{1×C}`,
`θ^res_l ∈ R^{n×C}`, `b^pre_l, b^post_l ∈ R^{1×n}`, `b^res_l ∈ R^{n×n}`, and

> "the scalars $\alpha_{l}^{\mathrm{pre}},\alpha_{l}^{\mathrm{post}}$ and
> $\alpha_{l}^{\mathrm{res}}\in\mathbb{R}$ are learnable gating factors **initialized to small
> values**." (§3, emphasis mine)

**Critical shape difference from mHC.** Here `θ^res_l ∈ R^{n×C}` acts on `x̃^T_l` (which is
`C×n`), so each stream's coefficients are produced from **that stream's own C-dim vector**. mHC
instead flattens to `nC` first (§1.3). This is a genuine architectural change, not a
reparameterization.

### 1.3 The mHC parameterization (§4.2) — the core of the spec

Flatten first:

> "Given the input hidden matrix $\mathbf{x}_{l}\in\mathbb{R}^{n\times C}$ at the $l$-th layer, we
> first flatten it into a vector
> $\vec{\mathbf{x}}_{l}=\text{vec}(\mathbf{x}_{l})\in\mathbb{R}^{1\times nC}$ **to preserve full
> context information**." (§4.2, emphasis mine)

Eq. (7) — raw (unconstrained) mappings. **Note there is no `tanh` here**, unlike HC's Eq. (5):

```latex
\begin{cases}\vec{\mathbf{x}}^{\prime}_{l}=\text{RMSNorm}(\vec{\mathbf{x}}_{l})\\
\tilde{\mathcal{H}}^{\mathrm{pre}}_{l}=\alpha_{l}^{\mathrm{pre}}\cdot(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{pre}}_{l})+\mathbf{b}_{l}^{\mathrm{pre}}\\
\tilde{\mathcal{H}}^{\mathrm{post}}_{l}=\alpha_{l}^{\mathrm{post}}\cdot(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{post}}_{l})+\mathbf{b}_{l}^{\mathrm{post}}\\
\tilde{\mathcal{H}}^{\mathrm{res}}_{l}=\alpha_{l}^{\mathrm{res}}\cdot\text{mat}(\vec{\mathbf{x}}^{\prime}_{l}\varphi^{\mathrm{res}}_{l})+\mathbf{b}_{l}^{\mathrm{res}},\\
\end{cases}   (7)
```

> "where $\varphi^{\mathrm{pre}}_{l},\varphi^{\mathrm{post}}_{l}\in\mathbb{R}^{nC\times n}$ and
> $\varphi^{\mathrm{res}}_{l}\in\mathbb{R}^{nC\times n^{2}}$ are linear projections for dynamic
> mappings and $\text{mat}(\cdot)$ is a reshape function from $\mathbb{R}^{1\times n^{2}}$ to
> $\mathbb{R}^{n\times n}$." (§4.2)

Eq. (8) — the manifold projections:

```latex
\begin{cases}\mathcal{H}^{\mathrm{pre}}_{l}=\sigma(\tilde{\mathcal{H}}^{\mathrm{pre}}_{l})\\
\mathcal{H}^{\mathrm{post}}_{l}=2\sigma(\tilde{\mathcal{H}}^{\mathrm{post}}_{l})\\
\mathcal{H}^{\mathrm{res}}_{l}=\text{Sinkhorn-Knopp}(\tilde{\mathcal{H}}^{\mathrm{res}}_{l}),\end{cases}   (8)
```

> "where $\sigma(\cdot)$ denotes the Sigmoid function. The $\text{Sinkhorn-Knopp}(\cdot)$ operator
> firstly makes all elements to be positive via an exponent operator and then conducts iterative
> normalization process that alternately rescales rows and columns to sum to 1. Specifically, given
> a positive matrix $\mathbf{M}^{(0)}=\exp(\tilde{\mathcal{H}}^{\mathrm{res}}_{l})$ as the start
> point, the normalization iteration proceeds as:" (§4.2)

```latex
\mathbf{M}^{(t)}=\mathcal{T}_{r}\left(\mathcal{T}_{c}(\mathbf{M}^{(t-1)})\right),   (9)
```

> "where $\mathcal{T}_{r}$ and $\mathcal{T}_{c}$ denote row and column normalization, respectively.
> This process converges to a doubly stochastic matrix
> $\mathcal{H}^{\mathrm{res}}_{l}=\mathbf{M}^{(t_{\text{max}})}$ as $t_{\text{max}}\to\infty$. We
> choose $t_{\text{max}}=20$ as a practical value in our experiments." (§4.2)

**Eq. (9) reads inner-to-outer as column-normalize then row-normalize.** The shipped kernel does the
opposite order and folds `exp`+row-normalize into a softmax. See §2.5 and the ambiguity list.

Why the manifold (§4.1), Eq. (6):

```latex
\mathcal{P}_{\mathcal{M}^{\mathrm{res}}}(\mathcal{H}^{\mathrm{res}}_{l})\coloneq\left\{\mathcal{H}^{\mathrm{res}}_{l}\in\mathbb{R}^{n\times n}\mid\mathcal{H}^{\mathrm{res}}_{l}\mathbf{1}_{n}=\mathbf{1}_{n},\ \mathbf{1}^{\top}_{n}\mathcal{H}^{\mathrm{res}}_{l}=\mathbf{1}^{\top}_{n},\ \mathcal{H}^{\mathrm{res}}_{l}\geqslant 0\right\},   (6)
```

Three stated properties: (1) "The spectral norm of a doubly stochastic matrix is bounded by 1 (i.e.,
$\|\mathcal{H}^{\mathrm{res}}_{l}\|_{2}\leq 1$)... the learnable mapping is non-expansive";
(2) "**Compositional Closure**: The set of doubly stochastic matrices is closed under matrix
multiplication"; (3) "**Geometric Interpretation via the Birkhoff Polytope**... the residual mapping
acts as a convex combination of permutations."

> "It is worth noting that when $n=1$, the doubly stochastic condition degenerates to the scalar
> $1$, thereby recovering the original identity mapping." (§4.1)

And on why `pre`/`post` get sigmoids at all:

> "Additionally, we impose non-negativity constraints on the input mappings
> $\mathcal{H}^{\mathrm{pre}}_{l}$ and output mappings $\mathcal{H}^{\mathrm{post}}_{l}$. This
> constrain prevents signal cancellation arising from the composition of positive and negative
> coefficients, which can also be considered as a special manifold projection." (§4.1)

### 1.4 The fused-kernel form (§4.3.1) — Eq. (10)–(19)

This is the form the reference code actually implements, and it is the most useful one to
reimplement from. Note two consolidations stated in the text:

> "In these kernels, the biases and linear projections are consolidated into $\mathbf{b}_{l}$ and
> $\varphi_{l}$, and **the RMSNorm weight is also absorbed in $\varphi_{l}$**." (§4.3.1)

> "Observing that RMSNorm in mHC imposes significant latency when operating on the high-dimensional
> hidden state $\vec{\mathbf{x}}_{l}\in\mathbb{R}^{1\times nC}$, we **reorder the dividing-by-norm
> operation to follow the matrix multiplication**. This optimization maintains mathematical
> equivalence while improving efficiency." (§4.3.1)

Inputs and parameters, with dtypes, Eq. (10)–(13):

```latex
\varphi_{l}                                              : \text{tfloat32}   [nC,\ n^{2}+2n]     (10)
\vec{\mathbf{x}}_{l}                                     : \text{bfloat16}   [1,\ nC]            (11)
\alpha_{l}^{\mathrm{pre}},\alpha_{l}^{\mathrm{post}},\alpha_{l}^{\mathrm{res}} : \text{float32}  Scalars  (12)
\mathbf{b}_{l}                                           : \text{float32}    [1,\ n^{2}+2n]      (13)
```

Compute, Eq. (14)–(19):

```latex
\left[\tilde{\tilde{\mathcal{H}}}^{\mathrm{pre}}_{l},\tilde{\tilde{\mathcal{H}}}^{\mathrm{post}}_{l},\tilde{\tilde{\mathcal{H}}}^{\mathrm{res}}_{l}\right] : \text{float32} = \vec{\mathbf{x}}_{l}\varphi_{l}   (14)
r : \text{float32} = \left\|\vec{\mathbf{x}}_{l}\right\|_{2}/\sqrt{nC}                                  (15)
\left[\tilde{\mathcal{H}}^{\mathrm{pre}}_{l},\tilde{\mathcal{H}}^{\mathrm{post}}_{l},\tilde{\mathcal{H}}^{\mathrm{res}}_{l}\right] : \text{float32} = 1/r\left[\alpha_{l}^{\mathrm{pre}}\tilde{\tilde{\mathcal{H}}}^{\mathrm{pre}}_{l},\alpha_{l}^{\mathrm{post}}\tilde{\tilde{\mathcal{H}}}^{\mathrm{post}}_{l},\alpha_{l}^{\mathrm{res}}\tilde{\tilde{\mathcal{H}}}^{\mathrm{res}}_{l}\right]+\mathbf{b}_{l}   (16)
\mathcal{H}^{\mathrm{pre}}_{l}  : \text{float32} = \sigma\left(\tilde{\mathcal{H}}^{\mathrm{pre}}_{l}\right)      (17)
\mathcal{H}^{\mathrm{post}}_{l} : \text{float32} = 2\sigma\left(\tilde{\mathcal{H}}^{\mathrm{post}}_{l}\right)     (18)
\mathcal{H}^{\mathrm{res}}_{l}  : \text{float32} = \text{Sinkhorn-Knopp}\left(\tilde{\mathcal{H}}^{\mathrm{res}}_{l}\right)   (19)
```

Read carefully: `r = ||x||_2 / sqrt(nC)` is the RMS, so `1/r` is the `rsqrt` factor, and **`1/r`
multiplies only the `α·(xφ)` term, not the bias `b_l`**. The bias is added after the
divide-by-norm. The reference code matches this exactly.

Five kernels total: one fused kernel for Eq. (14)–(15) (the two scans over `x_l`, with a fused
backward "comprising two matrix multiplications... eliminating redundant reloading"), one for
Eq. (16)–(18) ("lightweight operations on small coefficients... opportunistically fused"), one for
Eq. (19) (Sinkhorn, with "a custom backward kernel that recomputes the intermediate results on-chip
and traverses the entire iteration"), plus two application kernels:

> "one for $\mathcal{F}_{\mathrm{pre}}\coloneq\mathcal{H}^{\mathrm{pre}}_{l}\mathbf{x}_{l}$ and
> another for
> $\mathcal{F}_{\mathrm{post,res}}\coloneq\mathcal{H}^{\mathrm{res}}_{l}\mathbf{x}_{l}+\mathcal{H}_{l}^{\mathrm{post}\,\top}\mathcal{F}(\cdot,\cdot)$.
> Through fusing the application of $\mathcal{H}^{\mathrm{post}}_{l}$ and
> $\mathcal{H}^{\mathrm{res}}_{l}$ with residual merging, we reduce the number of elements read from
> $(3n+1)C$ to $(n+1)C$ and the number of elements written from $3nC$ to $nC$ for this kernel."
> (§4.3.1)

Implemented in TileLang (Wang et al., 2025), "excluding Eq. (14) to (15)".

### 1.5 Recompute and pipeline (§4.3.2, §4.3.3)

> "we discard the intermediate activations of the mHC kernels after the forward pass and recompute
> them on-the-fly in the backward pass, through re-executing the mHC kernels without the heavy layer
> function $\mathcal{F}$." (§4.3.2)

Table 3 (per-token activations): `x_{l0}` (`nC`, stored every `L_r` layers);
`F(H^pre_l x_l, W_l)` (`C`, stored every layer); `x_l` (`nC`), `H^pre_l x_l` (`C`),
`RMSNorm(H^pre_l x_l)` (`C`) — all three "Transient inside `L_r` layers".

Optimal recompute block size, Eq. (20):

```latex
L_{r}^{*}=\arg\min_{L_{r}}\left[nC\times\left\lceil\frac{L}{L_{r}}\right\rceil+(n+2)C\times L_{r}\right]\approx\sqrt{\frac{nL}{n+2}}.   (20)
```

> "Observing that the theoretical optimum $L_{r}^{*}$ typically aligns with the number of layers per
> pipeline stage, we choose to synchronize the recomputation boundaries with the pipeline stages."

DualPipe: "to prevent blocking the communication stream, we execute the
$\mathcal{F}_{\mathrm{post,res}}$ kernels of MLP (i.e. FFN) layers on a dedicated high-priority
compute stream. We further refrain from employing persistent kernels for long-running operations in
attention layers".

I/O accounting, Table 2 (per token, forward, excluding `F`'s internals):

| Method | Operation | Read | Write |
|---|---|---|---|
| Residual | Residual Merge | `2C` | `C` |
| | **Total** | **`2C`** | **`C`** |
| HC | Calculate `H^pre, H^post, H^res` | `nC` | `n²+2n` |
| | `H^pre` | `nC+n` | `C` |
| | `H^post` | `C+n` | `nC` |
| | `H^res` | `nC+n²` | `nC` |
| | Residual Merge | `2nC` | `nC` |
| | **Total** | **`(5n+1)C+n²+2n`** | **`(3n+1)C+n²+2n`** |

### 1.6 Where `H^pre` sits relative to the layer's own RMSNorm

The paper states this only implicitly, via the recompute table, which lists both `H^pre_l x_l` and
`RMSNorm(H^pre_l x_l)` as separate stored activations, and via §3.2 "Focusing on the widely adopted
pre-norm Transformer architecture". So the order is:

```
stream x_l  --H^pre-->  C-dim vector  --RMSNorm-->  F (attn or FFN)  --H^post-->  stream
```

i.e. **`H^pre` is applied first, then the block's own pre-norm RMSNorm on the collapsed C-dim
vector.** The reference code confirms this unambiguously (`x = self.hc_pre(...)` then
`x = self.attn_norm(x)`), as does HC's Algorithm 3 (`h = attn_norm(mix_h[...,0,:])`).

Note there are therefore **two distinct RMSNorms per sublayer**: the mHC-internal one over the
flattened `nC` stream (weight absorbed into `φ`, Eq. 7/15), and the block's ordinary pre-norm over
the collapsed `C`-dim vector.

### 1.7 Initialization — what the paper actually gives

**Only one init value is stated anywhere in the mHC paper**, in the Appendix A.1 hyper-parameter
table:

| Row | Value |
|---|---|
| `mHC/HC Expansion Rate n` | 4 |
| **`mHC/HC Gating Factor Init α`** | **0.01** |
| `mHC Sinkhorn-Knopp t_max` | 20 |
| `Layer Norm Type` | RMSNorm |
| `Layer Norm ε` | 1e-20 |

Everything else is *not stated*: there is no statement of how `b^pre`, `b^post`, `b^res` are
initialized, and none of how `φ^pre`, `φ^post`, `φ^res` are initialized. §3 only repeats HC's
"initialized to small values" for `α`. See §6.2 for why this is the single most consequential gap
and what the HC paper implies.

### 1.8 Stream expansion and collapse — what the paper says

The mHC paper does not restate either operation; it inherits §3's "expanded by a factor of $n$"
(i.e. repeat) and does not describe the collapse before the final norm/head at all. Both are
specified only in HC (§3, sum rows) and in the reference code (§2.4, §2.6), which disagree with each
other. Flagged in §6.2.

### 1.9 Experiments (§5, Appendix A.1)

MoE architectures "inspired by DeepSeek-V3". `n = 4` for both HC and mHC. Four variants:

| | 3B | 9B | 27B | 3B / 1T tokens |
|---|---|---|---|---|
| Vocab params | 331M | 496M | 662M | 331M |
| Active params | 612M | 1.66B | 4.14B | 612M |
| Total params | 2.97B | 9.18B | 27.0B | 2.97B |
| Layers | 12 | 18 | 30 | 12 |
| Routed experts | 64 | 64 | 72 | 64 |
| Dimension `C` | 1280 | 1920 | 2560 | 1280 |
| FFN dimension | 896 | 1280 | 1536 | 896 |
| Attention heads | 16 | 24 | 32 | 16 |
| Batch size | 320 | 512 | 1280 | 2560 |
| Training steps | 30000 | 50000 | 50000 | 100000 |
| **Training tokens** | **39.3B** | **105B** | **262B** | **1.05T** |
| Base LR | 8.6e-4 | 5.9e-4 | 4.0e-4 | 9.0e-4 |

Shared across all: 1 leading dense layer, 6 active + 2 shared experts, loss-free load balancing,
MLA with KV rank 512, attention head dim 128, RoPE dim 64, RoPE θ 10000, seq len 4096, vocab 129280,
warmup 2000 steps, AdamW (0.9, 0.95) with ε 1e-20 and weight decay 0.1, step LR scheduler decaying
at `[0.8×, 0.9×]` by `[0.316, 0.1]`.

**Baselines**: plain residual ("Baseline") and HC. **Metrics**: training loss (reported as absolute
loss gap), gradient norm, and 8 downstream benchmarks.

Main 27B results (Table 4):

| | BBH (3s EM) | DROP (3s F1) | GSM8K (8s EM) | HellaSwag (10s) | MATH (4s EM) | MMLU (5s) | PIQA (0s) | TriviaQA (5s) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 43.8 | 47.0 | 46.7 | 73.7 | 22.0 | 59.0 | 78.5 | 54.3 |
| w/ HC | 48.9 | 51.6 | 53.2 | 74.3 | **26.4** | 63.0 | 79.9 | 56.3 |
| w/ mHC | **51.0** | **53.9** | **53.8** | **74.7** | 26.0 | **63.4** | **80.5** | **57.6** |

> "mHC effectively mitigates the training instability observed in HC, achieving a final loss
> reduction of 0.021 compared to the baseline." (§5.2). Gains over HC: "2.1% on BBH... and 2.3% on
> DROP".

HC component ablation (Table 1; disabled mappings replaced by "uniform weights of $1/n$ for
$\mathcal{H}^{\mathrm{pre}}_{l}$, uniform weights of ones for $\mathcal{H}^{\mathrm{post}}_{l}$, and
the identity matrix for $\mathcal{H}^{\mathrm{res}}_{l}$"):

| `H^res` | `H^pre` | `H^post` | Absolute Loss Gap |
|---|---|---|---|
| | | | 0.0 |
| ✓ | | | −0.022 |
| ✓ | ✓ | | −0.025 |
| ✓ | ✓ | ✓ | −0.027 |

> "the residual mapping $\mathcal{H}^{\mathrm{res}}_{l}$ yields the most significant performance
> gain." (§3) — i.e. `res` buys −0.022 of the total −0.027; `pre` and `post` together add −0.005.

**There is no mHC-side ablation** of static vs dynamic, of `n`, or of the Sinkhorn iteration count.
The only ablation table in the paper is the HC one above.

### 1.10 The gain metric ("Amax Gain Magnitude")

Defined twice. §3.1:

> "The first, based on the maximum absolute value of the row sums of the composite mapping, captures
> the worst-case expansion in the forward pass. The second, based on the maximum absolute column
> sum, corresponds to the backward pass. We refer to these metrics as the Amax Gain Magnitude of the
> composite mapping."

Fig. 3 caption makes the reduction explicit:

> "The layer index $l$ (x-axis) unrolls each standard Transformer block into two independent layers
> (Attention and FFN). The Amax Gain Magnitude (y-axis) is calculated as the maximum absolute row sum
> (for the forward signal) and column sum (for the backward gradient), **averaged over all tokens in
> a selected sequence**."

Values: HC composite peaks at **~3000**; mHC composite "reaching a maximum value of approximately
**1.6**" — "reduces it by three orders of magnitude". mHC's *single-layer* backward gain "deviates
slightly from 1" because 20 iterations is only an approximate Sinkhorn solution. HC loss surge
"around the 12k step".

### 1.11 Overhead

> "mHC supports training at scale and introduces only a **6.7%** additional time overhead when
> expansion rate $n=4$." (§1; repeated §4.3 "a marginal training overhead of only 6.7%")

The V4 report sharpens what the 6.7% is measured against: "only 6.7% of the overlapped 1F1B pipeline
stage" (§3.4.2).

### 1.12 Heads, groups, channel splits, future variants

**There is no mention anywhere in the mHC paper of heads, groups, channel splits, or per-channel
mixing.** The word "group" appears only in §2.1 in the unrelated sense of "grouped convolutions"
and Grouped-Query Attention. The only future-work direction offered is *other manifolds*:

> "Although this work utilizes doubly stochastic matrices to ensure stability, the framework
> accommodates the exploration of diverse manifold constraints tailored to specific learning
> objectives. We anticipate that further investigation into distinct geometric constraints could
> yield novel methods that better optimize the trade-off between plasticity and stability." (§6)

So a multi-head / grouped mHC is **not** proposed by the paper; it is an extrapolation. The nearest
prior art for splitting along the channel axis is Frac-Connections (§4).

---

## 2. Reference inference code

### 2.1 Config values

| Key | V4.1-Flash | V4-Flash | V4-Pro |
|---|---|---|---|
| `hc_mult` | 4 | 4 | 4 |
| `hc_sinkhorn_iters` | 20 | 20 | 20 |
| `hc_eps` | 1e-06 | 1e-06 | 1e-06 |
| `hidden_size` | 5120 | 4096 | 7168 |
| `num_hidden_layers` | 40 | 43 | 61 |
| `rms_norm_eps` | 1e-20 | — | — |

`ModelArgs` in `DeepSeek-V4.1-Flash_model.py` (lines 102–105):

```python
# hyper-connections: the residual stream is carried as hc_mult parallel copies
hc_mult: int = 4
hc_sinkhorn_iters: int = 20
hc_eps: float = 1e-6
```

### 2.2 Parameters and dtypes (`Block.__init__`, V4.1-Flash lines 935–946)

```python
self.hc_mult = hc_mult = args.hc_mult
self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
self.hc_eps = args.hc_eps
mix_hc = (2 + hc_mult) * hc_mult
hc_dim = hc_mult * args.dim
with set_dtype(torch.float32):
    self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
    self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
    self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
    self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
    self.hc_attn_scale = nn.Parameter(torch.empty(3))
    self.hc_ffn_scale = nn.Parameter(torch.empty(3))
```

Mapping to the paper: `mix_hc = (2+n)·n = n²+2n` ✓ Eq. (10)/(13). So

- `hc_*_fn` ≡ `φ_l^T`, shape `[n²+2n, nC]` — **one projection for all three mappings**
- `hc_*_base` ≡ `b_l`, shape `[n²+2n]`
- `hc_*_scale` ≡ `(α^pre, α^post, α^res)`, shape `[3]` — **one triple shared across the whole layer**

All three are **fp32** (`set_dtype(torch.float32)`), matching Eq. (10)–(13) except that the paper
specifies `tfloat32` for `φ` (a compute format, not a storage format) and the code stores fp32.
There are **separate parameter sets for attention and FFN** (`hc_attn_*` vs `hc_ffn_*`), i.e. each
Transformer block is two mHC layers, consistent with Fig. 3's "unrolls each standard Transformer
block into two independent layers".

Parameter count per sublayer: `(n²+2n)·(nC + 1) + 3`. For `n=4`, `C=5120`: `24·20481 + 3 = 491,547`
per sublayer, `×2×40 layers ≈ 39.3M` for V4.1-Flash.

### 2.3 Coefficient generation (`Block.hc_mixes`, V4.1-Flash lines 948–955)

```python
def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
    pre / post / comb coefficients, split out of one projection of the flattened stream."""
    # normalized over the whole flattened hc*d stream, one statistic per token
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
```

Three things to note.

1. **The RMSNorm here has no learned weight** — it is a bare `rsqrt(mean(x²) + eps)`. This is
   exactly the paper's "the RMSNorm weight is also absorbed in `φ_l`".
2. The norm statistic is taken over the **whole flattened `hc*d` stream**, one scalar per token
   (`x.flatten(2)` then `mean(-1)`), matching Eq. (15)'s `r = ||x||_2/sqrt(nC)`.
3. The division by the norm is applied **after** the matmul (`F.linear(...) * rsqrt`), which is the
   reordering of Eq. (14)–(15). `hc_scale` and `hc_base` are applied later, inside the kernel, so
   the bias is added after the divide — matching Eq. (16).
4. Everything is cast to **fp32** before the projection.

### 2.4 Applying the mappings (V4.1-Flash lines 957–966)

```python
def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
    """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
    y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    return y.to(x.dtype)

def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
    x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)
```

`hc_post` is Eq. (3) exactly: `post` outer-producted with the sublayer output, plus `comb` applied to
the residual. Note the contraction in the `comb` term is over `dim=2` of
`comb[b,s,hc,hc] · residual[b,s,hc,1,d]` — i.e. over the **first** `hc` index of `comb`, so the code
computes `sum_j comb[j,k] · residual[j]`, which is `comb^T @ residual`.

### 2.5 The Sinkhorn kernel — `hc_split_sinkhorn` (`DeepSeek-V4.1-Flash_kernel.py` lines 407–474)

This is the authoritative statement of the projection, since the papers leave the iteration order
and the epsilons underspecified. TileLang source, verbatim:

```python
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), FP32],
        hc_base: T.Tensor[(mix_hc,), FP32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc])
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = mixes_shared[j * hc + k + hc * 2] * hc_scale[2] + hc_base[j * hc + k + hc * 2]

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            # comb = comb.softmax(-1) + eps
            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            # comb = comb / (comb.sum(-2) + eps)
            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                # comb = comb / (comb.sum(-1) + eps)
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])

    return hc_split_sinkhorn_kernel_
```

and the Python wrapper, which fixes the output shapes:

```python
def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6
):
    b, s, _ = mixes.size()
    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(mixes.view(-1, (2 + hc_mult) * hc_mult), hc_scale, hc_base, pre.view(-1, hc_mult), post.view(-1, hc_mult), comb.view(-1, hc_mult, hc_mult))
    return pre, post, comb
```

**Everything this pins down:**

- **Slicing of the `n²+2n` vector**: `pre` = elements `[0:n]`, `post` = `[n:2n]`,
  `comb[j,k]` = element `j*n + k + 2n`. So the layout is `[pre | post | res(row-major)]`.
- **`hc_scale` index assignment**: `scale[0]`→pre, `scale[1]`→post, `scale[2]`→res. Confirms
  `hc_scale == (α^pre, α^post, α^res)` in that order.
- **`pre = sigmoid(·) + eps`** — the code adds `eps` that Eq. (17) does not have.
- **`post = 2·sigmoid(·)`** — exactly Eq. (18), no eps.
- **The `exp` of Eq. (19) is fused into a row-wise softmax.** `softmax(-1)` over `dim=1` is
  `exp` followed by row-normalization, with max-subtraction for stability. So `M^(0) = exp(H̃res)`
  and the *first* `T_r` are one op.
- **Rows of `comb` are normalized before columns**, and the order per iteration is (row, then column).
- **Iteration count**: one softmax-row + one column, then `sinkhorn_iters - 1` further (row, column)
  pairs. With `sinkhorn_iters = 20` that is **20 row normalizations and 20 column normalizations**,
  with a *column* normalization last.
- **The kernel's ordering agrees with Eq. (9) once you account for the transpose.** `hc_post`
  computes `y[k] = Σ_j comb[j,k]·residual[j]`, so `comb[j,k] = H^res[k,j]`, i.e.
  **`comb = (H^res)^T`** (the same `A_r^T` convention as HC Eq. 4). A *row* normalization of `comb`
  is therefore a *column* normalization of `H^res`. Mapping the kernel into the paper's `H^res`
  space gives `T_c` then `T_r` per iteration — exactly Eq. (9)'s `M^(t) = T_r(T_c(M^(t-1)))`.
  I verified this numerically (`n=4`, 20 iterations, random input): the kernel's `comb^T` and a
  direct implementation of Eq. (9) on `H^res` agree to `5.1e-6`, the residual being entirely the
  kernel's epsilons.
- **Consequently `H^res`'s rows end up exact and its columns approximate**, which is what the paper
  reports: rows of `H^res` are the forward gain, columns the backward gain, and
  > "as shown in Fig. 7 (a), the backward gradient gain deviates slightly from 1" (§5.4)
  Measured on the toy run: `H^res` row sums all `0.999999`; column sums
  `0.99999942 / 0.99999908 / 0.99999906 / 0.99999844`. So the paper's diagnostic, Eq. (9), and the
  shipped kernel are all mutually consistent — but **only if you keep the transpose straight**, which
  is the single easiest thing to get wrong in a reimplementation.
- **`eps` (1e-6) is used three ways**: added to `pre` after the sigmoid; added to `comb` after the
  row softmax; and added to every row/column sum before dividing.
- The Sinkhorn runs **per token** (`T.Kernel(n)` over the flattened `b*s`), in **fp32**.

### 2.6 Block wiring — and the V4 / V4.1 divergence

This is the biggest thing the code adds over the papers, and the two model generations differ.

**V4.1-Flash (`Block.forward`, lines 968–995) — pre-mix shifting / "single pass":**

```python
def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
    sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
    previous layer's FFN produced and the FFN uses what this attention produced."""
    residual = x
    attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.hc_pre(x, pre_mix)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos, *attn_args)
    x = self.hc_post(x, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    x = self.hc_pre(x, attn_pre)
    x = self.ffn_norm(x)
    x = self.ffn(x, image_mask)
    x = self.hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre
```

The class docstring states the intent:

> "Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
> `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
> coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.
> **The coefficients a sublayer computes are used by the *next* one** -- see `forward`."

So each sublayer's `pre` is **deferred by one sublayer**, and the block returns `ffn_pre` alongside
the stream. The attention sublayer consumes the previous block's FFN `pre`; the FFN consumes this
block's attention `pre`; the final block's `ffn_pre` is used for the collapse before the head. The
very first `pre_mix` is a hard one-hot, not learned (see §2.7).

The shifting is what makes a single fused pass possible: `hc_mixes(x)` and `hc_pre(x, pre_mix)` both
read the same `x`, so the projection for the *next* sublayer and the collapse for *this* one can be
fused into one scan over the `nC` stream — which is exactly Eq. (14)–(15)'s "unified kernel that
fuses two scans on `x_l`".

**V4-Flash / V4-Pro (`Block.forward`, lines 688–702) — no shifting:**

```python
def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    x = self.hc_post(x, residual, post, comb)

    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x
```

where V4's `hc_pre` both generates *and* applies (lines 673–681):

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb
```

**V4 is the literal reading of the mHC paper** (`H^pre_l` applied to `x_l`, the same `l`).
**V4.1-Flash shifts `pre` forward by one sublayer.** The mHC paper's Eq. (3)/(7) describe the V4
behaviour; the shifting is a V4.1 change not documented in any of these papers.

### 2.7 Stream expansion and collapse

**Expansion** — identical in both generations, a plain repeat (V4.1-Flash lines 1257–1258):

```python
# Expand to hc_mult copies for Hyper-Connections
h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
```

So `x_0 = (e, e, ..., e)`, `n` identical copies of the embedding. Matches HC's `H^0`.

**Collapse — this is where the two generations differ most.**

V4.1-Flash reuses the last layer's shifted `pre`, with a hard one-hot seed (lines 1159–1163,
1260, 1268–1269):

```python
def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix
```

```python
pre_mix = make_identity_pre_mix(h, self.hc_mult)
for i, layer in enumerate(self.layers):
    ...
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
h = layer.hc_pre(h, pre_mix)
logits = self.head(self.norm(h))
```

So: seed `pre_mix` is `[1, 0, 0, 0]` (non-learned, exact one-hot on stream 0); the collapse before
the final norm is the *last FFN sublayer's* `ffn_pre`; and there are **no extra head parameters**.

V4-Flash / V4-Pro instead carry a **dedicated learned head collapse** (`Transformer.__init__`
lines 794–799, `ParallelHead.hc_head` lines 730–738):

```python
self.hc_mult = hc_mult = args.hc_mult
hc_dim = hc_mult * args.dim
with set_dtype(torch.float32):
    self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
    self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
    self.hc_head_scale = nn.Parameter(torch.empty(1))
```

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype)
```

i.e. a `pre`-only mHC layer: `φ_head ∈ R^{n×nC}`, `b_head ∈ R^n`, one scalar `α_head`, sigmoid plus
`eps`, no Sinkhorn. Applied **before** the final norm: `self.get_logits(norm(x))` — so the order is
collapse → final RMSNorm → unembedding, in both generations.

Neither generation sums or averages the streams for the head — **both use a learned weighted
collapse.** (This differs from the HC paper, which sums rows; see §3.4.)

### 2.8 Other places the `hc` stream is touched (V4.1-Flash only)

- **Engram** (lines 329–360) reads and writes the *expanded* stream: `x: [B, L, hc_mult, dim]`, and
  its `wkv` emits `args.dim * (args.hc_mult + 1)` — one `dim`-wide value plus an `hc_mult × dim` key,
  with `q_weight`/`k_weight` of shape `(hc_mult, dim)` initialised to ones. The gate is "a
  normalized dot product of stream against key", computed per stream.
- **MTP / DSpark** (`forward_embed`, `forward_head`) repeats the draft embedding to `hc_mult` copies
  and re-seeds `make_identity_pre_mix`, then collapses with `hc_pre` before its head — so the MTP
  stack is its own mHC stream with its own one-hot seed.
- The MTP hidden readout **averages** across streams: `main_hiddens.append(h.mean(dim=2))`.

### 2.9 What the code does *not* contain

`model.py` is inference-only: every `hc_*` parameter is `torch.empty(...)` and loaded from the
checkpoint. **There is no initialization code anywhere in the reference release**, so the code
cannot resolve the init gap in §1.7.

---

## 3. Hyper-Connections — arXiv 2409.19606 (Zhu et al.)

### 3.1 Static HC (§2.1)

Stream init and collapse, stated explicitly:

> "Initially, $\mathbf{h}^{0}\in\mathbb{R}^{d}$ is **replicated $n$ times** to form the initial hyper
> hidden matrix
> $\mathbf{H}^{0}=\begin{pmatrix}\mathbf{h}^{0}&\mathbf{h}^{0}&\dots&\mathbf{h}^{0}\end{pmatrix}^{\intercal}\in\mathbb{R}^{n\times d}$.
> ... Finally, we **sum the last hyper hidden matrix row-wise** to obtain the required hidden vector,
> which is then passed through a final projector to produce the final output of the network (i.e., a
> normalization layer and an unembedding layer in transformers)." (§2.1, emphasis mine)

The single matrix, Eq. (1):

```latex
\mathcal{HC}=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{B}\\
\mathbf{A_{m}}&\mathbf{A_{r}}\end{pmatrix}=\begin{pmatrix}0&\beta_{1}&\beta_{2}&\cdots&\beta_{n}\\
\alpha_{1,0}&\alpha_{1,1}&\alpha_{1,2}&\cdots&\alpha_{1,n}\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
\alpha_{n,0}&\alpha_{n,1}&\alpha_{n,2}&\cdots&\alpha_{n,n}\end{pmatrix}\in\mathbb{R}^{(n+1)\times(n+1)}.   (1)
```

Forward, Eqs. (2)–(5):

```latex
\mathbf{\hat{H}}=\mathcal{HC}(\mathcal{T},\mathbf{H})=\mathbf{B}^{\intercal}\mathcal{T}(\mathbf{H}^{\intercal}\mathbf{A_{m}})^{\intercal}+\mathbf{A_{r}}^{\intercal}\mathbf{H}.   (2)
\mathbf{h}_{0}^{\intercal}=\mathbf{A_{m}}^{\intercal}\mathbf{H},    (3)
\mathbf{H^{\prime}}=\mathbf{A_{r}}^{\intercal}\mathbf{H}.           (4)
\mathbf{\hat{H}}=\mathbf{B}^{\intercal}(\mathcal{T}\mathbf{h}_{0})^{\intercal}+\mathbf{H^{\prime}}.   (5)
```

Note the **transposes**: HC applies `A_r^T H`, mHC applies `H^res x`. The reference code's `hc_post`
contracts `comb` over its first index, i.e. `comb^T @ residual` — matching HC's convention.

The decomposition into depth- and width-connections, Eqs. (6)–(7):

```latex
\mathcal{DC}=\begin{pmatrix}\mathbf{B}\\ \text{diag}(\mathbf{A_{r}})\end{pmatrix}\in\mathbb{R}^{2\times n},   (6)
\mathcal{WC}=\begin{pmatrix}\mathbf{A_{m}}&\mathbf{A_{r}}\end{pmatrix}\in\mathbb{R}^{n\times(n+1)}.           (7)
```

`WC` is the fused `[pre | res]` matrix — HC computes `pre` and `res` in one matmul, and the
reference mHC code goes further by fusing `post` in too (`mix_hc = n²+2n`).

### 3.2 Dynamic HC (§2.2), Eqs. (8)–(13)

> "In practice, we combine the dynamic and static matrices to achieve DHC. The dynamic parameters
> are obtained through a linear transformation. To stabilize the training process, we introduce
> normalization before the linear transformation and apply the **tanh** activation function after
> it, scaling it by a **small initial learnable factor**." (§2.2)

```latex
\overline{\mathbf{H}} = \texttt{norm}(\mathbf{H})                                                       (10)
\mathcal{B}(\mathbf{H}) = s_{\beta}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{\beta})^{\intercal}+\mathbf{B}\in\mathbb{R}^{1\times n}   (11)
\mathcal{A}_{m}(\mathbf{H}) = s_{\alpha}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{m})+\mathbf{A}_{m}\in\mathbb{R}^{n\times 1}          (12)
\mathcal{A}_{r}(\mathbf{H}) = s_{\alpha}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{r})+\mathbf{A}_{r}\in\mathbb{R}^{n\times n}          (13)
```

Shapes from Appendix B Eq. (24): `W_β ∈ R^{d}`, `W_m ∈ R^{d}`, `W_r ∈ R^{d×n}` — so
`H̄ W_r` is `[n,d]@[d,n] = [n,n]`. **HC's dynamic weights are generated per stream from that
stream's own `d`-dim vector**, whereas mHC's come from the flattened `nC` vector. Note also
`s_α` is *shared* between `A_m` and `A_r` (one scalar), while mHC has three separate `α`s.

### 3.3 Initialization (§2.3) — the exact values

> "In order to make the initialization of the hyper-connections **equivalent to the Pre-Norm residual
> connections**, we adopt the following initialization strategy. The dynamic parameters
> $\mathbf{W}_{\beta}$, $\mathbf{W}_{m}$, and $\mathbf{W}_{r}$ in Eqs. 11, 12, and 13 are
> **initialized to 0**, while the static matrices are initialized as follows:"

```latex
\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{B}^{k}\\
\mathbf{A_{m}}^{k}&\mathbf{A_{r}}^{k}\end{pmatrix}=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{1}_{1\times n}\\
\mathbf{e}_{k\bmod n}&\mathbf{e}_{n\times n}\end{pmatrix},   (14)
```

> "where $k$ is the index of the layer. $\bmod$ denotes the modulo operation."

So, per layer `k`:

| Parameter | Init |
|---|---|
| `B^k` (≡ `H^post`) | `1_{1×n}` — **all ones** |
| `A_m^k` (≡ `H^pre`) | `e_{k mod n}` — **one-hot on stream `k mod n`** |
| `A_r^k` (≡ `H^res`) | `e_{n×n}` — **identity matrix** |
| `W_β`, `W_m`, `W_r` (≡ `φ`) | **0** |

The PyTorch reference (Appendix J, Algorithm 2) confirms and supplies the missing scalar value:

```python
self.static_beta = nn.Parameter(torch.ones((rate,), device=device))
init_alpha0 = torch.zeros((rate, 1), device=device)
init_alpha0[layer_id % rate, 0] = 1.
self.static_alpha = nn.Parameter(torch.cat([init_alpha0, torch.eye((rate), device=device)], dim=1))
if self.dynamic:
    self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate+1), device=device))
    self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
    self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim, ), device=device))
    self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
self.layer_norm = LayerNorm(dim)
```

**`s_α = s_β = 0.01`** — the same value mHC's Appendix A.1 gives for its `α`. Note the pseudocode
uses `LayerNorm`, while the paper text says `norm(·)` and mHC specifies `RMSNorm`.

Two further training details from §4 "Implementation":

> "The static component in Eqs. 1, 11, 12, 13 **does not utilize weight decay**, whereas the dynamic
> component does."

> "Since the hyper hidden vectors of the final transformer block are ultimately summed, we ensure
> that the standard deviation (std) of the output ... remains consistent with the original. At
> initialization, we **scale the std of the weights of the output module at all layers**, including
> those of the second linear layer of the feedforward network and the output projector of the
> attention module, **by a factor of $\sqrt{n}$**."

### 3.4 Where the norm sits, and the collapse (Appendix I/J)

Algorithm 3 pins the ordering — width connection first, then the block's own norm:

```python
# Attention Block
mix_h, beta = atten_hyper_connection.width_connection(h)
h = attn_norm(mix_h[...,0,:])
h = self_attention(h)
h = atten_hyper_connection.depth_connection(mix_h, dropout(h), beta)
# FFN Block
mix_h, beta = ffn_hyper_connection.width_connection(h)
h = ffn_norm(mix_h[...,0,:])
h = ffn(h)
h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta)
```

with

```python
mix_h = alpha.transpose(-1, -2) @ h          # [n+1, d]; slot 0 = layer input, slots 1: = streams
...
def depth_connection(self, mix_h, h_o, beta):
    h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
    return h
```

So HC does the `pre` and `res` mixing in **one matmul** producing `n+1` rows, uses row 0 as the
(pre-normed) layer input and rows `1:` as the transformed residual. **Same `pre`-then-norm ordering
as mHC.**

Algorithm 1 gives the collapse:

```latex
\mathbf{h}^{L}\leftarrow\text{sum rows of }\mathbf{H}^{L}
\mathbf{h}^{L}\leftarrow\text{Normalization Layer}(\mathbf{h}^{L})
\mathbf{y}\leftarrow\text{Output Layer}(\mathbf{h}^{L})
```

**HC collapses by an unweighted row sum**, compensated by the `√n` output-init rescale. The
DeepSeek code instead uses a learned `pre`-style collapse (§2.7).

### 3.5 The Pre-Norm / Post-Norm seesaw argument (§1, §3.1)

> "In contrast, Post-Norm applies normalization after the output of each residual block, reducing the
> influence of a hidden state on subsequent layers. This approach can alleviate the issue of
> representation collapse but also reintroduces the problem of vanishing gradients. **The vanishing
> gradient and the representation collapse are like two ends of a seesaw**, with these two variants
> making respective trade-offs between these issues. The key issue is that residual connections,
> including both Pre-Norm and Post-Norm variants, **predefine the strength of connections** between
> the output and input within a layer." (§1)

Both are `n=1` non-trainable HC matrices, Eqs. (15)–(16):

```latex
\mathcal{HC}_{PreNorm}=\begin{pmatrix}0&1\\ 1&1\\ \end{pmatrix},   (15)
\mathcal{HC}_{PostNorm}=\begin{pmatrix}0&\frac{1}{\sqrt{\sigma_{i}^{2}+\sigma_{o}^{2}+2\sigma_{io}}}\\ 1&\frac{1}{\sqrt{\sigma_{i}^{2}+\sigma_{o}^{2}+2\sigma_{io}}}\\ \end{pmatrix},   (16)
```

> "where $\sigma_{i}$ and $\sigma_{o}$ denote the standard deviations of the input and output of the
> neural network layer, respectively, and $\sigma_{io}$ is the covariance between them."

And why `n > 1` is required (§2 + Appendix F):

> "the seesaw effect persists when $n=1$, and experiments show that it does not improve performance"
> (§2)

> "Note that HC$\times 1$ does not support the pattern of $\Lambda$ in its mathematical formulation,
> where the connections to previous layers must be weakened or strengthened simultaneously. Thus,
> the lack of connection from the early layers to the final layers may suffer from gradient
> vanishing, like post-norm style transformers, which leads to performance degeneration."
> (Appendix F)

### 3.6 Expansion-rate ablation (Table 1, OLMo-1B, 500B tokens)

| Method | V2 Loss ↓ | V2 PPL ↓ | V3 Loss ↓ | V3 PPL ↓ | Downstream Avg Acc ↑ |
|---|---|---|---|---|---|
| OLMo-1B (baseline) | 2.811 | 18.023 | 2.544 | 14.229 | 62.5 |
| DHC×1 w/o tanh | 2.822 | 18.270 | 2.556 | 14.428 | 62.3 |
| DHC×2 w/o tanh | 2.792 | 17.663 | 2.537 | 14.033 | 63.8 |
| DHC×4 w/o tanh | 2.779 | 17.451 | 2.516 | 13.844 | **64.4** |
| DHC×8 w/o tanh | **2.777** | **17.425** | **2.514** | **13.819** | 63.8 |
| DHC×1 | 2.819 | 18.125 | 2.556 | 14.418 | 62.3 |
| DHC×2 | 2.802 | 17.950 | 2.534 | 14.114 | 63.0 |
| DHC×4 | 2.781 | 17.509 | 2.514 | 13.826 | 63.8 |
| DHC×8 | 2.778 | 17.445 | 2.516 | 13.843 | 62.8 |

> "with an expansion rate of $n=1$, the performance of DHC is inferior to the baseline. However, for
> $n>1$, DHC significantly outperforms the baseline, achieving superior results at $n=4$, with the
> increase to $n=8$ providing minimal additional benefits." (§4.1)

Component ablation (Table 3, DHC×4; "✗" = not trainable from initialization):

| `WC` | `B` | Tanh | V2 Loss ↓ | V3 Loss ↓ | Down Avg ↑ |
|---|---|---|---|---|---|
| ✗ | ✓ | ✗ | 2.804 | 2.537 | 62.5 |
| ✓ | ✗ | ✗ | 2.781 | 2.518 | 63.6 |
| ✓ | ✓ | ✗ | 2.779 | 2.516 | **64.4** |
| ✗ | ✓ | ✓ | 2.802 | 2.532 | 63.4 |
| ✓ | ✗ | ✓ | 2.783 | 2.520 | 63.4 |
| ✓ | ✓ | ✓ | 2.781 | 2.515 | 63.8 |

Width connections (`WC`) matter far more than the output weights (`B`) — the same conclusion mHC's
Table 1 reaches for `H^res`. Also: SHC vs DHC — "At an expansion rate of 2, the improvements of DHC
and SHC are similar. However, at an expansion rate of 4, DHC performs notably better than SHC."

Setup: OLMo/OLMoE recipes, dolma-v1.5-sample for dense, OLMOE-MIX for MoE, all runs 500B tokens,
1B ablations, 7B and MoE 1B/7B confirmations, plus ViT-B/L image classification (300 epochs, 224²,
`n=2`) and vision generation.

### 3.7 Parameter overhead (Appendix B), Eqs. (21)–(26)

```latex
\left|\theta_{\texttt{SHC}}\right|=|\theta_{\mathbf{B}}|+|\theta_{\mathbf{A}}|=n+n\cdot(n+1)=n\cdot(n+2),   (21)
P_{\texttt{extra}}=\left|\theta_{\texttt{SHC}}\right|\times 2\times L,                                     (22)
\left|\theta_{\texttt{DHC}}\right|=|\theta_{\texttt{norm}}|+d_{\texttt{model}}\times(n+2)+n\times(n+2)+2,   (25)
```

Examples given: OLMo-1B-SHC×4 → `P_extra = 4×(4+2)×2×16 = 768`; OLMo-1B-DHC×4 →
`(0 + 2048×(4+2) + 4×(4+2) + 2)×2×16 = 394,048`.

Note `n·(n+2) = n²+2n` — the same `mix_hc` the DeepSeek code uses. Memory: HC adds
`2nsbd_model L`; "For $n=2$, this contributes less than $15\%$"; with recompute "the additional
memory requirement is reduced to $nsbd_{\text{model}}$".

### 3.8 Channel-wise / group variants in HC

**None.** The words "channel-wise" and "per-channel" do not occur. "Group" appears only in
Appendix H's *sequential-parallel duality* argument, in a completely different sense — grouping
*layers*, not channels:

> "We define a parallel-arranged network such that $n$ adjacent layers form a group, with layers
> within a group being parallel and groups arranged sequentially." (Appendix H)

So HC never proposes splitting `d` into heads or groups. All `n` streams are full-width `d`.

---

## 4. Frac-Connections — arXiv 2503.14125 (ByteDance Seed, 18 Mar 2025)

This is the closest published relative of a "multi-head" HC, so it gets full treatment.

### 4.1 The split (§4.1, Eq. 6)

> "This is achieved by generalizing the expansion rate to fractional values. When $n=1$,
> frac-connections are equivalent to hyper-connections. For $0<n<1$, frac-connections can be viewed
> as a fractional variant of hyper-connections that **divides the hidden states into $m=1/n$ parts
> instead of replicating them $n$ times**, where $m$ (referred to as the **frac-rate**) represents the
> number of partitions." (§4.1)

```latex
\mathbf{H}=\begin{pmatrix}\mathbf{h}_{1}&\mathbf{h}_{2}&\dots&\mathbf{h}_{m}\end{pmatrix}^{\intercal}=\texttt{Reshape}(\mathbf{h},(m,d/m)),   (6)
```

> "where $\mathbf{h}_{i}\in\mathbb{R}^{d/m}$ for $i=1,2,\dots,m$."

**Shapes.** `H ∈ R^{m×(d/m)}`; each fraction is `d/m = n·d` wide. **The total residual width stays
`d`**, against HC's `n·d`. The paper never states that sentence outright, but it follows from Eq. (6)
versus HC's Eq. (1) `H^0 ∈ R^{n×d}`.

The "concat" is literally a reshape, per Algorithm 1 (lines 5, 7, 8, 10):

```latex
\mathbf{H}^{0}\leftarrow\texttt{Reshape}\big{(}\mathbf{h}^{0},(m,d/m)\big{)}^{\intercal}\in\mathbb{R}^{m\times(d/m)}
\mathbf{h_{0}}^{k-1}\leftarrow\texttt{Reshape}({\mathbf{Y}^{k}}^{\intercal}\mathbf{H}^{k-1},(d,))
\mathbf{H}^{k}\leftarrow{\mathbf{B}^{k}}^{\intercal}\texttt{Reshape}\big{(}\mathcal{T}^{k}(\mathbf{h_{0}}^{k-1}),(m,d/m)\big{)}+{\mathbf{A}^{k}}^{\intercal}\mathbf{H}^{k-1}
\mathbf{h}^{L}\leftarrow\texttt{Reshape}\big{(}\mathbf{H}^{L},(m,d/m)\big{)}
```

Note **there is no collapse at all** — the stream reshapes back to `(d,)`. Line 10's printed target
shape is evidently a typo for `(d,)`. Contrast HC's sum-pooling and mHC's learned `pre` collapse.

Fig. 3(c) caption: "Frac-connections split the hidden representations into smaller fractions and
process each fraction independently. … These fractions are concatenated (denoted as Cat) after
processing, followed by integration into the main network pipeline."

### 4.2 The mixing (§4.1, Eqs. 7–8)

FC uses `B` (depth/output, ≡ HC's `B`), `Y` (read-out, ≡ HC's `A_m`), `A` (residual mixing, ≡ HC's
`A_r`). Note the matrix is **not square**, unlike HC's `(n+1)×(n+1)`:

```latex
\mathcal{FC}
=\begin{pmatrix}\mathbf{0}_{1\times m}&\mathbf{B}\\ \mathbf{Y}&\mathbf{A}\end{pmatrix}\in\mathbb{R}^{(m+1)\times(2\times m)}
=\begin{pmatrix}0&\cdots&0&\beta_{1}&\cdots&\beta_{m}\\
\gamma_{1,1}&\cdots&\gamma_{1,m}&\alpha_{1,1}&\cdots&\alpha_{1,m}\\
\vdots&\ddots&\vdots&\vdots&\ddots&\vdots\\
\gamma_{m,1}&\cdots&\gamma_{m,m}&\alpha_{m,1}&\cdots&\alpha_{m,m}\end{pmatrix}.   (7)
```

So `B ∈ R^{1×m}`, `Y ∈ R^{m×m}`, `A ∈ R^{m×m}`. **`Y` is `m×m`, where HC's `A_m` is `n×1`** — the
read-out is a full matrix because the layer input must be reassembled from all `m` fractions rather
than reduced to one vector.

```latex
\mathbf{H}^{k}=\mathcal{FC}^{k}(\mathcal{T}^{k},\mathbf{H}^{k-1})
={\mathbf{B}^{k}}^{\intercal}\mathcal{T}^{k}\big{(}{\mathbf{Y}^{k}}^{\intercal}\mathbf{H}^{k-1}\big{)}+{\mathbf{A}^{k}}^{\intercal}\mathbf{H}^{k-1}.   (8)
```

Mechanically: `Y^T H ∈ R^{m×(d/m)}` mixes **across the `m` fraction slots only** — each output
fraction is a `γ`-weighted sum of the `m` input fractions **at the same intra-fraction offset**.
There is no cross-offset (cross-channel-within-fraction) mixing anywhere. Then reshape to `(d,)`,
feed `T^k`, reshape the output back to `(m, d/m)`, scale fraction `i` by `β_i`, add `A^T H`.

The `B^T(·)` notation is loose; the released pseudocode (Algorithm 2 `depth_connection`) is an
elementwise per-fraction broadcast, not a matmul:

```python
h = beta[..., None] * h_o.reshape(h_o_shape[:-1] + (self.rate, h_o_shape[-1]//self.rate)) + mix_h[..., self.rate:, :]
```

and, as in HC and the mHC code, `[Y | A]` is **fused into one matmul** of shape `(m, 2m)`
(Algorithm 3: `h = mix_h[...,:self.rate,:].reshape(...)` takes the first `m` rows as the layer input,
the last `m` as the residual).

**No softmax, no sum-to-one, no normalization of the weights themselves.** The words "softmax",
"identity" and "one-hot" never appear. The only stabilizers are on the dynamic term: a norm, a
`tanh`, and a small learnable scale.

### 4.3 Static vs dynamic (§4.2, Eqs. 9–14) and initialization (§4.3, Eq. 15)

> "1. **Static Frac-Connections**: The weights are learnable, but static during testing.
> 2. **Dynamic Frac-Connections**: The weights are dynamically computed based on the input, allowing
> greater flexibility." (§4.2)

> "In practice, we follow that of DHC [28], combining the dynamic and static matrices to achieve DFC.
> The dynamic parameters are obtained through a linear transformation. To stabilize the training
> process, we introduce normalization before the linear transformation and apply the tanh activation
> function after it, scaling it by a small initial learnable factor."

```latex
\overline{\mathbf{H}} =\texttt{norm}(\mathbf{H})                                                            (11)
\mathcal{B}(\mathbf{H}) =s_{\beta}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{\beta})^{\intercal}+\mathbf{B}\in\mathbb{R}^{1\times m}   (12)
\mathcal{Y}(\mathbf{H}) =s_{\alpha}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{\gamma})+\mathbf{Y}\in\mathbb{R}^{m\times m}            (13)
\mathcal{A}(\mathbf{H}) =s_{\alpha}\circ\texttt{tanh}(\overline{\mathbf{H}}\mathbf{W}_{\alpha})+\mathbf{A}\in\mathbb{R}^{m\times m}            (14)
```

Init, §4.3 — same "equivalent to the Pre-Norm residual connections" goal as HC, with `W_β, W_γ, W_α`
"initialized to 0":

```latex
\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{B}\\ \mathbf{Y}&\mathbf{A}\end{pmatrix}=\begin{pmatrix}\mathbf{0}_{1\times 1}&\mathbf{1}_{1\times m}\\ \mathbf{e}_{m\times m}&\mathbf{e}_{m\times m}\end{pmatrix}.   (15)
```

`e_{m×m}` is never defined in prose; Algorithm 2 shows it is the identity:

```python
self.static_beta = nn.Parameter(torch.ones((rate,), device=device))
self.static_alpha = nn.Parameter(torch.cat([torch.eye((rate), device=device), torch.eye((rate), device=device)], dim=1))
self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim // self.rate, rate*2), device=device))
self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim // self.rate, ), device=device))
self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
self.dynamic_beta_scale  = nn.Parameter(torch.ones(1, device=device) * 0.01)
self.layer_norm = LayerNorm(dim // self.rate)
```

So: `Y = A = I_m`, `B = ones`, dynamic projections 0, `s_α = s_β = 0.01` (same value as HC and mHC).
**There is no `k mod n` one-hot init here** — identity blocks only, because `Y` is now square and the
identity already reassembles the full vector. Static components get no weight decay, dynamic do
(§4.3), copying HC.

Note the dynamic projection input dim is `dim // rate = d/m`: **each fraction predicts from its own
`d/m`-dim vector**, through a shared `W`. Like HC (per-stream, `d`-dim), unlike mHC (flattened `nC`).
The norm is `LayerNorm(d/m)` in code, costed as RMSNorm in §4.4 ("For RMSNorm, `|θ_norm| = d_model/m`").

Parameter counts, Eqs. (16)–(20): `|θ_SFC| = m + m·m + m·m = m·(2m+1)` (printed as `θ_SHC`, a typo);
`P_extra = |θ| × 2 × L`; `|θ_DFC| = |θ_norm| + d_model/m × (2m+1) + m·(2m+1) + 2`. Examples:
SFC×4 → `P_extra = 1152`; DFC×4 → `P_extra = 165,056`.

### 4.4 Stated relationship to HC with `n` copies

The claim is an *interpolation*, and explicitly **not** an equivalence except at the boundary:

> "However, Hyper-Connections increase memory access costs by expanding the width of hidden states.
> In this paper, we propose Frac-Connections, a novel approach that divides hidden states into
> multiple parts rather than expanding their width. Frac-Connections **retain partial benefits** of
> Hyper-Connections while reducing memory consumption." (Abstract)

> "This approach extends the expansion rate $n$ of Hyper-Connections (HC) to the fractional domain.
> In particular, **when $n=1$, Frac-Connections and Hyper-Connections are equivalent** ... the
> similarity between adjacent hidden states in FC lies between that of HC and baseline (Pre-Norm),
> indicating that their representational capacity follows the order: **HC $>$ FC $>$ Pre-Norm**." (§1)

> "Our Frac-Connections build upon this design by **reducing the hidden size of each stream**,
> retaining the benefits of Hyper-Connections without increasing memory usage." (§2)

Fig. 1 caption: "Frac-connections correspond to $n\leq 1$, while Hyper-Connections are defined by
$n\geq 1$. The two connection types become identical when the expansion rate is $n=1$."

And the one head-to-head experiment (§5.2):

> "we observe that Hyper-Connections (OLMoE-7B-DHC$\times$4) **converge significantly faster** than
> Frac-Connections (OLMoE-7B-DFC$\times$4), suggesting that when applying HC or FC, a trade-off
> between memory consumption and performance needs to be considered."

Memory saving is never quantified in bytes. Only params (Table 1: +0.014% OLMo-1B2-DFC×4, +0.0024%
OLMoE-1B-7B-DFC×4) and FLOPs (Table 2: +0.044%, +0.056%) are tabulated.

### 4.5 Can it be read as "multi-head" HC?

**The paper never frames it that way.** The words **head, heads, group, grouped, chunk, channel,
per-channel, per-group do not occur anywhere** (the only "head" matches are inside "overhead").
Verified against both the tag-stripped render and an independent re-extraction.

The framings it does use: "divides hidden states into multiple parts rather than expanding their
width" (Abstract); "partitions the hidden states into multiple fractions" (§1); "split $\mathbf{h}$
into $m=1/n$ parts" (§4.1); "$m$ (referred to as the frac-rate) represents the number of partitions"
(§4.1); "process each fraction independently" (Fig. 3(c)); and once, "reducing the hidden size of
each **stream**" (§2). The nearest precedent it cites:

> "FractalNet [13] proposes partitioning the hidden states into multiple **segments**, each processed
> by networks of varying depths… Frac-Connections share a similar design principle; however, instead
> of assigning each partition to a different depth, we associate them with different connection
> weights." (§2)

**So: FC is a channel-split HC in mechanism but is presented as a fractional expansion rate, not as
multi-head.** Two structural facts matter for anyone reading it as a template for multi-head mHC:

1. The weights are **per-fraction, not per-channel**: `γ_{i,j}`, `α_{i,j}`, `β_i` are scalars indexed
   by fraction slot, and mixing happens only across the `m` slots at the same intra-fraction offset.
2. FC **replaces** width expansion rather than subdividing it. A "multi-head mHC" — `n` full-width
   streams whose mixing matrices differ per channel group — is a *different* object from FC, and is
   proposed by neither paper.

### 4.6 Experiments

> "for sparse models we study Sparse Mixture-of-Experts (MoE) models ... conducting ablation studies
> on **OLMoE-1.3B**, which has 1.3B total parameters with 260M activated parameters. We further
> validate the effectiveness of our approach on a larger sparse model, **OLMoE-7B**, which has 7B
> total parameters with 1.3B activated parameters. For dense models, we follow the OLMo2 training
> setup to pre-train a **1B2** parameter model. Importantly, all experiments were conducted without
> hyperparameter tuning" (§5)

- **Tokens**: 3T for OLMoE-7B; 2T for OLMo2-1B2. Ablation token budget **not stated**.
- **Frac-rates tried**: DFC×2, DFC×4 (`m = 2, 4`, i.e. `n = 1/2, 1/4`), plus SFC×4. No other `m`.
- **Baselines**: the Pre-Norm models, plus **OLMoE-7B-DHC×4** as an upper reference.
- Ablations (§5.1): "DFC$\times$2 demonstrates significant improvement over the baseline, while
  DFC$\times$4 offers only marginal additional gains"; DFC×4 training loss −0.014 vs baseline; DFC×4
  beats SFC×4. Component importance: "removing rescaling … causes the most severe performance
  degradation, followed by the removal of tanh activation …, while the absence of normalization …
  results in the least detrimental effect."
- MoE (§5.2): OLMoE-7B-DFC×4 training loss −0.012. Table 3 (3T tokens) avg 68.30 → 68.65;
  WinoGrande 67.64 → 68.59, MMLU Var 41.83 → 42.33, Commonsense QA 49.14 → 49.80; **BoolQ regresses**
  72.87 → 72.11.
- Dense (§5.3): Table 4 (2T tokens) avg 62.5 → 63.2; BoolQ +2.1%, WinoGrande +1.1%, SciQ +0.4%.

Complexity: "The primary computational cost of both SFC and DFC occurs in line 5 of Algorithm 1, with
a complexity of $\mathcal{O}(d_{\text{model}}\times 4m)$".

---

## 5. MUDDFormer — arXiv 2502.12170 (scoped to per-stream weight generation)

### 5.1 The generator

Section 2.2 gives the single-way dynamic generator, Eq. (6):

```latex
\mathcal{A}_{i}(X_{i})=\textrm{GELU}(\textrm{RMSNorm}(X_{i})W_{1})W_{2}+a_{i}   (6)
```

> "We instantiate $\mathcal{A}_{i}:\mathbb{R}^{D}\rightarrow\mathbb{R}^{i+1}$ with an MLP
> parameterized by $W_{1}$ and $W_{2}$ which computes connection weights **position-wise** ... We
> apply RMSNorm to $X_{i}$ before MLP to stabilize training. We also add a static weight vector
> $a_{i}$ acting as learnable prior for dense connectivity. The trainable parameters are
> $\theta_{i}^{d}=\{W_{1}\in\mathbb{R}^{D\times(i+1)},W_{2}\in\mathbb{R}^{(i+1)\times(i+1)},a_{i}\in\mathbb{R}^{i+1}\}$."

So: input is `X_i`, the **output of layer `i`** only (not a pooled or concatenated history);
**RMSNorm** first (not LayerNorm); **two matmuls with GELU between**; plus a learnable static prior
`a_i`. **No tanh, no softmax, no separate learnable output scale.**

Appendix A states what was deliberately left out, confirming the above:

> "• Keys are independent of input; • A learnable positional bias $a_{i}$ is used; • **Softmax is
> removed.** Instead, GELU activation is applied to query (more like linear attention); • $W^{V}$
> transformation is not used. ... we empirically found that adding more sophisticated ingredients in
> DA (e.g. input dependent keys, softmax) does not bring improvement and slow down training."

### 5.2 Multiway (per-stream Q/K/V/R) instantiation

§2.3:

> "we first turn a normal Transformer block $\operatorname{B}(X)$ into a multi-input one
> $\operatorname{B}^{\prime}(X^{Q},X^{K},X^{V},X^{R})$ by decoupling its input into four streams for
> query, key, value and residual, respectively (Eq. (7), Figure 2(e) bottom), and then instantiate
> **four DA modules, each specializing in one stream's dense connectivity** (Eq. (8), Figure 2(d))"

with the footnote "This is a logical view for clarity. In practice, these DSs can be combined for
efficiency."

The concrete multiway shapes appear **only in Appendix B's pseudocode** — one shared MLP emitting
all four streams at once, then reshaped:

```python
# B = batch_size; T = seq_len; D = model_dim
# L = layer_index; C = num_ways = 4; K = DA_hidden_dim = C*(L+1)
def generate_dw(x, mudd_theta):          # x: BxTxD
    w1, w2, a = mudd_theta               # w1: DxK, w2: Kx(C*(l+1)), a: Cx(l+1)
    dw = GELU(RMSNorm(x) @ w1) @ w2 + a
    dw = rearrange(dw, 'B T (C L)-> C B T L', C=4)
    return dw
```

called as `dw = generate_dw(Xs[-1], mudd_theta)` — i.e. on `X_i`. So for block `i`:
`W_1 ∈ R^{D×K_i}` with `K_i = 4(i+1)`, `W_2 ∈ R^{K_i×4(i+1)}`, `a ∈ R^{4×(i+1)}`.

Block and wiring, Eqs. (7)–(8):

```latex
\begin{split}X_{\textrm{A}}^{\prime}=\operatorname{MHA}(\textrm{LN}(X^{Q}),\textrm{LN}(X^{K}),\textrm{LN}(X^{V}))&+X^{R}\\
\operatorname{B}^{\prime}(X^{Q},X^{K},X^{V},X^{R})=\operatorname{FFN}(\textrm{LN}(X_{\textrm{A}}^{\prime}))&+X_{\textrm{A}}^{\prime}\\ \end{split}   (7)
```

```latex
\begin{split}\overline{X}_{0}^{Q}=\overline{X}_{0}^{K}&=\overline{X}_{0}^{V}=\overline{X}_{0}^{R}=X_{0}=\operatorname{Embedding(X)}\\
X_{i}&=\operatorname{B}^{\prime}_{i}(\overline{X}_{i-1}^{Q},\overline{X}_{i-1}^{K},\overline{X}_{i-1}^{V},\overline{X}_{i-1}^{R});\\
\overline{X}_{i}^{Q},..,\overline{X}_{i}^{R}&=\operatorname{DA}_{i}^{Q}(X_{:i}),..,\operatorname{DA}_{i}^{R}(X_{:i}),\;i\in[1,L]\\
&\operatorname{MUDDFormer}(X)=\overline{X}_{L}^{R}\end{split}   (8)
```

All four per-stream DAs take the **same** `X_{:i}` as values and the **same** `X_i` as the
weight-generating input; only the weights differ per stream.

### 5.3 Per-position dependence

Abstract:

> "Unlike existing dense connection approaches with static and shared connection weights, MUDD
> generates connection weights **dynamically depending on hidden states at each sequence position**
> and for each decoupled input stream (the query, key, value or residual) of a Transformer block."

§2.2:

> "Dynamic dense connections expand the connection weight for $X_{j}$ from a static scalar $a_{ij}$
> to a vector $A_{ij}\in\mathbb{R}^{T}$, allowing $X_{j}$ to contribute differentially to each
> position $t\in[1,T]$ of $\overline{X}_{i}$ based on the hidden state $X_{i}[t]\in\mathbb{R}^{D}$
> at that position."

Appendix A makes the position-locality explicit, Eq. (12):

```latex
(\operatorname{GELU}(X_{i}[t]W_{1})W_{2}+a_{i})X_{:i}[t]   (12)
```

### 5.4 Depth dimension and aggregation

Block `i` mixes over `i+1` previous outputs (`X_{:i} := {X_0,...,X_i}`, embedding included, all-to-all).
Per stream the weight tensor is `A_i ∈ R^{T×(i+1)}` — `i+1` scalars per position per stream; stacked
over the 4 streams, `4 × T × (i+1)`.

Dynamic aggregation, Eq. (5):

```latex
\begin{split}\overline{X}_{i}=&\operatorname{DA}_{i}^{\textrm{dynamic}}(X_{:i};\theta_{i}^{d})\\
= &\operatorname{wsum}(\stackrel{{T\times(i+1)}}{{A_{i}}}=\mathcal{A}_{i}(\stackrel{{T\times D}}{{X_{i}}}),\stackrel{{(i+1)\times T\times D}}{{X_{:i}}})\\
:= &\sum_{j=0}^{i}\stackrel{{T\times 1}}{{A_{ij}}}\odot\stackrel{{T\times D}}{{X_{j}}}\;(\textrm{with broadcasting})\end{split}   (5)
```

Static predecessor for contrast, Eq. (4): `\sum_{j=0}^{i} a_{ij} X_{j}` with `a_i ∈ R^{i+1}`.

The weights are **unnormalized** — they can be negative and need not sum to 1. This is precisely the
property mHC's `pre`/`post` sigmoids and `res` Sinkhorn are designed to remove, and it is why the
mHC paper groups MUDDFormer with HC as compromising identity mapping (§2.2 of mHC).

### 5.5 Initialization (§3, Implementation Details)

> "We initialize the MUDD connection weight generating parameters $W_{1}$ and $W_{2}$ with
> $\mathcal{N}(0,\frac{1}{D})$ and **0** respectively, and initialize the static weight vector
> $a_{i}$ with **1 at $a_{ii}$ and 0 elsewhere**. This reduces MUDDFormer to Transformer at the
> beginning of training, which is found to be **critical for good performance**."

Same design as HC's Eq. (14): zero the *last* projection so the dynamic term vanishes, and set the
static prior to the identity wiring.

### 5.6 Bottleneck dimension

`K_i = 4(i+1)`, tiny relative to `D` but *equal* to the output width, so not a bottleneck relative
to the output. The paper never uses "low-rank" or "bottleneck". Appendix C: average
`\overline{K}=4(\overline{L}+1)=4(\frac{L+1}{2}+1)=2(L+3)`.

### 5.7 Parameter and compute cost

Headline: "MUDDPythia-2.8B matches Pythia-6.9B in pretraining ppl and downstream tasks and even
rivals Pythia-12B in five-shot settings, while adding only **0.23% parameters and 0.4%
computation**."

With `η = (L+3)/D`, `ρ = T/D`, baseline params `12LD²`, baseline FLOPs `2LDT(12D+T)`:

```latex
R_{\Delta params}=\frac{\sum_{i=1}^{L}(\overbrace{DK_{i}}^{W_{1}}+\overbrace{K_{i}^{2}}^{W_{2}})}{12LD^{2}}   (13)
R_{\Delta params}\approx\frac{\overline{K}}{12D}=\frac{2(L+3)}{12D}=\frac{L+3}{6D}=\frac{\eta}{6}            (14)
R_{\Delta FLOPs}=\frac{\sum_{i=1}^{L}(\overbrace{2TDK_{i}+2TK_{i}^{2}}^{\text{generate dense weight}}+\overbrace{2TDK_{i}}^{\text{Depthwise Aggregate}})}{2LDT(12D+T)}   (15)
R_{\Delta FLOPs}\approx\frac{\eta}{3+\rho/4}                                                                  (16)
```

Table 1: 1.4B → 0.22% / 0.38%; 1.34B DeepNarrow → 0.49% / 0.8%; 2.8B → 0.23% / 0.4%;
6.9B → 0.14% / 0.26%.

**Measured wall-clock overhead is far larger than the FLOPs estimate** (Table 4): relative training
throughput 89.8% / 84.0% / 95.6% and inference 88.1% / 90.0% / 94.0% at 1.3B / 2.8B / 6.9B.

> "The overheads primarily stem from the series of small operations and additional I/O introduced by
> DA modules. We believe that kernel fusion techniques offer potential for further acceleration and
> leave it for future work."

This is the same diagnosis mHC §3.2 makes about HC — and mHC's answer (fused kernels) is the work
MUDDFormer leaves undone. Extra activation memory `6BTD + 2LBTD`, measured 17–29%.

---

## 6. DeepSeek-V4 report — arXiv 2606.19348

### 6.1 §2.2, the mHC restatement

HC in V4's notation, Eq. (1):

```latex
X_{l+1}=B_{l}X_{l}+C_{l}\mathcal{F}_{l}(A_{l}X_{l}),   (1)
```

with `X_l ∈ R^{n_hc×d}`, `A_l ∈ R^{1×n_hc}`, `B_l ∈ R^{n_hc×n_hc}`, `C_l ∈ R^{n_hc×1}`.

> "Note that the actual layer input $A_{l}X_{l}\in\mathbb{R}^{d}$ is also $d$-dimensional, so the
> expanded residual width does not influence the design of the inner layers. HC decouples the
> residual width from the actual hidden size, offering a complementary scaling axis with minimal
> computational overhead, as $n_{\text{hc}}$ is typically much smaller than the hidden size $d$."

Manifold, Eq. (2):

```latex
B_{l}\in\mathcal{M}\coloneq\{M\in\mathbb{R}^{n\times n}\mid M\mathbf{1}_{n}=\mathbf{1}_{n},\;\mathbf{1}_{n}^{T}M=\mathbf{1}_{n}^{T},\;M\geqslant 0\}.   (2)
```

Dynamic parameterization — "Given the input $X_{l}\in\mathbb{R}^{n_{\text{hc}}\times d}$, it is
first flattened and normalized:
$\hat{X}_{l}=\operatorname{RMSNorm}(\operatorname{vec}(X_{l}))\in\mathbb{R}^{1\times n_{\text{hc}}d}$",
Eqs. (3)–(5):

```latex
\tilde{A}_{l} =\alpha_{l}^{\mathrm{pre}}\cdot(\hat{X}_{l}W^{\mathrm{pre}}_{l})+S_{l}^{\mathrm{pre}},                  (3)
\tilde{B}_{l} =\alpha_{l}^{\mathrm{res}}\cdot\operatorname{Mat}(\hat{X}_{l}W^{\mathrm{res}}_{l})+S_{l}^{\mathrm{res}},  (4)
\tilde{C}_{l} =\alpha_{l}^{\mathrm{post}}\cdot(\hat{X}_{l}W^{\mathrm{post}}_{l})^{T}+S_{l}^{\mathrm{post}},            (5)
```

`W^pre_l, W^post_l ∈ R^{n_hc d × n_hc}`, `W^res_l ∈ R^{n_hc d × n_hc²}`; `S^pre_l ∈ R^{1×n_hc}`,
`S^post_l ∈ R^{n_hc×1}`, `S^res_l ∈ R^{n_hc×n_hc}`; and "$\alpha_{l}^{\mathrm{pre}}$,
$\alpha_{l}^{\mathrm{res}}$, $\alpha_{l}^{\mathrm{post}}\in\mathbb{R}$ are learnable gating factors
**initialized to small values**" — again no number.

Constraints, Eqs. (6)–(8):

```latex
A_{l} =\sigma(\tilde{A}_{l}),    (6)
C_{l} =2\sigma(\tilde{C}_{l}).   (7)
M^{(t)}=\mathcal{T}_{r}(\mathcal{T}_{c}(M^{(t-1)})),   (8)
```

> "the Sinkhorn-Knopp algorithm, which first applies an exponential function to $\tilde{B}_{l}$ to
> ensure positivity, getting $M^{(0)}=\exp(\tilde{B}_{l})$, and then iteratively performs column and
> row normalization ... We choose $t_{\text{max}}=20$ as a practical value."

Identical to the mHC paper, including the column-then-row reading of Eq. (8) that the kernel
contradicts. **No `tanh`**, consistent with mHC and against HC.

> "In addition, the input transformation $A_{l}$ and output transformation $C_{l}$ are also
> constrained to be non-negative and bounded via a Sigmoid function to avoid the risk of signal
> cancellation."

### 6.2 Infrastructure (§3.4.2)

> "Firstly, we carefully design and implement **fused kernels** of mHC for both training and
> inference. Secondly, we introduce a **recomputation strategy** that selectively checkpoints
> intermediate tensors. Specifically, we **recompute most hidden states between layers and all
> normalized layer inputs**, while avoiding recomputation of compute-intensive operations. ...
> Thirdly, we adjust the **DualPipe 1F1B** overlapping scheme to accommodate the increased pipeline
> communication and enable concurrent execution of some operations in mHC.
>
> Collectively, these optimizations constrain the wall-time overhead of mHC to only **6.7% of the
> overlapped 1F1B pipeline stage**."

### 6.3 Model setups (§4.2.1)

Both V4-Flash and V4-Pro: "the expansion factor $n_{\text{hc}}$ is set to 4, and the number of
Sinkhorn-Knopp iterations $t_{\text{max}}$ is set to 20." V4-Flash: 284B total / 13B active.
V4-Pro: 61 layers, `d = 7168`, 1.6T total / 49B active.

Optimizer (§4.2.2), relevant because it decides what the `hc_*` parameters are trained with:

> "We employ the **Muon** optimizer for the majority of parameters, but use the **AdamW** optimizer
> for the embedding module, the prediction head module, and **the weights of all RMSNorm modules**."

AdamW: `β₁=0.9, β₂=0.95, ε=1e-20, weight_decay=0.1`. Muon: momentum 0.95, weight decay 0.1, update
RMS rescaled to 0.18. The report does **not** say which optimizer the `hc_*` tensors use — the
`α`/`b` scalars and vectors are not matrices, so Muon does not naturally apply to them. Not stated.

### 6.4 Training instability (§4.2.3) — not mHC-related

The two techniques given (Anticipatory Routing; SwiGLU clamping to `[-10,10]` with gate capped at
10) are attributed to MoE routing outliers, **not** to mHC. mHC is presented as already-solved
stability. Worth noting: mHC did not remove all instability at the 1.6T scale.

---

## 7. Implementation checklist for a faithful small-scale mHC

Every choice needed to reimplement, with the source for each. `n` = expansion rate, `C` = width,
`L` = layers.

### 7.1 Shapes and parameters (per mHC sublayer — attention and FFN are separate sublayers)

| Parameter | Shape | dtype | Source |
|---|---|---|---|
| `φ_l` (fused `[pre\|post\|res]`) | `[nC, n²+2n]` (code stores `[n²+2n, nC]`) | fp32 storage; tf32 compute | mHC Eq. (10); code `hc_*_fn` |
| `b_l` (fused bias) | `[n²+2n]` | fp32 | mHC Eq. (13); code `hc_*_base` |
| `α^pre, α^post, α^res` | 3 scalars | fp32 | mHC Eq. (12); code `hc_*_scale` `[3]` |

Slice layout within the `n²+2n` vector, from the kernel: `pre = [0:n]`, `post = [n:2n]`,
`res[j,k] = [j*n + k + 2n]` (row-major). `α` index order is `(pre, post, res)`.

### 7.2 Forward, per sublayer

1. `x` is `[..., n, C]`. Flatten to `[..., nC]`, **cast to fp32**.
2. `rsqrt = 1/sqrt(mean(x²) + norm_eps)`, one scalar per token, over the **whole `nC` vector**.
   `norm_eps = 1e-20` (mHC Appendix A.1; V4.1 config `rms_norm_eps`).
   **No learned RMSNorm weight** — it is absorbed into `φ` (mHC §4.3.1; code has none).
3. `mixes = (x @ φ) * rsqrt` — norm divide **after** the matmul (mHC Eq. 14–15).
4. `raw = mixes * α[slice] + b` — bias added **after** the rsqrt scaling (mHC Eq. 16).
5. `pre   = sigmoid(raw_pre) + eps` (mHC Eq. 17 + code's `eps`; `eps = hc_eps = 1e-6`).
6. `post  = 2 * sigmoid(raw_post)` (mHC Eq. 18; no eps).
7. `res   = Sinkhorn(raw_res)`, per token, in fp32 (mHC Eq. 19), as:
   - `M = softmax(raw_res, dim=-1) + eps`   (fuses `exp` + first row-normalize)
   - `M = M / (M.sum(-2) + eps)`            (first column-normalize)
   - repeat `t_max - 1 = 19` times: `M = M / (M.sum(-1) + eps)`; `M = M / (M.sum(-2) + eps)`
   - `t_max = 20` (mHC §4.2 and Appendix A.1). Rows of `res` first, columns last — where `res` is
     stored **transposed** relative to the paper's `H^res` (`res[j,k] = H^res[k,j]`), so this is
     Eq. (9)'s `T_r(T_c(·))` on `H^res`. Getting this transpose wrong conserves the forward gain
     instead of the backward one, silently. See ambiguity 1.
8. Collapse: `y = sum_j pre[j] * x[j]` → `[..., C]`.
9. **Then** the block's own pre-norm RMSNorm on `y`, then `F` (attention or FFN).
10. Expand and merge: `x_out[k] = post[k] * F(...) + sum_j res[j,k] * x[j]`.
    Note the `res` contraction is over its **first** index (`comb^T @ residual` in the code).

### 7.3 Stream in and out

- **In**: repeat the embedding `n` times (`h.unsqueeze(2).repeat(1,1,n,1)`) — HC §2.1, both codebases.
- **Out**: a **learned** collapse, then the final RMSNorm, then the unembedding. Two shipped
  variants — pick one deliberately:
  - *V4*: a dedicated `pre`-only mHC layer in the head (`φ_head ∈ R^{n×nC}`, `b_head ∈ R^n`, one
    scalar `α_head`; `sigmoid + eps`; no Sinkhorn).
  - *V4.1*: reuse the last FFN sublayer's shifted `pre`; no head parameters.
  - *HC paper*: unweighted row **sum**, with all output-projection init stds scaled by `√n`. Do not
    mix this with the learned collapse — the `√n` rescale exists only to compensate the sum.

### 7.4 Pre-mix shifting (V4.1 only)

If imitating V4.1-Flash: each sublayer computes its mixes from `x` *before* `hc_pre`, applies the
`pre` it received from the previous sublayer, and passes its own `pre` forward. Seed with an exact
one-hot on stream 0 (`make_identity_pre_mix`, **not** learned, not a sigmoid). `Block.forward`
returns `(x, ffn_pre)`. If imitating V4/the papers, generate and apply `pre` in the same sublayer.

### 7.5 Initialization

Only `α^pre = α^post = α^res = 0.01` is stated by the mHC paper (Appendix A.1), matching HC's
`dynamic_alpha_scale = dynamic_beta_scale = 0.01`. Everything else must be carried over from HC by
analogy — see §7.7 for why this is not mechanical.

Target (HC Eq. 14, "equivalent to the Pre-Norm residual connections"):

| Quantity | HC target at init | mHC parameterization | Achievable? |
|---|---|---|---|
| `φ^{pre,post,res}` | `W = 0` | `φ = 0` | **Yes** — zero `φ` exactly; kills all dynamics at step 0 |
| `H^post` | `1_{1×n}` (ones) | `2σ(b_post)` | **Yes, exactly**: `b_post = 0` → `2σ(0) = 1` |
| `H^pre` | `e_{k mod n}` (one-hot) | `σ(b_pre)` | **No** — `σ` never reaches 0 or 1 |
| `H^res` | `I_n` | `Sinkhorn(exp(b_res))` | **No** — needs `b_res` diagonal → `+∞` |

That `H^post` lands *exactly* on HC's init value is almost certainly why the factor 2 is in
`2σ(·)`: it puts the HC init at the sigmoid's midpoint, where the gradient is largest. `b_post = 0`
is therefore a well-supported inference, even though the paper does not state it.

For the other two, reasonable faithful choices, all of which must be flagged as **inferred**:
- `b_pre = 0` → `H^pre = 0.5·1_n` (uniform). Compare mHC Table 1, which uses "uniform weights of
  $1/n$" as the *disabled* `H^pre`, so uniform is at least a sane baseline.
- `b_pre = c·e_{k mod n}` for a moderate `c` → a soft version of HC's one-hot. Note V4.1's
  non-learned seed mix *is* an exact one-hot on stream 0, which is weak evidence that a one-hot-ish
  `pre` is wanted at init.
- `b_res = τ·I_n` for a moderate `τ` → Sinkhorn gives a diagonally-dominant doubly stochastic matrix,
  approaching `I` as `τ` grows. `b_res = 0` gives the uniform matrix `1/n · J`, i.e. maximal mixing —
  the opposite of HC's identity init.

### 7.6 Training details worth carrying over

- Per-sublayer, not per-block: separate `φ`, `b`, `α` for attention and FFN.
- Static components (`b`) **without** weight decay; dynamic (`φ`) with it (HC §4 Implementation).
  Not restated by mHC.
- All mHC coefficient math in **fp32**; only the stream itself in bf16 (mHC Eq. 10–19; code casts
  `.float()` at every entry point).
- Recompute: store `x_{l0}` every `L_r ≈ sqrt(nL/(n+2))` layers, plus `F(·)` every layer; recompute
  `x_l`, `H^pre x_l`, `RMSNorm(H^pre x_l)` (mHC Table 3, Eq. 20). Irrelevant at small scale but
  needed if the point is to see the real mechanism.

### 7.7 Ambiguities between paper and reference code — flagged

1. **Sinkhorn iteration order — NOT an ambiguity, but a transpose trap.** At first reading the
   kernel (row-normalize first) looks to contradict Eq. (9) (`T_r(T_c(·))`, i.e. column first). It
   does not. Because `comb = (H^res)^T`, a row normalization of `comb` *is* a column normalization of
   `H^res`, so the kernel implements Eq. (9) exactly; verified numerically to `5.1e-6` (§2.5). Both
   also agree with §5.4's claim that the *backward* gain is the one that drifts. **The real risk is
   transposing `res` by accident**, which silently swaps which gain is conserved — forward instead of
   backward — while still producing a doubly-stochastic-looking matrix and no error. Decide the
   convention once: store `comb` as `(H^res)^T`, normalize rows-then-columns of `comb`, and apply it
   as `Σ_j comb[j,k]·x[j]`.
2. **Epsilons.** Eq. (17) is `σ(·)` but the kernel is `σ(·) + eps`. The kernel also adds `eps` after
   the row softmax and to every row/column sum before dividing. `hc_eps = 1e-6`, distinct from
   `norm_eps = 1e-20`. The papers never mention any epsilon in the projection.
3. **Where `pre` is applied (`V4` vs `V4.1`).** The papers' Eq. (3)/(7) apply `H^pre_l` to `x_l` at
   the same layer, which is V4/V4-Pro's code. V4.1-Flash defers `pre` by one sublayer and seeds with
   a non-learned one-hot. Undocumented in any paper; it is what makes the two-scan kernel fusion of
   Eq. (14)–(15) possible.
4. **How the stream is collapsed.** Three different answers across the sources: HC sums rows
   (+ `√n` output init rescale); V4 uses a dedicated learned `pre`-only head layer; V4.1 reuses the
   last sublayer's `pre`. The mHC paper says nothing at all.
5. **Initialization of `b^pre`, `b^post`, `b^res` and `φ`.** Not stated in the mHC paper, not
   stated in the V4 report, and absent from the code (inference-only, `torch.empty`). Only
   `α = 0.01` is given. `b_post = 0` is a safe inference; the rest are genuinely open, and
   HC's one-hot `pre` / identity `res` targets are **unreachable** under sigmoid/Sinkhorn.
   This is the largest single gap.
6. **RMSNorm learned weight.** The paper's Eq. (7) writes `RMSNorm(·)` and §4.3.1 says its weight is
   "absorbed in `φ_l`". The code has no weight at all. For a from-scratch implementation these are
   equivalent (a diagonal rescale immediately followed by a free linear map), so implement it
   without a weight — but note that a parameter named in the equation does not exist as a tensor.
7. **`tanh` presence.** HC Eq. (11)–(13) has `tanh` on the dynamic term; mHC Eq. (7) and V4
   Eq. (3)–(5) do not. The mHC paper says it "follow[s] the original HC formulation" while in fact
   dropping the `tanh`. HC's own Table 3 shows `tanh` is roughly neutral, so dropping it is
   defensible, but the "follows HC" phrasing is misleading. Implement **without** `tanh`.
8. **Where the dynamic weights are read from.** HC reads each stream's own `C`-dim row
   (`θ^res ∈ R^{n×C}`); mHC flattens to `nC` (`φ^res ∈ R^{nC×n²}`). This changes the parameter count
   from `O(nC)` to `O(n³C)`-ish and is a substantive change, stated only as "to preserve full context
   information".
9. **Optimizer assignment.** V4 uses Muon for most parameters, AdamW for embeddings, head and
   RMSNorm weights. Which group the `hc_*` tensors fall into is **not stated**.
10. **`α` sharing.** The code has one `hc_*_scale` of shape `[3]` per sublayer — one `α` each for
    pre/post/res, matching Eq. (12). HC instead shared a single `s_α` across `A_m` and `A_r`. If
    porting an HC implementation, split the scalars.
11. **No mHC-side ablations exist.** The paper's only ablation table is the *HC* component study
    (Table 1). There is no mHC ablation of static vs dynamic, of `n`, of `t_max`, or of the
    sigmoid/Sinkhorn choices. Any claim that a component "matters" in mHC is an inference from HC.
12. **Multi-head / grouped mHC is not in any of these sources.** Neither mHC, HC, nor
    Frac-Connections mentions heads, groups, or per-channel mixing weights — verified by keyword
    search of all three. mHC's only stated future direction is *other manifolds* (§6). The one
    channel-splitting precedent is Frac-Connections, but it splits `C` into `m` fractions
    *instead of* widening the stream (total width stays `C`), its weights are per-fraction scalars,
    and it mixes only across fraction slots at the same intra-fraction offset. A multi-head mHC —
    `n` full-width streams with mixing matrices that differ per channel group — is a distinct
    object that none of these papers proposes. Treat it as new design, not as a documented variant.

### 7.8 Cross-source comparison of the five design axes

| Axis | HC | Frac-Connections | mHC / V4 | V4.1-Flash code |
|---|---|---|---|---|
| Stream shape | `n × C` (repeat) | `m × C/m` (split) | `n × C` (repeat) | `n × C` (repeat) |
| Dynamic weights read from | each stream's own `C` | each fraction's own `C/m` | flattened `nC` | flattened `nC` |
| Activation on dynamic term | `tanh` | `tanh` | none | none |
| Constraint on `pre`/`post` | none | none | `σ`, `2σ` | `σ+eps`, `2σ` |
| Constraint on `res` | none | none | Sinkhorn (20 it.) | Sinkhorn (20 it., on `res^T`) |
| Collapse to head | row **sum**, `√n` init rescale | reshape (no collapse) | not stated | learned `pre` (shifted / dedicated) |
| `pre` applied at | layer `l` | layer `l` | layer `l` | layer `l−1`'s coefficients |
| Scalar init | `s = 0.01` | `s = 0.01` | `α = 0.01` | not in code |
| Static init | `post=1`, `pre=e_{k mod n}`, `res=I` | `B=1`, `Y=A=I` | not stated | not in code |
