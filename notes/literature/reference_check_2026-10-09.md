# Reference check, 2026-10-09

All 41 references cited in `paper/main.tex` were checked against their primary sources. For each one I checked:

- the metadata: title, authors, year, arXiv ID and version, and venue;
- every sentence of the paper that cites it.

The four ICLR 2027 submissions had been read only as abstracts. I have now read them in full from OpenReview.

## The four OpenReview submissions

### uHC, `Qnj7Lf8Bz2`

Submission 60332, posted 2026-09-19, modified 2026-10-05, 15 pages.

- **Read.** The group-wise readout uses G = n = 4 groups. The coefficients of each group come from that group's own slice of
  the n streams, through a sigmoid. That is our per-head read with a local predictor, and it has the parameter count of mHC.
- **Write.** The low-rank write-back has rank 8 and adds 2nCr parameters per layer. It has no parameter-matched control.
- **Mix.** `H_res` is the Sinkhorn mix of mHC, unchanged.
- **Setup.**
  - Code base: mHC-lite nanoGPT on OpenWebText.
  - Sizes: 46M, 128M and 363M parameters.
  - Learning rate: one per scale, shared by all methods.
  - No seed count is reported, and the best validation checkpoint is used.
- **Table 1.**

  | Scale | Residual | mHC | uHC |
  |---|---|---|---|
  | 46M | 3.4505 | 3.3805 | 3.3747 |
  | 128M | 3.1681 | 3.0892 | 3.0814 |
  | 363M | 2.8536 | 2.9470 | 2.8188 |

  The gain over mHC is 0.006 and 0.008 at the two smaller sizes and 0.128 at 363M. At 363M the mHC baseline is 0.093 worse than
  the residual.
- **Table 3, at 363M.**
  - Group readout alone: 2.8843, still worse than the residual.
  - Write-back alone: 2.8629.
  - Both: 2.8188.
- **Stability.** The evidence is gradient-norm curves and a probe of the AdamW state. There is no test at a high learning rate.

### MHAR, `tHhLEKa9YL`

Submission 11712, posted 2026-09-08, modified 2026-10-05, 24 pages.

- **Same paper as arXiv 2607.27230.** The title and the Table 1 numbers are the same.
- **Temperature control.**
  - At 350M, a single head with logits scaled by 1/√8 reaches 2.847, against 2.848 for MHAR and 2.876 for the single head.
  - At 1B on web data, the same scaling recovers 0.118 of the 0.141 gap.
  - So "most of the gain" means the gain of the split over a single query, not over the standard Transformer.
- **Per-method learning rates (Table 13).** FineWeb-Edu only, a grid of {1e-4, 5e-4, 1e-3}, single seed.
- **Scale (App. B).**
  - The gain over the baseline is non-monotonic in scale.
  - A 7B run gains only 0.010, and the authors leave it out of the scaling claim.

### SimpleHC, `i2WyUVJUJ2`

Submission 10018, posted 2026-09-06, modified 2026-10-05, 22 pages.

- **Results.**
  - At 15B-A1B: mHC 1.659, identityHC 1.660 and mHC without Sinkhorn 1.657.
  - At 30B-A2B: all three reach 1.561.
- **Credit.** It credits identityHC to oHC (Guo et al. 2026).
- **Stress test.** At 230B-A6B and 10 times the learning rate, mHC has gradient spikes of about 1.9e6.
- The sentence in the paper is supported.

### osHC, `1BC0eYN2uS`

Submission 27583, posted 2026-09-17, modified 2026-10-05, 19 pages.

- **Title.** The PDF reads "One Marginal Is Enough: The Price of Balanced Routing in Hyper-Connections". The OpenReview title is
  different, and the bib keeps the OpenReview one.
- **Theorems 4.1 and 4.2.** With one marginal constrained, the norm of the product of the mixes is at most √n at any depth. The
  bound is on the product, and the text now says so.

## Corrections to the paper

- **ViT-22B (`dehghani2023vit22b`).**
  - The attention-logit instability appeared at about 8B parameters, while the authors scaled to 22B. It was not seen "in a
    22B-parameter model".
  - The paper is also at ICML 2023.
- **Entropy collapse (`zhai2023entropycollapse`).** Zhai et al. named the failure. They are no longer cited for the 22B
  observation.
- **QK-norm (`henry2020qknorm`).** It was introduced for translation, and Dehghani et al. applied it to the logit instability.
  The text now cites each for its own part.
- **Qwen3.8-Next (`qiu2026qwen38next`).**
  - The stress test compares the new architecture and optimizer against the structure they replace. The paper had said that it
    tests every change.
  - The stress model is a 28-layer MoE with 25B parameters, 3B of them active.
  - Its spike count is relative to the median of a 201-step window.
- **QK-norm in OLMo 2 (`olmo2024olmo2`).**
  - OLMo 2 normalizes the whole query and key projections. Our QK-norm is per head, as in Gemma 3 and Qwen3.
  - The COLM 2025 paper is a shorter version.
  - The code comment in `model.py` is fixed as well.
- **Wortsman et al. (`wortsman2023proxies`).**
  - They tie loss spikes to the second-moment decay of Adam but do not say that the architecture plays no part, so that clause
    is dropped.
  - The sensitivity caps each loss at that of a uniform prediction, and the text now says so.
- **mHC (`xie2026mhc`).** The three coefficient sets come from three linear maps, not one.
- **MUDDFormer (`xiao2025muddformer`).** It predicts the depth weights from the hidden state, and DenseFormer learns them
  directly.
- **VWN and Frac-Connections (`seed2025vwn`, `zhu2025fracconnections`).** VWN widens the stream and splits it into groups.
  Frac-Connections splits the existing width.
- **mHC-SSM (`mutlu2026mhcssm`).** It carries a static form of mHC.
- **SiHC (`liang2026sihc`).** Its largest model has 954M parameters, so "large designs" became "recent designs". It is now at v2,
  with the abstract unchanged.
- **Stream use (`zhao2026howdoesmhc`).** The paper said that each site uses about two streams. It now says a typical site.
- **MHAR (`luo2026mhar`).** The gain was larger at 350M and 1B than at 100M. It did not grow steadily with scale.
- **uHC (`anon2026uhc`).**
  - The related-work paragraph and Table 1 now describe the full text: a local read, a low-rank write-back on top of a shared
    write, and the shared mix of mHC.
  - The gains and the weak 363M baseline are stated.
  - The limitation that it was available only as an abstract is replaced by the parts of it we did not test, the local read
    alone and the write-back.

## Metadata fixes

- `lyubinin2026tbpmhc`, `gu2026temper` and `wortsman2023proxies`: titles as on the source.
- `pagliardini2024denseformer`: Fleuret's first name is spelled with ç.
- `henry2020qknorm`: Shubham Shantaram Pawar.
- Several `checked` fields were updated with the current version, the venue or the names used in the source:
  - `liu2026shc`
  - `alimaskina2026streamcollapse`
  - `ge2026chimera`
  - `seed2025vwn`
  - `zhu2024hyperconnections`

## Checked with no change

- **HC family:** mHC-lite, KromHC, go-mHC, oHC, xHC, HC and Frac-Connections. The citing sentences are correct.
- **Attention Residuals:** Attention Residuals, Kimi K3 and Depth-Attention.
- **Controls and stability:** the residual-connection review of Kramer et al., DepthBench, Lourie et al. and the Adam instability
  of Molybog et al.
- **Models and gated attention:** Gemma 3, Qwen3, gated attention and the Transformer.

## Second pass, the same day

Every citing sentence was read again against the full text of its source.

- **TEMPER (`gu2026temper`).** It keeps the mix of mHC and redesigns the dense maps that predict the coefficients, so it is no
  longer in the list of works that re-parameterize the mix.
- **xHC (`zhang2026xhc`).** It updates only 4 of its 16 streams per sublayer.
- **SiHC (`liang2026sihc`) and identityHC (`anon2026simplehc`).** Both fix the mix to the identity. Table 1 now says so.
- **Kramer et al. (`kramer2026reviewresiduals`).** The gate is significant at 590M and only a trend at 1B.
- **Melis et al. (`melis2018evaluation`).** The claim is about properly regularized LSTMs.
- **Wortsman et al. (`wortsman2023proxies`).** The sensitivity caps each loss at the loss at initialization, which we take as
  ln 16384.
- **Qwen3.8-Next (`qiu2026qwen38next`).** The spike count uses a rolling 201-step median.
- **Gemma 3 and Qwen3.** The reports say they use QK-norm but not its form. The per-head RMSNorm before RoPE is in their code,
  so the paper no longer attributes the form to them.
- **Gated attention (`qiu2025gatedattention`).** It does not tune the learning rate of each design, so the paper now says only
  that the more stable design had the lower loss in the comparisons reported.
- **Multi-head attention (`vaswani2017attention`).** Described as several smaller heads with their own projections that beat a
  single full-width head at a similar cost.
- **QK-norm history.** Henry et al. introduced it for translation, and ViT-22B applied a LayerNorm form of it.
- **Attention Residuals (`kimi2026attnres`).** §5.3 tests per-head depth aggregation with 16 heads on Block AttnRes and finds it
  slightly worse (1.752 against 1.746). This is now cited in Related Work and in 6.3.
