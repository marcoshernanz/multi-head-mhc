# Design: what "multi-head mHC" can mean, and how each version is built

Notation, per token: the stream is X, an n × D matrix (n copies, width D). One mHC sublayer is

    u  = H_pre X                    (1×n)(n×D) -> 1×D, the layer's input
    y  = F(u)                       1×D
    X' = H_res^T X + H_post^T y     (n×n)(n×D) + (n×1)(1×D) -> n×D

with H_pre = σ(·), H_post = 2σ(·), H_res = Sinkhorn(exp(·)) doubly stochastic, all predicted per token from
RMSNorm(vec X) by a linear map φ plus a bias b, scaled by α (init 0.01).

Every coefficient in mHC is a scalar that multiplies a whole D-vector. "Multi-head" = give up that sharing along some axis.

## A. Channel-grouped heads (the main candidate; "MH-mHC")

Split the D channels into h groups of Dh = D/h ("heads"). Head k has its own H_pre^(k) (1×n), H_post^(k) (1×n), H_res^(k) (n×n):

    u^(k)  = H_pre^(k) X^(k)                          X^(k) = the n × Dh block of head k
    X'^(k) = H_res^(k)T X^(k) + H_post^(k)T y^(k)     y^(k) = head k's channels of F(u)

h = 1 is mHC. h = D is fully channel-wise (every channel its own routing), the analogue of going from a scalar gate to a per-channel
gate (Highway networks, LayerScale). Heads are nested when they divide each other (h | h').

**Guarantees carry over exactly.** On vec(X) the mixing operator is block-diagonal over heads, with block H_res^(k) ⊗ I_Dh.
- gain: the spectral norm of a block-diagonal matrix is the max over blocks; each block has norm ≤ 1 (doubly stochastic).
- sum over copies preserved per channel: every block is column-stochastic.
- closure: a product of such operators is again block-diagonal with the same blocks structure, each block a product of doubly
  stochastic matrices, hence doubly stochastic. So the whole-depth composite is again a member: no explosion, no loss of the mean.

**The basis question.** A transformer's residual channels have no built-in meaning: rotating the residual basis by an orthogonal Q
and absorbing Q into every matrix that reads or writes the stream leaves the function unchanged (only RMSNorm's per-channel gains and
Adam's per-coordinate step sizes are tied to the basis, which is why real models still develop privileged directions). mHC keeps this
symmetry: every coefficient multiplies whole D-vectors. Channel-group heads break it on purpose: routing becomes block-diagonal in a fixed
basis, so the network gains something only if it learns to place features that want the same route in the same group. MHAR relies on the
same thing and finds it works (optimum at 4–8 heads); ExoFormer finds per-channel mixing coefficients on the value path self-organize into
head blocks. Consequence for the experiments: contiguous slices vs a random channel permutation are equivalent at initialization (the
init distribution is rotation-invariant), so the choice of which channels form a head does not need its own ablation.

**What heads add, seen through depth.** Unroll the stream. Every sublayer's input is a weighted sum of the embedding and all earlier
sublayer outputs:

    u_l = Σ_{s ≤ l} c[l, s] · y_s
    c[l, s] = H_pre,l · (H_res,l-1ᵀ ⋯ H_res,sᵀ) · H_post,s-1ᵀ        (y_0 = the embedding, written into every copy: H_post,-1 = all ones)

- Plain residual: c ≡ 1 (every layer reads the plain sum). DenseFormer / MUDDFormer: c is a free lower-triangular matrix (static or per
  token).
- HC / mHC: the n copies are a small state carried through depth. H_post writes into it, H_res transitions it, H_pre reads it out: a
  linear state-space recurrence over depth with state size n, the structure Mamba-2 runs over time (write = B, transition = A,
  read-out = C). So c is n-semiseparable: every block strictly below the diagonal factors through the n-dim state and has rank ≤ n
  (rows l ≥ a and columns s < a give (H_pre,l H_res…ᵀ) · (H_res…ᵀ H_post,sᵀ), an (· × n)(n × ·) product). That is what n buys: depth
  patterns of rank ≤ n. The residual is the n = 1 case.
- With h channel-group heads every group has its own pre, post and res, so its own c^(g): **h independent rank-≤ n depth patterns instead
  of one pattern shared by all D channels.** In Mamba-2's words: mHC is a single-head depth-SSM with head dimension D, design A is a
  multi-head depth-SSM with head dimension D/h. Raising n instead (mHC n = 8) raises the rank of the single pattern and doubles the
  stream's memory; heads keep the memory and multiply the patterns (H6 compares the two).
- The dynamic predictor makes c depend on the token (a selective SSM over depth); the static variants have one c per head for all
  tokens.

I did not find this depth-SSM reading stated in the HC literature (checked 2026-09-26). The nearest are DenseFormer / MUDDFormer (the
depth-connectivity matrix c) and MHAR (per-head depth attention). `probe.depth_connectivity` measures c^(g) for every head, and
`depth_head_spread` measures how much the heads' patterns differ.

**Predictor choices** (how head k's 2n + n² raw coefficients are produced):
- `global`: from the whole normalized stream, φ ∈ R^{nD × h(2n+n²)}. h× the φ parameters and FLOPs of mHC.
  FLOPs per token per sublayer: 2·nD·h(2n+n²); at D=384, n=4, h=8 about 0.59 MFLOP, i.e. about +33% of the layer FLOPs at this small width
  (the fraction shrinks as D grows because the layer is O(D²) and this is O(D·h)).
- `local`: head k's coefficients from its own channels only, φ^(k) ∈ R^{n·Dh × (2n+n²)}: exactly mHC's parameter count and FLOPs.
  Heads then route on "their own" features; a head cannot see the rest of the token.
- `static`: biases only (no per-token prediction): per-head learned constants, the multi-head analogue of static HC.

**Costs.** Reads and writes touch the same n×D numbers as mHC (the same memory traffic). The coefficient tensors grow h×:
per token per sublayer h(2n+n²) floats (24h for n=4) and h Sinkhorn solves of n×n. For h ≤ 64 this is small next to n×D = 4D;
at h = D it is 24D per token per sublayer, 6× the stream itself, which is why h = D is impractical without fused kernels.

## B. Multi-read heads (MUDDFormer-style "multiway" reads)

Keep one res and one post, but give the attention sublayer three reads, for its query, key and value inputs:

    u_q = H_pre,q X,  u_k = H_pre,k X,  u_v = H_pre,v X      each normalized separately, then W_q, W_k, W_v

MUDDFormer found that separate dynamic dense connections for Q, K, V (and the residual) are what make its cross-layer connections work.
In HC terms: keys/values may want to read an older, more accumulated copy while queries read the freshest. Costs 2 extra reads (n×D each) and
2n extra coefficients per token. Guarantees are untouched (H_res unchanged).

## C. Attention-head-routed writes

Attention's output is a sum over its heads, y = Σ_j o_j W_o^(j). Instead of writing y into the copies with one post vector, write each
attention head's contribution with its own post vector: X'_i += Σ_j post_{j,i} o_j W_o^(j). Heads of the layer route their outputs into
different copies. Same FLOPs as W_o; needs the H per-head D-vectors in memory. Guarantees untouched (H_res unchanged).

## D. Cross-group mixing (HC × Frac-Connections)

Treat the n copies × h groups as n·h slots ("pieces") of width Dh and mix all of them with one (nh)×(nh) doubly stochastic matrix. This lets
information move between channel groups, as Frac-Connections mixes fractions of the hidden state. A block-diagonal H_res recovers A.

**Already exists in code (unpublished, no results reported):** lucidrains/hyper-connections, `mHCv2.py` with `num_fracs = m > 1` builds
exactly this joint index (stream, fraction) and runs Sinkhorn on an (n·m)×(n·m) matrix (verified by reading the source, 2026-09-26). It also
makes the reads and writes cross-group: branch-input group f2 reads from every (stream, group f1) piece (an (n·m)×m pre), and the write
β[s, f1, f2] = 2σ(·) sends branch-output group f1 into group f2 of stream s. Its init is far from the Pre-Norm residual when m > 1: static
res logits = I on 16×16 (diagonal only e/(e+15) ≈ 0.15 after Sinkhorn), pre = σ(1 or 0) on every entry, β = 2σ(1) ≈ 1.46 on every
(f1, f2) pair, so every group of the output is written into every group of every stream. The predictor is per row: each (stream, group)
piece predicts its own row of coefficients from its own Dh channels (HC's original per-stream dynamic weights), not mHC's flatten(X)·φ.

**What is lost.** A's operator is block-diagonal, so the per-channel sum over copies (the "mean stream") is conserved channel by channel.
D's operator is only column-stochastic over pieces, so what is conserved is the sum over all n·h pieces, a single Dh-vector: channel c of
group k can be moved into channel c of group k'. The gain bound (spectral norm ≤ 1) and closure still hold (D is an (nh)×(nh) doubly stochastic
matrix acting on Dh-vectors). Checked numerically in `analysis/checks/check_model.py` (item 9).

**This repo's version** (`hc_joint = True`): isolates the one question "block-diagonal vs joint res" by keeping A's per-head pre and post
and changing only res. Init next to A: logit τ on the diagonal, 0 between copies of the same group, −τ between groups, so each row starts
with ≈ 0.4% of its mass on other groups (τ = 4); the model has to learn to use cross-group moves. Predictor: `global` (φ_res ∈ R^{nD × (nh)²},
at D = 384, n = 4, h = 4 this is 393k parameters per sublayer, 2.7× A-global-h4's) or `static` (biases only).

**Cost.** (nh)² res coefficients per token and a Sinkhorn on nh × nh (256 entries for n = h = 4, vs h·n² = 64 for A).

## E. Mixtures / products of Sinkhorn heads for H_res (ruled out on theory)

Combining several doubly stochastic "heads" by a convex combination or a product stays doubly stochastic, but adds no expressivity: every
strictly positive doubly stochastic matrix M is already Sinkhorn(exp(log M)), so one dynamic Sinkhorn head reaches every matrix a mixture could.
It could only change the optimization landscape. Not tested here; already published in other forms: mHC-lite and BE-HC parameterize
H_res as a convex combination of permutation matrices (Birkhoff–von Neumann), and lucidrains' `num_dynamic_alpha_proposals` averages several
Sinkhorned proposals.

## Naming

DeepSeek's TileKernels code calls the n copies "heads" (`hc_mult` streams → per-"head" mixing). To avoid the collision, the report calls A
"per-subspace mHC" or "block-diagonal multi-head mHC": heads are channel subspaces, and every head routes over the same n copies.

## What is tested (see experiments/)

Priority: A (h sweep, global vs local vs static) against mHC and the plain residual, with paired seeds; then B and C; D if budget allows;
then a scale/depth and learning-rate stress check of the best variant against mHC.
