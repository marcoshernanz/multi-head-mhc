"""A small GPT whose residual connection is swappable: plain residual, mHC, or multi-head mHC.

Shape letters: B batch, T time, D model width, n copies of the stream (hc_mult),
h heads of the connection (hc_heads), Dh = D / h channels per connection head,
K coefficients per connection head (2n + n*n, or n for a read-only connection).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    mlp_dim: int = 1024
    max_seq_len: int = 1024
    conn: str = "mhc"  # "residual" or "mhc"
    hc_mult: int = 4  # n
    hc_heads: int = 1  # h; 1 is mHC as published
    hc_predictor: str = "global"  # "global", "local" (each head reads its own channels) or "static" (biases only)
    hc_sinkhorn_iters: int = 20
    hc_sinkhorn_form: str = "reduction"  # "reduction" (tensor ops; faster under torch.compile on a T4) or "elementwise" (entry by entry)
    hc_eps: float = 1e-6
    hc_alpha_init: float = 0.01
    hc_init: str = "hc"  # "hc": HC's Pre-Norm-equivalent init as far as sigmoid/Sinkhorn allow; "sym": uniform pre, random phi
    hc_pre_logit: float = 2.0  # "hc" init: pre bias +c on copy (sublayer index mod n), -c elsewhere
    hc_res_logit: float = 4.0  # "hc" init: res bias tau on the diagonal, so Sinkhorn(exp) starts diagonally dominant
    hc_attn_reads: int = 1  # 3 gives attention separate reads for its query, key and value inputs
    hc_head_writes: bool = False  # each attention head writes into the copies with its own post weights
    hc_joint: bool = False  # one doubly stochastic mix over all (copy, head) pairs, (n*h) x (n*h): channel groups can trade places
    hc_head_parts: str = "pre,post,res"  # which coefficients are per head when h > 1; the others are one set shared by all heads
    hc_local_gain: float = 1.0  # multiplies the local predictor's logits; a head reads 1/h of the stream, so under Adam it learns ~h x slower
    hc_layout: str = "tokens"  # "tokens": stream [B, T, n, D]; "streams": the same math on [n, D, B*T], which fills TPU tiles (see hidden_streams)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-20  # mHC's RMSNorm epsilon
    init_std: float = 0.02
    qk_norm: bool = False  # RMSNorm on each attention head's queries and keys before RoPE (as in Qwen3, Gemma 3, OLMo 2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def rope_tables(head_dim: int, max_len: int, theta: float) -> tuple[Tensor, Tensor]:
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)  # [Dhead/2]
    angles = torch.outer(torch.arange(max_len).float(), freqs)  # [T, Dhead/2]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # [B, heads, T, Dhead]
    x1, x2 = x.float().chunk(2, dim=-1)
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated.type_as(x)


def attention_stats(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:  # [B, heads, T, Dhead] each
    """The largest causal attention logit q.k / sqrt(Dhead), and the attention entropy averaged over heads and queries.

    Wortsman et al. (2023) trace one high-LR instability to attention logits that grow until the softmax collapses onto one key.
    """
    T = q.shape[2]
    with torch.autocast(q.device.type, enabled=False):  # float32 even inside a bf16 autocast region
        logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])  # [B, heads, T, T]
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        largest = logits.masked_fill(~causal, float("-inf")).max()  # not amax(): on XLA, amax() with no dims reduces none
        logp = torch.log_softmax(logits.masked_fill(~causal, float("-inf")), dim=-1)
        entropy = -(logp.exp() * logp.masked_fill(~causal, 0.0)).sum(-1).mean()
    return largest, entropy


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.per_head_output = False
        if config.qk_norm:  # ones-initialized, so adding them draws no random numbers: the other weights stay those of the seed
            self.q_norm = RMSNorm(self.head_dim, 1e-6)
            self.k_norm = RMSNorm(self.head_dim, 1e-6)
        self.qk_norm = config.qk_norm
        self.diag = None  # a list: the next forward appends its largest attention logit and mean attention entropy (diagnostics)
        cos, sin = rope_tables(self.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, xq: Tensor, xk: Tensor | None = None, xv: Tensor | None = None) -> Tensor:  # [B, T, D] each
        B, T, D = xq.shape
        if xk is None:
            qkv = self.qkv(xq)  # [B, T, 3D]
            q, k, v = qkv.split(D, dim=-1)  # each [B, T, D]
        else:
            w_q, w_k, w_v = self.qkv.weight.split(D, dim=0)  # each [D, D]
            q = F.linear(xq, w_q)  # [B, T, D]
            k = F.linear(xk, w_k)  # [B, T, D]
            v = F.linear(xv, w_v)  # [B, T, D]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, T, Dhead]
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, T, Dhead]
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, heads, T, Dhead]
        if self.qk_norm:
            q = self.q_norm(q).type_as(v)
            k = self.k_norm(k).type_as(v)
        q = apply_rope(q, self.cos[:T], self.sin[:T])
        k = apply_rope(k, self.cos[:T], self.sin[:T])
        if self.diag is not None:
            self.diag.append(attention_stats(q, k))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, heads, T, Dhead]
        o = o.transpose(1, 2)  # [B, T, heads, Dhead]
        if self.per_head_output:
            w_o = self.out.weight.view(D, self.n_heads, self.head_dim)  # [D, heads, Dhead]
            return torch.einsum("bthe,dhe->bthd", o, w_o)  # [B, T, heads, D], summing over heads gives out(o)
        o = o.reshape(B, T, D)  # [B, T, D]
        return self.out(o)


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.w_up = nn.Linear(config.dim, config.mlp_dim, bias=False)
        self.out = nn.Linear(config.mlp_dim, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # [B, T, D]
        return self.out(F.silu(self.w_gate(x)) * self.w_up(x))


class Sublayer(nn.Module):
    """F(u) = layer(norm(u)): what every connection wraps. With several reads, each input has its own norm."""

    def __init__(self, config: ModelConfig, layer: nn.Module, reads: int = 1) -> None:
        super().__init__()
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.extra_norms = nn.ModuleList(RMSNorm(config.dim, config.norm_eps) for _ in range(reads - 1))
        self.layer = layer
        self.diag = None  # a list: the next forward appends the RMS of its input u and of its output (diagnostics)

    def forward(self, u: Tensor, *extra: Tensor) -> Tensor:  # [B, T, D] each
        inputs = [self.norm(u), *(norm(e) for norm, e in zip(self.extra_norms, extra))]
        y = self.layer(*inputs)
        if self.diag is not None:
            self.diag.append((u.float().pow(2).mean().sqrt(), y.float().pow(2).mean().sqrt()))
        return y


def sinkhorn(logits: Tensor, iters: int, eps: float) -> Tensor:  # [..., n, n]
    """exp, then alternate row and column normalization: a doubly stochastic matrix.

    The same steps as DeepSeek's hc_split_sinkhorn kernel (row softmax + eps, column normalization, then
    iters - 1 rounds of row and column normalization). Written entry by entry, with the n x n entries as
    separate tensors, so that every step is elementwise and torch.compile fuses the whole loop into one kernel.
    """
    n = logits.shape[-1]
    m = [[logits[..., i, j] for j in range(n)] for i in range(n)]  # n x n tensors [...]
    for i in range(n):
        row_max = m[i][0]
        for j in range(1, n):
            row_max = torch.maximum(row_max, m[i][j])
        m[i] = [torch.exp(m[i][j] - row_max) for j in range(n)]
        total = sum(m[i])
        m[i] = [m[i][j] / total + eps for j in range(n)]
    m = _normalize_columns(m, eps)
    for _ in range(iters - 1):
        m = _normalize_rows(m, eps)
        m = _normalize_columns(m, eps)
    rows = [torch.stack(row, dim=-1) for row in m]  # n x [..., n]
    return torch.stack(rows, dim=-2)  # [..., n, n]


def _normalize_rows(m: list[list[Tensor]], eps: float) -> list[list[Tensor]]:
    out = []
    for row in m:
        total = sum(row) + eps
        out.append([entry / total for entry in row])
    return out


def _normalize_columns(m: list[list[Tensor]], eps: float) -> list[list[Tensor]]:
    n = len(m)
    totals = [sum(m[i][j] for i in range(n)) + eps for j in range(n)]
    return [[m[i][j] / totals[j] for j in range(n)] for i in range(n)]


def sinkhorn_reference(logits: Tensor, iters: int, eps: float, dims: tuple[int, int] = (-2, -1)) -> Tensor:  # [..., n, n]
    """The same computation written with tensor reductions. Measured on a T4 (exp01_bench): faster than sinkhorn() and 8x quicker to compile.

    dims = (i, j): entry [i, j] moves copy i into copy j; rows (fixed i) are normalized over j, columns over i.
    """
    i_dim, j_dim = dims
    m = logits.softmax(dim=j_dim) + eps
    m = m / (m.sum(dim=i_dim, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=j_dim, keepdim=True) + eps)
        m = m / (m.sum(dim=i_dim, keepdim=True) + eps)
    return m


class HyperConnection(nn.Module):
    """Multi-head mHC around one sublayer. With h = 1 it is mHC.

    The D channels are split into h heads; every head has its own read (pre), write (post)
    and doubly stochastic mixing (res) over the n copies, predicted per token.
    """

    def __init__(self, config: ModelConfig, read_only: bool = False, reads: int = 1, writers: int = 1, index: int = 0) -> None:
        super().__init__()
        n, h = config.hc_mult, config.hc_heads
        assert config.dim % h == 0
        assert writers == 1 or h == 1, "per-attention-head writes are built for one channel head"
        assert writers == 1 or config.hc_layout == "tokens", "per-attention-head writes are built for the token layout"
        self.n = n
        self.h = h
        self.dh = config.dim // h
        self.read_only = read_only
        self.index = index  # sublayer index, for the "hc" init of pre
        self.reads = reads  # pre vectors per head (one per input of the layer)
        self.writers = writers  # post vectors per head (one per attention head that writes)
        self.predictor = config.hc_predictor
        self.local_gain = config.hc_local_gain
        self.iters = config.hc_sinkhorn_iters
        self.sinkhorn = sinkhorn_reference if config.hc_sinkhorn_form == "reduction" else sinkhorn
        self.eps = config.hc_eps
        self.norm_eps = config.norm_eps
        self.joint = config.hc_joint and not read_only
        assert not self.joint or self.predictor != "local", "a joint mix couples the heads, so it has no local predictor"
        k = n if read_only else reads * n + writers * n + (0 if self.joint else n * n)
        self.k = k
        # Coefficient blocks in K order. A shared block (not in hc_head_parts) is predicted once and given to every head: its phi
        # rows exist once and its bias is head 0's. With all three per head (the default) this is design A exactly.
        self.parts = [("pre", n)] if read_only else [("pre", reads * n), ("post", writers * n)] + ([] if self.joint else [("res", n * n)])
        per_head = set(config.hc_head_parts.split(",")) - {""}
        assert per_head <= {"pre", "post", "res"}
        self.shared = {name for name, _ in self.parts if name not in per_head} if h > 1 else set()
        assert not self.shared or (self.predictor != "local" and writers == 1), "shared blocks: global or static predictor only"
        self.k_head = sum(size for name, size in self.parts if name not in self.shared)
        if self.predictor == "global":
            self.phi = nn.Parameter(torch.empty(h * self.k_head + k - self.k_head, n * config.dim))  # [h*K, n*D] if nothing is shared
        elif self.predictor == "local":
            self.phi = nn.Parameter(torch.empty(h, k, n * self.dh))  # [h, K, n*Dh]
        else:
            self.register_parameter("phi", None)
        self.base = nn.Parameter(torch.empty(h, k))  # [h, K]
        if self.joint:
            size = n * h  # joint index: copy s, head g -> s * h + g
            if self.predictor == "global":
                self.res_phi = nn.Parameter(torch.empty(size * size, n * config.dim))  # [(n*h)^2, n*D]
            else:
                self.register_parameter("res_phi", None)
            self.res_base = nn.Parameter(torch.empty(size, size))  # [n*h, n*h]
        self.scale = nn.Parameter(torch.full((1 if read_only else 3,), config.hc_alpha_init))
        self.init = config.hc_init
        self.pre_logit = config.hc_pre_logit
        self.res_logit = config.hc_res_logit
        self.init_std = config.init_std

    def reset_parameters(self) -> None:
        # The mHC paper only states alpha = 0.01; the rest follows HC's init (static part = Pre-Norm residual,
        # dynamic part zero), reached as closely as sigmoid and Sinkhorn allow. See notes/literature/core_papers_spec.md 7.5.
        n = self.n
        with torch.no_grad():
            self.base.zero_()  # post: 2 sigmoid(0) = 1, HC's all-ones write, exactly
            if self.joint:
                self.reset_joint()
            if self.init == "sym":
                # uniform pre (sigmoid(0) = 1/2), near-identity res, random phi to break the symmetry between copies
                if self.phi is not None:
                    nn.init.normal_(self.phi, std=self.init_std)
                if not self.read_only and not self.joint:
                    self.base[:, -n * n :] = torch.eye(n).flatten() * math.log(1e3)
                return
            if self.phi is not None:
                self.phi.zero_()  # HC: the dynamic weights start at zero
            if self.read_only:
                return
            one_hot = torch.full((n,), -self.pre_logit)
            one_hot[self.index % n] = self.pre_logit  # HC: sublayer k reads copy k mod n
            pre = self.base[:, : self.reads * n].view(self.h, self.reads, n)
            pre.copy_(one_hot.expand_as(pre))
            if not self.joint:
                self.base[:, -n * n :] = torch.eye(n).flatten() * self.res_logit  # HC: res = identity

    def reset_joint(self) -> None:
        # Starts next to the per-head mix: tau on the diagonal, 0 between copies of the same head,
        # -tau between different heads (exp(-tau) leak), so the joint mix has to learn to move channels across heads.
        size = self.n * self.h
        tau = math.log(1e3) if self.init == "sym" else self.res_logit
        group = torch.arange(size) % self.h  # [n*h], the head of each joint index
        same = group[:, None] == group[None, :]  # [n*h, n*h]
        logits = torch.where(same, 0.0, -tau) + torch.eye(size) * tau  # [n*h, n*h]
        self.res_base.copy_(logits)
        if self.res_phi is not None:
            if self.init == "sym":
                nn.init.normal_(self.res_phi, std=self.init_std)
            else:
                self.res_phi.zero_()

    def full_base(self) -> Tensor:  # [h, K]
        if not self.shared:
            return self.base
        pieces, start = [], 0
        for name, size in self.parts:
            piece = self.base[:, start : start + size]  # [h, size]
            pieces.append(piece[:1].expand_as(piece) if name in self.shared else piece)
            start += size
        return torch.cat(pieces, dim=1)  # [h, K]

    def per_head(self, mixes: Tensor, dim: int) -> Tensor:  # rows [h*K_head + K_shared] at dim -> [h, K] at dim, dim + 1
        if not self.shared:
            return mixes.unflatten(dim, (self.h, self.k))
        heads = mixes.narrow(dim, 0, self.h * self.k_head).unflatten(dim, (self.h, self.k_head))  # [.., h, K_head, ..]
        shared = mixes.narrow(dim, self.h * self.k_head, self.k - self.k_head).unsqueeze(dim)  # [.., 1, K_shared, ..]
        sizes = [-1] * shared.dim()
        sizes[dim] = self.h
        shared = shared.expand(*sizes)  # [.., h, K_shared, ..]
        pieces, at_head, at_shared = [], 0, 0
        for name, size in self.parts:
            if name in self.shared:
                pieces.append(shared.narrow(dim + 1, at_shared, size))
                at_shared += size
            else:
                pieces.append(heads.narrow(dim + 1, at_head, size))
                at_head += size
        return torch.cat(pieces, dim=dim + 1)  # [.., h, K, ..]

    def mixes(self, x: Tensor) -> Tensor:  # [B, T, n, D]
        B, T = x.shape[:2]
        if self.phi is None:
            return x.new_zeros(B, T, self.h, self.k)  # [B, T, h, K]
        flat = x.flatten(-2)  # [B, T, n*D]
        rsqrt = flat.pow(2).mean(-1, keepdim=True).add(self.norm_eps).rsqrt()  # [B, T, 1]
        if self.predictor == "global":
            mixes = F.linear(flat, self.phi)  # [B, T, h*K]
            mixes = mixes * rsqrt  # [B, T, h*K]
            return self.per_head(mixes, 2)  # [B, T, h, K]
        heads = x.unflatten(-1, (self.h, self.dh))  # [B, T, n, h, Dh]
        heads = heads.transpose(2, 3)  # [B, T, h, n, Dh]
        heads = heads.flatten(-2)  # [B, T, h, n*Dh]
        mixes = torch.einsum("bthi,hki->bthk", heads, self.phi)  # [B, T, h, K]
        mixes = mixes * self.local_gain  # [B, T, h, K]
        return mixes * rsqrt[..., None]  # [B, T, h, K]

    def joint_res(self, x: Tensor) -> Tensor:  # [B, T, n, D]
        B, T = x.shape[:2]
        size = self.n * self.h
        if self.res_phi is None:
            logits = self.res_base.expand(B, T, size, size)  # [B, T, n*h, n*h]
        else:
            flat = x.flatten(-2)  # [B, T, n*D]
            rsqrt = flat.pow(2).mean(-1, keepdim=True).add(self.norm_eps).rsqrt()  # [B, T, 1]
            logits = F.linear(flat, self.res_phi) * rsqrt  # [B, T, (n*h)^2]
            logits = logits.unflatten(-1, (size, size))  # [B, T, n*h, n*h]
            logits = logits * self.scale[2] + self.res_base  # [B, T, n*h, n*h]
        return sinkhorn_reference(logits, self.iters, self.eps)  # [B, T, n*h, n*h]; too many entries to unroll

    def coefficients(self, x: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:  # [B, T, n, D]
        n = self.n
        mixes = self.mixes(x)  # [B, T, h, K]
        if self.read_only:
            pre = torch.sigmoid(mixes * self.scale[0] + self.full_base()) + self.eps  # [B, T, h, n]
            return pre, None, None
        sizes = [self.reads * n, self.writers * n] + ([] if self.joint else [n * n])
        pre, post, *res = mixes.split(sizes, dim=-1)
        base_pre, base_post, *base_res = self.full_base().split(sizes, dim=-1)
        pre = torch.sigmoid(pre * self.scale[0] + base_pre) + self.eps  # [B, T, h, reads*n]
        pre = pre.unflatten(-1, (self.reads, n))  # [B, T, h, reads, n]
        post = 2 * torch.sigmoid(post * self.scale[1] + base_post)  # [B, T, h, writers*n]
        post = post.unflatten(-1, (self.writers, n))  # [B, T, h, writers, n]
        if self.joint:
            return pre, post, self.joint_res(x)  # res [B, T, n*h, n*h]
        res = res[0] * self.scale[2] + base_res[0]  # [B, T, h, n*n]
        res = res.unflatten(-1, (n, n))  # [B, T, h, n, n]
        res = self.sinkhorn(res, self.iters, self.eps)  # [B, T, h, n, n]
        return pre, post, res

    def read(self, x: Tensor, pre: Tensor) -> Tensor:  # [B, T, n, D], [B, T, h, n]
        heads = x.unflatten(-1, (self.h, self.dh))  # [B, T, n, h, Dh]
        weights = pre.transpose(-1, -2)  # [B, T, n, h]
        u = (weights[..., None] * heads).sum(dim=2)  # [B, T, h, Dh]
        return u.flatten(-2)  # [B, T, D]

    def mix(self, x: Tensor, res: Tensor) -> Tensor:  # [B, T, n, D], res [B, T, h, n, n] (or [B, T, n*h, n*h] joint)
        heads = x.unflatten(-1, (self.h, self.dh))  # [B, T, n, h, Dh]
        if self.joint:
            pieces = heads.flatten(2, 3)  # [B, T, n*h, Dh], joint index s * h + g
            mixed = torch.einsum("btij,btid->btjd", res, pieces)  # [B, T, n*h, Dh]; piece j gets sum_i res[i, j] x_i
            mixed = mixed.unflatten(2, (self.n, self.h))  # [B, T, n, h, Dh]
            return mixed.transpose(2, 3)  # [B, T, h, n, Dh]
        heads = heads.transpose(2, 3)  # [B, T, h, n, Dh]
        return (res[..., None] * heads[..., :, None, :]).sum(dim=-3)  # [B, T, h, n, Dh]; copy j gets sum_i res[i, j] x_i

    def write(self, x: Tensor, y: Tensor, post: Tensor, res: Tensor) -> Tensor:
        # x [B, T, n, D]; y [B, T, D], or [B, T, writers, D] with one D-vector per attention head; post [B, T, h, writers, n]
        mixed = self.mix(x, res)  # [B, T, h, n, Dh]
        if self.writers == 1:
            y_heads = y.unflatten(-1, (self.h, self.dh))  # [B, T, h, Dh]
            written = post[:, :, :, 0, :, None] * y_heads[..., None, :]  # [B, T, h, n, Dh]
        else:
            written = torch.einsum("btwn,btwd->btnd", post[:, :, 0], y)  # [B, T, n, D]; copy i gets sum_j post[j, i] y_j
            written = written[:, :, None]  # [B, T, 1, n, D] with h = 1
        out = mixed + written  # [B, T, h, n, Dh]
        out = out.transpose(2, 3)  # [B, T, n, h, Dh]
        return out.flatten(-2)  # [B, T, n, D]


    # The same connection on the stream laid out as [n, D, N], N = B * T tokens last (Model.hidden_streams). Every tensor
    # below has the tokens as its last dim and copies, heads and channels in front; heads split D, a leading dim, so
    # [n, D, N] -> [n, h, Dh, N] moves no data. Coefficients: pre [reads, n, h, N], post [n, h, N], res [n, n, h, N]
    # (joint [n*h, n*h, N]). phi, base and scale are indexed as in the token layout, so the two layouts share weights.

    def mixes_streams(self, s: Tensor) -> Tensor:  # [n, D, N]
        N = s.shape[-1]
        if self.phi is None:
            return s.new_zeros(self.h, self.k, N)  # [h, K, N]
        flat = s.flatten(0, 1)  # [n*D, N], row i * D + d, as flatten(-2) of [B, T, n, D]
        rsqrt = flat.pow(2).mean(0).add(self.norm_eps).rsqrt()  # [N]
        if self.predictor == "global":
            mixes = self.phi @ flat  # [h*K, N]
            mixes = mixes * rsqrt  # [h*K, N]
            return self.per_head(mixes, 0)  # [h, K, N]
        heads = s.unflatten(1, (self.h, self.dh))  # [n, h, Dh, N]
        heads = heads.transpose(0, 1)  # [h, n, Dh, N]
        heads = heads.flatten(1, 2)  # [h, n*Dh, N]
        mixes = torch.bmm(self.phi, heads)  # [h, K, N]
        mixes = mixes * self.local_gain  # [h, K, N]
        return mixes * rsqrt  # [h, K, N]

    def joint_res_streams(self, s: Tensor) -> Tensor:  # [n, D, N]
        N = s.shape[-1]
        size = self.n * self.h
        base = self.res_base[..., None]  # [n*h, n*h, 1]
        if self.res_phi is None:
            logits = base.expand(size, size, N)  # [n*h, n*h, N]
        else:
            flat = s.flatten(0, 1)  # [n*D, N]
            rsqrt = flat.pow(2).mean(0).add(self.norm_eps).rsqrt()  # [N]
            logits = (self.res_phi @ flat) * rsqrt  # [(n*h)^2, N]
            logits = logits.unflatten(0, (size, size))  # [n*h, n*h, N]
            logits = logits * self.scale[2] + base  # [n*h, n*h, N]
        return sinkhorn_reference(logits, self.iters, self.eps, dims=(0, 1))  # [n*h, n*h, N]

    def coefficients_streams(self, s: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:  # [n, D, N]
        n = self.n
        mixes = self.mixes_streams(s)  # [h, K, N]
        base = self.full_base()[..., None]  # [h, K, 1]
        if self.read_only:
            pre = torch.sigmoid(mixes * self.scale[0] + base) + self.eps  # [h, n, N]
            return pre.transpose(0, 1), None, None  # [n, h, N]
        sizes = [self.reads * n, n] + ([] if self.joint else [n * n])
        pre, post, *res = mixes.split(sizes, dim=1)
        base_pre, base_post, *base_res = base.split(sizes, dim=1)
        pre = torch.sigmoid(pre * self.scale[0] + base_pre) + self.eps  # [h, reads*n, N]
        pre = pre.unflatten(1, (self.reads, n))  # [h, reads, n, N]
        pre = pre.permute(1, 2, 0, 3)  # [reads, n, h, N]
        post = 2 * torch.sigmoid(post * self.scale[1] + base_post)  # [h, n, N]
        post = post.transpose(0, 1)  # [n, h, N]
        if self.joint:
            return pre, post, self.joint_res_streams(s)  # res [n*h, n*h, N]
        res = res[0] * self.scale[2] + base_res[0]  # [h, n*n, N]
        res = res.unflatten(1, (n, n))  # [h, n, n, N]
        res = res.permute(1, 2, 0, 3)  # [n, n, h, N]
        res = sinkhorn_reference(res, self.iters, self.eps, dims=(0, 1))  # [n, n, h, N]
        return pre, post, res

    def read_streams(self, s: Tensor, pre: Tensor) -> Tensor:  # [n, D, N], [n, h, N]
        heads = s.unflatten(1, (self.h, self.dh))  # [n, h, Dh, N]
        u = (pre[:, :, None, :] * heads).sum(dim=0)  # [h, Dh, N]
        return u.flatten(0, 1)  # [D, N]

    def write_streams(self, s: Tensor, y: Tensor, post: Tensor, res: Tensor) -> Tensor:
        # s [n, D, N]; y [D, N]; post [n, h, N]; res [n, n, h, N] (or [n*h, n*h, N] joint)
        heads = s.unflatten(1, (self.h, self.dh))  # [n, h, Dh, N]
        if self.joint:
            pieces = heads.flatten(0, 1)  # [n*h, Dh, N], joint index s * h + g
            mixed = (res[:, :, None, :] * pieces[:, None]).sum(dim=0)  # [n*h, Dh, N]; piece j gets sum_i res[i, j] x_i
            mixed = mixed.unflatten(0, (self.n, self.h))  # [n, h, Dh, N]
        else:
            mixed = (res[:, :, :, None, :] * heads[:, None]).sum(dim=0)  # [n, h, Dh, N]; copy j gets sum_i res[i, j] x_i
        y_heads = y.unflatten(0, (self.h, self.dh))  # [h, Dh, N]
        written = post[:, :, None, :] * y_heads  # [n, h, Dh, N]
        out = mixed + written  # [n, h, Dh, N]
        return out.flatten(1, 2)  # [n, D, N]


def hc_step(connection: HyperConnection, sublayer: Sublayer, stream: Tensor, coefficients: bool = False):  # [B, T, n, D]
    """One sublayer wrapped in its connection: read the copies, run the layer, mix the copies and write the output."""
    device_type = stream.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pre, post, res = connection.coefficients(stream)
        inputs = [connection.read(stream, pre[..., r, :]) for r in range(connection.reads)]  # each [B, T, D]
    y = sublayer(*inputs)  # [B, T, D], or [B, T, heads, D] for per-head writes
    with torch.autocast(device_type=device_type, enabled=False):
        stream = connection.write(stream, y.float(), post, res)  # [B, T, n, D]
    if coefficients:
        return stream, pre, post, res
    return stream


def hc_step_streams(connection: HyperConnection, sublayer: Sublayer, stream: Tensor, batch: int, time: int) -> Tensor:  # [n, D, N]
    """hc_step on the [n, D, N] layout; the sublayer still sees [B, T, D], so its input and output are transposed."""
    device_type = stream.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pre, post, res = connection.coefficients_streams(stream)
        inputs = [connection.read_streams(stream, pre[r]) for r in range(connection.reads)]  # each [D, N]
    inputs = [u.t().unflatten(0, (batch, time)) for u in inputs]  # each [B, T, D]
    y = sublayer(*inputs)  # [B, T, D]
    with torch.autocast(device_type=device_type, enabled=False):
        y = y.float().flatten(0, 1).t()  # [D, N]
        stream = connection.write_streams(stream, y, post, res)  # [n, D, N]
    return stream


def residual_step(sublayer: Sublayer, x: Tensor) -> Tensor:  # [B, T, D]
    return x + sublayer(x)


class Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.sublayers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.sublayers.append(Sublayer(config, Attention(config)))
            self.sublayers.append(Sublayer(config, FeedForward(config)))
        self.use_hc = config.conn == "mhc"
        if self.use_hc:
            self.connections = nn.ModuleList()
            for index, sublayer in enumerate(self.sublayers):
                is_attention = index % 2 == 0
                reads = config.hc_attn_reads if is_attention else 1
                writers = config.n_heads if is_attention and config.hc_head_writes else 1
                if reads > 1:
                    self.sublayers[index] = Sublayer(config, sublayer.layer, reads=reads)
                if writers > 1:
                    sublayer.layer.per_head_output = True
                self.connections.append(HyperConnection(config, reads=reads, writers=writers, index=index))
            self.head_connection = HyperConnection(config, read_only=True)
        self.out_norm = RMSNorm(config.dim, config.norm_eps)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.hc_step = hc_step  # replaced by compiled versions in compile_regions()
        self.hc_step_streams = hc_step_streams
        self.residual_step = residual_step
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = self.config.init_std
        residual_std = std / math.sqrt(2 * self.config.n_layers)
        # main weights first, so that every connection variant starts from the same main weights for a given seed
        for name, p in self.named_parameters():
            if "connection" in name:
                continue
            if name.endswith("out.weight"):
                nn.init.normal_(p, std=residual_std)
            elif p.dim() == 2:
                nn.init.normal_(p, std=std)
        if self.use_hc:
            for connection in [*self.connections, self.head_connection]:
                connection.reset_parameters()

    def compile_regions(self) -> None:
        """torch.compile one sublayer step at a time instead of the whole model.

        Every sublayer of a type runs the same code on the same shapes, so each step is compiled once and reused by all
        sublayers: a handful of small graphs instead of one graph with 17 unrolled Sinkhorn loops, whose compilation
        took tens of minutes on Kaggle's CPUs.
        """
        self.hc_step = torch.compile(hc_step)
        self.hc_step_streams = torch.compile(hc_step_streams)
        self.residual_step = torch.compile(residual_step)

    def hidden(self, ids: Tensor, record: list | None = None) -> Tensor:  # [B, T]
        x = self.embed(ids).float()  # [B, T, D]
        if not self.use_hc:
            for sublayer in self.sublayers:
                x = self.residual_step(sublayer, x)  # [B, T, D]
            return x
        if self.config.hc_layout == "streams" and record is None:
            return self.hidden_streams(x)
        n = self.config.hc_mult
        stream = x[:, :, None, :].expand(-1, -1, n, -1)  # [B, T, n, D]
        stream = stream.contiguous()  # one memory layout for every step, so a compiled step is reused
        for sublayer, connection in zip(self.sublayers, self.connections):
            if record is None:
                stream = self.hc_step(connection, sublayer, stream)  # [B, T, n, D]
                continue
            stream, pre, post, res = hc_step(connection, sublayer, stream, coefficients=True)
            record.append({"pre": pre.flatten(2, 3).detach(), "post": post.flatten(2, 3).detach(), "res": res.detach()})
        with torch.autocast(device_type=x.device.type, enabled=False):
            pre, _, _ = self.head_connection.coefficients(stream)
            x = self.head_connection.read(stream, pre)  # [B, T, D]
        if record is not None:
            record.append({"pre": pre.detach(), "stream": stream.detach()})
        return x

    def hidden_streams(self, x: Tensor) -> Tensor:  # [B, T, D]
        """The mHC stream as [n, D, N] with N = B * T: the math of hidden(), laid out for TPUs.

        A TPU stores the last two dims of an array in 8 x 128 tiles. In the token layout those dims are small: (n, D) for
        the stream, (Dh of a head, ...) and (n, n) for res, which a (8, 128) tile pads up to 64x (exp03: 2.4k tok/s for
        mHC h1 vs 125k for the residual; the head variants never finished compiling). Here the last dim is always the
        N tokens. The probe (record) keeps using the token layout; both layouts share the same weights.
        """
        B, T, _ = x.shape
        n = self.config.hc_mult
        stream = x.flatten(0, 1).t()  # [D, N]
        stream = stream[None].expand(n, -1, -1)  # [n, D, N]
        stream = stream.contiguous()
        for sublayer, connection in zip(self.sublayers, self.connections):
            stream = self.hc_step_streams(connection, sublayer, stream, B, T)  # [n, D, N]
        with torch.autocast(device_type=x.device.type, enabled=False):
            pre, _, _ = self.head_connection.coefficients_streams(stream)
            x = self.head_connection.read_streams(stream, pre)  # [D, N]
        return x.t().unflatten(0, (B, T))  # [B, T, D]

    def forward(self, ids: Tensor) -> Tensor:  # [B, T]
        x = self.hidden(ids)  # [B, T, D]
        x = self.out_norm(x)  # [B, T, D]
        return self.head(x)  # [B, T, vocab]
