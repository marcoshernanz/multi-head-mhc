"""Local correctness checks of the connection code (CPU, float32)."""

import math

import torch

from mhc_lab.model import HyperConnection, Model, ModelConfig, sinkhorn

torch.manual_seed(0)

# 1. Sinkhorn gives doubly stochastic matrices
m = sinkhorn(torch.randn(1000, 4, 4) * 3, iters=20, eps=1e-6)
print("sinkhorn row err", (m.sum(-1) - 1).abs().max().item(), "col err", (m.sum(-2) - 1).abs().max().item())

# 2. Multi-head with every head given the same coefficients equals single-head mHC
base = ModelConfig(dim=64, n_layers=2, n_heads=4, mlp_dim=128, vocab_size=256, max_seq_len=32)
single = HyperConnection(ModelConfig(**{**base.__dict__, "hc_heads": 1}))
multi = HyperConnection(ModelConfig(**{**base.__dict__, "hc_heads": 4}))
single.reset_parameters()
x = torch.randn(2, 8, 4, 64)
pre, post, res = single.coefficients(x)
pre = pre[..., 0, :]  # one read
u1 = single.read(x, pre)
y = torch.randn(2, 8, 64)
out1 = single.write(x, y, post, res)
u4 = multi.read(x, pre.expand(-1, -1, 4, -1))
out4 = multi.write(x, y, post.expand(-1, -1, 4, -1, -1), res.expand(-1, -1, 4, -1, -1))
print("read diff", (u1 - u4).abs().max().item(), "write diff", (out1 - out4).abs().max().item())

# 3. The mixing preserves the sum over copies for every channel (columns of res sum to one)
out_no_write = single.write(x, torch.zeros_like(y), post, res)
print("sum over copies preserved", (out_no_write.sum(2) - x.sum(2)).abs().max().item())

# 4. With heads, each head only mixes its own channels: perturb head 0's channels, others unchanged
multi.reset_parameters()
pre, post, res = multi.coefficients(x)
x2 = x.clone()
x2[..., :16] += 1.0
out_a = multi.write(x, y, post, res)
out_b = multi.write(x2, y, post, res)
print("other heads unchanged", (out_a[..., 16:] - out_b[..., 16:]).abs().max().item())

# 5. Whole models run forward and backward for every variant
for conn, heads, predictor in [("residual", 1, "global"), ("mhc", 1, "global"), ("mhc", 4, "global"),
                               ("mhc", 4, "local"), ("mhc", 16, "local"), ("mhc", 4, "static")]:
    cfg = ModelConfig(**{**base.__dict__, "conn": conn, "hc_heads": heads, "hc_predictor": predictor})
    model = Model(cfg)
    ids = torch.randint(0, 256, (2, 32))
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids.flatten())
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    hc_params = sum(p.numel() for n, p in model.named_parameters() if "connection" in n)
    print(f"{conn:8s} h={heads:2d} {predictor:6s} loss {loss.item():.4f} params {sum(p.numel() for p in model.parameters())} hc {hc_params} no-grad {no_grad}")

# 6. At init with identity res, the mHC model's output equals the plain residual's (copies stay identical)
cfg_r = ModelConfig(**{**base.__dict__, "conn": "residual"})
cfg_m = ModelConfig(**{**base.__dict__, "conn": "mhc", "hc_alpha_init": 0.0})
torch.manual_seed(1)
plain = Model(cfg_r)
torch.manual_seed(1)
mhc = Model(cfg_m)
missing = mhc.load_state_dict(plain.state_dict(), strict=False)
ids = torch.randint(0, 256, (2, 32))
print("init equivalence (alpha=0)", (plain(ids) - mhc(ids)).abs().max().item())

# 7. Multi-read attention with identical reads equals single-read; per-head writes with equal post equal one write
torch.manual_seed(5)
cfg1 = ModelConfig(**{**base.__dict__, "conn": "mhc", "hc_alpha_init": 0.0})
cfg3 = ModelConfig(**{**base.__dict__, "conn": "mhc", "hc_alpha_init": 0.0, "hc_attn_reads": 3})
cfgw = ModelConfig(**{**base.__dict__, "conn": "mhc", "hc_alpha_init": 0.0, "hc_head_writes": True})
torch.manual_seed(5)
m1 = Model(cfg1)
torch.manual_seed(5)
m3 = Model(cfg3)
torch.manual_seed(5)
mw = Model(cfgw)
ids = torch.randint(0, 256, (2, 32))
print("3 reads == 1 read at init", (m1(ids) - m3(ids)).abs().max().item())
print("head writes == one write at init", (m1(ids) - mw(ids)).abs().max().item())
for cfg in (cfg3, cfgw):
    cfg = ModelConfig(**{**cfg.__dict__, "hc_alpha_init": 0.01})
    model = Model(cfg)
    loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), ids.flatten())
    loss.backward()
    print("trains", loss.item(), [n for n, p in model.named_parameters() if p.grad is None])

# 8. The elementwise Sinkhorn equals the reduction form, values and gradients
from mhc_lab.model import sinkhorn_reference

logits = (torch.randn(64, 3, 4, 4) * 2).requires_grad_()
a = sinkhorn(logits, 20, 1e-6)
b = sinkhorn_reference(logits, 20, 1e-6)
weights = torch.randn_like(a)
ga, = torch.autograd.grad((a * weights).sum(), logits)
gb, = torch.autograd.grad((b * weights).sum(), logits)
print("sinkhorn forms agree", (a - b).abs().max().item(), "grads", (ga - gb).abs().max().item())

# 9. Joint (n*h) x (n*h) mixing: with a block-diagonal joint matrix it equals the per-head mix; it is doubly stochastic;
#    it preserves the sum over all (copy, head) pieces, but no longer the per-channel sum over copies once heads trade channels
cfg_a = ModelConfig(**{**base.__dict__, "hc_heads": 4})
cfg_d = ModelConfig(**{**base.__dict__, "hc_heads": 4, "hc_joint": True})
per_head = HyperConnection(cfg_a)
joint = HyperConnection(cfg_d)
per_head.reset_parameters()
joint.reset_parameters()
x = torch.randn(2, 8, 4, 64)
_, _, res_a = per_head.coefficients(x)  # [B, T, h, n, n]
size = 16
res_block = torch.zeros(2, 8, size, size)
for g in range(4):
    idx = torch.arange(4) * 4 + g  # copies of head g in the joint index s * h + g
    res_block[:, :, idx[:, None], idx[None, :]] = res_a[:, :, g]
print("joint with block-diagonal res == per-head mix", (joint.mix(x, res_block) - per_head.mix(x, res_a)).abs().max().item())
_, _, res_d = joint.coefficients(x)
print("joint res shape", tuple(res_d.shape), "row err", (res_d.sum(-1) - 1).abs().max().item(), "col err", (res_d.sum(-2) - 1).abs().max().item())
print("joint init: diagonal", res_d.diagonal(dim1=-2, dim2=-1).mean().item(), "cross-head mass per row",
      (res_d * (torch.arange(size)[:, None] % 4 != torch.arange(size)[None, :] % 4)).sum(-1).mean().item())
res_rand = torch.from_numpy(__import__("numpy").random.dirichlet([1.0] * size, size=(2, 8, size))).float()
res_rand = __import__("mhc_lab.model", fromlist=["x"]).sinkhorn_reference(res_rand.log(), 50, 0.0)
mixed = joint.mix(x, res_rand)  # [B, T, h, n, Dh]
pieces_in = x.unflatten(-1, (4, 16)).sum(dim=(2, 3))  # [B, T, Dh], sum over copies and heads
pieces_out = mixed.sum(dim=(2, 3))  # [B, T, Dh]
print("joint preserves sum over all pieces", (pieces_in - pieces_out).abs().max().item())
channels_out = mixed.transpose(2, 3).flatten(-2).sum(2)  # [B, T, D], per-channel sum over copies
print("joint per-channel sum over copies changes by", (channels_out - x.sum(2)).abs().max().item())
for predictor in ("global", "static"):
    cfg = ModelConfig(**{**cfg_d.__dict__, "conn": "mhc", "hc_predictor": predictor})
    model = Model(cfg)
    ids = torch.randint(0, 256, (2, 32))
    loss = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), ids.flatten())
    loss.backward()
    hc_params = sum(p.numel() for n, p in model.named_parameters() if "connection" in n)
    print(f"joint {predictor} trains {loss.item():.4f} hc params {hc_params} no-grad", [n for n, p in model.named_parameters() if p.grad is None])

# 11. Per-part heads (hc_head_parts): blocks left out are predicted once and shared by every head. With nothing per head,
#     h = 4 equals mHC h1 given the same phi rows and head 0's biases; with only some blocks per head, the shared blocks'
#     coefficients are identical across heads
torch.manual_seed(3)
small = ModelConfig(dim=64, n_layers=2, n_heads=4, mlp_dim=128, vocab_size=256, max_seq_len=32, hc_layout="streams")
h1 = Model(small)
h4 = Model(ModelConfig(**{**small.__dict__, "hc_heads": 4, "hc_head_parts": ""}))
with torch.no_grad():
    for p in h1.parameters():
        p.add_(torch.randn_like(p) * 0.05)
    state = h1.state_dict()
    for n, p in h4.named_parameters():
        p.copy_(state[n].expand_as(p) if n.endswith("base") else state[n])
ids = torch.randint(0, 256, (2, 32))
print("h4 with nothing per head == h1", (h4(ids) - h1(ids)).abs().max().item())
for parts in ("pre", "post", "res"):
    conn = HyperConnection(ModelConfig(**{**small.__dict__, "hc_heads": 4, "hc_head_parts": parts}))
    conn.reset_parameters()
    with torch.no_grad():
        for p in conn.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    coeffs = dict(zip(("pre", "post", "res"), conn.coefficients(torch.randn(2, 8, 4, 64))))
    spread = {name: (c - c[:, :, :1]).abs().max().item() for name, c in coeffs.items()}  # head axis is dim 2
    print(f"only {parts} per head: spread across heads", {k: f"{v:.1e}" for k, v in spread.items()})

# 12. qk-norm: the same seed gives the same weights as without it (the norms are ones and draw no random numbers); the model
#     trains in both layouts; attention_stats matches an explicit causal softmax (uniform attention: entropy mean log(t + 1))
from mhc_lab.model import Attention, attention_stats  # noqa: E402

for layout in ("tokens", "streams"):
    torch.manual_seed(4)
    plain = Model(ModelConfig(**{**small.__dict__, "hc_layout": layout, "hc_heads": 4}))
    torch.manual_seed(4)
    normed = Model(ModelConfig(**{**small.__dict__, "hc_layout": layout, "hc_heads": 4, "qk_norm": True}))
    shared = dict(plain.named_parameters())
    weights_differ = max((p - shared[n]).abs().max().item() for n, p in normed.named_parameters() if n in shared)
    extra = [n for n, _ in normed.named_parameters() if n not in shared]
    loss = torch.nn.functional.cross_entropy(normed(ids).flatten(0, 1), ids.flatten())
    loss.backward()
    print(f"qk-norm {layout}: same init {weights_differ:.1e}, new params {len(extra)} ({extra[:2]}), loss {loss.item():.4f}, "
          f"no-grad {[n for n, p in normed.named_parameters() if p.grad is None]}")
q = torch.zeros(1, 2, 6, 8)
largest, entropy = attention_stats(q, torch.randn(1, 2, 6, 8))
print("uniform attention: max logit", largest.item(), "entropy", entropy.item(), "expected", sum(math.log(t + 1) for t in range(6)) / 6)
q, k = torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
s = q @ k.transpose(-1, -2) / math.sqrt(8)
mask = torch.ones(7, 7, dtype=torch.bool).tril()
p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1).mean()
largest, entropy = attention_stats(q, k)
print("attention_stats vs explicit", abs(largest.item() - s.masked_fill(~mask, -1e9).max().item()), abs(entropy.item() - ent.item()))
model = Model(ModelConfig(**{**small.__dict__, "qk_norm": True}))
attentions = [m for m in model.modules() if isinstance(m, Attention)]
for m in attentions:
    m.diag = []
before = model(ids)
for m in attentions:
    m.diag = None
print("diag forward == plain forward", (before - model(ids)).abs().max().item(), "layers recorded", len(attentions))
