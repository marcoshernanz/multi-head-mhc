"""Readings of the learned connection coefficients on a fixed batch."""

import torch
from torch import Tensor


def rounded(t: Tensor, digits: int = 4) -> list:
    return torch.round(t.float().cpu(), decimals=digits).tolist()


def pairwise_head_distance(m: Tensor) -> float:  # [h, ...]
    h = m.shape[0]
    if h < 2:
        return 0.0
    flat = m.flatten(1)  # [h, F]
    d = torch.cdist(flat[None], flat[None])[0]  # [h, h]
    return (d.sum() / (h * (h - 1))).item()


def effective_copies(w: Tensor) -> Tensor:  # [..., n] nonnegative weights over the copies
    p = w / w.sum(-1, keepdim=True)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return entropy.exp()  # 1 = one copy only, n = all copies equally


def head_agreement(pre: Tensor) -> float:  # [B, T, h, n]: how often every head reads mostly from the same copy
    if pre.shape[2] < 2:
        return 1.0
    top = pre.argmax(-1)  # [B, T, h]
    same = (top == top[..., :1]).all(-1)  # [B, T]
    return same.float().mean().item()


def depth_connectivity(record: list) -> Tensor:
    """How much each reader takes from each source, per head, averaged over tokens: [h, L+1, L+1].

    Reader l = 0..L-1 is sublayer l, reader L is the final read before the output head. Source 0 is the embedding
    (written into every copy), source s = 1..L is the output of sublayer s-1. Unrolling the stream,
    u_l = sum_s c[l, s] y_s with c[l, s] = pre_l . (R_{l-1}^T ... R_s^T a_s), a_0 = all ones, a_s = post_{s-1}.
    For the plain residual every c is 1. Per head, because each head has its own pre, post and res.
    """
    pres = [entry["pre"] for entry in record]  # L+1 of [B, T, h, n]
    posts = [entry["post"] for entry in record[:-1]]  # L of [B, T, h, n]
    mixes = [entry["res"] for entry in record[:-1]]  # L of [B, T, h, n, n]
    L = len(posts)
    h = posts[0].shape[2]
    out = torch.zeros(h, L + 1, L + 1)
    for s in range(L + 1):
        a = torch.ones_like(posts[0]) if s == 0 else posts[s - 1]  # [B, T, h, n], weight of source s on each copy
        for l in range(s, L + 1):
            c = (pres[l].float() * a.float()).sum(-1)  # [B, T, h]
            out[:, l, s] = c.mean(dim=(0, 1)).cpu()
            if l < L:
                a = torch.einsum("bthij,bthi->bthj", mixes[l].float(), a.float())  # through sublayer l's mix: copy j gets sum_i res[i, j] a_i
    return out


@torch.no_grad()
def coefficient_report(model, ids: Tensor, amp_dtype) -> dict:
    model.eval()
    record: list = []
    with torch.autocast(device_type=ids.device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
        model.hidden(ids, record=record)
    model.train()
    n = model.config.hc_mult
    h = model.config.hc_heads
    joint = model.config.hc_joint
    size = n * h if joint else n  # side of each mixing matrix
    eye = torch.eye(size, device=ids.device)
    uniform = torch.full((size, size), 1.0 / size, device=ids.device)
    group = torch.arange(size, device=ids.device) % h  # joint index s * h + g -> head g
    other_group = (group[:, None] != group[None, :]).float()  # [n*h, n*h]
    layers = []
    composite = None  # [B, T, h, n, n] (or [B, T, n*h, n*h] joint), product of the mixing matrices so far
    for entry in record[:-1]:
        pre, post, res = entry["pre"], entry["post"], entry["res"]  # [B,T,h,n], [B,T,h,n], [B,T,h,n,n] or [B,T,nh,nh]
        composite = res if composite is None else composite @ res
        res_mean = res.mean(dim=(0, 1))  # [h, n, n] or [n*h, n*h]
        layers.append({
            "pre_mean": rounded(pre.mean(dim=(0, 1))),
            "post_mean": rounded(post.mean(dim=(0, 1))),
            "res_mean": rounded(res_mean),
            "pre_token_std": pre.std(dim=(0, 1)).mean().item(),
            "pre_eff_copies": effective_copies(pre).mean().item(),
            "post_eff_copies": effective_copies(post).mean().item(),
            "pre_head_agreement": head_agreement(pre),
            "post_token_std": post.std(dim=(0, 1)).mean().item(),
            "res_token_std": res.std(dim=(0, 1)).mean().item(),
            "res_dist_identity": (res - eye).flatten(-2).norm(dim=-1).mean().item(),
            "res_dist_uniform": (res - uniform).flatten(-2).norm(dim=-1).mean().item(),
            "res_diag_mass": res.diagonal(dim1=-2, dim2=-1).mean().item(),
            "row_err": (res.sum(-1) - 1).abs().max().item(),
            "col_err": (res.sum(-2) - 1).abs().max().item(),
            "head_dist_res": 0.0 if joint else pairwise_head_distance(res_mean),
            "head_dist_pre": pairwise_head_distance(pre.mean(dim=(0, 1))),
            "head_dist_post": pairwise_head_distance(post.mean(dim=(0, 1))),
            "head_token_std_res": res.std(dim=2).mean().item() if res.shape[2] > 1 and not joint else 0.0,
            "cross_head_mass": (res * other_group).sum(-1).mean().item() if joint else 0.0,
        })
    depth = None
    simple = not joint and model.config.hc_attn_reads == 1 and not model.config.hc_head_writes
    if simple:
        depth = depth_connectivity(record)  # [h, L+1, L+1]
    stream = record[-1]["stream"]  # [B, T, n, D]
    unit = stream / stream.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cos = unit @ unit.transpose(-1, -2)  # [B, T, n, n]
    off_diag = cos.sum(dim=(-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
    return {
        "layers": layers,
        "final_pre_mean": rounded(record[-1]["pre"].mean(dim=(0, 1))),
        "composite_forward_gain_max": composite.abs().sum(-1).amax().item(),
        "composite_backward_gain_max": composite.abs().sum(-2).amax().item(),
        "composite_dist_uniform": (composite - uniform).flatten(-2).norm(dim=-1).mean().item(),
        "composite_mean": rounded(composite.mean(dim=(0, 1))),
        "composite_cross_head_mass": (composite * other_group).sum(-1).mean().item() if joint else 0.0,
        "stream_copy_cosine": (off_diag / (n * (n - 1))).mean().item(),
        "stream_copy_norms": rounded(stream.norm(dim=-1).mean(dim=(0, 1))),
        "depth_connectivity": rounded(depth, 3) if simple else None,
        "depth_head_spread": depth_head_spread(depth) if simple else None,
    }


def depth_head_spread(depth: Tensor) -> float:  # [h, L+1, L+1]
    """Mean over the (reader, source) pairs of the std across heads: 0 when every head routes through depth the same way."""
    if depth.shape[0] < 2:
        return 0.0
    L1 = depth.shape[1]
    lower = torch.tril(torch.ones(L1, L1, dtype=torch.bool))  # reader l reads sources s <= l
    return depth.std(dim=0)[lower].mean().item()
