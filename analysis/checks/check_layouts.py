"""Check that the two stream layouts compute the same model: same weights, same logits, same gradients.

Usage: PYTHONPATH=src uv run python analysis/checks/check_layouts.py
The token layout ([B, T, n, D]) is what the GPU runs used; the streams layout ([n, D, B*T]) is what the TPU runs use.
"""

import torch

from mhc_lab.model import Model, ModelConfig

VARIANTS = {
    "mhc-h1": {},
    "global-h4": {"hc_heads": 4},
    "global-h16": {"hc_heads": 16},
    "local-h8": {"hc_heads": 8, "hc_predictor": "local"},
    "static-h8": {"hc_heads": 8, "hc_predictor": "static"},
    "joint-h4-global": {"hc_heads": 4, "hc_joint": True},
    "joint-h4-static": {"hc_heads": 4, "hc_joint": True, "hc_predictor": "static"},
    "n8": {"hc_mult": 8},
    "attn-reads-3": {"hc_attn_reads": 3},
    "sym-init-local-h4": {"hc_heads": 4, "hc_predictor": "local", "hc_init": "sym"},
    "global-h4-pre-only": {"hc_heads": 4, "hc_head_parts": "pre"},
    "global-h4-post-only": {"hc_heads": 4, "hc_head_parts": "post"},
    "global-h4-res-only": {"hc_heads": 4, "hc_head_parts": "res"},
    "global-h4-pre-post": {"hc_heads": 4, "hc_head_parts": "pre,post"},
    "static-h8-res-only": {"hc_heads": 8, "hc_predictor": "static", "hc_head_parts": "res"},
}


def run(model: Model, ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
    model.zero_grad(set_to_none=True)
    logits = model(ids)  # [B, T, vocab]
    loss = logits.square().mean()  # any scalar that depends on every logit
    loss.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    return logits.detach(), grads


def main() -> None:
    # The model keeps its stream in float32 (.float()); in float32 a few tiny, cancellation-dominated gradients (the head
    # connection's scale, ~1e-7) differ by ~1e-4 relative between layouts from summation order alone. Run in float64 instead.
    torch.Tensor.float = lambda self, *args, **kwargs: self.to(torch.float64)
    torch.manual_seed(0)
    small = {"vocab_size": 512, "dim": 96, "n_layers": 2, "n_heads": 4, "mlp_dim": 128, "max_seq_len": 32}
    ids = torch.randint(0, 512, (2, 32))
    for name, extra in VARIANTS.items():
        tokens = Model(ModelConfig(**small, **extra, hc_layout="tokens")).double()
        with torch.no_grad():  # move the dynamic weights off zero, so the check also covers the token-dependent path
            for p_name, p in tokens.named_parameters():
                if "connection" in p_name:
                    p.add_(torch.randn_like(p) * 0.05)
        streams = Model(ModelConfig(**small, **extra, hc_layout="streams")).double()
        streams.load_state_dict(tokens.state_dict())
        logits_t, grads_t = run(tokens, ids)
        logits_s, grads_s = run(streams, ids)
        logit_err = ((logits_t - logits_s).abs().max() / logits_t.abs().max()).item()
        grad_err = max(((grads_t[k] - grads_s[k]).abs().max() / grads_t[k].abs().max().clamp_min(1e-30)).item() for k in grads_t)
        verdict = "ok" if logit_err < 1e-12 and grad_err < 1e-9 else "MISMATCH"
        print(f"{name:18s} logits rel err {logit_err:.1e}  worst grad rel err {grad_err:.1e}  {verdict}")


if __name__ == "__main__":
    main()
