"""Time training steps (forward + backward + AdamW) of one model configuration on random tokens."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from mhc_lab.model import Model, ModelConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--conn", default="mhc")
    p.add_argument("--hc-heads", type=int, default=1)
    p.add_argument("--hc-predictor", default="global")
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument("--hc-joint", type=int, default=0)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--old-sinkhorn", type=int, default=0)  # 1: reduction form, 0: elementwise form
    p.add_argument("--compile", type=int, default=1)  # 0 eager, 1 one compiled graph per sublayer step, 2 whole model
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = ModelConfig(dim=args.dim, n_layers=args.layers, n_heads=args.heads, mlp_dim=args.mlp_dim,
                         max_seq_len=args.seq_len, conn=args.conn, hc_heads=args.hc_heads, hc_predictor=args.hc_predictor,
                         hc_mult=args.hc_mult, hc_joint=bool(args.hc_joint),
                         hc_sinkhorn_form="reduction" if args.old_sinkhorn else "elementwise")
    model = Model(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler()
    step_model = model
    if args.compile == 1:
        model.compile_regions()
    if args.compile == 2:
        step_model = torch.compile(model)
    ids = torch.randint(0, config.vocab_size, (args.micro_batch, args.seq_len + 1), device="cuda")
    times = []
    start = time.time()
    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = step_model(ids[:, :-1])
        loss = F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    steady = sorted(times[5:])[len(times[5:]) // 2]
    result = {"args": vars(args) | {"out": str(args.out), "data_dir": None}, "first_step_s": times[0],
              "median_step_s": steady, "tok_s": args.micro_batch * args.seq_len / steady,
              "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "total_s": time.time() - start}
    (args.out / "bench.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
