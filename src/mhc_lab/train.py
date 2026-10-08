"""Train one run and write metrics.jsonl and summary.json into --out."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mhc_lab.data import VOCAB_SIZE, TrainBatches, validation_batches
from mhc_lab.model import Model, ModelConfig
from mhc_lab.probe import coefficient_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--conn", default="mhc")
    p.add_argument("--hc-mult", type=int, default=4)
    p.add_argument("--hc-heads", type=int, default=1)
    p.add_argument("--hc-predictor", default="global")
    p.add_argument("--hc-init", default="hc")
    p.add_argument("--hc-pre-logit", type=float, default=2.0)
    p.add_argument("--hc-res-logit", type=float, default=4.0)
    p.add_argument("--hc-attn-reads", type=int, default=1)
    p.add_argument("--hc-head-writes", type=int, default=0)
    p.add_argument("--hc-joint", type=int, default=0)
    p.add_argument("--hc-alpha-init", type=float, default=0.01)
    p.add_argument("--hc-sinkhorn-form", default="reduction")
    p.add_argument("--hc-head-parts", default="pre,post,res")  # which coefficients are per head; the rest shared by all heads
    p.add_argument("--hc-local-gain", type=float, default=1.0)
    p.add_argument("--hc-layout", default="tokens")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp-dim", type=int, default=1024)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--micro-sync", type=int, default=0)  # train_xla only: 1 = one compiled graph per micro-step (less memory)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--hc-lr-mult", type=float, default=1.0)
    p.add_argument("--conn-lr-mult", type=float, default=1.0)  # LR multiplier for every connection parameter (phi, biases, alpha)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--final-eval-batches", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--compile", type=int, default=1)  # 0 eager, 1 one compiled graph per sublayer step, 2 whole model
    p.add_argument("--max-seconds", type=float, default=1e9)
    p.add_argument("--ckpt-every", type=int, default=0)  # train_xla only: save out/ckpt.pt every N steps and resume from it
    p.add_argument("--qk-norm", type=int, default=0)  # 1: RMSNorm on each attention head's queries and keys
    p.add_argument("--diag-every", type=int, default=0)  # train_xla only: every N steps, attention and output-logit statistics
    p.add_argument("--log-steps", type=int, default=0)  # train_xla only: 1 also logs every step's loss and grad norm
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_dim=args.mlp_dim,
        max_seq_len=args.seq_len,
        conn=args.conn,
        hc_mult=args.hc_mult,
        hc_heads=args.hc_heads,
        hc_predictor=args.hc_predictor,
        hc_init=args.hc_init,
        hc_pre_logit=args.hc_pre_logit,
        hc_res_logit=args.hc_res_logit,
        hc_attn_reads=args.hc_attn_reads,
        hc_head_writes=bool(args.hc_head_writes),
        hc_joint=bool(args.hc_joint),
        hc_alpha_init=args.hc_alpha_init,
        hc_sinkhorn_form=args.hc_sinkhorn_form,
        hc_local_gain=args.hc_local_gain,
        hc_head_parts=args.hc_head_parts,
        hc_layout=args.hc_layout,
        qk_norm=bool(getattr(args, "qk_norm", 0)),
    )


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, args.steps - args.warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cosine)


def param_groups(model: Model, args: argparse.Namespace) -> list[dict]:
    decay, no_decay, hc, phi = [], [], [], []
    conn = getattr(args, "conn_lr_mult", 1.0)
    for name, p in model.named_parameters():
        if "connection" in name and not name.endswith("phi"):
            hc.append(p)
        elif "connection" in name:
            # HC: weight decay on the dynamic weights, none on the static ones. phi gets its own group only when its LR differs,
            # so runs at --conn-lr-mult 1 keep exactly the earlier optimizer layout (and stay bit-identical to earlier runs).
            (phi if conn != 1.0 else decay).append(p)
        elif p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": args.wd, "lr_mult": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_mult": 1.0},
    ]
    if hc:
        groups.append({"params": hc, "weight_decay": 0.0, "lr_mult": args.hc_lr_mult * conn})
    if phi:
        groups.append({"params": phi, "weight_decay": args.wd, "lr_mult": conn})
    return groups


def evaluate(model, batches, device, amp_dtype) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = model(x)
            total += F.cross_entropy(logits.float().flatten(0, 1), y.flatten()).item()
    model.train()
    return total / len(batches)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config = build_config(args)
    model = Model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_hc_params = sum(p.numel() for n_, p in model.named_parameters() if "connection" in n_)
    groups = param_groups(model, args)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    step_model = model
    if args.compile == 1 and device.type == "cuda":
        model.compile_regions()
    if args.compile == 2 and device.type == "cuda":
        step_model = torch.compile(model)

    train = TrainBatches(args.data_dir, args.batch, args.seq_len, args.seed)
    val = validation_batches(args.data_dir, args.micro_batch, args.seq_len, args.final_eval_batches)
    accum = args.batch // args.micro_batch

    run_info = {"args": {k: str(v) for k, v in vars(args).items()}, "config": asdict(config),
                "params": n_params, "hc_params": n_hc_params,
                "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "torch": torch.__version__}
    (args.out / "run.json").write_text(json.dumps(run_info, indent=1))
    print(json.dumps({"params": n_params, "hc_params": n_hc_params}), flush=True)

    log = open(args.out / "metrics.jsonl", "w")
    start = time.time()
    timed_tokens, timed_seconds = 0, 0.0
    diverged = False
    losses = []
    step = 0
    for step in range(args.steps):
        t0 = time.time()
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_mult"]
        x, y = train.next()
        loss_sum = 0.0
        for micro in range(accum):
            xs = x[micro * args.micro_batch : (micro + 1) * args.micro_batch].to(device, non_blocking=True)
            ys = y[micro * args.micro_batch : (micro + 1) * args.micro_batch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = step_model(xs)
            loss = F.cross_entropy(logits.float().flatten(0, 1), ys.flatten()) / accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= 20:
            timed_tokens += args.batch * args.seq_len
            timed_seconds += dt
        losses.append(loss_sum)
        if not math.isfinite(loss_sum) or (step > 100 and loss_sum > 12):
            diverged = True
        if step % args.log_every == 0 or diverged:
            record = {"step": step, "loss": loss_sum, "lr": lr, "grad_norm": grad_norm,
                      "scale": scaler.get_scale() if scaler.is_enabled() else 1.0,
                      "tok_s": timed_tokens / timed_seconds if timed_seconds else None,
                      "elapsed": time.time() - start}
            log.write(json.dumps(record) + "\n")
            log.flush()
        if (step + 1) % args.eval_every == 0 and step + 1 < args.steps:
            val_loss = evaluate(step_model, val[: args.eval_batches], device, amp_dtype)
            log.write(json.dumps({"step": step + 1, "val_loss": val_loss, "elapsed": time.time() - start}) + "\n")
            log.flush()
            print(f"step {step + 1} val {val_loss:.4f} tok/s {timed_tokens / max(timed_seconds, 1e-9):.0f}", flush=True)
        if diverged or time.time() - start > args.max_seconds:
            break

    final_val = float("nan") if diverged else evaluate(step_model, val, device, amp_dtype)
    log.write(json.dumps({"step": step + 1, "val_loss": final_val, "final": True, "elapsed": time.time() - start}) + "\n")
    log.close()
    summary = {
        **run_info,
        "steps_done": step + 1,
        "diverged": diverged,
        "final_val_loss": final_val,
        "final_train_loss_avg100": sum(losses[-100:]) / len(losses[-100:]),
        "tok_s": timed_tokens / timed_seconds if timed_seconds else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "seconds": time.time() - start,
    }
    if config.conn == "mhc" and not diverged:
        summary["coefficients"] = coefficient_report(model, val[0][0][:4].to(device), amp_dtype)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("final_val_loss", "tok_s", "peak_mem_gb", "seconds", "diverged")}), flush=True)


if __name__ == "__main__":
    main()
