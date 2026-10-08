"""Train one run on one TPU chip through torch_xla. Same model, data order, schedule and output files as train.py.

Differences from the GPU loop, all forced by XLA's lazy graphs:
- bf16 autocast, no GradScaler (TPUs have no fast fp16);
- no host reads inside a step: losses and grad norms stay on the device and are fetched every --log-every steps;
- one torch_xla.sync() per step, so each step is one compiled graph (compiled once, then reused); with --micro-sync 1 also one
  after every micro-step (the compiler then cannot keep several micro-steps' activations alive at once);
- tokens/s is measured between fetches, after the first --timing-skip steps (which include compilation);
- the coefficient probe runs on the CPU copy of the final model (float32), to avoid compiling dozens of tiny graphs.

With --ckpt-every N (for preemptible Cloud TPUs) the run saves out/ckpt.pt every N steps (model, optimizer, the batch generator's
state, the per-step losses so far) and, if that file exists at start, resumes from it. Nothing in the training math changes, so a
resumed run continues the same trajectory (checked: analysis/checks/check_resume.py).
"""

import json
import math
import os
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.debug.metrics as met

from mhc_lab.data import TrainBatches, validation_batches
from mhc_lab.model import Model
from mhc_lab.probe import coefficient_report
from mhc_lab.train import build_config, lr_at, param_groups, parse_args

TIMING_SKIP = 30


COMPILES_BEFORE = 0  # set at the start of main(): several jobs can run one after another in one process (xla_pool)


def compiles() -> int:
    data = met.metric_data("CompileTime")
    total = data[0] if data else 0
    return total - COMPILES_BEFORE


def device_name() -> str:
    """The TPU type: TPU_ACCELERATOR_TYPE on Kaggle (v5litepod-8), MHC_DEVICE from cloud/runner.py on Cloud TPU VMs."""
    kind = os.environ.get("MHC_DEVICE") or os.environ.get("TPU_ACCELERATOR_TYPE", "")
    if kind.startswith("v5litepod"):
        return "TPU v5e chip (torch_xla)"
    return f"TPU {kind} chip (torch_xla)" if kind else "TPU chip (torch_xla)"


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


def evaluate(model, batches, device) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    with torch.no_grad():
        for x, y in batches:
            x = x.to(device)
            y = y.to(device)
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(x)
            total = total + F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            torch_xla.sync()
    model.train()
    return total.item() / len(batches)


def diagnostics(model, ids, device) -> dict:
    """Instability indicators on a fixed batch (no gradient): per attention layer the largest logit and the mean entropy, per
    sublayer the RMS of its input u (before its norm) and of its output, and the output logits' mean squared log-partition
    function z^2 (what z-loss penalizes) and largest absolute value."""
    from mhc_lab.model import Attention, Sublayer

    attentions = [m for m in model.modules() if isinstance(m, Attention)]
    sublayers = [m for m in model.modules() if isinstance(m, Sublayer)]
    for m in attentions + sublayers:
        m.diag = []
    model.eval()
    with torch.no_grad():
        with torch.autocast("xla", dtype=torch.bfloat16):
            logits = model(ids.to(device))
        logits = logits.float()
        z = torch.logsumexp(logits, dim=-1)
        stats = [torch.stack([m.diag[0][0], m.diag[0][1]]) for m in attentions]
        rms = [torch.stack([m.diag[0][0], m.diag[0][1]]) for m in sublayers]
        out = torch.stack([z.pow(2).mean(), logits.abs().max()])
        torch_xla.sync()
    model.train()
    for m in attentions + sublayers:
        m.diag = None
    attn, rms = torch.stack(stats).cpu().tolist(), torch.stack(rms).cpu().tolist()
    return {"attn_max_logit": [round(a[0], 3) for a in attn], "attn_entropy": [round(a[1], 4) for a in attn],
            "rms_in": [round(r[0], 4) for r in rms], "rms_out": [round(r[1], 4) for r in rms],
            "z2": round(out[0].item(), 3), "max_logit": round(out[1].item(), 3)}


def main() -> None:
    global COMPILES_BEFORE
    COMPILES_BEFORE = 0
    COMPILES_BEFORE = compiles()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch_xla.device()
    torch.manual_seed(args.seed)
    config = build_config(args)
    model = Model(config)  # initialized on the CPU, so the same seed gives the same weights as on the GPU
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_hc_params = sum(p.numel() for n_, p in model.named_parameters() if "connection" in n_)
    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    train = TrainBatches(args.data_dir, args.batch, args.seq_len, args.seed)
    val = validation_batches(args.data_dir, args.micro_batch, args.seq_len, args.final_eval_batches)
    accum = args.batch // args.micro_batch
    diag_ids = val[0][0][:4]  # the same 4 validation windows for every run and step

    run_info = {"args": {k: str(v) for k, v in vars(args).items() if k != "ckpt_every"}, "config": asdict(config),
                "params": n_params, "hc_params": n_hc_params, "device": device_name(),
                "torch": torch.__version__, "torch_xla": torch_xla.__version__, "precision": "bf16 autocast"}
    (args.out / "run.json").write_text(json.dumps(run_info, indent=1))
    print(json.dumps({"params": n_params, "hc_params": n_hc_params}), flush=True)

    ckpt_path = args.out / "ckpt.pt"
    pending = []  # (step, lr, loss, grad norm) not yet fetched; device tensors, or floats after a resume
    losses = []
    diverged = False
    first_step, elapsed_before, resumes = 0, 0.0, 0
    if args.ckpt_every and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        train.generator.set_state(state["generator"])
        first_step, losses, pending = state["step"], state["losses"], state["pending"]
        elapsed_before, resumes = state["elapsed"], state["resumes"] + 1
        torch_xla.sync()  # materialize the restored weights and optimizer state before the first step
        with open(args.out / "metrics.jsonl", "r+b") as f:
            f.truncate(state["metrics_bytes"])  # drop what was logged after the checkpoint
        print(f"resumed from step {first_step} (resume {resumes})", flush=True)
    log = open(args.out / "metrics.jsonl", "a" if first_step else "w")
    start = time.time() - elapsed_before
    timed_from = None  # (step, time) where throughput timing starts
    paused = 0.0  # seconds spent saving checkpoints since timed_from (left out of tokens/s)
    tok_s = None
    step = first_step
    for step in range(first_step, args.steps):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_mult"]
        x, y = train.next()
        loss_sum = torch.zeros((), device=device)
        for micro in range(accum):
            xs = x[micro * args.micro_batch : (micro + 1) * args.micro_batch].to(device)
            ys = y[micro * args.micro_batch : (micro + 1) * args.micro_batch].to(device)
            with torch.autocast("xla", dtype=torch.bfloat16):
                logits = model(xs)
            loss = F.cross_entropy(logits.float().flatten(0, 1), ys.flatten()) / accum
            loss.backward()
            loss_sum = loss_sum + loss.detach()
            if args.micro_sync:
                torch_xla.sync()  # cut the graph here, so only one micro-step's activations are ever alive
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch_xla.sync()
        pending.append((step, lr, loss_sum, grad_norm))
        last = step + 1 == args.steps
        if step % args.log_every == 0 or last:
            values = [(s, lr_s, float(l), float(g)) for s, lr_s, l, g in pending]  # blocks until the step is done
            now = time.time()
            pending = []
            if timed_from is None and step >= first_step + TIMING_SKIP:
                timed_from, paused = (step, now), 0.0
            elif timed_from is not None and step > timed_from[0]:
                tok_s = (step - timed_from[0]) * args.batch * args.seq_len / (now - timed_from[1] - paused)
            for s, lr_s, loss_value, grad_value in values:
                losses.append(loss_value)
                if not math.isfinite(loss_value) or (s > 100 and loss_value > 12):
                    diverged = True
            s, lr_s, loss_value, grad_value = values[-1]
            record = {"step": s, "loss": loss_value, "lr": lr_s, "grad_norm": grad_value, "tok_s": tok_s,
                      "elapsed": now - start, "compiles": compiles()}
            if args.log_steps:  # every step since the last record, for spike counts
                record["losses"] = [v[2] for v in values]
                record["grad_norms"] = [v[3] for v in values]
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 0:
                print(f"first step (with compilation) {now - start:.1f}s, compiles {compiles()}", flush=True)
        if args.diag_every and (step % args.diag_every == 0 or last):
            log.write(json.dumps({"step": step, "diag": diagnostics(model, diag_ids, device)}) + "\n")
            log.flush()
        if (step + 1) % args.eval_every == 0 and not last:
            val_loss = evaluate(model, val[: args.eval_batches], device)
            log.write(json.dumps({"step": step + 1, "val_loss": val_loss, "elapsed": time.time() - start}) + "\n")
            log.flush()
            print(f"step {step + 1} val {val_loss:.4f} tok/s {tok_s} compiles {compiles()}", flush=True)
        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and not last and not diverged:
            began = time.time()
            pending = [(s, lr_s, float(l), float(g)) for s, lr_s, l, g in pending]
            log.flush()
            state = {"model": to_cpu(model.state_dict()), "optimizer": to_cpu(optimizer.state_dict()),
                     "generator": train.generator.get_state(), "step": step + 1, "losses": losses, "pending": pending,
                     "elapsed": time.time() - start, "resumes": resumes, "metrics_bytes": log.tell()}
            torch.save(state, ckpt_path.with_suffix(".tmp"))
            os.replace(ckpt_path.with_suffix(".tmp"), ckpt_path)
            paused += time.time() - began
        if diverged or time.time() - start > args.max_seconds:
            break

    final_val = float("nan") if diverged else evaluate(model, val, device)
    log.write(json.dumps({"step": step + 1, "val_loss": final_val, "final": True, "elapsed": time.time() - start}) + "\n")
    log.close()
    summary = {
        **run_info,
        "steps_done": step + 1,
        "diverged": diverged,
        "final_val_loss": final_val,
        "final_train_loss_avg100": sum(losses[-100:]) / max(1, len(losses[-100:])),
        "tok_s": tok_s,
        "xla_compiles": compiles(),
        "seconds": time.time() - start,
        "resumes": resumes,
    }
    if config.conn == "mhc" and not diverged:
        cpu_model = model.to("cpu")
        summary["coefficients"] = coefficient_report(cpu_model, val[0][0][:4], torch.float32)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("final_val_loss", "tok_s", "xla_compiles", "seconds", "diverged")}), flush=True)


if __name__ == "__main__":
    main()
