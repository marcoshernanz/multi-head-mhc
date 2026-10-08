"""Run a list of training jobs on all TPU chips of one host with torch_xla.launch: one process per chip.

Usage: python -m mhc_lab.xla_pool <jobs.json> <out_dir> <data_dir> <deadline (unix time)> [--smoke]
Each process takes the next unclaimed job (a claim is a file created with O_EXCL, so two processes never take the same job),
runs train_xla.main() in-process on its own chip, and writes runs/<job>/log.txt. No job starts after the deadline. The processes never communicate:
every job is an independent single-chip run. --smoke only prints each process's ordinal and a matmul speed.
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path


def claim(claims: Path, name: str) -> bool:
    try:
        fd = os.open(claims / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.close(fd)
    return True


def run_job(job: dict, out: Path, data: Path) -> str:
    from mhc_lab import train_xla

    run_dir = out / "runs" / job["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.argv = ["train_xla", "--out", str(run_dir), "--data-dir", str(data), *job["args"]]
    log = os.open(run_dir / "log.txt", os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log, 1)  # the job's Python and C++ output both go to its log
    os.dup2(log, 2)
    began = time.time()
    try:
        train_xla.main()
        result = "exit 0"
    except Exception:
        traceback.print_exc()
        result = "exit 1"
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(saved_out, 1)
    os.dup2(saved_err, 2)
    os.close(log)
    return f"{result} in {time.time() - began:.0f}s"


def worker(index: int, jobs_path: str, out: str, data: str, deadline: float, smoke: bool) -> None:
    import torch
    import torch_xla
    import torch_xla.runtime as xr

    ordinal = xr.global_ordinal()
    if smoke:
        device = torch_xla.device()
        a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        c = a @ a
        torch_xla.sync()
        c.cpu()
        t = time.time()
        for _ in range(20):
            c = a @ a
        torch_xla.sync()
        c.cpu()
        tflops = 20 * 2 * 4096**3 / (time.time() - t) / 1e12
        local = getattr(xr, "addressable_runtime_device_count", lambda: "?")()
        print(f"SMOKE ordinal {ordinal} index {index} local devices {local} tflops {tflops:.1f}", flush=True)
        return
    out_dir = Path(out)
    claims = out_dir / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    for job in json.loads(Path(jobs_path).read_text()):
        if time.time() > deadline:
            return
        if not claim(claims, job["name"]):
            continue
        result = run_job(job, out_dir, Path(data))
        (claims / job["name"]).write_text(f"{result} on chip {ordinal}")
        print(f"[chip {ordinal}] {job['name']}: {result}", flush=True)


def main() -> None:
    import torch_xla

    jobs_path, out, data, deadline = sys.argv[1:5]
    smoke = "--smoke" in sys.argv
    torch_xla.launch(worker, args=(jobs_path, out, data, float(deadline), smoke))


if __name__ == "__main__":
    main()
