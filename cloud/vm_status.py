"""On a TPU VM: print one JSON line describing every job in ~/runs (for cloud/controller.py)."""
import json
import os
from pathlib import Path

out = {}
for run in sorted((Path.home() / "runs").glob("*")):
    info = {"exit_code": None, "alive": False, "last": None, "val": None, "chip": None}
    if (run / "chip").exists():
        info["chip"] = int((run / "chip").read_text())
    if (run / "exit_code").exists():
        info["exit_code"] = (run / "exit_code").read_text().strip()
    if (run / "pid").exists():
        try:
            os.kill(int((run / "pid").read_text()), 0)
            info["alive"] = True
        except (OSError, ValueError):
            pass
    metrics = run / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text().splitlines()[-30:]:
            r = json.loads(line)
            if "loss" in r:
                info["last"] = {k: r[k] for k in ("step", "loss", "grad_norm", "tok_s", "elapsed")}
            elif "val_loss" in r:
                info["val"] = (r["step"], r["val_loss"])
    out[run.name] = info
print(json.dumps(out))
