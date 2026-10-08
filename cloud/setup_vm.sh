#!/bin/bash
# Prepare a Cloud TPU v6e VM (Ubuntu 22.04, runtime v2-alpha-tpuv6e) for train_xla.py. Idempotent.
# Pins the software stack of the Kaggle TPU v5e image the earlier batches ran on (from their logs): Python 3.12, torch 2.8.0+cpu,
# torch_xla 2.8.0, libtpu 0.0.17, numpy 2.5.0. Then only the chip differs.
set -euo pipefail
if [ ! -x "$HOME/.local/bin/uv" ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
if [ ! -x "$HOME/venv/bin/python" ]; then uv venv -q --python 3.12 "$HOME/venv"; fi
PY="$HOME/venv/bin/python"
if ! "$PY" -c "import torch_xla" 2>/dev/null; then
  uv pip install -q --python "$PY" torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$PY" torch_xla==2.8.0 libtpu==0.0.17 numpy==2.5.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi
"$PY" - <<'PY'
import importlib.metadata as m
print({p: m.version(p) for p in ("torch", "torch_xla", "libtpu", "numpy")})
PY
# Data: the same 20 train shards + validation shard 0 as every TPU batch (TrainBatches draws shard ids from the shard count).
mkdir -p "$HOME/data"
"$PY" - <<'PY'
import sys, pathlib, concurrent.futures as cf, urllib.request, time
REPO = "https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full/resolve/main"
data = pathlib.Path.home() / "data"
names = [f"train_{i:05d}.npy" for i in range(20)] + ["validation_00000.npy"]
def fetch(name):
    path = data / name
    if path.exists():
        return
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(f"{REPO}/{name}", data / (name + ".part"))
            (data / (name + ".part")).rename(path)
            return
        except Exception as error:
            print("retry", name, error, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(name)
with cf.ThreadPoolExecutor(8) as pool:
    list(pool.map(fetch, names))
print("data ok:", len(list(data.glob("*.npy"))), "shards")
PY
