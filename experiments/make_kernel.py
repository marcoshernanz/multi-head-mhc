"""Build a Kaggle kernel folder for one experiment batch.

Usage: uv run python experiments/make_kernel.py experiments/<batch>
The batch folder holds jobs.py, which defines JOBS, TRAIN_SHARDS, DEADLINE_SECONDS and SLUG, and optionally
ACCELERATOR = "tpu" (default "gpu": 2x T4).
Then: kaggle kernels push -p experiments/<batch>/kernel
"""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
batch = Path(sys.argv[1]).resolve()
spec = runpy.run_path(str(batch / "jobs.py"))
sources = {str(p.relative_to(ROOT / "src")): p.read_text() for p in sorted((ROOT / "src" / "mhc_lab").glob("*.py"))}
tpu = spec.get("ACCELERATOR", "gpu") == "tpu"
template = (ROOT / "experiments" / ("launcher_tpu_template.py" if tpu else "launcher_template.py")).read_text()
for marker in ("SOURCES = {}", "JOBS = []", "TRAIN_SHARDS = 0", "DEADLINE_SECONDS = 0"):
    assert template.count(marker) == 1, marker
script = (
    template.replace("SOURCES = {}", f"SOURCES = {sources!r}")
    .replace("JOBS = []", f"JOBS = {spec['JOBS']!r}")
    .replace("TRAIN_SHARDS = 0", f"TRAIN_SHARDS = {spec['TRAIN_SHARDS']}")
    .replace("DEADLINE_SECONDS = 0", f"DEADLINE_SECONDS = {spec['DEADLINE_SECONDS']}")
)
kernel_dir = batch / "kernel"
kernel_dir.mkdir(exist_ok=True)
slug = spec["SLUG"]
(kernel_dir / f"{slug}.py").write_text(script)
metadata = {
    "id": f"marcoshernanz/{slug}",
    "title": slug,
    "code_file": f"{slug}.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": not tpu,
    "enable_tpu": tpu,
    "enable_internet": True,
    "keywords": [],
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
    "machine_shape": "TpuV5E8" if tpu else "NvidiaTeslaT4",
}
if tpu:  # the TPU VM image my earlier TPU kernels ran on (torch 2.8 + torch_xla 2.8 + jax)
    metadata["docker_image"] = "gcr.io/kaggle-private-byod/python-tpuvm@sha256:a2111cb9be558ea4bc187754bb95d7b65e90d8259434f1eb0e0ab1193ff498c0"
(kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
(batch / "jobs.json").write_text(json.dumps(spec["JOBS"], indent=1))
print(f"{slug}: {len(spec['JOBS'])} jobs, script {len(script)} chars")
