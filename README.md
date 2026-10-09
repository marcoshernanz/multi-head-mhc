# Multi-Head mHC: No Gain at Equal Parameters

Code, raw results and paper source for the technical report *Multi-Head mHC: No Gain at Equal Parameters* (Marcos Hernanz,
October 2026).

Manifold-constrained hyper-connections (mHC) widen the residual stream of a transformer into n = 4 copies that each layer reads
from, writes to and mixes with a doubly stochastic matrix. One coefficient per copy is shared by all channels. Multi-head mHC
splits the channels into h groups and gives each group its own read, write and mix, while keeping the guarantees of mHC.

We trained GPT-style models of 27M and 112M parameters on FineWeb-Edu, 629 TPU runs in all, and compared each variant at its own
best learning rate against mHC (same parameter count) or against an MLP control that spends the same extra parameters on a wider
feed-forward layer. The main findings:

- Multi-head mHC gives no gain at equal parameters. With mHC's parameter count it is no better than mHC, and when it adds
  parameters it is no better than a wider feed-forward layer.
- At 27M this also holds with QK-norm, with four times the training tokens (393M), and for each per-head block on its own.
- At one shared learning rate the heads seem to improve on mHC, but a wider feed-forward layer with the same parameters gains as
  much.
- Without QK-norm the heads degrade more gently when the learning rate is too high. The failure is attention-logit growth, which
  the heads do not prevent, and QK-norm removes both the failure and the advantage of the heads.

## Layout

| Path | Contents |
|---|---|
| `src/mhc_lab/` | The model and training code: `model.py` (a small GPT with the residual connection, mHC and multi-head mHC), `train.py` (GPU or CPU), `train_xla.py` (TPU, used for every TPU run), `data.py`, `probe.py` |
| `experiments/` | One folder per batch of runs (`exp00` to `exp20`). `jobs.py` lists every run of the batch with its exact arguments and says, at the top, why the batch was run. Batches run on Kaggle also keep the exact script that was pushed (`kernel/`) |
| `results/` | The raw output of every run, one folder per batch (format below) |
| `analysis/` | The scripts that compute every number, table and figure of the paper from `results/`, and correctness checks in `checks/` |
| `paper/` | LaTeX source of the paper and its figures |
| `cloud/` | The runner for Google Cloud spot TPU VMs, the VM setup scripts and the ledger of every VM created (`ledger.jsonl`, summed by `cost.py`) |
| `notes/` | `plan.md` (hypotheses and decision rules, each dated before the results it decides), `design.md` (the variants and their math), `gcp.md` and `kaggle.md` (how runs were launched, and why results from different chips are comparable), `literature/` (the literature search) |
| `LOG.md` | The research log, in order: every step, decision and result |
| `REPORT.md` | The working report written during the experiments; the paper supersedes it |
| `figures/` | Exploratory plots made by the `analysis/` scripts and used in `REPORT.md` |

## Setup

The analysis needs Python 3.13, NumPy, Matplotlib and PyTorch, installed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Training data is FineWeb-Edu sample-10BT tokenized with a 16,384-token BPE, in 10M-token shards on Hugging Face
([`marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full`](https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full)).
The runs from `exp04` on read the first 20 train shards (the pilots read fewer, as `TRAIN_SHARDS` in each `jobs.py` says), and all
of them score validation on shard 0. Download exactly 20 train shards, because the batch sampler draws shard ids from the number
of shards present:

```bash
PYTHONPATH=src uv run python -c "from pathlib import Path; from mhc_lab.data import download; download(Path('data'), 20)"
```

## Reproducing the paper from the raw results

No training is needed: every script reads `results/`. Run them from `analysis/`.

| Paper | Command |
|---|---|
| Figures 1, 3, 4 and 6; Table 4 (27M and pooled 112M columns) | `uv run python paper_figures.py` |
| Figure 5 | `uv run python stability_figure.py --partial` |
| Figure 2 | `paper/figures/src/method.tex` (TikZ) |
| Table 4 (112M per-chip columns), Tables 10 and 11 | `uv run python lr_sweep.py ../results/exp11_v6e_check ../results/exp12_d384_lr_grid ../results/exp15_d768_v6e --seeds 0,1,2`, and for TPU v5e `uv run python lr_sweep.py ../results/exp04_main ../results/exp06_controls ../results/exp07_scale ../results/exp09_d768_lr ../results/exp10_d384_lr` |
| Table 5 | `uv run python long_training.py ../results/exp13_d384_long ../results/exp04_main ../results/exp06_controls ../results/exp08_local` |
| Table 6 | `uv run python params_frontier.py ../results/exp04_main ../results/exp06_controls ../results/exp07_scale ../results/exp08_local` |
| Table 7 | `uv run python stability.py`, and `uv run python stability.py --width 768` for 112M |
| Table 8 | `uv run python head_parts.py ../results/exp11_v6e_check ../results/exp12_d384_lr_grid ../results/exp14_head_parts ../results/exp17_parts_best_lr` |
| Table 9 | `uv run python hardware_check.py ../results/exp11_v6e_check ../results/exp04_main ../results/exp06_controls ../results/exp07_scale ../results/exp09_d768_lr ../results/exp10_d384_lr` |

`paper_figures.py` and `stability_figure.py` write into `paper/figures/`. Build the paper with `latexmk` in `paper/`; the PDF is
`paper/build/main.pdf`.

## Results format

Each run has a folder `results/<batch>/runs/<run>/` with

- `run.json`: the command-line arguments, the model configuration, the parameter counts and the device;
- `metrics.jsonl`: the training log, one JSON record per line (training loss and gradient norm, every step in `exp19`;
  validation loss at fixed intervals; in `exp19` also diagnostics of the attention and output logits every 250 steps);
- `summary.json`: the final validation loss, throughput and whether the run diverged;
- `log.txt` and `exit_code`.

Batches run on Google Cloud also keep `controller.log` and `status.json` from `cloud/controller.py`; batches run on Kaggle keep the
kernel log.

## Batches

| Batch | Question | Hardware |
|---|---|---|
| `exp00_smoke` | Environment, compilation, throughput and memory | Kaggle T4 GPU |
| `exp01_bench` | Throughput of the connection implementations | Kaggle T4 GPU |
| `exp02_pilot` | Learning rate and initialization for the residual and mHC | Kaggle T4 GPU |
| `exp03_tpu_probe` | Running eight single-chip processes; compile time and speed of each variant | Kaggle TPU v5e |
| `exp04_main` | Every variant at 27M, h = 1 to 16, both predictors | Kaggle TPU v5e |
| `exp05_gpu_focus` | The deficit of the local predictor | Kaggle T4 GPU |
| `exp06_controls` | MLP controls, a first 112M check, four times the learning rate | Kaggle TPU v5e |
| `exp07_scale` | 112M with parameter controls | Kaggle TPU v5e |
| `exp08_local` | Local heads at 27M (same parameters as mHC) | Kaggle TPU v5e |
| `exp09_d768_lr` | Learning-rate sweep at 112M | Google Cloud TPU v5e |
| `exp10_d384_lr` | Learning-rate dependence at 27M | Kaggle TPU v5e |
| `exp11_v6e_check` | Whether TPU v6e results match TPU v5e | Google Cloud TPU v6e |
| `exp12_d384_lr_grid` | Learning-rate grid at 27M | Google Cloud TPU v6e |
| `exp13_d384_long` | Four times the tokens at 27M | Google Cloud TPU v5e |
| `exp14_head_parts` | Each per-head block on its own | Google Cloud TPU v6e |
| `exp15_d768_v6e` | Learning-rate grid at 112M | Google Cloud TPU v6e |
| `exp16_conn_lr` | A higher learning rate for the connection parameters (exploratory) | Google Cloud TPU v6e |
| `exp17_parts_best_lr` | The per-head blocks at the best learning rate | Google Cloud TPU v6e |
| `exp18_d768_parts` | The per-head blocks at 112M | Google Cloud TPU v6e |
| `exp19_stability` | Behavior above the optimal learning rate, with and without QK-norm | Google Cloud TPU v6e |
| `exp20_d768_qknorm` | 112M with QK-norm (stopped before any run finished, for lack of spot capacity) | Google Cloud TPU v6e |

## Running experiments

The exact arguments of every run are in its `run.json` and in the batch's `jobs.py`. A single run on a GPU or CPU:

```bash
PYTHONPATH=src uv run python -m mhc_lab.train --out runs/demo --data-dir data --hc-heads 4 --steps 6000 --lr 2.1e-3
```

TPU runs use `python -m mhc_lab.train_xla` with the same arguments, under `torch_xla` 2.8. `notes/kaggle.md` and `notes/gcp.md`
describe how batches were launched on Kaggle and on Google Cloud spot TPU VMs. The Google Cloud runner creates and deletes VMs in
your own project and costs money; read `notes/gcp.md` before using it. Do not use fused gradient accumulation on TPU v6e with
`torch_xla` 2.8: it computes wrong gradients (`notes/gcp.md`).

## Citation

```bibtex
@misc{hernanz2026multihead,
  title  = {Multi-Head {mHC}: No Gain at Equal Parameters},
  author = {Hernanz, Marcos},
  year   = {2026},
  note   = {Technical report},
  url    = {https://github.com/marcoshernanz/multi-head-mhc}
}
```

## License

MIT (`LICENSE`). The training data is derived from FineWeb-Edu, released under ODC-By 1.0. `notes/literature/raw/` keeps copies of
DeepSeek's reference code under DeepSeek's MIT license (`notes/literature/raw/LICENSE-DeepSeek`).
