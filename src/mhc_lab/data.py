import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

REPO_URL = "https://huggingface.co/datasets/marcoshernanz/llm-lab-fineweb-edu-sample10bt-bpe-16384-full/resolve/main"
VOCAB_SIZE = 16384  # the BPE the shards were tokenized with
SHARD_TOKENS = 10_000_000


def download(data_dir: Path, train_shards: int, validation_shards: int = 1) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    names = [f"train_{i:05d}.npy" for i in range(train_shards)]
    names += [f"validation_{i:05d}.npy" for i in range(validation_shards)]
    for name in names:
        path = data_dir / name
        if path.exists():
            continue
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{REPO_URL}/{name}", partial)
        partial.rename(path)


def load_shards(data_dir: Path, split: str) -> list[np.ndarray]:
    paths = sorted(data_dir.glob(f"{split}_*.npy"))
    return [np.load(path, mmap_mode="r") for path in paths]


class TrainBatches:
    """Random windows from the train shards; the same seed gives the same batches in every run."""

    def __init__(self, data_dir: Path, batch_size: int, seq_len: int, seed: int) -> None:
        self.shards = load_shards(data_dir, "train")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.generator = torch.Generator().manual_seed(seed)

    def next(self) -> tuple[Tensor, Tensor]:
        shard_ids = torch.randint(len(self.shards), (self.batch_size,), generator=self.generator)
        max_start = SHARD_TOKENS - self.seq_len - 1
        starts = torch.randint(max_start, (self.batch_size,), generator=self.generator)
        rows = []
        for shard_id, start in zip(shard_ids.tolist(), starts.tolist()):
            window = self.shards[shard_id][start : start + self.seq_len + 1]
            rows.append(torch.from_numpy(window.astype(np.int64)))
        tokens = torch.stack(rows)  # [B, T+1]
        return tokens[:, :-1], tokens[:, 1:]


def validation_batches(data_dir: Path, batch_size: int, seq_len: int, num_batches: int) -> list[tuple[Tensor, Tensor]]:
    """Fixed, contiguous, non-overlapping windows from the start of the first validation shard."""
    shard = load_shards(data_dir, "validation")[0]
    batches = []
    for b in range(num_batches):
        rows = []
        for r in range(batch_size):
            start = (b * batch_size + r) * (seq_len + 1)
            window = shard[start : start + seq_len + 1]
            rows.append(torch.from_numpy(window.astype(np.int64)))
        tokens = torch.stack(rows)  # [B, T+1]
        batches.append((tokens[:, :-1], tokens[:, 1:]))
    return batches
