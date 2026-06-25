"""Data loading for pretraining — next-token prediction."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, IterableDataset


class PretrainDataset(Dataset):
    """Map-style dataset for pretraining.

    Expects pre-tokenized data as a 1D tensor of token indices or list[int].

    Each sample is a sequence of `seq_len` tokens. The target is the same sequence
    shifted by one position (handled in the model's loss computation, not here).

    Args:
        tokens: 1D tensor/list of token indices (the full corpus).
        seq_len: Sequence length (default 2048).
    """

    def __init__(self, tokens: torch.Tensor | list[int], seq_len: int = 2048) -> None:
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        if tokens.dim() != 1:
            raise ValueError(f"Expected 1D tensor, got {tokens.dim()}D")

        self.tokens = tokens
        self.seq_len = seq_len
        # Number of non-overlapping sequences
        self._len = (len(tokens) - 1) // seq_len

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len

        input_ids = self.tokens[start:end]  # (seq_len,)
        labels = self.tokens[start:end]     # same, shifted handled by model

        # Pad if needed (last chunk might be shorter)
        if input_ids.shape[0] < self.seq_len:
            pad_len = self.seq_len - input_ids.shape[0]
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "labels": labels}


class PretrainIterableDataset(IterableDataset):
    """Streaming dataset for pretraining — avoids loading full corpus into memory.

    Each worker loads a different shard of the corpus.

    Args:
        data_files: List of paths to .bin files containing token indices.
        seq_len: Sequence length.
        shuffle_buffer_size: Shuffle buffer size in samples.
    """

    def __init__(
        self,
        data_files: list[str],
        seq_len: int = 2048,
        shuffle_buffer_size: int = 10000,
    ) -> None:
        self.data_files = data_files
        self.seq_len = seq_len
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        import random
        from itertools import cycle

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.data_files
        else:
            files = [
                f for i, f in enumerate(self.data_files)
                if i % worker_info.num_workers == worker_info.id
            ]
        if not files:
            return

        buffer = []

        for file_path in cycle(files):
            tokens = _memmap_tokens(file_path)

            for i in range(0, len(tokens) - self.seq_len, self.seq_len):
                chunk = tokens[i:i + self.seq_len]
                sample = {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }

                if len(buffer) >= self.shuffle_buffer_size:
                    idx = random.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = sample
                else:
                    buffer.append(sample)

        random.shuffle(buffer)
        yield from buffer


def _memmap_tokens(path: str) -> "np.ndarray":
    """Memory-map a binary file of uint16 token indices."""
    import numpy as np
    return np.memmap(path, dtype=np.uint16, mode="r")


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with sensible defaults for language modeling."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and isinstance(dataset, Dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
