"""Unit tests for data loading and tokenization."""

import pytest
import torch
import tempfile
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PretrainDataset, make_dataloader


class TestPretrainDataset:
    def test_basic(self):
        tokens = list(range(4096))
        ds = PretrainDataset(tokens, seq_len=128)
        assert len(ds) == 31  # (4096-1)//128 = 31

        sample = ds[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (128,)
        assert sample["labels"].shape == (128,)

    def test_last_chunk_padded(self):
        tokens = list(range(300))  # 299 valid => (299-1)//128 = 2 full + partial
        ds = PretrainDataset(tokens, seq_len=128)
        # Last sample should be padded to 128
        sample = ds[-1]
        assert sample["input_ids"].shape == (128,)
        # Padded positions should have label=-100
        assert (sample["labels"][-1] == -100)

    def test_tensor_input(self):
        tokens = torch.arange(5000)
        ds = PretrainDataset(tokens, seq_len=128)
        assert len(ds) > 0

    def test_dataloader_integration(self):
        tokens = torch.arange(6000)
        ds = PretrainDataset(tokens, seq_len=128)
        loader = make_dataloader(ds, batch_size=4, shuffle=False, num_workers=0)

        batch = next(iter(loader))
        assert batch["input_ids"].shape == (4, 128)
        assert batch["labels"].shape == (4, 128)


class TestMakeDataloader:
    def test_shuffle_flag(self):
        tokens = torch.arange(4096)
        ds = PretrainDataset(tokens, seq_len=128)

        loader_shuffle = make_dataloader(ds, batch_size=8, shuffle=True, num_workers=0)
        loader_no_shuffle = make_dataloader(ds, batch_size=8, shuffle=False, num_workers=0)

        batch1 = next(iter(loader_shuffle))
        batch2 = next(iter(loader_no_shuffle))
        # With shuffle=True, batches might differ from sequential
        # Just verify shapes
        assert batch1["input_ids"].shape == (8, 128)
        assert batch2["input_ids"].shape == (8, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
