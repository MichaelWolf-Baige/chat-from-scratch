"""Distributed training utilities (DDP wrapper).

Phase 1: single-GPU, this module is unused.
Phase 2-3: DDP for multi-GPU data-parallel training.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get rank of current process."""
    if not is_distributed():
        return 0
    return dist.get_rank()


def setup_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """Initialize distributed process group.

    Call once at the start of training for DDP.
    """
    if is_distributed():
        return

    dist.init_process_group(backend=backend, init_method=init_method)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def cleanup_distributed() -> None:
    """Destroy distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module) -> nn.Module:
    """Wrap model in DDP for data-parallel training."""
    if not is_distributed():
        return model

    return nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist.get_rank() % torch.cuda.device_count()],
        output_device=dist.get_rank() % torch.cuda.device_count(),
        find_unused_parameters=False,      # strict: all params must receive grads
        gradient_as_bucket_view=True,       # memory efficiency
    )
