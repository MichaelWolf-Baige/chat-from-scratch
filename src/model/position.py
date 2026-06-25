"""
Rotary Position Embedding (RoPE).

Reference: RoFormer: Enhanced Transformer with Rotary Position Embedding
Paper: https://arxiv.org/abs/2104.09864

Supports:
    - Standard RoPE (base theta)
    - NTK-aware scaling (for extended context)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding.

    Applies rotary embeddings to query and key tensors.

    Args:
        d_head: Dimension per head (must be even).
        max_seq_len: Maximum sequence length (for precomputed cache).
        theta: Base frequency (default 10000.0). Larger values (e.g. 500000)
               slow frequency decay → better for long context.
    """

    def __init__(
        self, d_head: int, max_seq_len: int = 2048, theta: float = 10000.0
    ) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head ({d_head}) must be even for RoPE.")

        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute cos/sin for positions [0, max_seq_len)
        # Frequency per pair of dimensions
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )  # (d_head//2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)

        # [max_seq_len, d_head//2]
        angles = torch.outer(positions, freqs)

        # Expand to full d_head by repeating each frequency for the pair
        # (max_seq_len, d_head)
        angles_full = angles.repeat_interleave(2, dim=-1)

        self.register_buffer("cos_cached", angles_full.cos(), persistent=False)
        self.register_buffer("sin_cached", angles_full.sin(), persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given positions.

        Args:
            x: Input tensor [batch, seq_len, n_heads, d_head]
            position_ids: [batch, seq_len] or None.

        Returns:
            cos, sin: each broadcastable to x's shape.
        """
        if position_ids is not None:
            cos = self.cos_cached[position_ids]  # (B, S, d_head)
            sin = self.sin_cached[position_ids]
        else:
            seq_len = x.shape[1]
            cos = self.cos_cached[:seq_len].unsqueeze(0)  # (1, S, d_head)
            sin = self.sin_cached[:seq_len].unsqueeze(0)

        # Insert head dim: (B, S, 1, d_head) for broadcasting with (B, S, H, d_head)
        cos = cos.unsqueeze(2).to(x.dtype)
        sin = sin.unsqueeze(2).to(x.dtype)
        return cos, sin


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding to tensor x.

    Args:
        x: [batch, seq_len, n_heads, d_head]
        cos, sin: broadcastable to x's shape [..., seq_len, ..., d_head]

    Returns:
        Tensor of same shape as x with rotary encoding applied.
    """
    # Split into even/odd positions
    x_even = x[..., 0::2]  # (B, S, H, D//2)
    x_odd = x[..., 1::2]

    # cos and sin have shape (..., S, 1, D_head) but we need (..., S, 1, D//2)
    # since we split x — take every other element
    cos_half = cos[..., 0::2]  # (B, S, 1, D//2)
    sin_half = sin[..., 0::2]

    # Rotate: (x_e + i*x_o) * (cos + i*sin) → real and imag parts
    x_rotated_even = x_even * cos_half - x_odd * sin_half
    x_rotated_odd = x_even * sin_half + x_odd * cos_half

    # Interleave back
    result = torch.empty_like(x)
    result[..., 0::2] = x_rotated_even
    result[..., 1::2] = x_rotated_odd
    return result
