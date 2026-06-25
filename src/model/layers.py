"""
Transformer building blocks: RMSNorm, SwiGLU FFN, TransformerBlock.

Llama-style architecture components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import Attention


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm (no mean subtraction, no bias).
    Used in Llama, Mistral, and most modern LLMs.

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) or (B, S, H, D)
        dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return (self.weight * x).to(dtype)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Architecture: x → gate(x) * up(x) → down(…) → output
    where gate and up are linear projections, gate uses SiLU activation.

    Standard FFN:       2 * d_model * d_ff  params
    SwiGLU FFN:          3 * d_model * d_ff  params (gate, up, down)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # SiLU activation on gate
        up = self.up_proj(x)
        hidden = gate * up                 # element-wise gating
        hidden = self.down_proj(hidden)
        return self.dropout(hidden)


class TransformerBlock(nn.Module):
    """A single Llama-style Transformer block.

    Layout (pre-norm):
        x ← x + Attention(RMSNorm(x))
        x ← x + SwiGLUFFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        d_ff: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        rms_norm_eps: float = 1e-6,
        use_flash_attention: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.attention = Attention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_qk_norm=use_qk_norm,
        )
        self.ffn_norm = RMSNorm(d_model, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Self-attention with pre-norm
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        attn_output, cache = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        # FFN with pre-norm
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)

        return hidden_states, cache
