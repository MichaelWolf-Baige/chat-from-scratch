"""
Multi-Head Attention with GQA support and FlashAttention fallback.

Supports:
    - MHA (Phase 1-2): n_heads == n_kv_heads
    - GQA (Phase 3+):  n_heads >  n_kv_heads
    - FlashAttention-2 (optional, auto-fallback to SDPA)
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.position import RotaryEmbedding, apply_rotary_emb


class Attention(nn.Module):
    """Grouped-Query Multi-Head Attention.

    When n_heads == n_kv_heads, this is standard MHA.
    When n_kv_heads < n_heads, KV heads are shared (GQA).

    Args:
        d_model: Model dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (≤ n_heads).
        d_head: Dimension per head (d_model // n_heads).
        max_seq_len: Maximum sequence length for RoPE cache.
        rope_theta: RoPE base frequency.
        dropout: Attention dropout (0.0 for pretraining).
        use_flash_attention: Try torch.nn.functional.scaled_dot_product_attention.
        use_qk_norm: Apply RMSNorm to Q, K before attention (stability).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_head: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        use_flash_attention: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.n_rep = n_heads // n_kv_heads  # KV head repetition factor
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        # Q projection: all heads
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        # K, V projection: only KV heads
        self.k_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        # Output projection
        self.o_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            d_head=d_head, max_seq_len=max_seq_len, theta=rope_theta
        )

        # Optional QK normalization (stability for small models)
        self.q_norm = nn.RMSNorm(d_head, eps=1e-6) if use_qk_norm else None
        self.k_norm = nn.RMSNorm(d_head, eps=1e-6) if use_qk_norm else None

        self._check_flash_available()

    def _check_flash_available(self) -> None:
        """Log availability of fused attention backend."""
        if not self.use_flash_attention:
            return
        try:
            # Dry-run: SDPA backend will be selected at runtime
            pass
        except Exception:
            warnings.warn("FlashAttention not available; falling back to manual.", stacklevel=2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass.

        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: [batch, 1, seq_len, seq_len] causal mask
            position_ids: [batch, seq_len] for RoPE

        Returns:
            output: [batch, seq_len, d_model]
            cache: (key, value) tuple or None (only if use_cache=True)
        """
        B, S, D = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(B, S, self.n_heads, self.d_head)
        k = self.k_proj(hidden_states).view(B, S, self.n_kv_heads, self.d_head)
        v = self.v_proj(hidden_states).view(B, S, self.n_kv_heads, self.d_head)

        # Apply RoPE (only to Q and K)
        cos, sin = self.rotary_emb(k, position_ids)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Optional QK norm
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # Repeat KV heads for GQA
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.d_head)
            k = k.reshape(B, S, self.n_heads, self.d_head)
            v = v.unsqueeze(3).expand(B, S, self.n_kv_heads, self.n_rep, self.d_head)
            v = v.reshape(B, S, self.n_heads, self.d_head)

        cache = (k, v) if use_cache else None

        # Transpose to (B, H, S, D) for attention
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention
        if self.use_flash_attention and attention_mask is None:
            # Use PyTorch's fused SDPA (calls FlashAttention-2 if available)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        elif self.use_flash_attention and attention_mask is not None:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Manual attention (for debugging / no flash attn)
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                scores = scores + attention_mask

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = F.dropout(
                attn_weights, p=self.dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, v)

        # Merge heads: (B, H, S, D) → (B, S, H*D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)

        # Output projection
        output = self.o_proj(attn_output)
        return output, cache
