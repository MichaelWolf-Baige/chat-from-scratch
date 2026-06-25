"""
Model configuration with parameter budget verification.

Llama-style decoder-only Transformer.
Parameter formula:
    Embedding:  vocab_size * d_model
    Per layer:
        Attention (Q/K/V/O): 4 * d_model^2        [no bias]
        FFN (SwiGLU gate/up/down): 3 * d_model * d_ff  [no bias]
        RMSNorm (attn + ffn):  2 * d_model
    Final RMSNorm: d_model
    LM Head: shared with Embedding (0 extra params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for a Llama-style Transformer model.

    All sizes are integers; d_model must be divisible by n_heads.
    d_ff should ideally be a multiple of 256 for tensor core alignment.

    Usage:
        # Phase 1: ~13.8M
        config = ModelConfig.phase1()

        # Phase 2: ~48M
        config = ModelConfig.phase2()

        # Phase 3: ~150M
        config = ModelConfig.phase3()

        # Custom
        config = ModelConfig(vocab_size=16384, d_model=512, ...)
    """

    # ── vocabulary ──
    vocab_size: int = 8192
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ── dimensions ──
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6          # GQA: set < n_heads for Phase 3+
    d_ff: int = 1024             # SwiGLU intermediate, 8/3*d_model rounded to 256x

    # ── sequence ──
    max_seq_len: int = 2048

    # ── position encoding ──
    rope_theta: float = 10000.0
    position_embedding_type: Literal["rope", "none"] = "rope"

    # ── regularisation ──
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0          # 0.0 for pretraining, 0.1 for SFT
    tie_word_embeddings: bool = True

    # ── attention ──
    attention_bias: bool = False  # Llama uses no bias in Q/K/V/O
    attention_dropout: float = 0.0
    use_flash_attention: bool = True

    # ── activation ──
    activation: Literal["silu", "gelu"] = "silu"

    # ── stability (toggles for Phase 2 experimentation) ──
    use_qk_norm: bool = False
    use_z_loss: bool = False
    z_loss_coeff: float = 1e-4

    # ── initialisation ──
    initializer_range: float = 0.02

    # ── derived (computed in __post_init__) ──
    d_head: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate constraints and set derived fields."""
        self.d_head = self.d_model // self.n_heads

        # 1. d_model must be divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        # 2. n_heads must be divisible by n_kv_heads (for GQA)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )

        # 3. d_head should be >= 32 for tensor core efficiency
        if self.d_head < 32:
            raise ValueError(f"d_head ({self.d_head}) should be >= 32 for efficiency")

    # ── parameter counting ──────────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per component."""
        V, D = self.vocab_size, self.d_model
        L, H, Dk = self.n_layers, self.n_heads, self.d_head
        n_kv = self.n_kv_heads
        F = self.d_ff
        tie = self.tie_word_embeddings

        embedding = V * D  # token embedding
        lm_head = 0 if tie else V * D

        # Per-layer attention: Q, K, V, O projections
        # Q: D * (H * Dk), K: D * (n_kv * Dk), V: D * (n_kv * Dk), O: (H * Dk) * D
        attn_per_layer = (
            D * H * Dk           # Q (no bias)
            + D * n_kv * Dk      # K
            + D * n_kv * Dk      # V
            + H * Dk * D         # O
        )

        # SwiGLU FFN: gate, up, down
        ffn_per_layer = (
            D * F   # gate
            + D * F  # up
            + F * D  # down
        )

        rmsnorm_per_layer = 2 * D  # attn_norm + ffn_norm
        final_rmsnorm = D

        total_attn = L * attn_per_layer
        total_ffn = L * ffn_per_layer
        total_norm = L * rmsnorm_per_layer + final_rmsnorm

        total = embedding + lm_head + total_attn + total_ffn + total_norm

        return {
            "embedding": embedding,
            "lm_head": lm_head,
            "attention": total_attn,
            "ffn": total_ffn,
            "rmsnorm": total_norm,
            "total": total,
        }

    @property
    def total_params(self) -> int:
        return self.count_parameters()["total"]

    # ── preset builders ─────────────────────────────────────────────────

    @classmethod
    def phase1(cls) -> "ModelConfig":
        """~13.8M params — pipeline verification."""
        return cls(
            vocab_size=8192,
            d_model=384,
            n_layers=6,
            n_heads=6,
            n_kv_heads=6,        # MHA (no GQA)
            d_ff=1024,
            max_seq_len=2048,
            rope_theta=10000.0,
            dropout=0.0,
            use_flash_attention=True,
        )

    @classmethod
    def phase2(cls) -> "ModelConfig":
        """~49M params — stability + efficiency."""
        return cls(
            vocab_size=16384,
            d_model=576,
            n_layers=10,
            n_heads=9,
            n_kv_heads=9,        # MHA
            d_ff=1536,            # 256*6 ≈ 8/3*576
            max_seq_len=2048,
            rope_theta=10000.0,
            dropout=0.0,
            use_flash_attention=True,
        )

    @classmethod
    def phase3(cls) -> "ModelConfig":
        """~150M params — scaling + capability probing."""
        return cls(
            vocab_size=16384,
            d_model=896,
            n_layers=16,
            n_heads=14,
            n_kv_heads=7,        # GQA (2:1)
            d_ff=2432,            # 256*9.5 ≈ 8/3*896
            max_seq_len=4096,
            rope_theta=500000.0,  # larger theta for longer context
            dropout=0.0,
            use_flash_attention=True,
        )
