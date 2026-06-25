"""
Full Llama-style Decoder-only Transformer.

Assembles TokenEmbedding → [TransformerBlock × N] → RMSNorm → LM Head.

Usage:
    config = ModelConfig.phase1()
    model = Transformer(config)
    logits, _ = model(input_ids)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.layers import RMSNorm, TransformerBlock


class Transformer(nn.Module):
    """Llama-style decoder-only Transformer.

    Args:
        config: ModelConfig with architecture hyperparameters.

    Forward:
        input_ids: [batch, seq_len] — token indices
        → logits:  [batch, seq_len, vocab_size]
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                d_head=config.d_head,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                rms_norm_eps=config.rms_norm_eps,
                use_flash_attention=config.use_flash_attention,
                use_qk_norm=config.use_qk_norm,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        # LM head (shares weights with embedding when tie_word_embeddings=True)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Z-loss auxiliary head (optional stability)
        self.use_z_loss = config.use_z_loss
        self.z_loss_coeff = config.z_loss_coeff

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following Llama conventions."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, 1, seq_len, seq_len] or None (use causal)
            position_ids: [batch, seq_len] or None (auto-generate)
            labels: [batch, seq_len] for loss computation (shifted internally)
            use_cache: Return KV cache for autoregressive generation.

        Returns:
            logits:  [batch, seq_len, vocab_size]
            outputs: dict with 'loss' (if labels provided), 'z_loss', 'logits'
        """
        B, S = input_ids.shape

        # Embeddings
        hidden_states = self.embed_tokens(input_ids)  # (B, S, D)

        # Build causal mask if none provided
        if attention_mask is None:
            # Create causal mask: (S, S) with -inf in upper triangle
            causal_mask = torch.triu(
                torch.full((S, S), float("-inf"), device=hidden_states.device),
                diagonal=1,
            )
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(
                S, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(B, -1)

        # Pass through transformer blocks
        all_caches = [] if use_cache else None
        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
            )
            if use_cache and cache is not None:
                all_caches.append(cache)

        # Final norm
        hidden_states = self.norm(hidden_states)  # (B, S, D)

        # LM head → logits
        logits = self.lm_head(hidden_states)  # (B, S, V)

        # Compute loss
        outputs: dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Shift for next-token prediction:
            #   logits[:, :-1, :] predicts labels[:, 1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            outputs["loss"] = loss

            # Optional Z-loss for stability (prevents logit drift)
            if self.use_z_loss:
                z_loss = self.z_loss_coeff * torch.mean(logits ** 2)
                outputs["z_loss"] = z_loss
                outputs["loss"] = loss + z_loss

        return logits, outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive generation.

        Args:
            input_ids: [batch, seq_len] prompt tokens.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).
            top_k: Top-K sampling filter.
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation when this token is produced.

        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
            generated_tokens_only: [batch, max_new_tokens]
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.eval()
        B = input_ids.shape[0]
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if generated.shape[1] > self.config.max_seq_len:
                generated = generated[:, -self.config.max_seq_len:]

            logits, _ = self(generated)  # (B, S, V)

            # Take logits of the last position
            next_logits = logits[:, -1, :] / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                for b in range(B):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    next_logits[b, indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop if all sequences hit EOS
            if (next_token == eos_token_id).all():
                break

        new_tokens = generated[:, input_ids.shape[1]:]
        return generated, new_tokens
