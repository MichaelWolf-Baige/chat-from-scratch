"""Training stability monitoring.

Small Transformers (< 100M params) are more prone to:
    - Attention collapse (entropy → 0)
    - Rank collapse (hidden representations lose diversity)
    - Gradient imbalance (early/late layers diverge)

This module computes diagnostic metrics during training.
Use at low frequency (every 100-500 steps) to avoid overhead.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class StabilityMonitor:
    """Collect stability metrics during training.

    Usage:
        monitor = StabilityMonitor(model)
        metrics = monitor.compute(batch)
        # log metrics to wandb/tensorboard
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._attention_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._attention_maps: dict[int, torch.Tensor] = {}  # layer_idx → (B, H, S, S)

    def compute(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, float]:
        """Run a forward pass and collect stability metrics.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional mask.

        Returns:
            dict of metric_name → value.
        """
        self.model.eval()
        metrics: dict[str, Any] = {}

        with torch.no_grad():
            # Forward pass with hooks to capture per-layer outputs
            layer_outputs: dict[int, torch.Tensor] = {}

            def _hook(layer_idx: int):
                def _fn(_module, _input, output):
                    # output from TransformerBlock is (hidden_states, cache)
                    hidden = output[0] if isinstance(output, tuple) else output
                    layer_outputs[layer_idx] = hidden.detach()
                return _fn

            hooks = []
            for i, layer in enumerate(self.model.layers):
                hooks.append(layer.register_forward_hook(_hook(i)))

            logits, _ = self.model(input_ids, attention_mask=attention_mask)

            for hook in hooks:
                hook.remove()

            # 1. Per-layer hidden representation rank (approximate)
            for i, hidden in layer_outputs.items():
                # Compute singular values via covariance matrix (approximate rank)
                # [B, S, D] → [B*S, D] → covariance → eigenvalues
                flat = hidden.reshape(-1, hidden.shape[-1]).float()
                # Use a subsample to save compute
                if flat.shape[0] > 512:
                    indices = torch.randperm(flat.shape[0], device=flat.device)[:512]
                    flat = flat[indices]

                # Covariance singular values (via SVD on centered data)
                centered = flat - flat.mean(dim=0, keepdim=True)
                _, S, _ = torch.linalg.svd(centered, full_matrices=False)
                # Effective rank = exp(entropy of normalized singular values)
                s_normalized = S / (S.sum() + 1e-10)
                entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum()
                effective_rank = torch.exp(entropy).item()
                metrics[f"stability/layer_{i}_rank"] = effective_rank

                # 2. Hidden representation norm
                metrics[f"stability/layer_{i}_hidden_norm"] = hidden.float().norm(dim=-1).mean().item()

            # 3. Output embedding norm (warning sign before loss spike)
            output_norm = logits.float().norm(dim=-1).mean().item()
            metrics["stability/output_embedding_norm"] = output_norm

            # 4. Per-layer gradient norm ratio (approximate from current step)
            # Requires backward — skip in this no_grad pass, compute separately

        return metrics

    def compute_gradient_metrics(
        self,
    ) -> dict[str, float]:
        """Compute per-layer gradient norm ratios (call after backward).

        Warning: only call after optimizer step when gradients exist.
        """
        metrics: dict[str, Any] = {}

        for i, layer in enumerate(self.model.layers):
            total_norm = 0.0
            num_params = 0
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
                    num_params += 1
            if num_params > 0:
                metrics[f"stability/layer_{i}_grad_norm"] = total_norm ** 0.5

        return metrics

    def check_attention_collapse(
        self, attention_weights: torch.Tensor, layer_idx: int, threshold: float = 0.01
    ) -> dict[str, float]:
        """Check if attention has collapsed (too peaked).

        Args:
            attention_weights: [B, H, S, S] or [B, H, S] (with source dim)
            layer_idx: Which layer.
            threshold: Entropy below this is considered collapsed.

        Returns:
            dict with 'attention_entropy_l{layer_idx}' and 'attention_collapsed_l{layer_idx}'.
        """
        # Compute per-head entropy
        # attention_weights: (B, H, S_src, S_tgt) — average over query positions
        if attention_weights.dim() == 4:
            avg_attn = attention_weights.mean(dim=2)  # (B, H, S_tgt)
        else:
            avg_attn = attention_weights

        # Entropy of attention distribution
        eps = 1e-10
        entropy = -(avg_attn * torch.log(avg_attn + eps)).sum(dim=-1).mean().item()

        return {
            f"stability/attn_entropy_l{layer_idx}": entropy,
            f"stability/attn_collapsed_l{layer_idx}": 1.0 if entropy < threshold else 0.0,
        }
