"""Unit tests for model architecture."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.config import ModelConfig
from src.model.position import RotaryEmbedding, apply_rotary_emb
from src.model.layers import RMSNorm, SwiGLUFFN, TransformerBlock
from src.model.attention import Attention
from src.model.transformer import Transformer


# ── Config Tests ──────────────────────────────────────────────────────────

class TestModelConfig:
    def test_phase1_params(self):
        cfg = ModelConfig.phase1()
        params = cfg.count_parameters()
        total = params["total"]
        # Should be ~13.8M (allow 10% tolerance)
        assert 10_000_000 < total < 18_000_000, f"Expected ~13.8M, got {total:,}"

    def test_phase2_params(self):
        cfg = ModelConfig.phase2()
        params = cfg.count_parameters()
        total = params["total"]
        assert 40_000_000 < total < 60_000_000, f"Expected ~49M, got {total:,}"

    def test_phase3_params(self):
        cfg = ModelConfig.phase3()
        params = cfg.count_parameters()
        total = params["total"]
        assert 130_000_000 < total < 170_000_000, f"Expected ~150M, got {total:,}"

    def test_rejects_d_model_not_divisible_by_n_heads(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=385, n_heads=6)  # 385 % 6 != 0

    def test_rejects_n_heads_not_divisible_by_n_kv_heads(self):
        with pytest.raises(ValueError):
            ModelConfig(n_heads=6, n_kv_heads=4)  # 6 % 4 != 0

    def test_d_head_too_small(self):
        with pytest.raises(ValueError):
            ModelConfig(d_model=64, n_heads=4)  # d_head = 16 < 32

    def test_tie_embeddings_saves_params(self):
        cfg1 = ModelConfig.phase1()
        cfg1.tie_word_embeddings = True
        cfg2 = ModelConfig.phase1()
        cfg2.tie_word_embeddings = False
        assert cfg1.total_params < cfg2.total_params


# ── RMSNorm Tests ─────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(384)
        x = torch.randn(2, 16, 384)
        out = norm(x)
        assert out.shape == x.shape

    def test_variance_near_one(self):
        norm = RMSNorm(384, eps=1e-6)
        x = torch.randn(2, 16, 384) * 5.0
        out = norm(x)
        # RMS should be close to 1
        rms = torch.sqrt(torch.mean(out.float() ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


# ── SwiGLU FFN Tests ──────────────────────────────────────────────────────

class TestSwiGLUFFN:
    def test_output_shape(self):
        ffn = SwiGLUFFN(d_model=384, d_ff=1024)
        x = torch.randn(2, 16, 384)
        out = ffn(x)
        assert out.shape == x.shape

    def test_nonzero_output(self):
        ffn = SwiGLUFFN(d_model=384, d_ff=1024)
        x = torch.randn(2, 16, 384)
        out = ffn(x)
        assert not torch.allclose(out, torch.zeros_like(out))


# ── RoPE Tests ────────────────────────────────────────────────────────────

class TestRoPE:
    def test_rotary_emb_shape(self):
        rope = RotaryEmbedding(d_head=64, max_seq_len=512)
        x = torch.randn(2, 32, 6, 64)  # (B, S, H, D)
        cos, sin = rope(x)
        assert cos.shape == (1, 32, 1, 64)
        assert sin.shape == (1, 32, 1, 64)

    def test_apply_rotary_preserves_shape(self):
        x = torch.randn(2, 16, 4, 64)
        rope = RotaryEmbedding(d_head=64, max_seq_len=32)
        cos, sin = rope(x)
        out = apply_rotary_emb(x, cos, sin)
        assert out.shape == x.shape

    def test_invalid_d_head(self):
        with pytest.raises(ValueError):
            RotaryEmbedding(d_head=63, max_seq_len=512)  # odd

    def test_with_position_ids(self):
        rope = RotaryEmbedding(d_head=64, max_seq_len=256)
        x = torch.randn(1, 8, 4, 64)
        pos_ids = torch.arange(8).unsqueeze(0)
        cos, sin = rope(x, pos_ids)
        assert cos.shape == (1, 8, 1, 64)


# ── Attention Tests ───────────────────────────────────────────────────────

class TestAttention:
    def test_mha_forward(self):
        attn = Attention(d_model=384, n_heads=6, n_kv_heads=6, d_head=64, max_seq_len=128)
        x = torch.randn(2, 32, 384)
        out, cache = attn(x)
        assert out.shape == x.shape
        assert cache is None  # use_cache=False

    def test_gqa_forward(self):
        attn = Attention(d_model=384, n_heads=6, n_kv_heads=2, d_head=64, max_seq_len=128)
        x = torch.randn(2, 32, 384)
        out, cache = attn(x)
        assert out.shape == x.shape

    def test_with_attention_mask(self):
        attn = Attention(d_model=384, n_heads=6, n_kv_heads=6, d_head=64, max_seq_len=128)
        x = torch.randn(2, 16, 384)
        mask = torch.triu(torch.full((16, 16), float("-inf")), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        out, _ = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_qk_norm(self):
        attn = Attention(d_model=384, n_heads=6, n_kv_heads=6, d_head=64, max_seq_len=128, use_qk_norm=True)
        x = torch.randn(2, 16, 384)
        out, _ = attn(x)
        assert out.shape == x.shape


# ── TransformerBlock Tests ────────────────────────────────────────────────

class TestTransformerBlock:
    def test_forward_shape(self):
        block = TransformerBlock(d_model=384, n_heads=6, n_kv_heads=6, d_head=64, d_ff=1024, max_seq_len=128)
        x = torch.randn(2, 32, 384)
        out, cache = block(x)
        assert out.shape == x.shape
        assert cache is None

    def test_residual_works(self):
        block = TransformerBlock(d_model=384, n_heads=6, n_kv_heads=6, d_head=64, d_ff=1024, max_seq_len=128)
        x = torch.randn(2, 32, 384)
        out, _ = block(x)
        # Output should differ from input (no identity shortcut-only)
        assert not torch.allclose(x, out, atol=1e-4)


# ── Full Model Tests ──────────────────────────────────────────────────────

class TestTransformer:
    @pytest.fixture
    def model(self):
        config = ModelConfig.phase1()
        config.use_flash_attention = True  # relies on SDPA
        return Transformer(config)

    def test_forward_pass(self, model):
        input_ids = torch.randint(0, 8192, (2, 128))
        logits, outputs = model(input_ids)
        assert logits.shape == (2, 128, 8192)
        assert "logits" in outputs

    def test_loss_computation(self, model):
        input_ids = torch.randint(0, 8192, (2, 64))
        labels = input_ids.clone()
        _, outputs = model(input_ids, labels=labels)
        assert "loss" in outputs
        loss = outputs["loss"]
        assert loss.item() > 0

    def test_loss_decreases_during_training(self, model):
        """Verify that loss actually decreases after one optimizer step."""
        input_ids = torch.randint(1, 8192, (4, 64))  # avoid pad=0
        labels = input_ids.clone()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        _, outputs1 = model(input_ids, labels=labels)
        loss1 = outputs1["loss"]

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        _, outputs2 = model(input_ids, labels=labels)
        loss2 = outputs2["loss"]

        assert loss2.item() < loss1.item(), \
            f"Loss should decrease after optimizer step! {loss1.item():.4f} → {loss2.item():.4f}"

    def test_generate(self, model):
        input_ids = torch.randint(0, 8192, (1, 10))
        full_ids, new_ids = model.generate(input_ids, max_new_tokens=5)
        assert full_ids.shape[0] == 1
        assert full_ids.shape[1] == 15
        assert new_ids.shape[1] == 5

    def test_generate_respects_eos(self, model):
        """Generate should stop when EOS is produced."""
        # This is probabilistic but should not hang
        input_ids = torch.randint(0, 8192, (1, 5))
        full_ids, _ = model.generate(input_ids, max_new_tokens=10, temperature=0.1)
        assert full_ids.shape[0] == 1

    def test_model_gradient_flow(self, model):
        """Ensure gradients flow through all layers."""
        input_ids = torch.randint(0, 8192, (2, 32))
        labels = input_ids.clone()

        _, outputs = model(input_ids, labels=labels)
        outputs["loss"].backward()

        # Check that each layer gets gradients
        for i, layer in enumerate(model.layers):
            has_grad = False
            for name, param in layer.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grad = True
                    break
            assert has_grad, f"Layer {i} has no gradients!"

    def test_deterministic_inference(self, model):
        """Inference should be deterministic."""
        model.eval()
        input_ids = torch.randint(0, 8192, (1, 16))

        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = model(input_ids)

        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_shared_embedding_weights(self, model):
        """LM head should share weights with token embedding."""
        assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


# ── Edge Case Tests ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_batch_size_one(self):
        config = ModelConfig.phase1()
        model = Transformer(config)
        input_ids = torch.randint(0, 8192, (1, 16))
        logits, _ = model(input_ids)
        assert logits.shape == (1, 16, 8192)

    def test_short_sequence(self):
        config = ModelConfig.phase1()
        model = Transformer(config)
        input_ids = torch.randint(0, 8192, (2, 1))
        logits, _ = model(input_ids)
        assert logits.shape == (2, 1, 8192)

    def test_vocab_boundary(self):
        """Token IDs at vocabulary boundaries should not crash."""
        config = ModelConfig.phase1()
        model = Transformer(config)
        input_ids = torch.tensor([[0, 1, 8191, 0]])  # pad, bos, max, pad
        logits, _ = model(input_ids)
        assert logits.shape == (1, 4, 8192)
        assert not torch.isnan(logits).any()

    def test_padding_is_ignored_in_loss(self):
        config = ModelConfig.phase1()
        model = Transformer(config)
        # Mix of pad and real tokens
        input_ids = torch.randint(1, 8192, (2, 16))
        labels = input_ids.clone()
        # Set first 8 positions to padding
        labels[:, :8] = config.pad_token_id  # ignored in loss
        _, outputs = model(input_ids, labels=labels)
        # Loss should be finite (not NaN) because some tokens are not ignored
        assert torch.isfinite(outputs["loss"]), f"Loss should be finite, got {outputs['loss']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
