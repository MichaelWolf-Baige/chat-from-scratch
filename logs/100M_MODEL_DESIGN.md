# 100M 模型架构升级方案

> 从 14M 升级到 ~105M，一次性加入 GQA、优化 depth/width 配比

---

## 参数预算核算

目标：~100M，使用 Llama-style architecture（RoPE + RMSNorm + SwiGLU + GQA）

### 配置确定

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 8192 | 保持现有 tokenizer |
| d_model | 512 | 适中，确保 d_head >= 64 |
| n_layers | 14 | 深度足够用于多轮对话上下文跟踪 |
| n_heads | 8 | d_head = 512/8 = 64 |
| n_kv_heads | 4 | **GQA: 2:1 比例**，每2个Q头共享1组KV |
| d_ff | 1408 | 256×5.5 ≈ 8/3×512 ≈ 1365，取 256 的倍数 = 1408 |
| max_seq_len | 2048 | 支持长对话 |
| tie_word_embeddings | True | 节省参数 |
| rope_theta | 100000.0 | 更大 theta 支持更长上下文 |

### 参数核算

```
Embedding:  8192 × 512 = 4,194,304
Per layer:
  Attention Q: 512 × 512 = 262,144          (8 heads × 64 dim)
  Attention K: 512 × (4×64) = 512 × 256 = 131,072   (GQA: only 4 KV heads!)
  Attention V: 512 × (4×64) = 131,072
  Attention O: 512 × 512 = 262,144
  FFN gate:    512 × 1408 = 720,896
  FFN up:      512 × 1408 = 720,896
  FFN down:    1408 × 512 = 720,896
  RMSNorm ×2:  2 × 512 = 1,024
  Per-layer total: 2,950,144

14 layers: 14 × 2,950,144 = 41,302,016
Final RMSNorm: 512
LM Head: 0 (shared with embedding)

TOTAL: 4,194,304 + 41,302,016 + 512 = 45,496,832
```

45M — 偏低。微调到 20 层：

```
14→20 layers: 20 × 2,950,144 = 59,002,880
TOTAL: 4,194,304 + 59,002,880 + 512 = 63,197,696
```

63M — 还是偏低。增大 d_ff 到 1792（256×7）：

```
20 layers × 3,707,904 per layer = 74,158,080
TOTAL: 4,194,304 + 74,158,080 = 78,352,384
```

~78M。调到 24 层 + d_ff=1792：

```
24 layers: 24 × 3,707,904 = 88,989,696
TOTAL: 4,194,304 + 88,989,696 = 93,183,488
```

~93M。再微调 d_ff 到 2048：

```
Per layer: Q(262K) + K(131K) + V(131K) + O(262K) + gate(1,048K) + up(1,048K) + down(1,048K) + norm(1K)
         = 786,432 (attn) + 3,145,728 (ffn) + 1,024 (norm)
         = 3,933,184

24 layers: 24 × 3,933,184 = 94,396,416
TOTAL: 4,194,304 + 94,396,416 = 98,590,720
```

~98.6M ✅

### 最终配置

| 参数 | 值 |
|------|-----|
| vocab_size | 8192 |
| d_model | 512 |
| n_layers | 24 |
| n_heads | 8 |
| **n_kv_heads** | **4 (GQA, 2:1)** |
| d_head | 64 |
| d_ff | 2048 |
| max_seq_len | 2048 |
| **rope_theta** | **100000.0** |
| **tie_word_embeddings** | **True** |
| **attention_bias** | **False** |
| **activation** | **silu (SwiGLU)** |
| **norm** | **RMSNorm (pre-norm)** |
| total_params | **~98.6M** |

### 与 14M 对比

| 指标 | 14M | 100M | 提升 |
|------|-----|------|------|
| d_model | 384 | 512 | 1.3× |
| n_layers | 6 | 24 | **4×** |
| n_heads | 6 | 8 | 1.3× |
| d_ff | 1024 | 2048 | 2× |
| GQA | ❌ MHA | ✅ 2:1 | 推理KV节省50% |
| depth/width ratio | 6/384 = 0.016 | 24/512 = **0.047** | 3×更深 |
| 上下文长度 | 512 | 2048 | 4× |
| Embedding占比 | 3.1M/14M=22% | 4.2M/99M=**4.2%** | 参数更集中在 Transformer 层 |

**关键改进**：
- 深度 4 倍增长（6→24 层）——核心驱动力，多层抽象对上下文跟踪至关重要
- GQA 减少推理 KV cache 50%——部署友好
- Embedding 占比从 22% 降到 4.2%——参数利用效率大幅提升
- rope_theta 增大 10 倍——支持更长上下文外推

---

## 实施步骤

### 1. 修改 ModelConfig（5分钟）

在 `src/model/config.py` 添加 `phase4_100m()`：

```python
@classmethod
def phase4_100m(cls):
    return cls(
        vocab_size=8192, d_model=512, n_layers=24,
        n_heads=8, n_kv_heads=4,  # GQA 2:1
        d_ff=2048, max_seq_len=2048,
        rope_theta=100000.0, dropout=0.0,
        use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, initializer_range=0.02,
    )
```

### 2. 数据准备（重点）

基于 14M 阶段的教训，100M 预训练数据策略：

| 数据源 | 比例 | 作用 |
|--------|------|------|
| 模板引擎（扩充版） | 40% | 高重复度核心语言模式 |
| 中文维基采样 | 30% | 真实文本分布 |
| 合成推理/知识文本 | 20% | 结构化知识注入 |
| 代码/英文 | 10% | 多语言能力 |

目标：500M-1B tokens，覆盖 4000+ unique token types

### 3. 训练配置

```
GPU: 4x RTX 3090 DDP
Global batch: 128 × 2048
LR: 5e-4 WSD (warmup 10%, stable 75%, decay 15%)
Epochs: 3-5 (取决于数据量)
Estimated time: 10-20 hours for 1B tokens
```

### 4. 对齐阶段

```
SFT: 100M基座 + 6K-10K 对话数据（含多轮）
DPO + LoRA: 实验性（有偏好数据后再做）
```

---

## 可行性确认

- [x] 当前 Attention 类已支持 GQA（n_kv_heads < n_heads 自动分组）
- [x] 当前 RoPE、SwiGLU、RMSNorm 已实现
- [x] 现有 DDP 训练脚本可直接复用
- [x] 99M 模型 fp32 ~400MB，单卡 3090 轻松训练
- [ ] 需确认：服务器磁盘空间（99M checkpoint ~800MB）
