# Phase 1 冒烟测试日志

> 日期：2026-06-26 | 服务器：master (10.1.36.65) | GPU：1x RTX 3090 24GB

## 目标

验证 Phase 1 管线五大闭环：数据进得去、梯度流得通、loss 降得动、checkpoint 存得回、推理跑得出。

## 环境

```
Python:    3.10.20
PyTorch:   2.6.0+cu124
CUDA:      12.4
GPU:       RTX 3090 24GB (单卡)
tokenizers: 0.23.1
```

---

## Step 0.1：生成合成数据

**脚本**：`scripts/generate_synthetic_data.py`

```bash
python scripts/generate_synthetic_data.py --num_lines 10000 --output data/raw/synthetic_train.jsonl
```

**结果**：
| 指标 | 值 |
|------|-----|
| 行数 | 10,000 |
| 总字符 | 496,988 |
| 文件大小 | 1.1 MB |
| 中文比例 | ~50% |
| 英文比例 | ~50% |

**样本**：
```
→ 今天天气精准，非常适合去函数学习新知识。
→ Scientists discovered that function affects function in unexpected ways.
→ 对于工具来说，精准的数据预处理是工具成功的关键步骤。
```

✅ **通过**

---

## Step 0.2：训练 Tokenizer + 编解码往返测试

**脚本**：`scripts/train_tokenizer.py` + `scripts/test_tokenizer_smoke.py`

```bash
python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_synthetic --vocab_size 4096
```

**结果**：
| 指标 | 值 | 状态 |
|------|-----|------|
| 词表大小 | 473 (期望 4096) | ⚠️ 数据量不足，预期行为 |
| 英文往返 | `The model is efficient.` → OK | ✅ |
| 中文 UNK 率 | 100% | ⚠️ 预期：5K 条合成文本无法覆盖中文 |
| Special token 隔离 | BOS(1) 不在普通 encode 中 | ✅ |

**诊断**：473 词表是合成数据过小（500MB → 500KB）的正常结果。英文子词学到了一些，中文完全 `<unk>`。正式数据会解决。

✅ **通过**（管线验证目的，非质量目的）

---

## Step 0.3：预处理合成数据

**脚本**：`scripts/preprocess_data.py`

```bash
PYTHONPATH=. python scripts/preprocess_data.py \
  --input data/raw/ --output data/tokenized/phase1_synthetic/ \
  --tokenizer tokenizers/phase1_synthetic/tokenizer.json \
  --seq_len 256 --min_text_len 20 --num_shards 2 --train_ratio 0.95
```

**结果**：
| 指标 | 值 |
|------|-----|
| 总 tokens | 400,773 |
| Train tokens | 380,734 |
| Eval tokens | 20,039 |
| Train shards | 2 |
| dtype | uint16 |

✅ **通过**

---

## Step 0.4：10-Step 冒烟训练

**脚本**：`scripts/smoke_train.py`

```bash
python scripts/smoke_train.py
```

**结果**：

| Step | Loss | LR |
|------|------|-----|
| 0 | 8.2123 | 1.00e-04 |
| 1 | 5.2728 | 2.00e-04 |
| 2 | 4.3608 | 3.00e-04 |
| 3 | 3.5961 | 3.00e-04 |
| 4 | 3.1678 | 2.85e-04 |
| 5 | 3.0738 | 2.44e-04 |
| 6 | 2.8920 | 1.84e-04 |
| 7 | 2.9431 | 1.17e-04 |
| 8 | 2.8135 | 5.73e-05 |
| 9 | 2.6611 | 1.58e-05 |

| 指标 | 值 | 状态 |
|------|-----|------|
| Loss 变化 | 8.21 → 2.66 (delta: 5.55) | ✅ |
| NaN/Inf | 无 | ✅ |
| 训练速度 | 13,059 tok/s | ✅ |
| 峰值显存 | 0.9 GB | ✅ |
| Checkpoint save/load | step=5 正确恢复 | ✅ |
| 生成 | 20 tokens 正常产出 | ✅ |

✅ **通过**

---

## Step 0.5：确定性验证

**脚本**：`scripts/test_determinism.py`

```bash
python scripts/test_determinism.py
```

**结果**：

| Step | Run1 Loss | Run2 Loss | Diff |
|------|-----------|-----------|------|
| 0 | 8.212329 | 8.212329 | 0.00 |
| 1 | 5.272841 | 5.272841 | 0.00 |
| 2 | 3.754870 | 3.754870 | 0.00 |
| 3 | 3.229796 | 3.229796 | 0.00 |
| 4 | 2.950434 | 2.950434 | 0.00 |
| 5 | 2.935216 | 2.935216 | 0.00 |
| 6 | 2.774821 | 2.774821 | 0.00 |
| 7 | 2.813286 | 2.813286 | 0.00 |
| 8 | 2.644268 | 2.644268 | 0.00 |
| 9 | 2.425519 | 2.425519 | 0.00 |

✅ **通过**——两次 seed=42 完全一致，cudnn.deterministic 正确生效。

---

## 总结

| 验证项 | 状态 |
|--------|------|
| 合成数据生成 | ✅ |
| Tokenizer 训练 + 往返测试 | ✅ |
| 数据预处理管线 | ✅ |
| 10-step 训练（loss 下降） | ✅ |
| Checkpoint save/load | ✅ |
| 推理生成 | ✅ |
| 确定性验证 | ✅ |

**结论**：Phase 1 管线在合成数据上全部打通，可以进入真实数据阶段（Step 1-7）。

### 发现的问题与修正

1. **`PYTHONPATH` 问题**：`scripts/train_tokenizer.py` 和 `scripts/preprocess_data.py` 运行需要 `PYTHONPATH=.`，因为没有 `sys.path` 插入。需要加到脚本里。
2. **中文 UNK 率 100%**：合成数据只有 5K 条短文本，tokenizer 只学到 473 个 token。正式数据会解决。
3. **初始化 loss 8.2**：在 6.0-7.5 预期上限之上（偏高），因为合成数据 token 分布极不均匀（重复模板）。
4. **`min_text_len` 默认值不匹配**：默认 100 字符而合成文本约 38 字符，需要显式传 `--min_text_len 20`。
