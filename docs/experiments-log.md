# 实验日志

> 项目：chat-from-scratch (100M 中文预训练模型)
> 训练脚本：`scripts/train_single.py`（单卡，bs=8, seq_len=1024, 2 epochs, LR=5e-4）
> 测试脚本：`scripts/gen_test.py`
> 硬件：NVIDIA RTX 3090 × 1，每次实验约 31 分钟

---

## 实验总览

| # | 日期 | 实验 | 数据 | 比例 | VAL PPL | 备注 |
|---|------|------|------|------|---------|------|
| A | 06-27 | 蒸馏基线 | distill_merged.jsonl (50K) | 100% | **5** | 首次验证 |
| B | 06-27 | MiniMind 等量 | minimind_sampled.jsonl (62.6K) | 100% | 11 | 对比基线 |
| C1 | 06-27 | 蒸馏 100% | mixed/ratio_100_0.jsonl (50K) | 100% | **5** | 对照复现 |
| C2 | 06-27 | 蒸馏+原始Wiki 80/20 | mixed/ratio_80_20.jsonl | 80/20 | 8 | 原始Wiki |
| C3 | 06-27 | 蒸馏+原始Wiki 60/40 | mixed/ratio_60_40.jsonl | 60/40 | 12 | 原始Wiki |
| RW-20 | 06-27 | 蒸馏+改写Wiki 80/20 | mixed/ratio_80_20_rw.jsonl | 80/20 | 6.8 | 改写Wiki |
| RW-40 | 06-27 | 蒸馏+改写Wiki 60/40 | mixed/ratio_40_60_rw.jsonl | 60/40 | 9 | 改写Wiki |
| RW-50 | 06-27 | 蒸馏+改写Wiki 50/50 | mixed/ratio_50_50_rw.jsonl | 50/50 | 10 | 改写Wiki |

---

## 详细记录

### 实验 A：蒸馏数据首次验证

**命令**：
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_single.py \
  -d data/distill_merged.jsonl -o checkpoints/p3_ours.pt -e 2 --max_docs 50000
```

**数据**：50,000 texts → 13,350,118 tokens（从 87,395 条中取前 50K）

**结果**：
| 指标 | 值 |
|------|-----|
| Loss 起点 | 9.13 |
| Loss 终点 | 1.55 |
| VAL PPL | **5** |
| 训练时间 | 31.0 min |
| 吞吐 | 14K tok/s |

**生成测试**：
- 人工智能是 → ✅ "计算机科学的一个分支…模拟人类智能"
- 什么是机器学习？→ ✅ "人工智能的一个分支…从数据中自动学习和改进"
- 北京是中国的 → ❌ "中国文化史中心…第一个王朝…"（事实错误）
- 春天来了 → ⚠️ 通顺但有重复
- 1+1等于几 → ❌ "位数组中的元素"（完全跑偏）

**评价**：AI/学术类精准，常识类差。

---

### 实验 B：MiniMind 等量数据对比

**采样脚本**：`scripts/sample_minimind.py`
```bash
python scripts/sample_minimind.py --target_tokens 13350000 -o data/minimind_sampled.jsonl
```

**数据**：62,643 texts → 13,349,945 tokens（从 1.2GB MiniMind 数据中等 token 量采样）
**注意**：MiniMind 单条更短（~213 tokens/条 vs distill ~267 tokens/条），所以需要更多条数

**训练**：
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_single.py \
  -d data/minimind_sampled.jsonl -o checkpoints/p3_minimind.pt -e 2 --max_docs 65000
```

**结果**：
| 指标 | 值 |
|------|-----|
| Loss 起点 | 9.14 |
| Loss 终点 | 2.24 |
| VAL PPL | **11** |
| 训练时间 | 31.1 min |

**生成测试**：
- 人工智能是 → ⚠️ "什么?人工智能是什么?…模拟人类思考"
- 北京是中国的 → ✅ "首都吗?是，北京是中国的首都"
- 今天天气 → ✅ "晴朗，阳光明媚，气温适宜"
- 1+1等于几 → ❌ "1+2*3等于2"

**评价**：常识类比蒸馏数据好，AI 类不如蒸馏数据精准。PPL 更高但某些生成更自然。
**结论**：蒸馏数据"效率高"但覆盖面偏科；MiniMind 数据"多样化"但单条质量更低。

---

### 实验 C1：蒸馏 100%（对照复现）

**数据**：mixed/ratio_100_0.jsonl，49,987 docs → 13,350,010 tokens

**训练**：
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_single.py \
  -d data/mixed/ratio_100_0.jsonl -o checkpoints/mix_100_0.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **5**（和实验 A 一致，复现成功）

---

### 实验 C2：蒸馏 80% + 原始 Wiki 20%

**数据来源**：
- Wiki 原始：`data/wiki_zh_clean.jsonl`（1,361,809 条，从 fjcanyue/wikipedia-zh-cn 下载）
- 格式转换为 `{"text": "标题\n\n正文"}`

**数据**：mixed/ratio_80_20.jsonl，45,775 docs → 13,354,333 tokens

**训练**：
```bash
CUDA_VISIBLE_DEVICES=2 python scripts/train_single.py \
  -d data/mixed/ratio_80_20.jsonl -o checkpoints/mix_80_20.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **8**

**生成测试**：
- 人工智能是 → ✅ "计算机科学的一个分支…"（好）
- 春天来了 → ❌ "我来了一个新的一天的房间"（风格混乱）
- 北京是中国的 → ❌ "国际贸易中心…"（事实错误）
- 机器学习 → ❌ "计算机视觉是一种AI技术…"（概念搞混）

**评价**：PPL 比纯蒸馏差，生成质量下降。Wiki 的百科体风格和蒸馏的对话体冲突。

---

### 实验 C3：蒸馏 60% + 原始 Wiki 40%

**数据**：mixed/ratio_60_40.jsonl，41,550 docs → 13,352,529 tokens

**训练**：
```bash
CUDA_VISIBLE_DEVICES=3 python scripts/train_single.py \
  -d data/mixed/ratio_60_40.jsonl -o checkpoints/mix_60_40.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **12**

**评价**：PPL 进一步恶化，loss 震荡剧烈。

---

### 实验 RW-20：蒸馏 80% + 改写 Wiki 20%

**改写脚本**：`scripts/rewrite_wiki.py`
- 用 Qwen2.5-1.5B-Instruct 把 Wiki 条目改写为教程/对话体
- 5 种改写 prompt 模板随机选用
- 4 卡并行（GPU 4-7），各 750 条，3.3min/卡

**数据**：mixed/ratio_80_20_rw.jsonl，wiki_rewritten.jsonl (2,352 条) → 混合后 42,356 docs

**训练**：
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_single.py \
  -d data/mixed/ratio_80_20_rw.jsonl -o checkpoints/mix_rw.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **6.8**

**对比 C2 (原始 Wiki 20% → PPL=8)**：
- 改写 Wiki 比原始 Wiki 提升了 1.2 PPL
- 验证了 Phi-4 "web rewrites" 路线：改写后风格一致，模型吸收更好
- 但仍不如纯蒸馏（PPL=5）

---

### 实验 RW-40：蒸馏 60% + 改写 Wiki 40%

**改写数据**：wiki_rw_all.jsonl（17,660 条，约 6.5M tokens）
- 4 卡并行（GPU 4-7），各 5000 条，~20min/卡

**数据**：mixed/ratio_40_60_rw.jsonl，44,540 docs → 13,350,370 tokens

**训练**：
```bash
CUDA_VISIBLE_DEVICES=4 python scripts/train_single.py \
  -d data/mixed/ratio_40_60_rw.jsonl -o checkpoints/mix_40_60_rw.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **9**

---

### 实验 RW-50：蒸馏 50% + 改写 Wiki 50%

**数据**：mixed/ratio_50_50_rw.jsonl，42,671 docs → 13,160,688 tokens

**训练**：
```bash
CUDA_VISIBLE_DEVICES=5 python scripts/train_single.py \
  -d data/mixed/ratio_50_50_rw.jsonl -o checkpoints/mix_50_50_rw.pt -e 2 --max_docs 100000
```

**结果**：VAL PPL = **10**

---

## 📊 数据总览

| Wiki 比例 | 原始 Wiki PPL | 改写 Wiki PPL | 改写提升 |
|-----------|-------------|-------------|---------|
| 0% | **5** | **5** | — |
| 20% | 8 | 6.8 | +1.2 |
| 40% | 12 | 9 | +3.0 |
| 50% | — | 10 | — |

**规律**：
1. Wiki 比例越高，PPL 越差（单调，无相变）
2. 改写 Wiki 在每个比例都优于原始 Wiki
3. 纯蒸馏始终是最优

---

## 🎯 核心发现

### 1. 数据质量 > 数据量
同样 13M tokens，蒸馏数据 PPL=5 vs MiniMind PPL=11——蒸馏胜出 2 倍以上。

### 2. 风格一致性对小模型至关重要
100M 模型容量有限，混入不同风格的数据（百科体 vs 对话体）会混淆学习信号。
改写 Wiki（统一为对话体）始终优于原始 Wiki。但即使改写，仍然不如纯蒸馏。

### 3. 不存在相变
根据 Gu et al. (NeurIPS 2025) 的理论预测，从 20%→40%→50% 都没有观测到知识获取的突变。
可能原因：100M 模型对这个数据源的阈值比例 > 50%。

### 4. 瓶颈是模型大小
100M 参数的容量不足以同时学好两种数据风格。模型被迫做容量分配——
主流数据（蒸馏）抢走了大部分容量，少数数据（Wiki/改写Wiki）的 token 被"浪费"。

### 5. 改写 Wiki 路线被验证有效但有限
Phi-4 的 "web rewrites" 策略在我们 100M 模型上也验证了方向正确性。
但 100M 模型的容量天花板限制了这个路线的上限。

---

## 🛠 工具产出

| 脚本 | 用途 |
|------|------|
| `scripts/sample_minimind.py` | 按 token 数等量采样 MiniMind 数据 |
| `scripts/rewrite_wiki.py` | 用 Teacher 模型改写 Wiki 条目为对话体 |
| `scripts/train_single.py` | 单卡训练（所有实验共用） |
| `scripts/gen_test.py` | 生成测试（所有实验共用） |

### 新增数据文件

| 文件 | 内容 | 大小 |
|------|------|------|
| `data/wiki_zh_clean.jsonl` | 中文维基百科（fjcanyue/wikipedia-zh-cn, 2026-05-01） | 2.2GB |
| `data/wiki_rewritten.jsonl` | 第1批改写 Wiki（2352条） | 3.1MB |
| `data/wiki_rw_all.jsonl` | 全部改写 Wiki（17,660条） | ~25MB |
| `data/minimind_sampled.jsonl` | MiniMind 等量采样（62,643条） | ~45MB |
| `data/mixed/ratio_*.jsonl` | 各种比例的混合数据集 | 各 ~47MB |

---

## 📚 论文引用

| 论文 | 相关发现 |
|------|---------|
| Gu et al. (NeurIPS 2025) "Data Mixing Can Induce Phase Transitions" | 模型容量阈值效应 |
| Kang et al. (EMNLP 2025) "Demystifying Synthetic Data" | 合成数据 30% 黄金比例 |
| Phi-4 Technical Report (Microsoft, 2024) | Web rewrites 策略 |
| Yam & Paek (BabyLM 2024) | 小模型多样性 vs 复杂度权衡 |
| "Register Always Matters" (2025) | 语域匹配对小模型的重要性 |

---

> 最后更新：2026-06-27
> 总计实验：8 次训练 × ~31min = ~4 小时 GPU 时间
> 零崩溃（全部单卡训练）
