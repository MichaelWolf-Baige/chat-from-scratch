# 新对话接手文档

> 项目：chat-from-scratch — 100M 中文预训练模型从零训练
> 最后更新：2026-06-29 23:00
> GitHub：https://github.com/MichaelWolf-Baige/chat-from-scratch

---

## 一、服务器连接

```bash
ssh school  # 别名已配好（10.1.36.65, user: b23113_）
```

**环境激活**：
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate minimind
```

**GPU**：9× RTX 3090 (24GB)，全部空闲。

---

## 二、项目状态总览

### 当前阶段

预训练阶段基本完成。**发现**：
- p3_ours (蒸馏数据, PPL=5) 生成质量良好，在对话式和续写式下均有合理输出
- Wiki 训练的模型（PPL=19）生成全是垃圾——格式不匹配导致
- 预训练 → SFT 的转换是下一步关键

### 近期重大发现（2026-06-29 session）

1. **模板崩塌是误判**：test_results.json 的 Mad Libs 是因为用了裸聊天 prompt，模型训练数据是"A:...\n\nB:..."格式。换对话式/续写式 prompt 后正常。
2. **蒸馏数据非常健康**：Simpson 多样性 = 1.0，87K 条几乎每条独特，不存在模板化。
3. **架构消融（Wiki 数据）无效**：所有架构变体 PPL 挤在 18-19，生成全是垃圾。架构实验需要在蒸馏数据上重跑才有效。
4. **纯 Wiki 预训练对对话生成完全无效**：Wiki 100M tokens 无法让 100M 模型学会对话。

---

## 三、已完成实验全量数据

| # | 实验 | 架构 | 数据 | 比例 | Epochs | VAL PPL | 阶段 |
|---|------|------|------|------|--------|---------|------|
| A | 蒸馏基线 | 100M | distill 50K | 100% | 2 | **5** | 基线 |
| B | MiniMind对比 | 100M | MiniMind 62.6K | 100% | 2 | **11** | 基线 |
| C1 | 蒸馏对照 | 100M | 13M tok | 100% | 2 | **5** | 混合 |
| C2 | +原始Wiki | 100M | 13M tok | 80/20 | 2 | **8** | 混合 |
| C3 | +原始Wiki | 100M | 13M tok | 60/40 | 2 | **12** | 混合 |
| RW-20~50 | +改写Wiki | 100M | 13M tok | 80/20~50/50 | 2 | 6.8~10 | 混合 |
| E1-E4 | 缩放/多轮 | 100M | 13M~29M | 混合 | 2~8 | 5~8 | 缩放 |
| S1 | Deep Narrow | d=512 L=48 193M | 23M | 100% | 2 | **4** | 架构 |
| S2 | Wide Shallow | d=1024 L=14 207M | 23M | 100% | 2 | **4** | 架构 |
| C50 | 纯Wiki | 100M | 50M tok | 100% | 2 | **23** | 容量 |
| C100 | 纯Wiki | 100M | 100M tok | 100% | 2 | **19** | 容量 |
| C200 | 纯Wiki | 100M | 200M tok | 100% | 2 | **18.4** | 容量 |

### 架构消融（Wiki 100M，2026-06-29 完成）

| Checkpoint | 架构 | 参数 | Wiki PPL |
|-----------|------|------|---------|
| cap_wiki_100M | d=512 L=24 (基线) | 99M | 18.93 |
| arch_deep_thin | d=576 L=30 | 111M | 19.48 |
| arch_shallow_wide | d=768 L=16 | 116M | 19.09 |
| arch_extreme_deep | d=512 L=36 | 117M | 19.05 |
| arch_extreme_wide | d=896 L=12 | 119M | 18.75 |
| arch_mid_d768L28 | d=768 L=28 | 188M | 18.13 |

> ⚠️ 这些实验用纯 Wiki 数据，所有架构生成都是垃圾。此批实验结果价值有限。

### Benchmark 评估结果（2026-06-29，47 题 × 3 格式 × 3 解码参数）

| Checkpoint | PPL | 续写 | 对话 | 聊天 |
|-----------|-----|------|------|------|
| p3_ours (蒸馏) | 5.4 | ✅ 合理 | ✅ 流畅多轮 | ✅ 能直接聊 |
| p3_minimind | 11.2 | ⚠️ 不稳定 | ⚠️ 有内容 | ⚠️ 有时偏题 |
| cap_wiki_100M | 18.9 | ❌ 胡说 | ❌ 重复prompt | ❌ TV show 幻觉 |
| arch_extreme_wide | 18.7 | ❌ 胡说 | ❌ 重复prompt | ❌ TV show 幻觉 |

核心结论：PPL 不能跨数据分布比较。蒸馏 PPL=5 vs Wiki PPL=19 的实际生成质量差距远超 5→19 这个数字。

---

## 四、数据资产

| 文件 | 路径 | 大小 |
|------|------|------|
| 蒸馏数据 | `~/chat-from-scratch/data/distill_merged.jsonl` | 80MB, 87K条 |
| 中文维基百科 | `~/chat-from-scratch/data/wiki_zh_clean.jsonl` | 2.2GB, 1.36M条 |
| 改写Wiki | `~/chat-from-scratch/data/wiki_rw_all.jsonl` | 25MB, 17K条 |
| MiniMind原文 | `~/minimind-master/dataset/pretrain_t2t_mini.jsonl` | 1.2GB |
| 纯Wiki 50M/100M/200M | `~/chat-from-scratch/data/pure_wiki/` | 173MB~693MB |
| 混合数据 | `~/chat-from-scratch/data/mixed/` | ~47MB each |
| Tokenizer | `~/chat-from-scratch/tokenizers/phase1_8k_real/tokenizer.json` | 580KB |
| Benchmark 数据 | `data/benchmark/{completion,dialogue,chat}.json` | 47 prompts |

---

## 五、关键脚本

| 脚本 | 用途 |
|------|------|
| `scripts/train_single.py` ⭐ | 单卡训练 |
| `scripts/gen_test.py` | 旧版生成测试（8 prompt，已过时） |
| `scripts/eval_benchmark.py` ⭐🆕 | 新版综合 benchmark 评估（47 题，3 格式，4 解码预设） |
| `scripts/analyze_benchmark.py` 🆕 | Benchmark 结果对比分析 |
| `scripts/rewrite_wiki.py` | Wiki改写 (Phi-4路线) |
| `scripts/sample_minimind.py` | MiniMind等量采样 |
| `scripts/extract_logs.py` | 提取loss CSV |
| `scripts/plot_loss.py` | 画训练曲线 |

### 用法示例

```bash
# Benchmark 评估
CUDA_VISIBLE_DEVICES=0 python scripts/eval_benchmark.py \
  -c checkpoints/p3_ours.pt checkpoints/cap_wiki_100M.pt \
  --arch 100M \
  --benchmarks completion dialogue chat \
  --decoding default creative conservative \
  -o logs/benchmarks/my_results.json

# 分析结果
python scripts/analyze_benchmark.py logs/benchmarks/*.json
```

---

## 六、下一步

### 🔴 新方向：CommonCrawl 数据工程 Pipeline

完整方案见 **`docs/cc-pipeline-design.md`**。

#### 阶段 0：中文适配验证（先做，半天）
- 正文提取器对比（Trafilatura vs jusText vs readability-lxml，200-500 中页）
- MinHash n-gram 策略对比（字符 5-gram vs jieba 分词 token n-gram）
- 跨 snapshot 重复率估算

#### 后续阶段
- 阶段 1：数据获取（3-5 个 snapshot）
- 阶段 2：WARC → 文本提取 + 语言识别
- 阶段 3：Gopher 质量过滤（中文适配版）+ 自定义规则 + 截断检测
- 阶段 4：MinHash 去重 + PII
- 阶段 5：后处理、Tokenizer 训练、数据配比

#### 对照实验价值
完成 CC pipeline 后用同一 100M 架构对比：
- 蒸馏数据（p3_ours，已有）vs CC 手工清洗数据
- 直接回答"手工数据工程 vs Teacher 蒸馏"的效果差异

### 🟡 可选的后续

- SFT 实验设计（教模型数学/逻辑/安全等当前短板）
- 在蒸馏数据上重跑架构消融（Wiki 数据的结果作废）

### ❌ 不建议做的

- 继续用 Wiki 做预训练主力数据（已证明对对话无效）
- 用 Teacher 同族模型做 judge（循环论证）
- 不经验证直接套用英文数据清洗结论到中文

---

## 七、文件导航

| 文档 | 内容 |
|------|------|
| `HANDOFF.md` ⭐ | **本文档——新对话第一读** |
| `STATE.md` | 项目状态摘要（部分过时） |
| `docs/experiments-log.md` | 15 次实验完整记录 |
| `docs/pretrain-data-best-practices.md` | 17 篇论文调研报告 |
| `docs/cc-pipeline-design.md` 🆕 | CommonCrawl 数据清洗 Pipeline 完整方案 |
| `data/benchmark/` 🆕 | Benchmark prompt（47 题，3 格式） |
| `logs/experiments/` | 13 组实验 raw.log + loss.csv |
| `logs/benchmarks/` 🆕 | Benchmark 评估结果 |
| `debates/` | 辩论结果存档 |
