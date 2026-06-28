# 项目当前状态

> 日期: 2026-06-27 | 下次对话从这里继续

---

## 核心成果

### 1. 从零造了自己的预训练数据集（路径A完成 ✅）

用 Qwen2.5-1.5B 作为 Teacher，2000+ seed prompts 蒸馏出 80MB 中文文本。

**产出**：`data/distill_merged.jsonl`（87,395 条，80MB，342 字/条，8 个领域）

**方法论**：和 MiniMind 本质相同——大模型蒸馏。只是 MiniMind 用更大的 Teacher（7B 级别）+ 更多 seed prompts 产出更多数据。

**关键区别**：这份数据完全自主生成，没有用第三方数据集。

### 2. 100M 模型架构验证通过

d=512, L=24, GQA 2:1, QK-Norm, ~99M params。烟测试 5 项全过。

### 3. 真实数据 vs 模板数据 — 结论明确

同样 100M 模型：真实数据 PPL=7 但生成正确 ✅，模板引擎虽然 500 句式但生成全是垃圾 ❌。**PPL 不是好指标。**

### 4. 解决了 DDP 退出崩溃问题

根因：PyTorch DDP `destroy_process_group` 要求所有 rank 同步。我们的代码在 `if rank==0` 后 save，其他 rank 在 barrier 之前就退了。
修复：单卡训练（`train_single.py`），或 `dist.barrier()` + `try/finally`。

---

## ✅ 完整实验记录

参见 **`docs/experiments-log.md`** — 8 次训练实验的完整记录，含命令、数据、PPL、生成测试。

### 实验矩阵总览

| # | 实验 | Wiki 比例 | Wiki 类型 | VAL PPL |
|---|------|----------|-----------|---------|
| A | 蒸馏基线 | 0% | — | **5** |
| B | MiniMind 等量 | 0% | — | 11 |
| C1 | 蒸馏 100%（对照） | 0% | — | **5** |
| C2 | 蒸馏+原始Wiki | 20% | 原始 | 8 |
| C3 | 蒸馏+原始Wiki | 40% | 原始 | 12 |
| RW-20 | 蒸馏+改写Wiki | 20% | 改写 | 6.8 |
| RW-40 | 蒸馏+改写Wiki | 40% | 改写 | 9 |
| RW-50 | 蒸馏+改写Wiki | 50% | 改写 | 10 |

### 核心结论

1. **纯蒸馏数据 PPL=5 最优**——同 token 量下碾压 MiniMind（11）和混合 Wiki 方案
2. **改写 Wiki 始终优于原始 Wiki**——Phi-4 "web rewrites" 路线验证有效
3. **Wiki 比例越高 PPL 越差，无相变**——100M 模型的容量分配阈值可能在 >50%
4. **瓶颈是模型大小**：100M 参数容量不足同时学好两种数据风格
5. **风格一致性 > 数据多样性**（对小模型）——容量有限时，统一风格 > 多样化

---

## 关键脚本

| 文件 | 用途 |
|------|------|
| `scripts/train_single.py` | 单卡训练，不会崩 |
| `scripts/gen_test.py` | 加载 checkpoint，生成测试 |
| `scripts/sample_minimind.py` | 按 token 数等量采样 MiniMind 数据 |
| `scripts/rewrite_wiki.py` | 用 Teacher 改写 Wiki 为对话体（Phi-4 路线）|
| `scripts/distill_pipeline.py` | 蒸馏数据生成（已完成使命）|
| `scripts/smoke_test.py` | 最小 pipeline 验证（500 文本/50 步）|

## 服务器关键路径

| 路径 | 内容 |
|------|------|
| `~/chat-from-scratch/` | 项目根目录 |
| `~/chat-from-scratch/data/distill_merged.jsonl` | 我们的蒸馏数据 (80MB, 87,395条) |
| `~/chat-from-scratch/data/wiki_zh_clean.jsonl` | 中文维基百科 (2.2GB, 1.36M条) |
| `~/chat-from-scratch/data/wiki_rw_all.jsonl` | 改写 Wiki (17,660条, ~25MB) |
| `~/chat-from-scratch/data/minimind_sampled.jsonl` | MiniMind 等量采样 (~45MB, 62,643条) |
| `~/chat-from-scratch/data/mixed/` | 各种比例混合数据集 |
| `~/minimind-master/dataset/pretrain_t2t_mini.jsonl` | MiniMind 原始数据 (1.2GB) |
| `~/chat-from-scratch/tokenizers/phase1_8k_real/tokenizer.json` | Tokenizer (580KB) |
| `~/chat-from-scratch/checkpoints/p3_ours.pt` | 实验 A：蒸馏 100M (PPL=5) |
| `~/chat-from-scratch/checkpoints/p3_minimind.pt` | 实验 B：MiniMind 100M (PPL=11) |
| `~/chat-from-scratch/checkpoints/mix_*.pt` | C1-C3, RW-20~50 checkpoints |

## SSH 连接

```bash
ssh school  # 已配好 KexAlgorithms 兼容
```

---

## GPU 状态（上次检查）

9x RTX 3090，全部空闲。GPU 0 有僵尸进程残留时需要 `nvidia-smi --query-compute-apps=pid | xargs kill -9`。

## 📚 数据方案调研

参见 **`docs/pretrain-data-best-practices.md`** — 2026-06-27 完成，覆盖 17 篇顶会论文和开源项目的全面调研。

**核心结论**：
1. 合成数据黄金比例 20-30%，纯合成有泛化崩塌风险（717 倍差距）
2. Easy→Hard 数据课程一致最优
3. 防幻觉必须在蒸馏阶段做（RAG 蒸馏 + 双遍验证）
4. 数据质量 > 数据量，尤其对小模型

## 下一步

基于 12 次实验的核心结论：**PPL=5 是当前 Teacher（Qwen2.5-1.5B）的天花板。**
瓶颈不是模型大小也不是数据组合，是 Teacher 模型的知识边界。

两个可突破方向：

| 路线 | 做法 | 预期 |
|------|------|------|
| **A. 扩 prompt 蒸馏** | 当前 8 领域→14 领域，几何级扩大 prompt 覆盖面 | 突破 PPL=5 天花板 |
| **B. 升级 Teacher** | 用 Qwen2.5-7B 或开源 7B+ 模型做蒸馏 | 知识准度跃升，事实错误减少 |

## 不做的

- ❌ 不要用多卡 DDP 做简单训练（退出同步问题反复出现）
- ❌ 不要在一个脚本里塞 train + save + generate
- ❌ 不要用模板数据做预训练主力（12 次实验已经证明不行）
- ❌ ModelScope 数据集全部 404，不要再去试了

## 已解决的问题

| # | 问题 | 解法 |
|---|------|------|
| P19 | SSH 10.0 证书不兼容 | KexAlgorithms 旧算法 |
| P21 | PPL=2 但生成全是垃圾 | 罕见 token marginalization → 用真数据 |
| P23 | 真数据突破 | MiniMind pretrain_t2t_mini |
| P25 | DDP 退出 SIGABRT | 单卡训练 / try-finally |
| P28 | 蒸馏数据制作 | Qwen2.5-1.5B + 2000 prompts |

## GitHub

https://github.com/MichaelWolf-Baige/chat-from-scratch — 全部代码已推送
`logs/PHASE1_ISSUE_LOG.md` — 28 个问题的完整记录
