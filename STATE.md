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

## ✅ A/B 对比实验 — 完成！

同 token 量（~13.35M）对比我们的蒸馏数据 vs MiniMind 数据。

| 指标 | 实验 A (我们的蒸馏) | 实验 B (MiniMind 等量) |
|------|--------------------|-----------------------|
| 文档数 | 50,000 | 62,643 |
| Token 数 | 13,350,118 | 13,349,945 |
| **VAL PPL** | **5** 🔥 | **11** |
| 最终 Loss | ~1.55 | ~2.24 |
| AI/ML 生成 | ✅ 精准专业 | ⚠️ 啰嗦 |
| 常识生成 | ❌ 事实错误 | ✅ 自然流畅 |
| 训练时间 | 31min | 31min |

### 结论

**我们的蒸馏数据 PPL 碾压 MiniMind（5 vs 11），但生成质量各有胜负：**

- **优势**：AI/学术类话题更精准——蒸馏数据学术风格一致，模型学得更快
- **劣势**：常识/生活类话题差——seed prompts 偏向 AI 领域，覆盖面不够
- **下一步**：扩大蒸馏 prompt 领域覆盖（常识、地理、天气、数学），不是单纯加数据量

### 关键发现

1. 13M tokens 蒸馏数据就能达到 VAL PPL=5，生成质量在 AI 领域超过 MiniMind 等量数据
2. MiniMind 1.2GB 数据多样性更好但"效率"更低——需要更多 token 才能达到同样 PPL
3. **数据质量 > 数据量**：80MB 高质量蒸馏 ≫ 1.2GB 低质数据（在匹配 token 量时）
4. 单卡训练完全稳定，31 分钟，零崩溃——DDP 问题已彻底绕过

---

## 关键脚本

| 文件 | 用途 |
|------|------|
| `scripts/train_single.py` | 单卡训练，不会崩 |
| `scripts/gen_test.py` | 加载 checkpoint，生成测试 |
| `scripts/sample_minimind.py` | 按 token 数等量采样 MiniMind 数据 |
| `scripts/distill_pipeline.py` | 蒸馏数据生成（已完成使命）|
| `scripts/smoke_test.py` | 最小 pipeline 验证（500 文本/50 步）|

## 服务器关键路径

| 路径 | 内容 |
|------|------|
| `~/chat-from-scratch/` | 项目根目录 |
| `~/chat-from-scratch/data/distill_merged.jsonl` | 我们的蒸馏数据 (80MB) |
| `~/chat-from-scratch/data/minimind_sampled.jsonl` | MiniMind 等量采样 (~45MB, 62,643条) |
| `~/minimind-master/dataset/pretrain_t2t_mini.jsonl` | MiniMind 原始数据 (1.2GB) |
| `~/chat-from-scratch/tokenizers/phase1_8k_real/tokenizer.json` | Tokenizer (580KB) |
| `~/chat-from-scratch/checkpoints/p3_ours.pt` | 实验 A：蒸馏数据 100M checkpoint (0.4GB, PPL=5) |
| `~/chat-from-scratch/checkpoints/p3_minimind.pt` | 实验 B：MiniMind 等量 100M checkpoint (0.4GB, PPL=11) |
| `~/chat-from-scratch/checkpoints/p2_realdata/step_6000.pt` | Plan B 100M checkpoint (1.2GB) |

## SSH 连接

```bash
ssh school  # 已配好 KexAlgorithms 兼容
```

---

## GPU 状态（上次检查）

9x RTX 3090，全部空闲。GPU 0 有僵尸进程残留时需要 `nvidia-smi --query-compute-apps=pid | xargs kill -9`。

## 下一步

1. **扩大蒸馏 prompt 领域覆盖**：加常识、地理、天气、数学等 prompt，补上当前短板
2. **全量蒸馏数据训练**：用全部 87,395 条（~23M tokens）训练，看 PPL 和生成是否继续提升
3. **混合训练**：蒸馏数据 + MiniMind 数据混合，取两者的优势
4. **Tokenizer 升级**：当前 8K 词表偏小，后续可升级到 16K+ 提升中文编码效率

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
