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

## 待做：A/B 对比实验

目标：同 token 量（~13M）对比我们的蒸馏数据 vs MiniMind 数据。

### 实验 A：我们的蒸馏数据 ← 下一步

```bash
# 单卡训练 (~15分钟)
ssh school "source ~/miniconda3/etc/profile.d/conda.sh && conda activate minimind && cd ~/chat-from-scratch && CUDA_VISIBLE_DEVICES=0 python scripts/train_single.py -d data/distill_merged.jsonl -o checkpoints/p3_ours.pt -e 2 --max_docs 50000"

# 生成测试
ssh school "... && CUDA_VISIBLE_DEVICES=0 python scripts/gen_test.py -c checkpoints/p3_ours.pt"
```

### 实验 B：MiniMind 等量采样

需要先写一个采样脚本，从 `pretrain_t2t_mini.jsonl` 中等量抽取 ~50K 条文档（匹配 A 组的 token 量）。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/train_single.py -d data/minimind_sampled.jsonl -o checkpoints/p3_minimind.pt -e 2 --max_docs 50000
CUDA_VISIBLE_DEVICES=1 python scripts/gen_test.py -c checkpoints/p3_minimind.pt
```

### 对比维度

| 实验 | 数据源 | token 量 | 模型 | 对比 |
|------|--------|---------|------|------|
| A | 我们的蒸馏 | ~13M | 100M | 基线 |
| B | MiniMind 等量 | ~13M | 100M | 同龄质量对比 |

---

## 关键脚本

| 文件 | 用途 |
|------|------|
| `scripts/train_single.py` | 单卡训练，不会崩 |
| `scripts/gen_test.py` | 加载 checkpoint，生成测试 |
| `scripts/distill_pipeline.py` | 蒸馏数据生成（已完成使命）|
| `scripts/smoke_test.py` | 最小 pipeline 验证（500 文本/50 步）|

## 服务器关键路径

| 路径 | 内容 |
|------|------|
| `~/chat-from-scratch/` | 项目根目录 |
| `~/chat-from-scratch/data/distill_merged.jsonl` | 我们的蒸馏数据 (80MB) |
| `~/minimind-master/dataset/pretrain_t2t_mini.jsonl` | MiniMind 数据 (1.2GB) |
| `~/chat-from-scratch/tokenizers/phase1_8k_real/tokenizer.json` | Tokenizer (580KB) |
| `~/chat-from-scratch/checkpoints/p2_realdata/step_6000.pt` | Plan B 100M checkpoint (1.2GB) |

## SSH 连接

```bash
ssh school  # 已配好 KexAlgorithms 兼容
```

---

## GPU 状态（上次检查）

9x RTX 3090，全部空闲。GPU 0 有僵尸进程残留时需要 `nvidia-smi --query-compute-apps=pid | xargs kill -9`。

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
