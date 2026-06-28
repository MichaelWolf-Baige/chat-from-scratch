# 新对话接手文档

> 项目：chat-from-scratch — 100M 中文预训练模型从零训练
> 最后更新：2026-06-28 20:00
> GitHub：https://github.com/MichaelWolf-Baige/chat-from-scratch

---

## 一、服务器连接

```bash
ssh school  # 别名已配好（10.1.36.65, user: b23113_）
```

SSH 可能断开，但服务器上进程自存活。断开后重连即可。

**环境激活**：
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate minimind
```

**GPU**：9× RTX 3090 (24GB)，0/2/7 占用中，其余空闲。`nvidia-smi` 查看状态。

---

## 二、当前在跑实验（服务器上）

运行时间统计截止 19:50，后续检查 checkpoint 确认是否完成：

| GPU | 实验名 | 架构 | 数据 | 预计 | 确认方式 |
|-----|--------|------|------|------|---------|
| 0 | Deep-Thin | d=576 L=30 111M | Wiki 100M | 超时 | `ls checkpoints/arch_deep_thin.pt` |
| 1 | Shallow-Wide | d=768 L=16 116M | Wiki 100M | 重跑中 | `ls checkpoints/arch_shallow_wide.pt` |
| 2 | Extreme Deep | d=512 L=36 117M | Wiki 100M | 超时 | `ls checkpoints/arch_extreme_deep.pt` |
| 3 | Extreme Wide | d=896 L=12 119M | Wiki 100M | 重跑中 | `ls checkpoints/arch_extreme_wide.pt` |
| 7 | Mid | d=768 L=28 188M | Wiki 100M | 超时 | `ls checkpoints/arch_mid_d768L28.pt` |
| 6 | C200 | 100M基线 | Wiki 200M | ❌崩了 | **需重跑** |
| 4,5 | 空闲 | — | — | — | 随时可用 |

**查实验结果**：
```bash
ssh school "source ~/miniconda3/etc/profile.d/conda.sh && conda activate minimind && python -c \"
import torch
ckpt = torch.load('/wuzhou/pentafleet/b23113_/chat-from-scratch/checkpoints/<文件名>.pt', map_location='cpu', weights_only=False)
print(f'VAL PPL: {ckpt[\"val_ppl\"]} | Steps: {ckpt[\"steps\"]} | Data: {ckpt[\"data\"]}')
\""
```

---

## 三、已完成实验全量数据

| # | 实验 | 架构 | 数据 | 比例 | Epochs | VAL PPL | 阶段 |
|---|------|------|------|------|--------|---------|------|
| A | 蒸馏基线 | 100M | distill 50K | 100% | 2 | **5** | 基线 |
| B | MiniMind对比 | 100M | MiniMind 62.6K | 100% | 2 | **11** | 基线 |
| C1 | 蒸馏对照 | 100M | 13M tok | 100% | 2 | **5** | 混合 |
| C2 | +原始Wiki | 100M | 13M tok | 80/20 | 2 | **8** | 混合 |
| C3 | +原始Wiki | 100M | 13M tok | 60/40 | 2 | **12** | 混合 |
| RW-20 | +改写Wiki | 100M | 13M tok | 80/20 | 2 | **6.8** | 混合 |
| RW-40 | +改写Wiki | 100M | 13M tok | 60/40 | 2 | **9** | 混合 |
| RW-50 | +改写Wiki | 100M | 13M tok | 50/50 | 2 | **10** | 混合 |
| E1 | 蒸馏放大 | 100M | 23M tok | 100% | 2 | **5** | 缩放 |
| E2 | 混合放大 | 100M | 29M tok | 80/20rw | 2 | **5** 🔥 | 缩放 |
| E3 | 蒸馏多轮 | 100M | 13M tok | 100% | **8** | **6** ⚠️ | 缩放 |
| E4 | 混合多轮 | 100M | 13M tok | 80/20rw | **8** | **8** ⚠️ | 缩放 |
| S1 | Deep Narrow | d=512 L=48 193M | 23M tok | 100% | 2 | **4** | 架构 |
| S2 | Wide Shallow | d=1024 L=14 207M | 23M tok | 100% | 2 | **4** | 架构 |
| C50 | 纯Wiki | 100M | 50M tok | 100% | 2 | **23** | 容量 |
| C100 | 纯Wiki | 100M | 100M tok | 100% | 2 | **19** 🔥 | 容量 |

**每个实验生成测试结果**：见 `docs/experiments-log.md` 逐题对比。关键生成亮点：
- C1 蒸馏：AI 精准、常识错误（"北京是四大发明"）
- C100 Wiki："北京是中国的首都"——第一次正确

---

## 四、核心发现

### 1. 数据质量 > 数据量
蒸馏 PPL=5 vs MiniMind PPL=11（同 token 量）

### 2. 改写 Wiki 始终优于原始 Wiki
验证了 Phi-4 "web rewrites" 路线

### 3. PPL=5 不是容量天花板——被数据量缩放证伪
- 纯蒸馏：13M→23M→29M，PPL 不动（蒸馏数据饱和）
- **纯 Wiki：50M→100M，PPL 23→19（-18.5%）**——100M 模型远未到极限
- 200M 数据实验崩了，需重跑

### 4. 架构：蒸馏场景下宽度更划算
- S1 Deep/S2 Wide PPL 持平 (4 vs 4)，Wide 训练快 34%
- MobileLLM "deep is better" 适用于自然文本大量训练，蒸馏数据改变了需求
- PPL 层面宽/深等价，生成质量 Wide 略优

### 5. 2 epochs 最优，多了过拟合
- 8 epochs：PPL 5→6（纯蒸馏）、6.8→8（混合）
- 100M 小模型数据量小时容易过拟合

### 6. 所有模型共享 Teacher 天花板
- Qwen2.5-1.5B 的"北京是四大发明"偏见被所有模型继承
- 突破需要更强 Teacher 或 RAG 蒸馏

---

## 五、数据资产

| 文件 | 路径 | 大小 |
|------|------|------|
| 蒸馏数据 | `~/chat-from-scratch/data/distill_merged.jsonl` | 80MB, 87K条 |
| 中文维基百科 | `~/chat-from-scratch/data/wiki_zh_clean.jsonl` | 2.2GB, 1.36M条 |
| 改写Wiki | `~/chat-from-scratch/data/wiki_rw_all.jsonl` | 25MB, 17K条 |
| MiniMind原文 | `~/minimind-master/dataset/pretrain_t2t_mini.jsonl` | 1.2GB |
| 纯Wiki 50M | `~/chat-from-scratch/data/pure_wiki/wiki_50M.jsonl` | 173MB |
| 纯Wiki 100M | `~/chat-from-scratch/data/pure_wiki/wiki_100M.jsonl` | 347MB |
| 纯Wiki 200M | `~/chat-from-scratch/data/pure_wiki/wiki_200M.jsonl` | 693MB |
| 各种混合数据 | `~/chat-from-scratch/data/mixed/` | ~47MB each |
| Tokenizer | `~/chat-from-scratch/tokenizers/phase1_8k_real/tokenizer.json` | 580KB |

---

## 六、关键脚本

| 脚本 | 用途 | 重要参数 |
|------|------|---------|
| `scripts/train_single.py` ⭐ | 单卡训练 | `-d DATA -o OUT -e 2 --d_model 512 --n_layers 24 --n_heads 8 --n_kv_heads 4 --d_ff 2048 -b 8` |
| `scripts/gen_test.py` | 生成测试 | `-c CHECKPOINT --d_model 512 --n_layers 24 ...` |
| `scripts/rewrite_wiki.py` | Wiki改写(Phi-4路线) | `--n 5000 -o OUT` |
| `scripts/sample_minimind.py` | MiniMind等量采样 | `--target_tokens N -o OUT` |
| `scripts/extract_logs.py` | 提取loss CSV | `python scripts/extract_logs.py` |
| `scripts/plot_loss.py` | 画训练曲线 | `python scripts/plot_loss.py --all` |

### 架构参数对照表

| 模型 | 命令参数 |
|------|---------|
| 100M基线 | `--d_model 512 --n_layers 24 --n_heads 8 --n_kv_heads 4 --d_ff 2048` |
| Deep-Thin 111M | `--d_model 576 --n_layers 30 --n_heads 9 --n_kv_heads 3 --d_ff 1536` |
| Shallow-Wide 116M | `--d_model 768 --n_layers 16 --n_heads 12 --n_kv_heads 4 --d_ff 2304` |
| Extreme Deep 117M | `--d_model 512 --n_layers 36 --n_heads 8 --n_kv_heads 4 --d_ff 1536` |
| Extreme Wide 119M | `--d_model 896 --n_layers 12 --n_heads 14 --n_kv_heads 7 --d_ff 2560` |
| Mid 188M | `--d_model 768 --n_layers 28 --n_heads 12 --n_kv_heads 6 --d_ff 2048` |
| Deep 193M (S1) | `--d_model 512 --n_layers 48 --n_heads 8 --n_kv_heads 4 --d_ff 2048` |
| Wide 207M (S2) | `--d_model 1024 --n_layers 14 --n_heads 16 --n_kv_heads 8 --d_ff 3584` |

---

## 七、常见问题

### SSH 断开导致本地 bash wrapper 报 255
**服务器上的训练不受影响。** 重连后 `nvidia-smi` 和 `ps aux | grep train_single` 检查即可。

### Save 阶段 OOM
原因：`torch.save` 序列化时内存峰值超 24GB。**已修复**（`cpu().clone() + empty_cache()`）。新对话启动训练前确保 `train_single.py` 是最新版。

### 上一个 SSH heredoc 执行报错
SSH heredoc（`ssh school << 'EOF'`）有时不稳定。用 `ssh school "command"` 单行更可靠。

### GPU 有僵尸进程
```bash
ssh school "nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9 2>/dev/null"
```

---

## 八、下一步

### 🔴 立即
1. **检查服务器上 GPU 0/2/7 是否已产出 checkpoint**
   ```bash
   ssh school "ls -lt ~/chat-from-scratch/checkpoints/arch_*.pt"
   ```
2. **重跑 C200**（100M 基线 × 200M Wiki，bs=4）
   ```bash
   CUDA_VISIBLE_DEVICES=4 python scripts/train_single.py -d data/pure_wiki/wiki_200M.jsonl -o checkpoints/cap_wiki_200M.pt -e 2 -b 4 --max_docs 500000
   ```

### 🟡 后续
3. 分析架构消融结果（Deep-Thin vs Shallow-Wide vs Extreme Deep vs Extreme Wide vs Mid）
4. 给 100M 灌更多 Wiki 数据（400M/800M），追踪 PPL 下降曲线
5. 验证蒸馏+Wiki 混合在更大数据量下是否反超纯蒸馏
6. 如果确认 Teacher 天花板：换 Qwen2.5-7B 蒸馏或中间层特征蒸馏

---

## 九、文件导航

| 文档 | 内容 |
|------|------|
| `HANDOFF.md` ⭐ | **本文档——新对话第一读** |
| `STATE.md` | 项目状态摘要 |
| `docs/experiments-log.md` | 15 次实验完整记录（命令/配置/PPL/生成/分析） |
| `docs/pretrain-data-best-practices.md` | 17 篇论文调研报告 |
| `logs/experiments/` | 13 组实验的 raw.log + loss.csv + meta.json |
| `logs/plots/training_curves.png` | 四宫格训练曲线 |
| `debates/` | 辩论结果存档 |
