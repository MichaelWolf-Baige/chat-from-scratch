# Chat from Scratch — 完整运行手册

> 从零训练一个 14M 参数的对话模型。环境：RTX 3090 ×4, PyTorch 2.6, Python 3.10

---

## 快速开始（5 分钟，直接聊天）

用已训练好的模型直接对话，不需要重新训练。

### 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MichaelWolf-Baige/chat-from-scratch.git
cd chat-from-scratch

# 2. 安装依赖
pip install torch tokenizers pyyaml numpy tqdm

# 3. 确认模型文件存在
ls saved_models/sft_chat_final.pt
ls saved_models/tokenizers/phase1_8k_real_tokenizer.json
```

### 启动对话

```bash
python scripts/chat_test.py
```

如果需要交互式对话，可以用下面这个脚本：

```python
# scripts/interactive_chat.py — 交互式对话
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

tok = Tokenizer.from_file("saved_models/tokenizers/phase1_8k_real_tokenizer.json")
cfg = ModelConfig(vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
                   d_ff=1024, max_seq_len=512, dropout=0.0, use_flash_attention=True)
model = Transformer(cfg).cuda()
ckpt = torch.load("saved_models/sft_chat_final.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

print("=" * 40)
print("Chat from Scratch — 14M 中文对话助手")
print("输入 'quit' 退出")
print("=" * 40)

while True:
    user_input = input("\n👤 你: ")
    if user_input.lower() in ("quit", "exit", "退出"):
        break
    text = f"用户：{user_input}\n助手："
    ids = [1] + tok.encode(text).ids
    pid = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        full, _ = model.generate(pid, max_new_tokens=80, temperature=0.8,
                                  top_k=35, top_p=0.9, eos_token_id=2)
    result = tok.decode(full[0].tolist(), skip_special_tokens=True)
    response = result.split("助手：")[-1].strip()
    print(f"🤖 助手: {response}")
```

---

## 完整训练流程（从零开始，约 2 小时）

### 阶段概览

```
Step 0: 环境验证 — 10M 模型 + 合成数据跑通全链路           (30分钟)
Step 1: 训练 Tokenizer — 用中文维基训练 8192 BPE            (5分钟)
Step 2: 下载数据 — 中文维基百科 436MB                        (10分钟)
Step 3: TinyStories 基线 — 纯英文验证 Pipeline 无 bug        (5分钟)
Step 4: Chinese TinyStories — 生成 100K 中文脚本 + 预训练   (20分钟)
Step 5: SFT — 对话数据微调                                  (5分钟)
Step 6: 聊天测试 — 加载模型进行多轮对话
```

### Step 0: 环境验证

```bash
# 查看 GPU
nvidia-smi

# 冒烟测试：10M 合成数据跑通全链路
python scripts/generate_synthetic_data.py --num_lines 10000
python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_synthetic --vocab_size 4096
PYTHONPATH=. python scripts/preprocess_data.py --input data/raw/ --output data/tokenized/smoke/ --tokenizer tokenizers/phase1_synthetic/tokenizer.json --seq_len 256 --min_text_len 20 --num_shards 2
python scripts/smoke_train.py

# 确定性验证
python scripts/test_determinism.py
```

### Step 1: 训练 Tokenizer

需要约 500MB 中文文本。先下载数据，然后训练。

```bash
# 下载中文维基百科（需联网）
python scripts/download_real_data.py --output data/raw/ --target_total_mb 500

# 训练 BPE tokenizer (8192 词表)
python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_8k_real --vocab_size 8192 --max_files 0
```

### Step 2: 预处理数据

```bash
PYTHONPATH=. python scripts/preprocess_data.py \
  --input data/raw/ \
  --output data/tokenized/phase1_real/ \
  --tokenizer tokenizers/phase1_8k_real/tokenizer.json \
  --seq_len 2048 --num_shards 10
```

### Step 3: TinyStories 基线验证 (4卡 DDP)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/run_tinystories_ddp.py
```

期望 VAL PPL < 30。如果 > 100，先排查环境问题。

### Step 4: Chinese TinyStories 预训练 (4卡 DDP)

核心步骤——用 100K 条模板生成的中文文本训练 14M 模型。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_chinese_tinystories.py
```

**数据引擎说明**：脚本内置了 200+ 实体词汇池、100+ 模板、8 个领域（故事/对话/问答/新闻/推理/指令/描述/代码注释）。生成约 9M tokens，覆盖 900+ unique token 类型。

模型保存到 `checkpoints/chinese_tinystories/final.pt`。

### Step 5: SFT 对话微调 (4卡 DDP)

加载预训练模型，用 4000+ 条对话数据微调。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/sft_train.py
```

模型保存到 `checkpoints/sft_chat/final.pt`。

### Step 6: 聊天测试

```bash
python scripts/chat_test.py
```

---

## 项目结构

```
chat-from-scratch/
├── src/                        # 核心库
│   ├── model/                  # Transformer (config, attention, RoPE, layers, transformer)
│   ├── data/                   # 数据集、tokenizer 工具
│   ├── train/                  # 训练循环、优化器、分布式
│   ├── eval/                   # 评估 (perplexity, stability)
│   └── utils/                  # 日志、checkpoint
├── scripts/                    # 可执行脚本入口
│   ├── train_chinese_tinystories.py   # 核心预训练
│   ├── sft_train.py                   # SFT 微调
│   ├── chat_test.py                   # 对话测试
│   ├── run_tinystories_ddp.py         # TinyStories 基线
│   ├── train_tokenizer.py             # Tokenizer 训练
│   ├── download_real_data.py          # 数据下载
│   ├── generate_synthetic_data.py     # 冒烟数据生成
│   └── smoke_train.py                 # 冒烟训练
├── configs/                    # YAML 配置
├── saved_models/               # 已训练好的模型 (需从服务器下载)
│   ├── sft_chat_final.pt              # 对话模型 (158MB)
│   ├── chinese_tinystories_final.pt   # 预训练基座 (158MB)
│   └── tokenizers/                    # Tokenizer 文件
├── logs/                       # 操作日志
│   ├── PHASE1_ISSUE_LOG.md           # 18+ 问题完整记录
│   └── phase1_smoke_test.md          # 冒烟测试日志
├── checkpoints/                # 训练中间产物 (本地 .gitignore)
├── RUNBOOK.md                  # 本文件
└── README.md
```

---

## 常见问题

### 1. 脚本报 `ModuleNotFoundError: No module named 'src'`

解决方法：在脚本顶部加了 `sys.path.insert(0, str(Path(__file__).parent.parent))`。如果仍有问题，加 `PYTHONPATH=.`.

### 2. DDP 启动报错

确保用 `torchrun --nproc_per_node=N` 而不是 `python` 启动：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/xxx.py
```

### 3. GPU OOM

降低 `bs` (batch size per GPU) 参数。14M 模型一般不会 OOM (单卡只需 ~5GB)，如果 OOM 可能是其他进程占用。检查：
```bash
nvidia-smi
```

### 4. HuggingFace 下载失败 (DNS)

在脚本或命令行中设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 5. 恢复训练

所有 checkpoint 支持断点续训。使用 `src/utils/checkpoint.py` 中的 `load_checkpoint`。

---

## 模型规格

| 版本 | 参数量 | 架构 | 用途 |
|------|--------|------|------|
| Chinese TinyStories | 13.8M | d=384, L=6, h=6 | 预训练基座 |
| SFT Chat | 13.8M | 同上 | 对话助手 |

```
Config:
    vocab_size=8192, d_model=384, n_layers=6, n_heads=6
    d_ff=1024, max_seq_len=512, RoPE θ=10000
    Gated SiLU FFN, RMSNorm pre-norm
    Shared embedding weights (weight tying)
```

---

## 训练记录摘要

最终策略源自 20+ 次实验：

| 实验 | 数据 | PPL | 结论 |
|------|------|-----|------|
| Wiki 全量 146K 篇 | 中文百科 | **2200** ❌ | Token 分布太稀疏 |
| TinyStories (EN) | 英文童书 | **6** ✅ | Pipeline 验证通过 |
| 方案 E: 精选百科×5 | 1000 篇 | 84 | 有改善，不够 |
| 方案 F: 纯模板 | 5000 条 | 1 | 太窄，背题 |
| 方案 G: 短文本 | 5000 条 | 78 | 有改善，不够 |
| **Chinese TinyStories** | **100K 条模板** | **1** ✅ | **最终方案** |
| SFT | 4338 条对话 | — | 可对话 ✅ |

---

## 关键设计决策

1. **小模型不能直接用真实文本** — 14M 参数的容量不足以学会 8192 词表中所有 token 的分布。需要控制数据多样性，确保每个 token 出现足够多次。

2. **Template 比 Wikipedia 好** — TinyStories 论文验证了小模型在「简单重复」数据上效果最好。我们的中文版证明了同理。

3. **TinyStories 是 Pipeline 健康度的基准** — 在没有已知基线的情况下，先用 TinyStories 验证代码正确性，再切换到中文数据。

4. **WSD LR schedule 替代 cosine decay** — 小模型训练步数少，cosine decay 的「有效学习窗口」太短。WSD (Warmup-Stable-Decay) 给模型 70%+ 的全速学习时间。

5. **四卡 DDP 加速比 ~3.8x** — 单卡 95K tok/s → 四卡 360K tok/s。
