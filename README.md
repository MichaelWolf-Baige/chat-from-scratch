# Chat from Scratch

从零训练一个对话模型——以学习为目的，从第一行代码开始。

## 目标

- 从零实现 Llama-style Decoder-only Transformer
- 分三阶段：14M 管线验证 → 48M 稳定性 → 150M+ 规模化
- 每个阶段遇到并解决真实的工程问题
- 最终产出：一个能进行基础中英对话的模型

## 项目结构

```
chat-from-scratch/
├── src/                    # 核心库代码
│   ├── model/              # Transformer 架构
│   │   ├── config.py       # ModelConfig dataclass
│   │   ├── attention.py    # Multi-head attention
│   │   ├── position.py     # RoPE 位置编码
│   │   ├── layers.py       # RMSNorm, SwiGLU FFN, TransformerBlock
│   │   └── transformer.py  # 完整 Llama-style 模型
│   ├── data/               # 数据处理
│   │   ├── dataset.py      # 预训练数据集
│   │   └── tokenizer_utils.py
│   ├── train/              # 训练基础设施
│   │   ├── trainer.py      # 训练循环
│   │   ├── optimizer.py    # AdamW + scheduler
│   │   └── distributed.py  # DDP 封装
│   ├── eval/               # 评估
│   │   ├── metrics.py      # Perplexity, loss
│   │   └── stability.py    # 训练稳定性监控
│   └── utils/              # 工具
│       ├── logging.py      # TensorBoard/WandB
│       └── checkpoint.py   # 保存/恢复
├── scripts/                # 可执行脚本入口
├── configs/                # YAML 配置文件
├── tests/                  # 单元测试
└── notebooks/              # 分析 notebook
```

## 阶段规划

| 阶段 | 参数量 | 核心目标 |
|------|--------|---------|
| Phase 1 | ~14M | 管线正确性 + 诊断体系 |
| Phase 2 | ~48M | 训练稳定性 + 效率优化 |
| Phase 3 | ~150M | 数据规模化 + 能力边界探测 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 训练 tokenizer
python scripts/train_tokenizer.py

# 预处理数据
python scripts/preprocess_data.py

# 训练模型
python scripts/train.py --config configs/train/phase1.yaml

# 生成文本
python scripts/generate.py --checkpoint checkpoints/phase1_best.pt --prompt "你好"
```

## 环境要求

- Python 3.10+
- PyTorch 2.4+
- CUDA 12.x (推荐)
- 单张 RTX 3090 即可运行 Phase 1-2
