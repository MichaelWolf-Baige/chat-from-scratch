#!/bin/bash
# ============================================================
# Chat-from-Scratch 一键复现脚本
# 从零训练 14M 中文对话模型
#
# 硬件要求: 4x RTX 3090 (单卡亦可, 手动改参数)
# 预计时间: ~3小时 (4卡)
# ============================================================
set -e

echo "========================================"
echo " Chat-from-Scratch: 一键复现"
echo "========================================"

# ── 环境检查 ──────────────────────────────
echo "[1/6] 检查环境..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, GPUs {torch.cuda.device_count()}')"
python -c "import tokenizers; print(f'  tokenizers {tokenizers.__version__}')"

GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$GPUS" -lt 1 ]; then
    echo "  ⚠️  没有检测到 GPU，将使用 CPU 训练（很慢）"
    DDP_CMD="python"
    DDP_ARGS=""
else
    echo "  检测到 $GPUS 张 GPU"
    if [ "$GPUS" -ge 4 ]; then
        DDP_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4"
    elif [ "$GPUS" -ge 2 ]; then
        DDP_CMD="torchrun --nproc_per_node=$GPUS"
    else
        DDP_CMD="python"
        DDP_ARGS=""
    fi
fi

# ── 数据准备 ──────────────────────────────
echo "[2/6] 准备数据..."
mkdir -p data/raw data/tokenized
python scripts/generate_synthetic_data.py --num_lines 10000 --output data/raw/synthetic_train.jsonl

# ── Tokenizer ─────────────────────────────
echo "[3/6] 训练 Tokenizer (如果已有则跳过)..."
if [ -f "tokenizers/phase1_8k_real/tokenizer.json" ]; then
    echo "  Tokenizer 已存在，跳过训练"
else
    # 需要真实数据训练 tokenizer。如果 data/raw/ 没有文本，下载中文维基采样
    if [ ! -f "data/raw/wiki_zh.jsonl" ]; then
        echo "  下载中文维基百科数据..."
        HF_ENDPOINT=https://hf-mirror.com python scripts/download_real_data.py --output data/raw/ --target_total_mb 500
    fi
    python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_8k_real --vocab_size 8192 --max_files 0
fi

# ── 冒烟测试 ─────────────────────────────
echo "[4/6] 冒烟测试 (10MB 合成数据跑通全链路)..."
python scripts/train_tokenizer.py --data_dir data/raw/ --output tokenizers/phase1_synthetic --vocab_size 4096
PYTHONPATH=. python scripts/preprocess_data.py \
    --input data/raw/ --output data/tokenized/smoke/ \
    --tokenizer tokenizers/phase1_synthetic/tokenizer.json \
    --seq_len 256 --min_text_len 20 --num_shards 2
python scripts/smoke_train.py
python scripts/test_determinism.py
echo "  ✅ 冒烟测试通过"

# ── 预训练 Chinese TinyStories ─────────────────────
echo "[5/6] 14M 预训练 (Chinese TinyStories)..."
$DDP_CMD scripts/train_chinese_tinystories.py
echo "  ✅ 预训练完成: checkpoints/chinese_tinystories/final.pt"

# ── SFT ChatML ─────────────────────────────
echo "[6/6] SFT 对话微调 (ChatML 格式)..."
$DDP_CMD scripts/sft_train_token.py
echo "  ✅ SFT 完成: checkpoints/sft_v5/final.pt"

echo ""
echo "========================================"
echo " ✅ 全部完成!"
echo ""
echo " 运行对话: python scripts/chat_test.py"
echo "========================================"
