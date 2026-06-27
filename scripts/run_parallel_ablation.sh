#!/bin/bash
cd /wuzhou/pentafleet/b23113_/chat-from-scratch
source /wuzhou/pentafleet/b23113_/miniconda3/etc/profile.d/conda.sh
conda activate minimind

echo '========================================'
echo '  PARALLEL ABLATION - 5 experiments x 9 GPUs'
echo "  Started: $(date)"
echo '========================================'

RESULTS_DIR="checkpoints/ablation"
mkdir -p "$RESULTS_DIR"
LOGDIR="/tmp/ablation_logs"
mkdir -p "$LOGDIR"

# ── Experiment 0: ratio 100/0, GPUs 0,1, port 29500 ──
echo 'Launching ratio_100_0: GPUs 0,1 port 29500'
CUDA_VISIBLE_DEVICES="0,1" torchrun --master_port=29500 --nproc_per_node=2 \
    scripts/train_ablation.py \
    --data_file data/mixed/ratio_100_0.jsonl \
    --run_name ablation_100_0 \
    --epochs 10 --seq_len 512 --bs 8 \
    --output_dir "$RESULTS_DIR" \
    --val_file data/val/val_set.jsonl \
    --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
    > "$LOGDIR/ablation_100_0.log" 2>&1 &
echo "  PID=$!"

# ── Experiment 1: ratio 90/10, GPUs 2,3, port 29501 ──
echo 'Launching ratio_90_10: GPUs 2,3 port 29501'
CUDA_VISIBLE_DEVICES="2,3" torchrun --master_port=29501 --nproc_per_node=2 \
    scripts/train_ablation.py \
    --data_file data/mixed/ratio_90_10.jsonl \
    --run_name ablation_90_10 \
    --epochs 10 --seq_len 512 --bs 8 \
    --output_dir "$RESULTS_DIR" \
    --val_file data/val/val_set.jsonl \
    --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
    > "$LOGDIR/ablation_90_10.log" 2>&1 &
echo "  PID=$!"

# ── Experiment 2: ratio 80/20, GPUs 4,5, port 29502 ──
echo 'Launching ratio_80_20: GPUs 4,5 port 29502'
CUDA_VISIBLE_DEVICES="4,5" torchrun --master_port=29502 --nproc_per_node=2 \
    scripts/train_ablation.py \
    --data_file data/mixed/ratio_80_20.jsonl \
    --run_name ablation_80_20 \
    --epochs 10 --seq_len 512 --bs 8 \
    --output_dir "$RESULTS_DIR" \
    --val_file data/val/val_set.jsonl \
    --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
    > "$LOGDIR/ablation_80_20.log" 2>&1 &
echo "  PID=$!"

# ── Experiment 3: ratio 70/30, GPUs 6,7, port 29503 ──
echo 'Launching ratio_70_30: GPUs 6,7 port 29503'
CUDA_VISIBLE_DEVICES="6,7" torchrun --master_port=29503 --nproc_per_node=2 \
    scripts/train_ablation.py \
    --data_file data/mixed/ratio_70_30.jsonl \
    --run_name ablation_70_30 \
    --epochs 10 --seq_len 512 --bs 8 \
    --output_dir "$RESULTS_DIR" \
    --val_file data/val/val_set.jsonl \
    --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
    > "$LOGDIR/ablation_70_30.log" 2>&1 &
echo "  PID=$!"

# ── Experiment 4: ratio 50/50, GPU 8, port 29504 ──
echo 'Launching ratio_50_50: GPU 8 port 29504'
CUDA_VISIBLE_DEVICES="8" torchrun --master_port=29504 --nproc_per_node=1 \
    scripts/train_ablation.py \
    --data_file data/mixed/ratio_50_50.jsonl \
    --run_name ablation_50_50 \
    --epochs 10 --seq_len 512 --bs 8 \
    --output_dir "$RESULTS_DIR" \
    --val_file data/val/val_set.jsonl \
    --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
    > "$LOGDIR/ablation_50_50.log" 2>&1 &
echo "  PID=$!"

echo ''
echo "All 5 launched! Monitor: tail -f $LOGDIR/ablation_*.log"
echo ''

wait

echo ''
echo '========================================'
echo '  ALL DONE - Results:'
echo '========================================'
for RATIO in 100_0 90_10 80_20 70_30 50_50; do
    F="$RESULTS_DIR/ablation_${RATIO}_results.json"
    if [ -f "$F" ]; then
        PPL=$(python -c "import json; print(f\"{json.load(open('$F'))['best_val_ppl']:.1f}\")" 2>/dev/null)
        echo "  ratio_${RATIO}: best VAL PPL = ${PPL}"
    else
        echo "  ratio_${RATIO}: no results"
    fi
done
echo "  Finished: $(date)"
