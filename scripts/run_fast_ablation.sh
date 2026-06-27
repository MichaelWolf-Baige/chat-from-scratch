#!/bin/bash
# Run all 5 ablation experiments in parallel, each on 1 GPU with bs=40
# Fast mode: bs=40 (5x larger), 5 epochs (not 10), 1 GPU each
# Estimated: ~10 min total vs ~3 hours before

cd /wuzhou/pentafleet/b23113_/chat-from-scratch
source /wuzhou/pentafleet/b23113_/miniconda3/etc/profile.d/conda.sh
conda activate minimind

echo '========================================'
echo '  FAST PARALLEL ABLATION'
echo '  5 experiments x 1 GPU x bs=40 x 5 epochs'
echo "  Started: $(date)"
echo '========================================'

RESULTS_DIR="checkpoints/ablation"
LOGDIR="/tmp/ablation_fast_logs"
rm -rf "$LOGDIR"
mkdir -p "$RESULTS_DIR" "$LOGDIR"

launch() {
    local RATIO=$1
    local GPU=$2
    local PORT=$3
    local LOG="$LOGDIR/ablation_${RATIO}.log"
    echo "  ratio_${RATIO}: GPU=$GPU port=$PORT bs=40"
    CUDA_VISIBLE_DEVICES="$GPU" torchrun --master_port=$PORT --nproc_per_node=1 \
        scripts/train_ablation.py \
        --data_file "data/mixed/ratio_${RATIO}.jsonl" \
        --run_name "ablation_${RATIO}" \
        --epochs 5 --seq_len 512 --bs 40 \
        --output_dir "$RESULTS_DIR" \
        --val_file data/val/val_set.jsonl \
        --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json \
        > "$LOG" 2>&1 &
    echo "    PID=$!"
}

echo ''
launch "100_0" "0" 29500
launch "90_10" "1" 29501
launch "80_20" "2" 29502
launch "70_30" "3" 29503
launch "50_50" "4" 29504

echo ''
echo "All launched. Monitor: tail -f $LOGDIR/ablation_*.log"
wait

echo ''
echo '========================================'
echo '  RESULTS (best VAL PPL)'
echo '========================================'
for RATIO in 100_0 90_10 80_20 70_30 50_50; do
    F="$RESULTS_DIR/ablation_${RATIO}_results.json"
    if [ -f "$F" ]; then
        PPL=$(python -c "import json; d=json.load(open('$F')); print(f\"{d['best_val_ppl']:.1f}\")" 2>/dev/null)
        echo "  ratio_${RATIO}: best VAL PPL = ${PPL}"
    else
        echo "  ratio_${RATIO}: no results"
    fi
done
echo "  Finished: $(date)"
