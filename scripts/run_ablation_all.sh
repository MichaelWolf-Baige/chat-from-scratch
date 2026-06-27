#!/bin/bash
set -e
cd /wuzhou/pentafleet/b23113_/chat-from-scratch
source /wuzhou/pentafleet/b23113_/miniconda3/etc/profile.d/conda.sh
conda activate minimind

echo '======================================================================'
echo '  FULL ABLATION EXPERIMENT SUITE'
echo '  5 ratios x 10 epochs x 4 GPUs'
echo "  Started: $(date)"
echo '======================================================================'

RESULTS_DIR="checkpoints/ablation"
mkdir -p "$RESULTS_DIR"

RATIOS=("100_0" "90_10" "80_20" "70_30" "50_50")
DATAFILES=("data/mixed/ratio_100_0.jsonl"
           "data/mixed/ratio_90_10.jsonl"
           "data/mixed/ratio_80_20.jsonl"
           "data/mixed/ratio_70_30.jsonl"
           "data/mixed/ratio_50_50.jsonl")
GPUS="0,1,2,3"
NPROC=4

for i in "${!RATIOS[@]}"; do
    RATIO="${RATIOS[$i]}"
    DATAFILE="${DATAFILES[$i]}"

    echo ''
    echo "=== [${i}/5] Ablation: ratio_${RATIO} ==="
    echo "    Started: $(date)"

    CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc_per_node=$NPROC \
        scripts/train_ablation.py \
        --data_file "$DATAFILE" \
        --run_name "ablation_${RATIO}" \
        --epochs 10 \
        --seq_len 512 \
        --bs 8 \
        --output_dir "$RESULTS_DIR" \
        --val_file data/val/val_set.jsonl \
        --tokenizer_path saved_models/tokenizers/phase1_8k_real_tokenizer.json || true

    echo "    Finished: $(date)"
done

echo ''
echo '======================================================================'
echo '  ALL DONE - Results:'
echo '======================================================================'
for RATIO in "${RATIOS[@]}"; do
    F="$RESULTS_DIR/ablation_${RATIO}_results.json"
    if [ -f "$F" ]; then
        PPL=$(python -c "import json; print(f\"{json.load(open('$F'))['best_val_ppl']:.1f}\")" 2>/dev/null || echo "N/A")
        echo "  ratio_${RATIO}: best VAL PPL = ${PPL}"
    else
        echo "  ratio_${RATIO}: no results yet"
    fi
done
echo "  Finished: $(date)"
