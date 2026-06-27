#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Full Data Pipeline — Build improved pretraining dataset end-to-end
# ═══════════════════════════════════════════════════════════════════════════
#
# This script runs the complete improved data pipeline:
#   Phase 0: Build validation set (if not exists)
#   Phase 1: Generate template data (150K texts)
#   Phase 2: Download real data (50K texts)
#   Phase 3: Build mixed datasets (5 ablation ratios)
#   Phase 4: Generate SFT data (8000 conversations)
#
# After running this, use scripts/train_ablation.py on each mixed dataset
# to find the optimal template/real ratio.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Requirements:
#   - Python 3.10+
#   - pip install datasets tokenizers torch
#   - ~2 GB disk space
# ═══════════════════════════════════════════════════════════════════════════

set -e
TIMEFORMAT="Elapsed: %R sec"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "======================================================================"
echo "  MiniChat Improved Data Pipeline"
echo "  Started at: $(date)"
echo "======================================================================"

# ── Phase 0: Validation Set ─────────────────────────────────────────────
echo ""
echo "=== Phase 0: Validation Set ==="
if [ -f "data/val/val_set.jsonl" ]; then
    echo "  Validation set already exists. Skipping."
    wc -l data/val/val_set.jsonl
else
    echo "  Building validation set (1000 texts)..."
    python scripts/build_validation_set.py --num_total 1000 --output_dir data/val
    echo "  Done!"
fi

# ── Phase 1: Template Data ──────────────────────────────────────────────
echo ""
echo "=== Phase 1: Template Data ==="
if [ -f "data/pretrain/template_150k.jsonl" ]; then
    echo "  Template data already exists. Skipping."
    wc -l data/pretrain/template_150k.jsonl
else
    echo "  Generating 150K template texts..."
    python scripts/generate_template_data.py \
        --num_texts 150000 \
        --output data/pretrain/template_150k.jsonl \
        --no_dedup
    echo "  Done!"
fi

# ── Phase 2: Real Data ──────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Real Data ==="
WIKI_EXISTS=$(test -f "data/real/wiki_clean.jsonl" && echo "yes" || echo "no")
if [ "$WIKI_EXISTS" = "yes" ]; then
    echo "  Real data already exists. Skipping."
    wc -l data/real/*.jsonl
else
    echo "  Downloading 50K real Chinese texts..."
    python scripts/download_real_data_v2.py \
        --num_texts 50000 \
        --output_dir data/real
    echo "  Done!"
fi

# ── Phase 3: Mixed Datasets ─────────────────────────────────────────────
echo ""
echo "=== Phase 3: Mixed Datasets (Ablation Ratios) ==="
if [ -f "data/mixed/ratio_50_50.jsonl" ]; then
    echo "  Mixed datasets already exist. Skipping."
    wc -l data/mixed/*.jsonl
else
    echo "  Building 5 mixed datasets at different ratios..."
    python scripts/build_mixed_dataset.py \
        --num_total 100000 \
        --template_file data/pretrain/template_150k.jsonl \
        --real_dir data/real \
        --ratios "100/0,90/10,80/20,70/30,50/50"
    echo "  Done!"
fi

# ── Phase 4: SFT Data ───────────────────────────────────────────────────
echo ""
echo "=== Phase 4: SFT Dialog Data ==="
if [ -f "data/sft/sft_dialogs.jsonl" ]; then
    echo "  SFT data already exists. Skipping."
    wc -l data/sft/sft_dialogs.jsonl
else
    echo "  Generating 8000 SFT conversations..."
    python scripts/generate_sft_data.py \
        --num_dialogs 8000 \
        --output data/sft/sft_dialogs.jsonl
    echo "  Done!"
fi

# ── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE"
echo "======================================================================"
echo "  Data files:"
echo ""
echo "  Validation:"
ls -lh data/val/ 2>/dev/null || echo "    (none)"
echo ""
echo "  Pretrain Template:"
ls -lh data/pretrain/ 2>/dev/null || echo "    (none)"
echo ""
echo "  Real Data:"
ls -lh data/real/ 2>/dev/null || echo "    (none)"
echo ""
echo "  Mixed (Ablation):"
ls -lh data/mixed/ 2>/dev/null || echo "    (none)"
echo ""
echo "  SFT:"
ls -lh data/sft/ 2>/dev/null || echo "    (none)"
echo ""
echo "======================================================================"
echo "  NEXT STEPS:"
echo "  1. Run ablation experiments on GPU server:"
echo "     for ratio in 100_0 90_10 80_20 70_30 50_50; do"
echo "       CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \\"
echo "         scripts/train_ablation.py --data_file data/mixed/ratio_\${ratio}.jsonl \\"
echo "         --run_name ablation_\${ratio}"
echo "     done"
echo ""
echo "  2. Compare validation PPLs to find optimal ratio"
echo "  3. Generate full dataset at optimal ratio + SFT + final training"
echo "======================================================================"
echo "  Finished at: $(date)"
