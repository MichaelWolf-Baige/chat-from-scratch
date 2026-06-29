# Architecture Ablation Results — 2026-06-29

## Experiment Setup

- **Data**: pure_wiki/wiki_100M.jsonl (100M tokens of Chinese Wikipedia)
- **Epochs**: 2
- **Batch size**: 4 × 1024
- **Learning rate**: 5e-4 (cosine schedule with warmup)
- **Sequence length**: 1024
- **Vocabulary**: 8192 (phase1_8k_real tokenizer)
- **Base architecture**: Llama-style (SwiGLU, RoPE, RMSNorm, Pre-LN)
- **GPUs**: 9 × RTX 3090 (24GB)
- **Training script**: scripts/train_single.py (with `--save_every 500` for intermediate ckpts)

## Results Summary

| Experiment | d_model | n_layers | Params | VAL PPL | Rank |
|-----------|---------|----------|--------|---------|------|
| Extreme Wide | 896 | 12 | 119M | **18.75** | 1 |
| Mid | 768 | 28 | 188M | **18.13** | 🔥 |
| Extreme Deep | 512 | 36 | 117M | 19.05 | 3 |
| Shallow-Wide | 768 | 16 | 116M | 19.09 | 4 |
| Deep-Thin | 576 | 30 | 111M | 19.48 | 5 |
| C200 (200M Wiki) | 512 | 24 | 99M | ⏳ running | - |

### Historical Baselines for Comparison

| Experiment | Data | VAL PPL | Notes |
|-----------|------|---------|-------|
| C100 (old) | wiki_100M | 19.0 | Original baseline, 100M params |
| C50 (old) | wiki_50M | 23.2 | Half data, same model |
| S1 Deep (old) | distill_23M | 4.3 | 193M params, distilled data |
| S2 Wide (old) | distill_23M | 4.2 | 207M params, distilled data |

## Key Findings

### 1. Width > Depth on Pure Wiki
Extreme Wide (d=896 L=12, PPL=18.75) outperforms Deep-Thin (d=576 L=30, PPL=19.48) by **3.7%** at similar parameter counts (119M vs 111M). This reproduces and extends the earlier finding from distilled data.

### 2. Extreme Deep = Shallow-Wide (boundary found)
Extreme Deep (19.05) and Shallow-Wide (19.09) are statistically tied — suggesting the width/depth tradeoff boundary lies between d=768 and d=512 at ~115M params.

### 3. Larger model wins (expected)
Mid (188M, PPL=18.13) beats all ~115M variants — expected scaling behavior, but the margin is modest (+3.3% over Extreme Wide) for +58% more parameters.

### 4. Deep-Thin is the worst direction
Narrow+deep (d=576 L=30, PPL=19.48) is worse than the original 100M baseline (PPL=19.0) — making models deeper without widening them is harmful for this scale.

## Code Changes

### Critical Bug Fix: rotate_ckpts lexicographic sort

**Bug**: `sorted(glob.glob(pattern))` uses string sort. `step10000.pt` sorts before `step8500.pt`, causing the just-saved checkpoint to be immediately deleted.

**Fix**: `sorted(glob.glob(pattern), key=os.path.getmtime)` — sort by actual modification time instead of filename.

**Impact**: This bug killed all 6 experiments in the previous run at step 10000. Without the fix, intermediate checkpoint saving silently corrupted `last_good_ckpt` reference.

### Other improvements
- `copy.deepcopy(opt.state_dict())` — prevents Adam optimizer state corruption during save
- `--save_every N` — optional intermediate checkpoint saving (default 0 = original behavior)
- `--keep_ckpts N` — rotate old checkpoints, keeping last N

## Files

- `scripts/train_single.py` — Fixed training script with checkpoint protection
- `checkpoints/arch_extreme_wide.pt` — Best architecture variant (1.4GB)
- `checkpoints/arch_mid_d768L28.pt` — Best PPL overall (2.2GB)
