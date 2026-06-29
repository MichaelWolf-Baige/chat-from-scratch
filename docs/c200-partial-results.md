# C200 Partial Results — 2026-06-29 17:45 CST

**Status**: Running (93% complete, should finish in ~20min)
**Experiment**: 100M baseline model × wiki_200M tokens × 2 epochs

## Current Progress

- Step: 86,100 / 92,772 (93%)
- Best validation loss: **2.7276** (at step 86,000)
- Epoch: 0 (still in first epoch — 92,772 total steps = 2 epochs)
- NaN events: 0

## Loss Trend

Recent steps show loss oscillating around 2.5-3.5 (PPL ~12-33):
- step 85750: loss=3.68 ppl=40
- step 85800: loss=3.42 ppl=30  
- step 85850: loss=3.02 ppl=21
- step 85900: loss=2.81 ppl=17
- step 85950: loss=2.48 ppl=12
- step 86000: loss=3.30 ppl=27
- step 86050: loss=3.18 ppl=24
- step 86100: loss=3.34 ppl=28

## Projected Final VAL PPL

Based on best_val trajectory (3.14→3.05→2.99→2.90→2.76→2.73),
expected final VAL PPL ≈ **17-18** (best_val ≈ 2.7 → real VAL PPL ~16-18 range).

## Resume Instructions

After server update, resume C200 from best checkpoint:
```bash
# TODO: implement --resume flag in train_single.py
CUDA_VISIBLE_DEVICES=4 python scripts/train_single.py \
  --resume checkpoints/cap_wiki_200M_best.pt \
  -d data/pure_wiki/wiki_200M.jsonl \
  -o checkpoints/cap_wiki_200M.pt \
  -e 2 -b 4 --max_docs 500000 --save_every 500
```

Note: `--resume` flag not yet implemented. Will need to load optimizer state from checkpoint to continue training from step 86,000.
