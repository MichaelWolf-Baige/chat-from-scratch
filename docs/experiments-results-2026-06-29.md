# Architecture Ablation Results — 2026-06-29

## Experiment Setup

- **Data**: pure_wiki/wiki_100M.jsonl (100M tokens of Chinese Wikipedia)
- **Epochs**: 2
- **Batch size**: 4 × 1024
- **Learning rate**: 5e-4 (cosine schedule, 10% warmup, 85% decay start)
- **Sequence length**: 1024
- **Vocabulary**: 8192 (phase1_8k_real tokenizer)
- **Architecture**: Llama-style (SwiGLU, RoPE, RMSNorm, Pre-LN, GQA)
- **Hardware**: 9 × RTX 3090 (24GB), single GPU per experiment
- **Training script**: scripts/train_single.py (with `--save_every 500`)

## Results: Architecture Ablation (wiki_100M × 2 epochs)

| # | Experiment | d_model | n_layers | Params | VAL PPL | vs Baseline |
|---|-----------|---------|----------|--------|---------|-------------|
| 1 | **Extreme Wide** | 896 | 12 | 119M | **18.75** | -1.3% |
| 2 | Shallow-Wide | 768 | 16 | 116M | 19.09 | +0.5% |
| 3 | Extreme Deep | 512 | 36 | 117M | 19.05 | +0.3% |
| 4 | Deep-Thin | 576 | 30 | 111M | 19.48 | +2.6% |
| 5 | **Mid** | 768 | 28 | 188M | **18.13** | -4.6% |
| 6 | C200 (200M Wiki) | 512 | 24 | 99M | ⏳ 93% | 2.76 best_val |

## Results: Data Scaling (100M baseline)

| Experiment | Data | Tokens | VAL PPL | PPL Drop |
|-----------|------|--------|---------|---------|
| C50 (old) | wiki_50M | 50M | 23.23 | — |
| C100 (old) | wiki_100M | 100M | 18.93 | -18.5% |
| C200 (new) | wiki_200M | 200M | ⏳ | — |

## Key Findings

### 1. Width > Depth on Pure Wiki (reproduced)
Extreme Wide (PPL=18.75) beats Deep-Thin (PPL=19.48) by **3.7%** at ~115M params.

### 2. Extreme Deep ≈ Shallow-Wide: boundary found
At this scale, the width/depth tradeoff is balanced around d=512-768.

### 3. Larger model wins (expected but modest)
Mid (188M, PPL=18.13) beats Extreme Wide by 3.3% for +58% params.

### 4. Deep-Thin is worst direction
Going deeper without widening (d=576 L=30) is **worse than original baseline**.

## Critical Bug Fix: rotate_ckpts Lexicographic Sort

**Symptom**: All experiments silently died around step 10000 in previous run.

**Root cause**: `sorted(glob.glob(pattern))` uses string sort. `step10000.pt` 
sorts before `step8500.pt` → just-saved checkpoint immediately deleted by rotate.

**Fix**: `sorted(glob.glob(pattern), key=os.path.getmtime)`

### Other code improvements
- `copy.deepcopy(opt.state_dict())` — prevent opt.state_dict() tensor sharing bug
- `--save_every N` — intermediate checkpoint every N steps (default 0 = off)
- `--keep_ckpts N` — keep last N intermediate checkpoints
- NaN detection: skip bad step (zero_grad + continue)

## C200 Status (incomplete)

- Current best_val: 2.76 at step 85500
- If training trend holds, expected final VAL PPL ≈ **17-18**
- Need to resume from best checkpoint after server update
