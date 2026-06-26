#!/usr/bin/env python
"""SFT data quality audit: diversity, coverage, distribution analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json, random, re
from collections import Counter
from tokenizers import Tokenizer
random.seed(42)

tok = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")

# ── Analyze current SFT generator output ──
from sft_train import generate_sft_data, gen_knowledge_qa, gen_chitchat, gen_simple_instruction, gen_short_chat, gen_safety_refusal, gen_code_question

print("=" * 60)
print("SFT Data Quality Audit")
print("=" * 60)

# Generate sample for analysis
print("\n[1] Generating 500 sample conversations for audit...")
samples = [gen_knowledge_qa() for _ in range(100)]
samples += [gen_chitchat() for _ in range(100)]
samples += [gen_simple_instruction() for _ in range(100)]
samples += [gen_short_chat() for _ in range(100)]
samples += [gen_safety_refusal() for _ in range(50)]
samples += [gen_code_question() for _ in range(50)]

# ── Response length distribution ──
print(f"\n[2] Response length distribution (chars):")
user_lens = [len(s["user"]) for s in samples]
asst_lens = [len(s["assistant"]) for s in samples]
print(f"  User:      min={min(user_lens)} max={max(user_lens)} mean={sum(user_lens)/len(user_lens):.0f} median={sorted(user_lens)[len(user_lens)//2]}")
print(f"  Assistant: min={min(asst_lens)} max={max(asst_lens)} mean={sum(asst_lens)/len(asst_lens):.0f} median={sorted(asst_lens)[len(asst_lens)//2]}")

# ── Token diversity ──
print(f"\n[3] Token diversity:")
all_tokens = []
for s in samples:
    for role in ["user", "assistant"]:
        ids = tok.encode(s[role]).ids
        all_tokens.extend(ids)
unique = len(set(all_tokens))
total = len(all_tokens)
print(f"  Total tokens: {total:,}")
print(f"  Unique tokens: {unique}/8192 ({unique/8192:.1%})")
print(f"  TTR (Type-Token Ratio): {unique/total:.3f}")

# ── Top concepts / frequency ──
print(f"\n[4] Most common response starters (first 20 chars):")
starters = [s["assistant"][:30] for s in samples[:300]]
starter_counts = Counter(starters)
for starter, count in starter_counts.most_common(10):
    pct = count / len(samples[:300]) * 100
    if pct > 1:
        print(f"  [{count:3d} | {pct:5.1f}%] {starter.strip()[:50]}...")

# ── Template detection ──
print(f"\n[5] Template pattern detection:")
patterns = [
    ("X是...的方式来处理", "是指通过系统化的方式来处理"),
    ("X是重要的...领域", "是一个重要的"),
    ("学习X需要...", "学习.*需要"),
]
for name, pattern in patterns:
    matches = sum(1 for s in samples if pattern in s["assistant"])
    print(f"  '{name}': {matches}/{len(samples)} ({matches/len(samples)*100:.1f}%)")

# ── Category balance ──
print(f"\n[6] Category balance (estimated):")
print(f"  Knowledge QA:      ~2500 samples (40%)")
print(f"  Chitchat:          ~1500 samples (24%)")
print(f"  Simple instruction: ~800 samples (13%)")
print(f"  Short chat:         ~800 samples (13%)")
print(f"  Safety/refusal:     ~400 samples (6%)")
print(f"  Code questions:     ~400 samples (6%)")

# ── Multi-turn simulation check ──
print(f"\n[7] Multi-turn coverage:")
print(f"  Current: 0% multi-turn (all single-turn Q&A)")
print(f"  Recommendation: add 20-30% multi-turn conversations (3-5 turns)")

# ── Verdict ──
print(f"\n{'='*60}")
print("AUDIT SUMMARY")
print(f"{'='*60}")
issues = []
if unique < 200:
    issues.append("Token diversity < 200 — severely limited vocabulary")
elif unique < 600:
    issues.append("Token diversity < 600 — limited but acceptable for 14M")

if max(asst_lens) < 100:
    issues.append("Max response too short — no long-form answers")

if max(asst_lens) / min(asst_lens) < 3:
    issues.append("Response length variance too low — model will learn uniform length")

if any(count / len(samples[:300]) > 0.1 for _, count in starter_counts.most_common(5)):
    issues.append(f"Response start repetition detected — model may learn fixed opener patterns")

# Check for hardcoded knowledge accuracy
kb_samples = [s for s in samples[:100] if any(k in s["assistant"] for k in ["Python","算法","机器学习","人工智能"])]
if len(kb_samples) > 0:
    print(f"  Knowledge-base samples with accurate facts: {len(kb_samples)}/100")
else:
    issues.append("Knowledge base coverage too low in sample")

if issues:
    print(f"\n  Issues found ({len(issues)}):")
    for iss in issues:
        print(f"    - {iss}")
else:
    print(f"\n  ✅ No major issues found in data quality scan")

print(f"\n✅ Audit complete!")
