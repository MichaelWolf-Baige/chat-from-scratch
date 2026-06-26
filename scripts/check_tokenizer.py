#!/usr/bin/env python
"""Tokenizer quality audit: compression rate, char coverage, vs Qwen2.

Usage: python scripts/check_tokenizer.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizers import Tokenizer
import json, random, time

random.seed(42)

# ── Load tokenizers ──
ours = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")
print(f"Our tokenizer: vocab={ours.get_vocab_size()}")

try:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from transformers import AutoTokenizer
    qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    print(f"Qwen2-0.5B tokenizer: vocab={qwen_tok.vocab_size}")
    HAS_QWEN = True
except Exception as e:
    print(f"Qwen2 tokenizer unavailable (network issue): {type(e).__name__}")
    HAS_QWEN = False

# ── Test texts ──
test_texts = {
    "greeting": "你好，今天天气真好！",
    "tech_cn": "机器学习是人工智能的一个重要分支，深度学习技术推动了这一领域的快速发展。",
    "code": "def hello_world():\n    print('Hello, world!')\n    return 42",
    "mixed": "Python 3.12 发布于 2024 年 10 月，带来了新的 f-string 语法改进。",
    "long_zh": "自然语言处理（NLP）是计算机科学和人工智能领域的交叉学科。它研究如何让计算机理解、生成和处理人类语言。主要任务包括文本分类、情感分析、机器翻译、问答系统和对话生成等。近年来，基于Transformer的预训练模型如BERT和GPT系列在NLP领域取得了突破性进展。",
    "rare_chars": "觊觎、貔貅、饕餮、龘龘 —— 这些生僻字在日常文本中很少出现。",
    "numbers": "2024年11月15日下午3点30分，温度22.5度，湿度65%，风速3.2m/s。",
    "dialogue": "用户：请问Python怎么安装？\n助手：你可以从python.org下载安装包，或者使用Anaconda发行版。\n用户：哪个更适合初学者？\n助手：Anaconda更适合，它自带了很多常用的科学计算库。",
}

print(f"\n{'='*65}")
print(f"{'Text':<15} {'Chars':<8} {'Our tok':<10} {'Ratio':<10} {'Qwen tok':<10} {'Qwen Ratio':<10}")
print(f"{'='*65}")

total_chars = 0
total_ours = 0
total_qwen = 0

for name, text in test_texts.items():
    chars = len(text)
    our_ids = ours.encode(text).ids
    our_n = len(our_ids)
    our_ratio = chars / our_n if our_n > 0 else 0

    if HAS_QWEN:
        qwen_ids = qwen_tok.encode(text)
        qwen_n = len(qwen_ids)
        qwen_ratio = chars / qwen_n if qwen_n > 0 else 0
    else:
        qwen_n = 0
        qwen_ratio = 0

    total_chars += chars
    total_ours += our_n
    if HAS_QWEN:
        total_qwen += qwen_n

    print(f"{name:<15} {chars:<8} {our_n:<10} {our_ratio:<10.2f} "
          f"{qwen_n:<10} {qwen_ratio:<10.2f}" if HAS_QWEN else
          f"{our_n:<10} {our_ratio:<10.2f}")

print(f"{'='*65}")
print(f"{'TOTAL':<15} {total_chars:<8} {total_ours:<10} {total_chars/total_ours:<10.2f} "
      f"{total_qwen:<10} {total_chars/total_qwen if HAS_QWEN and total_qwen > 0 else 'N/A':<10}")
print()

# ── Verdict ──
avg_ratio = total_chars / total_ours
print(f"Average compression: {avg_ratio:.2f} chars/token")
if avg_ratio > 1.5:
    print("  ✅ EXCELLENT — each token represents >1.5 Chinese chars on average")
elif avg_ratio > 1.0:
    print("  ⚠️  OK — acceptable compression")
elif avg_ratio > 0.7:
    print("  ⚠️  POOR — tokenizer may be a bottleneck for long contexts")
else:
    print("  ❌ BAD — tokenizer is fragmenting text too much")

print()

# ── High-frequency character coverage ──
HF_CHARS = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"

print(f"High-frequency character coverage check:")
print(f"  Testing {len(HF_CHARS)} most common Chinese characters...")
covered = 0
uncovered = []
for ch in HF_CHARS:
    ids = ours.encode(ch).ids
    # If the character itself (as 1 token) is in vocab, it's covered
    decoded = ours.decode(ids)
    if ch in decoded and len(ids) == 1:
        covered += 1
    elif len(ids) > 3:
        uncovered.append(ch)

print(f"  Single-token coverage: {covered}/{len(HF_CHARS)} ({covered/len(HF_CHARS):.1%})")
if uncovered:
    print(f"  Poorly covered chars (>{len(uncovered)} tokens each): {uncovered[:20]}...")
print()

# ── Special token check ──
print("Special token isolation:")
for name, tok_id in [("PAD", 0), ("BOS", 1), ("EOS", 2), ("UNK", 3),
                      ("<|system|>", 4), ("<|user|>", 5), ("<|assistant|>", 6)]:
    exists = tok_id < ours.get_vocab_size()
    print(f"  ID {tok_id} ({name}): {'present' if exists else 'MISSING'}")

print()
print("✅ Tokenizer audit complete!")
