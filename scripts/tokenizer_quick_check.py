"""Quick tokenizer metrics without network dependencies."""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizers import Tokenizer

ours = Tokenizer.from_file("tokenizers/phase1_8k_real/tokenizer.json")
v = ours.get_vocab_size()
print(f"Vocab size: {v}")

# Core compression test
tests = {
    "greeting": "你好，今天天气真好！",
    "tech_cn": "机器学习是人工智能的一个重要分支，深度学习技术推动了这一领域的快速发展。",
    "code": "def hello_world():\n    print('Hello')\n    return 42",
    "mixed": "Python 3.12 发布于 2024 年，带来了新的语法改进。",
    "long_zh": "自然语言处理是计算机科学和人工智能领域的交叉学科，研究如何让计算机理解人类语言。",
    "dialogue": "用户：请问Python怎么安装？\n助手：你可以从官网下载安装包。",
}
total_chars = 0
total_tokens = 0
for name, text in tests.items():
    ids = ours.encode(text).ids
    ratio = len(text) / len(ids) if ids else 0
    total_chars += len(text)
    total_tokens += len(ids)
    print(f"  {name:<12}: {len(text)} chars -> {len(ids)} tokens = {ratio:.2f} c/t")

avg = total_chars / total_tokens if total_tokens else 0
print(f"\n  TOTAL: {total_chars} chars -> {total_tokens} tokens = {avg:.2f} c/t")

# HF Chinese chars coverage
HF = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"
covered = sum(1 for ch in HF if len(ours.encode(ch).ids) == 1)
print(f"\nHF Chinese chars (top 500): {covered}/500 single-token = {covered/500:.1%}")

# Rare char test
rare = "觊觎貔貅饕餮"
ids = ours.encode(rare).ids
unk_count = ids.count(3)
print(f"Rare chars [{rare}]: {len(ids)} tokens, UNK rate={unk_count/len(ids):.0%}")

# Verdict
print()
if avg > 1.2:
    print(f"Compression {avg:.2f} c/t -- EXCELLENT (>1.2)")
elif avg > 0.8:
    print(f"Compression {avg:.2f} c/t -- OK (0.8-1.2)")
else:
    print(f"Compression {avg:.2f} c/t -- POOR (<0.8)")

if covered / 500 > 0.8:
    print(f"HF char coverage {covered/500:.0%} -- GOOD (>80%)")
elif covered / 500 > 0.6:
    print(f"HF char coverage {covered/500:.0%} -- ACCEPTABLE (60-80%)")
else:
    print(f"HF char coverage {covered/500:.0%} -- LOW (<60%), consider larger vocab")
