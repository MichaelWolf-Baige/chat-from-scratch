#!/usr/bin/env python
"""Evaluation benchmark: 50 Chinese prompts across 4 dimensions.

Usage:
    python scripts/eval_benchmark.py
Output: eval_results.json — model responses + scoring template
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, json, time
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

# ── 50 prompts across 4 dimensions ──
BENCHMARK = {
    "chitchat": [
        "你好！",
        "今天天气真好",
        "谢谢你帮我",
        "再见，下次聊",
        "最近过得怎么样？",
        "我心情不太好",
        "晚安",
        "你叫什么名字？",
        "我觉得今天很幸运",
        "好久不见，你还好吗？",
    ],
    "knowledge_qa": [
        "什么是机器学习？",
        "Python是什么语言？",
        "中国的首都是哪里？",
        "地球绕太阳转一圈要多久？",
        "水的化学式是什么？",
        "世界上最高的山是哪座？",
        "什么是人工智能？",
        "计算机网络的IP地址是什么意思？",
        "李白是谁？",
        "太阳系有几大行星？",
        "什么是数据库？",
        "物理学是研究什么的？",
        "互联网是如何工作的？",
        "英语里的past tense是什么意思？",
        "人体正常体温是多少？",
    ],
    "instruction": [
        "请把'你好世界'翻译成英文",
        "帮我数一下从1到10",
        "请列出三种常见的水果",
        "今天北京的天气怎么样？（假设你不知道，如实说）",
        "帮我写一封简短的道歉信",
        "请用'春天'造一个句子",
        "告诉我怎么煮鸡蛋",
        "推荐三本适合小学生看的书",
        "帮我计算 15 + 27",
        "解释一下什么叫'节约用水'",
    ],
    "multiturn": [
        # Turn 1
        "我最近想学一门新的编程语言，你有什么建议吗？",
        # Turn 2 (depends on turn 1)
        "那你说的这个语言有什么优点呢？",
        # Turn 1
        "帮我推荐一个好玩的游戏",
        # Turn 2
        "这个游戏适合小学生玩吗？",
        # Turn 1
        "我明天要去面试，好紧张",
        # Turn 2
        "你说的对，那我应该准备哪些问题呢？",
    ],
}

SCORING_GUIDE = """
评分标准 (1-3分):
  1 = 完全不相关 / 语法错误 / 答非所问
  2 = 基本合理但生硬 / 模板化 / 过于简短
  3 = 自然流畅 / 内容相关 / 表达得体

请对每个回答评分。
"""


def load_model(checkpoint_path="saved_models/sft_v2_final.pt", tokenizer_path="saved_models/tokenizers/phase1_8k_real_tokenizer.json"):
    """Load trained SFT model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(tokenizer_path)
    cfg = ModelConfig(
        vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
        d_ff=1024, max_seq_len=512, dropout=0.0,
        use_flash_attention=(device.type == "cuda"),
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model, tok, device


def generate_response(model, tok, device, prompt, max_tokens=80):
    """Generate a response for a single prompt."""
    text = f"用户：{prompt}\n助手："
    ids = [1] + tok.encode(text).ids
    pid = torch.tensor([ids], device=device)
    tokens = []
    with torch.no_grad():
        for token_id, is_done in model.generate_stream(
            pid, max_new_tokens=max_tokens, temperature=0.8,
            top_k=35, top_p=0.9, eos_token_id=2
        ):
            tokens.append(token_id)
            if is_done:
                break
    return tok.decode(tokens, skip_special_tokens=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="saved_models/sft_v2_final.pt")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    print("Loading model...")
    model, tok, device = load_model(args.checkpoint)
    print(f"Model loaded on {device}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    results = {}
    total = sum(len(v) for v in BENCHMARK.values())
    i = 0

    print(f"\nRunning {total} evaluations...")
    t0 = time.time()

    multiturn_context = {}

    for category, prompts in BENCHMARK.items():
        print(f"\n--- {category} ({len(prompts)} prompts) ---")
        results[category] = []

        for prompt in prompts:
            # For multiturn, accumulate context
            if category == "multiturn":
                # Even indices are turn-1, odd indices are turn-2
                idx = len(results[category])
                if idx % 2 == 0:
                    # New conversation starts
                    context = f"用户：{prompt}\n助手："
                    response = generate_response(model, tok, device, prompt)
                    multiturn_context[prompt] = response
                else:
                    # Continue previous conversation
                    prev_prompt = prompts[idx - 1]
                    prev_response = multiturn_context.get(prev_prompt, "")
                    context = f"用户：{prev_prompt}\n助手：{prev_response}\n用户：{prompt}\n助手："
                    response = generate_response(model, tok, device,
                                                  f"{prev_prompt}\n助手：{prev_response}\n用户：{prompt}")
            else:
                response = generate_response(model, tok, device, prompt)

            i += 1
            elapsed = time.time() - t0
            print(f"  [{i}/{total}] {prompt[:40]}... -> {response[:60]}... "
                  f"({elapsed/i:.1f}s/q)")

            results[category].append({
                "prompt": prompt,
                "response": response,
                "score": None,  # to be filled by human evaluator
                "notes": "",
            })

    # ── Save results ──
    output = {
        "model": args.checkpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scoring_guide": SCORING_GUIDE.strip(),
        "n_total": total,
        "avg_time_per_query": (time.time() - t0) / total,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"Evaluation complete!")
    print(f"  Total: {total} queries in {time.time()-t0:.0f}s")
    print(f"  Saved to: {args.output}")
    print(f"\nNext step: open {args.output} and fill in 'score' (1-3)")
    print(f"  Then run: python scripts/eval_scorer.py")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
