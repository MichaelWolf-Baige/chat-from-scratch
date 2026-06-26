#!/usr/bin/env python
"""P0.5: SFT v5 baseline evaluation — 30 prompts across 3 categories.

Usage (server or local):
    python scripts/eval_sft_v5.py --checkpoint checkpoints/sft_v5/final.pt
    python scripts/eval_sft_v5.py --checkpoint saved_models/sft_v5_final.pt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, json, time, argparse
from tokenizers import Tokenizer
from src.model.config import ModelConfig
from src.model.transformer import Transformer

# ── 30 prompts ──────────────────────────────────────────────────────
PROMPTS = {
    "knowledge_known": [
        "什么是Python？",
        "什么是机器学习？",
        "什么是算法？",
        "什么是数据库？",
        "什么是人工智能？",
        "什么是互联网？",
        "什么是操作系统？",
        "什么是深度学习？",
        "什么是编程？",
        "什么是Linux？",
    ],
    "knowledge_unknown": [
        "中国的首都是哪里？",
        "李白是谁？",
        "1+1等于几？",
        "水的化学式是什么？",
        "地球绕太阳转一圈要多久？",
        "太阳系有几大行星？",
        "世界上最高的山是哪座？",
        "人体正常体温是多少？",
        "今天北京的天气怎么样？",
        "帮我计算 15 加 27 等于多少？",
    ],
    "chitchat": [
        "你好！",
        "谢谢你帮我",
        "再见",
        "我今天心情不好",
        "最近过得怎么样？",
        "你叫什么名字？",
    ],
    "instruction": [
        "请把'你好世界'翻译成英文",
        "帮我数一下从1到10",
        "请用'春天'造一个句子",
        "推荐一本好书",
    ],
}

SCORING = {
    "OK": "Single clean response, no ghost turns, relevant to question",
    "WRONG_TOPIC": "Response is about wrong topic (e.g. asked about capital, got 'internet' definition)",
    "IDK_OK": "Clean rejection (I don't know / can't answer) — acceptable for unknown topics",
    "IDK_WRONG": "Rejected a question the model SHOULD know (known topic in knowledge_known)",
    "GHOST_TURN": "Hallucinated extra assistant turns in the same response",
    "TEMPLATE": "Falls back to template pattern (X是指通过系统化的方式...)",
    "GIBBERISH": "Unreadable / nonsensical output",
}

BOS, EOS, USR, AST = 1, 2, 5, 6


def evaluate(checkpoint_path, tokenizer_path="tokenizers/phase1_8k_real/tokenizer.json"):
    """Run evaluation and return results."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok = Tokenizer.from_file(tokenizer_path)
    cfg = ModelConfig(
        vocab_size=8192, d_model=384, n_layers=6, n_heads=6, n_kv_heads=6,
        d_ff=1024, max_seq_len=512, dropout=0.0,
        use_flash_attention=(device.type == "cuda"),
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        pad_token_id=0, bos_token_id=BOS, eos_token_id=EOS,
    )
    model = Transformer(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    results = []
    total = sum(len(v) for v in PROMPTS.values())
    i = 0
    t0 = time.time()

    for category, prompts in PROMPTS.items():
        print(f"--- {category} ---")
        for prompt in prompts:
            user_ids = tok.encode(prompt).ids
            pid_list = [BOS, USR] + user_ids + [AST]
            pid = torch.tensor([pid_list], device=device)

            tokens = []
            for tid, is_done in model.generate_stream(
                pid, max_new_tokens=60, temperature=0.8, top_k=35, top_p=0.9,
                eos_token_id=EOS
            ):
                tokens.append(tid)
                if is_done:
                    break
            resp = tok.decode(tokens, skip_special_tokens=True)

            # Auto-score
            status = "OK"
            if "助手" in resp:
                status = "GHOST_TURN"
            elif "X是指通过系统化的方式" in resp or "是指通过系统化的方式来处理" in resp:
                status = "TEMPLATE"
            elif category == "knowledge_known" and (
                "不太了解" in resp or "暂时无法" in resp or "知识范围" in resp
            ):
                status = "IDK_WRONG"
            elif category in ("knowledge_unknown", "chitchat", "instruction") and (
                "不太了解" in resp or "暂时无法" in resp or "知识范围" in resp
            ):
                status = "IDK_OK"

            i += 1
            results.append({
                "category": category,
                "prompt": prompt,
                "response": resp,
                "status": status,
                "score_note": SCORING.get(status, ""),
            })

            elapsed = time.time() - t0
            print(f"  [{i:2d}/{total}] [{status:<12}] {prompt[:30]:<30} -> {resp[:60]}... ({elapsed/i:.1f}s/q)")

    # ── Summary ──
    status_counts = {}
    for r in results:
        s = r["status"]
        status_counts[s] = status_counts.get(s, 0) + 1

    print(f"\n{'='*55}")
    print("P0.5: SFT v5 Baseline Summary")
    print(f"{'='*55}")
    for s, n in sorted(status_counts.items()):
        pct = n / len(results) * 100
        bar = "█" * int(pct / 5)
        print(f"  {s:<15} {n:2d} ({pct:5.1f}%) {bar}")

    # Category-level breakdown
    print(f"\n  Category breakdown:")
    for cat in PROMPTS:
        cat_results = [r for r in results if r["category"] == cat]
        ok = sum(1 for r in cat_results if r["status"] in ("OK", "IDK_OK"))
        print(f"    {cat:<25}: {ok}/{len(cat_results)} ({ok/len(cat_results):.0%})")

    clean = sum(1 for r in results if r["status"] in ("OK", "IDK_OK"))
    print(f"\n  Total clean: {clean}/{len(results)} ({clean/len(results):.0%})")
    print(f"  Total time:  {time.time()-t0:.0f}s")

    return results, clean / len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sft_v5/final.pt")
    parser.add_argument("--output", default="eval_v5_baseline.json")
    args = parser.parse_args()

    results, score = evaluate(args.checkpoint)

    output = {
        "model": "chat-from-scratch SFT v5 (ChatML)",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "score": score,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {args.output}")
    print(f"  Score: {score:.0%} clean")


if __name__ == "__main__":
    main()
