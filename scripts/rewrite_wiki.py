#!/usr/bin/env python
"""Rewrite Wikipedia entries into tutorial/conversational style using Teacher model.
Usage: CUDA_VISIBLE_DEVICES=0 python scripts/rewrite_wiki.py --n 2000 -o data/wiki_rewritten.jsonl
"""
import json, random, time, argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)

REWRITE_PROMPTS = [
    "请将以下百科内容改写为一篇通俗易懂的中文科普文章，用对话式的语言，约200字。保留所有关键事实：\n\n{text}",
    "用一问一答的形式，把下面这段百科知识变成一段有趣的中文教学对话，约200字：\n\n{text}",
    "假设你是一个AI助教，请用简单易懂的中文向高中生解释以下内容，保留关键信息，约200字：\n\n{text}",
    "把下面这段内容用讲故事的方式重新表达，保留所有事实和数据，语言要生动有趣，约200字：\n\n{text}",
    "将以下百科条目改写为适合中文预训练语料的通顺文本，语言自然流畅，保留所有重要信息，约200字：\n\n{text}",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('-o', required=True)
    parser.add_argument('-b', type=int, default=32)
    args = parser.parse_args()

    data_dir = Path.home() / 'chat-from-scratch/data'
    wiki_path = data_dir / 'wiki_zh_clean.jsonl'

    print(f'Sampling {args.n} wiki entries...')
    reservoir = []
    with open(wiki_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < args.n:
                reservoir.append(json.loads(line)['text'])
            else:
                j = random.randint(0, i)
                if j < args.n:
                    reservoir[j] = json.loads(line)['text']

    wiki_texts = [t for t in reservoir if 100 <= len(t) <= 2000]
    wiki_texts = [t[:1500] for t in wiki_texts]
    print(f'  {len(wiki_texts)} valid entries')

    model_id = 'Qwen/Qwen2.5-1.5B-Instruct'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map='cuda:0',
        local_files_only=True, attn_implementation='sdpa',
    ).eval()
    print('  Teacher loaded')

    prompts = []
    for wt in wiki_texts:
        template = random.choice(REWRITE_PROMPTS)
        prompts.append(template.format(text=wt))

    all_results = []
    t0 = time.time()

    for b_start in range(0, len(prompts), args.b):
        batch = prompts[b_start:b_start + args.b]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300, temperature=0.8, top_p=0.9,
                do_sample=True, pad_token_id=tok.eos_token_id,
            )
        for j, out in enumerate(outputs):
            in_len = inputs['input_ids'][j].shape[0]
            text = tok.decode(out[in_len:], skip_special_tokens=True).strip()
            if len(text) >= 50:
                all_results.append({'text': text})

        b_num = b_start // args.b + 1
        if b_num <= 3 or b_num % 20 == 0:
            elapsed = time.time() - t0
            chars = sum(len(r['text']) for r in all_results)
            total_b = (len(prompts) + args.b - 1) // args.b
            print(f'  [{b_num}/{total_b}] {len(all_results)} texts | {chars/max(elapsed,1):.0f} c/s | {elapsed/60:.0f}min')

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    elapsed = time.time() - t0
    print(f'\nDone! {len(all_results)} texts | {out_path.stat().st_size/1e6:.0f}MB | {elapsed/60:.1f}min')

if __name__ == '__main__':
    main()
