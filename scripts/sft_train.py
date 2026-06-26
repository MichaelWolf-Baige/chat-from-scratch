#!/usr/bin/env python
"""SFT: Fine-tune the pretrained Chinese TinyStories model into a chat assistant.

Uses template-generated dialogue data (same controlled vocab as pretraining).
High repetition × diverse patterns = efficient SFT at small scale.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/sft_train.py
"""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np, random, time, json
from datetime import datetime

from src.model.config import ModelConfig
from src.model.transformer import Transformer
from src.data.dataset import PretrainDataset
from src.utils.checkpoint import save_checkpoint

random.seed(42)

# ═══════════════════════════════════════════════════════════════
# SFT DATA GENERATOR — Chinese chat dialogues
# ═══════════════════════════════════════════════════════════════

GREETINGS = ["你好！","您好！","嗨！","早上好！","下午好！"]
GOODBYES = ["再见！","下次见！","拜拜！","回头聊！","祝你有美好的一天！"]
THANKS = ["谢谢！","非常感谢！","太感谢了！","多谢！"]
ACKNOWLEDGMENTS = ["不客气！","不用谢！","很高兴能帮到你！","随时欢迎！"]
TOPICS = [
    ("学习", ["数学","语文","英语","编程","物理","化学","历史","地理"]),
    ("工作", ["项目","报告","会议","计划","团队","客户","代码","测试"]),
    ("生活", ["健身","做饭","旅游","阅读","音乐","电影","运动","游戏"]),
    ("技术", ["Python","算法","数据库","网络","前端","后端","AI","云计算"]),
]

def sft_conversation():
    """Generate a multi-turn Chinese chat conversation."""
    turns = []
    user_name = random.choice(["小明","小红","小华","同学A","用户"])
    topic_cat, topic_items = random.choice(TOPICS)
    topic = random.choice(topic_items)

    # Turn 1: Greeting + ask something
    turns.append({
        "role": "user",
        "content": f"{random.choice(GREETINGS)}请问你能帮我了解一下关于{topic}的知识吗？"
    })
    turns.append({
        "role": "assistant",
        "content": f"{random.choice(GREETINGS)}当然可以！{topic}是一个很有趣的话题。{topic}的基本概念是{random.choice(['关于','围绕','基于'])}{random.choice(['数据','原理','方法','技术','理论'])}展开的。你想了解{topic}的哪个方面呢？比如基础知识、应用场景，还是学习方法？"
    })

    # Turn 2: Specific follow-up
    turns.append({
        "role": "user",
        "content": f"我想知道{topic}的主要应用场景有哪些？能不能举几个例子？"
    })
    turns.append({
        "role": "assistant",
        "content": f"好的，{topic}的应用场景非常广泛。首先是{topic_items[random.randint(0,len(topic_items)-1)]}领域，{topic}可以帮助提高效率和准确性。其次是{topic_items[random.randint(0,len(topic_items)-1)]}方面，{topic}能够解决传统方法难以处理的问题。此外在{topic_items[random.randint(0,len(topic_items)-1)]}中也有重要应用。这些应用都体现了{topic}的核心价值——让复杂的事情变得简单高效。"
    })

    # Turn 3: Practical advice
    turns.append({
        "role": "user",
        "content": f"听起来很有意思！如果我想要学习{topic}，应该从哪里开始呢？"
    })
    turns.append({
        "role": "assistant",
        "content": f"学习{topic}的建议如下：第一步，打好基础——了解{topic}的核心概念和基本原理。可以通过阅读入门教程或观看教学视频来建立初步理解。第二步，动手实践——找一些简单的项目或练习来应用所学的知识。实践是检验理解的唯一标准。第三步，深入学习——选择{topic}的一个细分方向进行专门研究。记住，学习是一个循序渐进的过程，不要急于求成。保持好奇心和学习热情最重要！"
    })

    # Turn 4: Wrap up
    turns.append({
        "role": "user",
        "content": f"非常感谢你的建议！我学到了很多关于{topic}的知识。"
    })
    turns.append({
        "role": "assistant",
        "content": f"不客气！很高兴能和你讨论{topic}。如果你在学习过程中遇到任何问题，随时可以来找我交流。{random.choice(GOODBYES)}"
    })

    return turns

def sft_qa():
    """Generate a single-turn QA."""
    q_templates = [
        "请解释一下{concept}的基本概念。",
        "{concept}的主要特点是什么？",
        "为什么{concept}很重要？",
        "如何正确理解{concept}？",
        "请举一个{concept}的实际例子。",
    ]
    concepts = ["模型","算法","系统","架构","接口","协议","框架","方法","策略","标准"]
    concept = random.choice(concepts)

    question = random.choice(q_templates).format(concept=concept)
    answer = f"{concept}是指通过系统化的方式来处理问题的一套方法。{concept}的核心包括明确的目标、合理的步骤和有效的评估机制。在实际应用中，一个好的{concept}可以帮助我们提高效率、降低错误率，并且方便团队协作。理解{concept}的关键在于掌握其基本原理并能够在实践中灵活运用。"

    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

def generate_sft_data(n_conversations=2000, n_qa=3000):
    """Generate SFT training data."""
    data = []
    for _ in range(n_conversations):
        data.extend(sft_conversation())
    for _ in range(n_qa):
        data.extend(sft_qa())
    # Group back into conversations
    conversations = []
    current = []
    for turn in data:
        current.append(turn)
        if turn["role"] == "assistant" and len(current) >= 2:
            # Decide whether to end the conversation here
            if random.random() < 0.3 or len(current) >= 8:
                conversations.append({"messages": current})
                current = []
    if current:
        conversations.append({"messages": current})
    return conversations


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank(); world = dist.get_world_size()
    local_r = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_r); device = torch.device(f"cuda:{local_r}")
    seed = 42 + rank
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    random.seed(seed)

    if rank == 0:
        print("=" * 55)
        print(f" SFT: Chinese Chat Assistant ({world}×RTX3090 DDP)")
        print("=" * 55)

    # ── Generate SFT data ──
    if rank == 0: print("Generating SFT conversation data...")
    sft_data = generate_sft_data(n_conversations=2000, n_qa=3000)
    if rank == 0:
        print(f"  Generated {len(sft_data)} conversations")
        # Show sample
        sample = sft_data[0]
        for turn in sample["messages"][:4]:
            print(f"  [{turn['role']}]: {turn['content'][:80]}...")
        print()

    # ── Tokenize in chat format ──
    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")

    # Convert conversations to token sequences
    # Format: <s>user: ...\nassistant: ...</s>
    all_ids = []
    for conv in sft_data:
        all_ids.append(1)  # BOS
        for turn in conv["messages"]:
            prefix = "用户：" if turn["role"] == "user" else "助手："
            text = prefix + turn["content"] + "\n"
            ids = tok.encode(text).ids
            all_ids.extend(ids)
        all_ids.append(2)  # EOS

    tokens = torch.tensor(all_ids, dtype=torch.long)

    if rank == 0:
        unique = len(torch.unique(tokens))
        print(f"  Tokens: {len(tokens):,} | Unique: {unique}/8192 ({unique/8192:.1%})")

    # ── Load pretrained model ──
    cfg = ModelConfig(
        vocab_size=tok.get_vocab_size(), d_model=384, n_layers=6,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512,
        dropout=0.0, use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)

    # Load pretrained weights
    ckpt_path = Path("checkpoints/chinese_tinystories/final.pt")
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if rank == 0: print(f"  Loaded pretrained weights from {ckpt_path}")
    else:
        if rank == 0: print(f"  ⚠️  No pretrained checkpoint found, training from scratch")

    model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    # ── Train/Val ──
    seq_len = 384; bs = 8  # per GPU, global=32
    usable = (len(tokens) // seq_len) * seq_len
    tokens_flat = tokens[:usable].view(-1, seq_len)
    split = int(len(tokens_flat) * 0.95)
    train_t, val_t = tokens_flat[:split], tokens_flat[split:]

    train_ds = PretrainDataset(train_t.flatten(), seq_len=seq_len)
    val_ds = PretrainDataset(val_t.flatten(), seq_len=seq_len)
    train_s = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
    val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False, drop_last=True)
    train_l = torch.utils.data.DataLoader(train_ds, batch_size=bs, sampler=train_s,
                                           num_workers=2, pin_memory=True, prefetch_factor=2,
                                           persistent_workers=True)
    val_l = torch.utils.data.DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                         num_workers=2, pin_memory=True, prefetch_factor=2,
                                         persistent_workers=True)

    # ── SFT Training (lower LR, fewer epochs — fine-tuning not pretraining) ──
    epochs = 5; max_lr = 2e-4
    total_steps = len(train_l) * epochs
    warmup = total_steps // 10; decay_start = int(total_steps * 0.85)

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\n  SFT Training: {epochs} epochs, LR={max_lr} WSD")
        print(f"  Global batch: {bs*world}×{seq_len}")
        print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")

    for epoch in range(epochs):
        train_s.set_epoch(epoch)
        for batch in train_l:
            if gs >= total_steps: break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            _, out = model(input_ids, labels=labels); loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()

            if gs < warmup: lr = max_lr*(gs+1)/warmup
            elif gs < decay_start: lr = max_lr
            else:
                p = min((gs-decay_start)/max(total_steps-decay_start,1), 1.0)
                lr = max_lr*0.01+0.5*max_lr*(1.0+np.cos(np.pi*p))
            for pg in opt.param_groups: pg["lr"] = lr
            gs += 1

            if rank == 0 and (gs <= 10 or gs % 100 == 0):
                elapsed = time.time() - t0
                print(f"  step {gs:4d}/{total_steps} | loss={loss.item():.4f} | "
                      f"ppl={np.exp(loss.item()):.0f} | {(gs*bs*world*seq_len/max(elapsed,0.01))/1000:.0f}K tok/s")

            if gs % 300 == 0 and rank == 0:
                model.eval(); et = []
                with torch.no_grad():
                    for ei, eb in enumerate(val_l):
                        if ei >= 10: break
                        _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                        et.append(eo["loss"].item())
                print(f"  >>> SFT VAL PPL @ {gs}: {np.exp(np.mean(et)):.0f} <<<")
                model.train()

    # ── Final + Chat Demo ──
    if rank == 0:
        model.eval()
        elapsed = time.time() - t0

        print(f"\n{'='*55}")
        print(f"SFT Complete! Time: {elapsed/60:.1f}min")
        print(f"{'='*55}")

        # Chat demo
        test_prompts = [
            "你好！请介绍一下什么是算法。",
            "我想学习编程，应该从哪里开始？",
            "谢谢你今天的帮助！",
            "什么是模型？它有什么应用？",
        ]
        for prompt in test_prompts:
            # Build input: <s>用户：{prompt}\n助手：
            prefix = "用户：" + prompt + "\n助手："
            pid = [1] + tok.encode(prefix).ids
            pid_t = torch.tensor([pid], device=device)
            with torch.no_grad():
                full, _ = model.module.generate(pid_t, max_new_tokens=80, temperature=0.8, top_k=35, top_p=0.9,
                                                 eos_token_id=2)
            # Extract only the generated part (after the prompt)
            full_text = tok.decode(full[0].tolist(), skip_special_tokens=True)
            # Find where the assistant response starts
            parts = full_text.split("助手：")
            response = parts[-1].strip() if len(parts) > 1 else full_text[len(prefix):].strip()
            print(f"  👤 {prompt}")
            print(f"  🤖 {response[:200]}")
            print()

        # Save
        ckpt_dir = Path("checkpoints/sft_chat")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None, step=gs, epoch=0, config={"phase":"SFT"})
        print(f"  ✅ Saved: {ckpt_dir / 'final.pt'}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
