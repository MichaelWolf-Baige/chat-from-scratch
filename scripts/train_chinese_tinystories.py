#!/usr/bin/env python
"""Chinese TinyStories: generate 100K+ diverse Chinese texts, train 14M model.

Strategy (proven by TinyStories + our template experiments):
  - Controlled vocabulary (200 entities × 40 adj × 30 verbs)
  - High token repetition (each entity appears 1000+ times)
  - Template diversity (100+ templates across 8 domains)
  - 2M tokens × 15 epochs = sufficient exposure

Then SFT with dialogue data for the final chat model.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/train_chinese_tinystories.py
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

# ═══════════════════════════════════════════════════════════════
# CHINESE TEXT GENERATOR (200+ entity pool, 100+ templates, 8 domains)
# ═══════════════════════════════════════════════════════════════

# --- Large entity pools for diversity ---
PERSONS = ["小明","小红","小华","小丽","小刚","小美","老师","妈妈","爸爸","姐姐",
           "弟弟","爷爷","奶奶","同学","朋友","邻居","医生","警察","科学家","厨师"]
ANIMALS = ["小猫","小狗","小鸟","小鱼","兔子","乌龟","蝴蝶","蜜蜂","蚂蚁","松鼠",
           "小熊","小猴","小鹿","狐狸","大象","老虎","熊猫","海豚","天鹅","鹦鹉"]
PLACES  = ["学校","公园","超市","医院","图书馆","动物园","海边","山上","花园",
           "操场","厨房","卧室","书店","游乐场","游泳池","森林","河边","城市","农村"]
OBJECTS = ["书本","铅笔","玩具","积木","气球","自行车","风筝","雨伞","杯子","帽子",
           "书包","手机","电脑","电视","冰箱","洗衣机","钢琴","吉他","画笔","相机"]
FOODS   = ["苹果","香蕉","面包","蛋糕","牛奶","果汁","糖果","饼干","冰淇淋","西瓜",
           "草莓","葡萄","橙子","胡萝卜","番茄","鸡蛋","米饭","面条","饺子","巧克力"]
SUBJECTS = ["数学","语文","英语","科学","音乐","美术","体育","历史","地理","自然"]
ACTIONS = ["跑步","游泳","画画","唱歌","跳舞","读书","写字","做饭","种花","钓鱼",
           "爬山","骑车","拍照","下棋","折纸","堆雪人","放风筝","捉迷藏","跳绳","踢球"]
EMOTIONS = ["开心","难过","兴奋","紧张","惊讶","感动","骄傲","担心","好奇","满足"]
WEATHER = ["晴天","下雨","刮风","下雪","阴天","多云","彩虹","闪电","雾天","冰雹"]
COLORS = ["红色","蓝色","绿色","黄色","白色","黑色","粉色","紫色","橙色","灰色"]
SIZES = ["大大的","小小的","高高的","矮矮的","胖胖的","瘦瘦的","长长的","短短的"]

def pick(p): return random.choice(p)

def _v(rng):
    """Pick all commonly used variables at once."""
    return {
        'p': pick(PERSONS), 'a': pick(ANIMALS), 'pl': pick(PLACES),
        'o': pick(OBJECTS), 'f': pick(FOODS), 's': pick(SUBJECTS),
        'ac': pick(ACTIONS), 'e': pick(EMOTIONS), 'w': pick(WEATHER),
        'c': pick(COLORS), 'sz': pick(SIZES),
    }

def gen_diverse_chinese(n_samples, seed=42):
    rng = random.Random(seed)
    texts = []
    for _ in range(n_samples):
        r = rng.random()
        # 8 domains × many templates each
        if r < 0.15:
            texts.append(_story(rng))
        elif r < 0.30:
            texts.append(_dialogue(rng))
        elif r < 0.42:
            texts.append(_description(rng))
        elif r < 0.54:
            texts.append(_qa(rng))
        elif r < 0.65:
            texts.append(_instruction(rng))
        elif r < 0.75:
            texts.append(_news(rng))
        elif r < 0.87:
            texts.append(_reasoning(rng))
        else:
            texts.append(_code_comment(rng))
    return texts

def _story(rng):
    v = _v(rng); p=v['p']; a=v['a']; pl=v['pl']; o=v['o']; f=v['f']
    s=v['s']; ac=v['ac']; e=v['e']; w=v['w']; c=v['c']; sz=v['sz']

    templates = [
        f"{w}的一天，{p}带着{sz}的{o}去{pl}{ac}。路上遇到了{a}，{a}看起来非常{e}。{p}把{f}分给了{a}，它们一起开心地{ac}。回到家后，{p}把今天的事告诉了妈妈，妈妈夸{p}是个好孩子。",
        f"{p}有一只{sz}的{a}，它最喜欢在{pl}{ac}。一天，{a}发现了一个{c}的{o}，非常{e}。{p}帮{a}把{o}带回家，它们成为了最好的朋友。从那以后，每天放学{p}都会和{a}一起在{pl}玩。",
        f"{p}的{s}成绩很好，老师让{p}在班上分享学习经验。{p}说：'我每天都认真{ac}，还会用{o}帮助记忆。'同学们听了都很佩服，纷纷向{p}请教。大家决定放学后一起去{pl}复习{s}。",
        f"周末，{p}和{pl}的{pick(PERSONS)}一起去{ac}。天气{w}，大家都很{e}。突然，{p}发现了一只受伤的{a}。{p}小心地把{a}抱起来，带去了{pick(PERSONS)}那里。医生说只要好好照顾，{a}很快就会好起来。",
        f"{p}一直想要一个{c}的{o}，但是妈妈说要等{p}的{s}考到一百分才行。{p}每天努力{ac}，终于在一次考试中得了满分。妈妈高兴地给{p}买了那个{sz}的{o}，{p}感到非常{e}和自豪。",
        f"在{pl}里，{p}和好朋友们在玩捉迷藏的游戏。大家说说笑笑，非常热闹。这时，{p}想到了一个好主意：'我们一起用{o}来做一个新游戏吧！'大家都很赞同，开始动手做起来。",
        f"{p}早上起来发现外面{w}，兴奋地穿上衣服跑出去。在{pl}，{p}和大家一起{ac}，堆了一个{sz}的雪人。还给雪人戴上了{sz}的帽子，雪人看起来可爱极了。",
        f"今天是{p}的生日，妈妈准备了{sz}的{c}蛋糕和很多{f}。朋友们都来到{pl}参加生日派对，送给{p}各种礼物，有{o}、{pick(OBJECTS)}和{pick(OBJECTS)}。{p}感到无比幸福。",
    ]
    return rng.choice(templates)

def _dialogue(rng):
    v = _v(rng); p=v['p']; pl=v['pl']; o=v['o']; f=v['f']; s=v['s']
    ac=v['ac']; sz=v['sz']; w=v['w']

    templates = [
        f"{p}：你好！你今天看起来很开心啊。\n{pick(PERSONS)}：是啊！我今天在{pl}学到了一个有趣的知识，关于{s}的。\n{p}：真的吗？快给我讲讲！\n{pick(PERSONS)}：老师告诉我们，{s}其实和我们的日常生活息息相关。比如用{o}做的实验就证明了这一点。\n{p}：好厉害！我也想去{pl}学习{s}。",
        f"顾客：你好，请问这里有{f}吗？\n店员：当然有！我们这里的{f}非常新鲜，是今天早上刚到的。\n顾客：太好了，我要买一些。另外，这个{o}怎么卖？\n店员：这个{sz}的{o}是我们的新品，很多顾客都说好用。\n顾客：那我一起买了，谢谢！",
        f"孩子：妈妈，我今天在学校学到了{ac}。\n妈妈：真棒！{ac}是很好的运动，对你身体有好处。\n孩子：但是我不小心把{o}弄丢了。\n妈妈：没关系，我们一起去找找。下次记得把东西放在固定的地方哦。\n孩子：好的妈妈，我记住了。",
        f"学生：老师，这道{s}题我不太明白，可以再讲一遍吗？\n老师：当然可以。你看，这个问题其实是关于{ac}的，关键是要理解其中的逻辑。来，我们用{o}来演示一下。\n学生：原来是这样！我明白了。谢谢老师！\n老师：不客气，有不懂的地方随时来问。",
        f"小朋友：你好，可以和我一起{ac}吗？\n新朋友：好啊！我叫{pick(PERSONS)}，你呢？\n小朋友：我叫{p}。你喜欢在{pl}{ac}吗？\n新朋友：喜欢！我最喜欢在{pl}用{o}{ac}了。\n小朋友：那我们一起玩吧！天气{w}，正适合{ac}呢。",
    ]
    return rng.choice(templates)

def _description(rng):
    v = _v(rng); a=v['a']; pl=v['pl']; c=v['c']; o=v['o']; f=v['f']
    sz=v['sz']; ac=v['ac']; w=v['w']

    templates = [
        f"{a}是一种非常可爱的动物。它有着{sz}的身体和{sz}的尾巴，全身覆盖着柔软的毛发。{a}喜欢在{pl}{ac}，最喜欢的食物是{f}。每天，{a}都会和伙伴们一起玩耍，它们在草地上追逐嬉戏，给人们带来很多快乐。",
        f"{pl}是一个很漂亮的地方。这里有{sz}的树木和{sz}的花朵，空气非常清新。每天都有很多人来这里{ac}、散步或野餐。春天的时候，{c}的花开满了整个{pl}，远远看去就像一幅美丽的画。",
        f"{o}是我们生活中常见的物品。它通常是{sz}的形状，颜色是{c}的。我们可以用{o}来{ac}，也可以把它当作{pick(OBJECTS)}送给朋友。一个好的{o}可以用很长时间，所以选购的时候要注意质量。",
        f"今天天气{w}，天空是{c}的，云朵像{sz}的{f}飘在空中。远处的{pl}传来{sz}的声音，人们在{ac}，孩子们在追逐蝴蝶。空气中弥漫着{f}的香味，让人感到非常惬意。",
    ]
    return rng.choice(templates)

def _qa(rng):
    v = _v(rng); s=v['s']; o=v['o']; a=v['a']; ac=v['ac']; pl=v['pl']
    f=v['f']; sz=v['sz']; c=v['c']; p=v['p']

    templates = [
        f"问题：为什么{s}很重要？\n答案：{s}在我们的生活中非常重要。首先，学好{s}可以帮助我们更好地理解世界。其次，{s}的知识可以用在{ac}和{pick(ACTIONS)}等活动中。最后，良好的{s}基础能让我们在未来的学习中更加顺利。",
        f"问题：如何正确使用{o}？\n答案：使用{o}的方法很简单。第一步，检查{o}是否完好无损。第二步，按照说明书上的步骤进行操作。第三步，使用后及时清理和存放。记住，安全永远是第一位的。",
        f"问题：{a}的主要特点是什么？\n答案：{a}主要有以下几个特点：第一，{a}的外形是{sz}的，颜色通常是{c}的。第二，{a}以{f}为食，喜欢生活在{pl}。第三，{a}有{ac}的习性，非常受到人们的喜爱。",
        f"问题：在{pl}应该注意什么？\n答案：在{pl}需要注意以下几点：保持安静，不要打扰别人；保持卫生，不乱扔垃圾；注意安全，不要做危险的事情，比如在{pl}里{pick(ACTIONS)}。如果遇到困难，可以向{pick(PERSONS)}求助。",
    ]
    return rng.choice(templates)

def _instruction(rng):
    v = _v(rng); f=v['f']; o=v['o']; pl=v['pl']; sz=v['sz']; ac=v['ac']

    templates = [
        f"如何制作美味的{f}：首先，准备好需要的材料，包括新鲜的{f}、{pick(OBJECTS)}和调味料。然后，将{f}清洗干净，切成适当的大小。接着，在锅里放入少许油，待油热后将{f}放入锅中翻炒。最后，加入{sz}的{pick(OBJECTS)}和调味料，翻炒均匀即可出锅。简单又美味的{f}就做好了！",
        f"学习{ac}的步骤：第一步，找到一位好的指导者，可以是{pick(PERSONS)}或者有经验的朋友。第二步，准备好必要的装备，比如{o}和{pick(OBJECTS)}。第三步，从基础动作开始练习，不要急于求成。第四步，坚持每天练习，慢慢地你就会发现自己的进步。",
        f"参观{pl}的注意事项：首先，提前了解{pl}的开放时间和门票信息。其次，穿着舒适的衣服和鞋子，因为可能需要{pick(ACTIONS)}。到达后，按照指示牌参观，不要触摸展品。最后，可以买一些纪念品，比如{sz}的{o}，作为美好的回忆。",
    ]
    return rng.choice(templates)

def _news(rng):
    v = _v(rng); p=v['p']; pl=v['pl']; o=v['o']; c=v['c']; sz=v['sz']; ac=v['ac']

    templates = [
        f"据最新消息，{pl}近日举办了一场{sz}的活动，吸引了大量市民参与。活动中，大家展示了各自的{ac}才能，气氛非常热烈。一位参与者{p}表示，这样的活动丰富了大家的业余生活，希望以后能经常举办。",
        f"本地新闻：我市{pick(PLACES)}小学的{p}同学在全国{ac}比赛中获得了第一名。{p}从去年开始学习{ac}，每天坚持练习两小时。{p}的老师说：'{p}非常努力，这次获奖是实至名归。'",
        f"科普小知识：专家介绍，{ac}对身体有很多好处。每天坚持{ac}可以增强体质，提高免疫力。此外，{ac}还能让人保持愉快的心情，是缓解压力的好方法。",
        f"好物推荐：最近在{pl}新开了一家小店，专门出售各种{o}。这些{o}不仅外观{c}，而且质量很好。店主说开店的想法来自于自己对{ac}的热爱。如果你也在找好的{o}，不妨去{pl}看看。",
    ]
    return rng.choice(templates)

def _reasoning(rng):
    v = _v(rng); s=v['s']; c=v['c']; e=v['e']; ac=v['ac']; f=v['f']
    sz=v['sz']; p=v['p']; w=v['w']; a=v['a']; pl=v['pl']; o=v['o']

    templates = [
        f"问题：为什么{p}在{w}的天气里感到{e}？\n推理：首先，{p}原本计划去{pl}{ac}，但是{w}的天气让计划无法实现，所以{p}会感到{e}。其次，{p}期待了很久的{ac}活动被取消，自然心情不好。结论：{p}的情绪变化是因为天气影响了计划。",
        f"问题：{a}为什么会喜欢在{pl}生活？\n推理：首先，{pl}有{a}需要的食物{f}和{pick(OBJECTS)}。其次，{pl}的环境适合{a}{ac}和休息。再次，{pl}有其他{a}的同伴，可以一起生活。结论：{pl}满足了{a}的生存和社交需求。",
        f"问题：选择{c}的{o}还是{sz}的{pick(OBJECTS)}？\n推理：首先，要明确自己的需求——如果注重美观，{c}的{o}更合适；如果注重实用，{sz}的{pick(OBJECTS)}更好。其次，要考虑使用场景。结论：根据具体情况选择，没有绝对的好坏。",
    ]
    return rng.choice(templates)

def _code_comment(rng):
    ac = pick(ACTIONS)
    n = rng.choice(range(1,10))
    fn = pick(['计算','获取','显示','更新'])
    cn = pick(['学生','老师','动物','植物','课程'])
    templates = [
        f"# 这是一个简单的{ac}函数\ndef {fn}_数据(输入):\n    # 检查输入是否有效\n    if 输入 is None:\n        return dict(失败='输入不能为空')\n    # 处理数据\n    结果 = []\n    for 项目 in 输入:\n        if 项目 > {n}:\n            结果.append(项目)\n    # 返回结果\n    print(f'处理完成')\n    return 结果",
        f"class {cn}管理系统:\n    def __init__(self, 名称):\n        self.名称 = 名称\n        self.列表 = []\n    def 添加(self, 项目):\n        self.列表.append(项目)\n    def 查找(self, 关键词):\n        return [x for x in self.列表 if 关键词 in str(x)]\n    def 统计(self):\n        return len(self.列表)",
    ]
    return rng.choice(templates)


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
        print("="*55)
        print(f"Chinese TinyStories: 14M Model, {world}×RTX3090 DDP")
        print("="*55)

    # ── Generate data ──
    if rank == 0: print("Generating 100K Chinese texts...")
    texts = gen_diverse_chinese(100000, seed=42)
    if rank == 0:
        print(f"  Generated {len(texts):,} texts")
        for i in range(3): print(f"  Sample {i}: {texts[i][:120]}...")

    # ── Tokenize with our Chinese BPE ──
    from tokenizers import Tokenizer as HFTok
    tok = HFTok.from_file("tokenizers/phase1_8k_real/tokenizer.json")
    all_ids = []
    for text in texts:
        ids = tok.encode(text).ids
        all_ids.append(1); all_ids.extend(ids); all_ids.append(2)
    tokens = torch.tensor(all_ids, dtype=torch.long)

    if rank == 0:
        unique = len(torch.unique(tokens))
        print(f"  Tokens: {len(tokens):,} | Unique: {unique}/8192 ({unique/8192:.1%})")
        print(f"  Avg: {len(tokens)/len(texts):.0f} tok/text")

    # ── Model: 14M ──
    cfg = ModelConfig(
        vocab_size=tok.get_vocab_size(), d_model=384, n_layers=6,
        n_heads=6, n_kv_heads=6, d_ff=1024, max_seq_len=512,
        dropout=0.0, use_flash_attention=True, tie_word_embeddings=True,
        rms_norm_eps=1e-6, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    model = Transformer(cfg).to(device)
    model = DDP(model, device_ids=[local_r], find_unused_parameters=False,
                gradient_as_bucket_view=True)
    model.train()

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"  Model: {n:,} params | d={cfg.d_model} L={cfg.n_layers} d_ff={cfg.d_ff}")

    # ── Train/Val ──
    seq_len = 512; bs = 12  # per GPU, global=48
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

    # ── Train: WSD, 10 epochs ──
    epochs = 10; max_lr = 8e-4
    total_steps = len(train_l) * epochs
    warmup = total_steps // 10; decay_start = int(total_steps * 0.85)

    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95))
    gs = 0; t0 = time.time()

    if rank == 0:
        print(f"\n  Epochs: {epochs} | Steps: ~{total_steps} | LR: {max_lr} WSD")
        print(f"  Global batch: {bs*world}×{seq_len} | Train seqs: {len(train_t):,}")
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

            if rank == 0 and (gs <= 20 or gs % 200 == 0):
                elapsed = time.time() - t0
                tps = gs * bs * world * seq_len / max(elapsed, 0.01)
                print(f"  step {gs:5d}/{total_steps} | loss={loss.item():.4f} "
                      f"ppl={np.exp(loss.item()):.0f} | {tps/1000:.0f}K tok/s")

            if gs % 500 == 0 and rank == 0:
                model.eval(); et = []
                with torch.no_grad():
                    for ei, eb in enumerate(val_l):
                        if ei >= 12: break
                        _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                        et.append(eo["loss"].item())
                print(f"  >>> VAL PPL @ {gs}: {np.exp(np.mean(et)):.0f} <<<")
                model.train()

    # ── Final ──
    if rank == 0:
        model.eval(); et = []
        with torch.no_grad():
            for ei, eb in enumerate(val_l):
                if ei >= 25: break
                _, eo = model(eb["input_ids"].to(device), labels=eb["labels"].to(device))
                et.append(eo["loss"].item())
        val_ppl = np.exp(np.mean(et))
        elapsed = time.time() - t0

        print(f"\n{'='*55}")
        print(f"Chinese TinyStories Complete!")
        print(f"  VAL PPL: {val_ppl:.0f}")
        print(f"  Time: {elapsed/60:.1f}min | Speed: {gs*bs*world*seq_len/elapsed:.0f} tok/s")

        # Generate demo
        model.eval()
        prompts = ["今天天气真好", "小明和小红一起去", "妈妈告诉我"]
        for prompt in prompts:
            pid = tok.encode(prompt).ids
            pid_t = torch.tensor([[1] + pid], device=device)
            with torch.no_grad():
                full, _ = model.module.generate(pid_t, max_new_tokens=60, temperature=0.8, top_k=35, top_p=0.9)
            resp = tok.decode(full[0].tolist(), skip_special_tokens=True)
            print(f"  P: {prompt}")
            print(f"  G: {resp[:200]}")
            print()

        # Save
        ckpt_dir = Path("checkpoints/chinese_tinystories")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(ckpt_dir / "final.pt", model.module, opt, None,
                        step=gs, epoch=0, config={"val_ppl": float(val_ppl), "n_texts": len(texts)})
        with open(ckpt_dir / "summary.json", "w") as f:
            json.dump({"val_ppl": float(val_ppl), "n_texts": len(texts),
                       "tokens": len(tokens), "time_min": elapsed/60,
                       "model_params": sum(p.numel() for p in model.parameters())}, f, indent=2)
        print(f"\n  Saved: {ckpt_dir / 'final.pt'}")
        print(f"✅ Done!")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
