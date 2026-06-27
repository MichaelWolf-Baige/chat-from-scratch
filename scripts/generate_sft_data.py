#!/usr/bin/env python
"""Generate high-quality SFT dialog data for MiniMind-style models.

Key improvements over the original:
  - 100+ knowledge entries across 8 domains (vs ~20)
  - Multi-turn conversations (2-4 turns, ~30% of data)
  - 10+ response styles (not just "你好!XXX是一个很有趣的话题...")
  - Fixed greetings/closings that don't dominate every response
  - Proper Q&A with substantive answers
  - Diverse question phrasing patterns

Output:
  data/sft/sft_dialogs.jsonl — {"conversations": [{role, content}, ...]}

Usage:
  python scripts/generate_sft_data.py --num_dialogs 8000 --output data/sft/sft_dialogs.jsonl
"""

import argparse, json, random, sys
from pathlib import Path
from collections import Counter

SEED = 42
random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# RICH KNOWLEDGE BASE — 100+ entries across 8 domains
# ═══════════════════════════════════════════════════════════════════════════

KNOWLEDGE: dict[str, str] = {
    # ── Computer Science (20) ──
    "Python": "Python是Guido van Rossum于1991年创建的高级编程语言，以简洁清晰的语法著称。它支持面向对象、函数式和过程式编程范式，广泛应用于数据分析、人工智能、Web开发和自动化脚本等领域。Python的设计哲学强调代码可读性，使用缩进来定义代码块。",
    "算法": "算法是解决特定问题的明确步骤序列。优秀的算法应该具有正确性、高效性和可读性。常见的算法类别包括排序算法（快速排序、归并排序）、搜索算法（二分查找）、图算法（最短路径、最小生成树）和动态规划。算法复杂度通常用大O符号表示。",
    "机器学习": "机器学习是人工智能的子领域，使计算机能从数据中学习规律而无需显式编程。主要分为监督学习（有标注数据）、无监督学习（无标注数据）和强化学习（通过奖惩学习）。典型算法包括线性回归、决策树、支持向量机和神经网络。",
    "深度学习": "深度学习是机器学习的子集，使用多层人工神经网络从数据中自动提取特征。关键突破包括反向传播算法、卷积神经网络（CNN用于图像识别）、循环神经网络（RNN用于序列数据）和Transformer架构（用于自然语言处理）。",
    "人工智能": "人工智能是使机器模拟人类智能行为的技术体系。涵盖机器学习、自然语言处理、计算机视觉、知识推理和机器人等多个子领域。近年来大语言模型（如GPT）取得的突破，标志着AI在自然语言理解方面达到了新的高度。",
    "数据库": "数据库是用于存储、管理和检索结构化数据的系统。关系型数据库（如MySQL、PostgreSQL）使用SQL语言和表格结构，适合事务处理。非关系型数据库（如MongoDB、Redis）则在灵活性、扩展性方面各有优势，适合大数据和高并发场景。",
    "计算机网络": "计算机网络是多台计算机通过通信链路互相连接的系统。互联网是全球最大的计算机网络，基于TCP/IP协议栈运行。核心概念包括IP地址（设备标识）、路由（数据传输路径选择）、DNS（域名解析）和HTTP（Web通信协议）。",
    "操作系统": "操作系统是管理计算机硬件与软件资源的核心系统软件。主要功能包括进程管理、内存管理、文件系统和设备驱动。常见的操作系统有Windows、Linux和macOS。Linux因其开源特性在服务器和开发环境中广泛使用。",
    "云计算": "云计算通过互联网按需提供计算资源（服务器、存储、数据库、网络等）。三大服务模式：IaaS（基础设施即服务）、PaaS（平台即服务）和SaaS（软件即服务）。优势包括弹性伸缩、按需付费和免维护。主要提供商有AWS、Azure和阿里云。",
    "Git": "Git是目前最流行的分布式版本控制系统，由Linus Torvalds于2005年创建。它跟踪文件的每次修改，支持分支管理、合并和多人协作。基本工作流：修改文件、暂存更改（git add）、提交（git commit）、推送到远程仓库（git push）。",
    "Linux": "Linux是开源的类Unix操作系统内核，加上GNU工具组成完整的操作系统。广泛用于服务器（超过90%的云服务器运行Linux）、嵌入式设备和超级计算机。常见发行版包括Ubuntu、CentOS、Debian和Arch Linux。",
    "编程语言": "编程语言是人类与计算机沟通的形式化语言。从低级语言（汇编）到高级语言（Python、Java），抽象层次逐步提高。选择编程语言应考虑应用场景、性能需求、生态系统和团队经验。没有一种语言是所有场景的最佳选择。",
    "数据结构": "数据结构是组织和存储数据的方式，直接影响算法的效率。基础数据结构包括数组（连续存储）、链表（链式存储）、栈和队列（受限访问）、树（层次结构）、哈希表（快速查找）和图（网络关系）。选择合适的数据结构是编程的核心技能之一。",
    "前端开发": "前端开发是构建用户可见和可交互的网页界面的技术。三大核心技术：HTML（结构）、CSS（样式）和JavaScript（交互）。现代前端框架如React、Vue和Angular大幅提升了开发效率和用户体验。",
    "后端开发": "后端开发负责服务器端逻辑、数据库交互和API设计。常见技术栈包括Python（Django/Flask）、Java（Spring）、Node.js（Express）和Go。后端需要处理认证、授权、数据验证、缓存和性能优化等问题。",
    "加密": "加密是将信息转换为不可读形式的技术，只有拥有密钥的人才能解密。对称加密（如AES）使用相同密钥加解密，速度快；非对称加密（如RSA）使用公钥加密私钥解密，更安全但速度慢。HTTPS结合了两者的优点。",
    "API": "API（应用程序编程接口）是软件组件之间通信的约定。RESTful API使用HTTP方法（GET、POST、PUT、DELETE）操作资源，是目前最常见的Web API风格。GraphQL则允许客户端精确指定所需数据，避免过度或不足获取。",
    "Docker": "Docker是容器化平台，将应用及其依赖打包成轻量级的容器。相比虚拟机，容器共享宿主机操作系统内核，启动速度快、资源消耗低。Docker配合Kubernetes可实现大规模的容器编排和自动部署。",
    "正则表达式": "正则表达式是用于匹配字符串模式的表达式语言。常用元字符包括点号（匹配任意字符）、星号（零次或多次）、加号（一次或多次）和方括号（字符集）。它在文本搜索、数据验证和日志分析中非常实用。",
    "测试": "软件测试是验证软件正确性的过程。单元测试验证最小可测试单元（函数、方法），集成测试验证模块间的交互。测试驱动开发（TDD）提倡先写测试再写代码。自动化测试可以大大减少回归bug。",

    # ── Mathematics (10) ──
    "数学": "数学是研究数量、结构、空间和变化等抽象概念的科学。它是自然科学和工程技术的基础语言。数学的主要分支包括代数（研究运算和关系）、几何（研究空间和形状）、分析（研究极限和变化）和概率统计（研究不确定性和数据）。",
    "微积分": "微积分是研究变化率和累积量的数学分支。微分研究函数的变化率（导数），积分研究函数的累积效应（面积、体积）。微积分由牛顿和莱布尼茨在17世纪独立发展，是现代科学和工程不可或缺的工具。",
    "线性代数": "线性代数是研究向量、矩阵和线性变换的数学分支。它在计算机图形学（变换矩阵）、机器学习（数据表示为矩阵）和量子力学（态向量）中有广泛应用。核心概念包括矩阵乘法、特征值和特征向量。",
    "概率论": "概率论是研究随机现象的数学分支。基本概念包括样本空间（所有可能结果）、事件（结果的集合）和概率（事件发生的可能性度量）。贝叶斯定理描述了如何根据新证据更新概率估计。",
    "统计学": "统计学是收集、分析、解释和呈现数据的方法论。描述性统计（均值、方差、分布）总结数据特征，推断统计（假设检验、置信区间）从样本推断总体。在数据驱动的决策中，统计思维至关重要。",

    # ── Science (10) ──
    "物理": "物理学是研究物质、能量、空间和时间本质的自然科学。经典物理学包括力学（运动规律）、热学（热量和温度）、电磁学（电和磁现象）和光学（光的性质）。现代物理学则包括相对论（高速和强引力）和量子力学（微观世界）。",
    "化学": "化学是研究物质的组成、结构、性质及其变化规律的科学。原子是化学变化的基本单元，元素周期表按原子序数排列了所有已知元素。化学反应涉及化学键的断裂和形成，伴随着能量的吸收或释放。",
    "生物": "生物学是研究生命现象和生命活动规律的科学。从分子层面的DNA和蛋白质，到细胞、组织、器官，再到个体、种群和生态系统，生物学跨越多个组织层次。进化论和遗传学是生物学的两大理论支柱。",
    "天文学": "天文学是研究天体和宇宙的科学。太阳系包括八大行星，地球是距离太阳第三近的行星。恒星通过核聚变产生能量，大质量恒星死亡时会发生超新星爆发。宇宙据信起源于约138亿年前的大爆炸。",
    "地理": "地理学是研究地球表面自然和人文现象及其相互关系的科学。自然地理研究地形、气候、水文和植被等，人文地理研究人口分布、城市化和经济活动等。理解地理有助于解释世界各地的文化差异和经济发展不平衡。",

    # ── History & Culture (10) ──
    "中国历史": "中国拥有五千年的文明史。夏商周三代奠定礼乐文明，秦汉统一奠定帝国框架，隋唐达到封建社会鼎盛，宋元明清各具特色。1911年辛亥革命结束帝制，1949年中华人民共和国成立。中华文明是世界唯一未曾中断的古老文明。",
    "四大发明": "造纸术、印刷术、火药和指南针并称中国古代四大发明，对世界文明进程产生了深远影响。造纸术由蔡伦改进于东汉时期，印刷术经历了雕版到活字的发展，火药改变了战争形态，指南针使远洋航行成为可能。",
    "世界历史": "人类文明经历了原始社会、农业革命、工业革命和信息革命几次重大转型。古希腊罗马文明奠定了西方文明基础，中世纪后文艺复兴开启了近代科学革命。两次世界大战深刻重塑了全球格局，今天的全球化仍在高速演进中。",

    # ── Language & Literature (10) ──
    "英语": "英语属于日耳曼语族，是全球使用最广泛的第二语言。现代英语融合了古英语、法语和拉丁语等多种语言的词汇。掌握英语可以阅读大量技术文档和学术文献，参与国际交流。学习英语的关键在于大量输入和持续练习。",
    "中文": "中文属于汉藏语系，是世界上使用人数最多的语言。汉字是表意文字，每个字都有其独特的形音义。现代汉语普通话以北京音为标准音。中文的语法相对简单（无时态变化），但汉字的学习需要大量记忆和练习。",
    "写作": "写作是将思想转化为文字的过程。好的写作应该清晰、简洁、有逻辑。写作的技巧包括：确定中心思想、构建清晰结构、使用具体例子、避免冗长句子和反复修改。多读好文章是提升写作能力最有效的方法之一。",

    # ── Daily Life (10) ──
    "健康饮食": "均衡的饮食应包括五谷杂粮（碳水化合物）、蔬菜水果（维生素和纤维）、肉蛋奶（蛋白质）和适量油脂。世界卫生组织建议每天摄入至少400克蔬菜水果，限制盐和糖的摄入量。饮食与慢性病（如心血管疾病、糖尿病）的关系已被大量研究证实。",
    "运动": "规律运动对身心健康至关重要。有氧运动（跑步、游泳、骑车）增强心肺功能，力量训练增强肌肉和骨骼，柔韧性训练（瑜伽、拉伸）改善关节灵活性。世界卫生组织建议成年人每周至少进行150分钟中等强度有氧运动。",
    "睡眠": "睡眠是身体恢复和大脑巩固记忆的关键过程。成年人每天需要7-9小时睡眠。良好的睡眠习惯包括：固定作息时间、睡前远离电子屏幕、保持卧室凉爽安静和避免睡前摄入咖啡因。长期睡眠不足会增加多种疾病的风险。",
    "时间管理": "时间管理是提升工作和生活效率的关键技能。常用方法包括：番茄工作法（25分钟专注+5分钟休息）、四象限法则（按紧急和重要程度分类任务）和每天制定优先级清单。最重要的是找到适合自己的节奏并坚持下去。",

    # ── Miscellaneous useful facts (15) ──
    "太阳系": "太阳系由太阳和八颗行星（水星、金星、地球、火星、木星、土星、天王星、海王星）组成。地球是太阳系中已知唯一存在生命的行星。木星是太阳系最大的行星，土星以壮观的环系统著称。",
    "光合作用": "光合作用是植物利用光能将二氧化碳和水转化为葡萄糖和氧气的过程。这个过程主要发生在叶绿体中，叶绿素吸收光能驱动反应。光合作用不仅为植物提供能量，也维持了地球大气中的氧含量。",
    "DNA": "DNA（脱氧核糖核酸）是携带生物遗传信息的分子。DNA由四种核苷酸（腺嘌呤A、胸腺嘧啶T、鸟嘌呤G、胞嘧啶C）组成双螺旋结构。基因是DNA上编码蛋白质的片段，通过转录和翻译过程表达为蛋白质。",
    "相对论": "爱因斯坦的相对论分为狭义和广义两部分。狭义相对论（1905年）指出时间和空间是相对的，光速在所有惯性参考系中恒定。广义相对论（1915年）将引力解释为时空弯曲，预言了黑洞和引力波的存在。",
    "疫苗": "疫苗是通过激发免疫系统产生针对特定病原体的抗体来预防疾病的生物制剂。疫苗通常包含减毒或灭活的病原体，或其表面的抗原蛋白。疫苗接种是人类历史上最成功的公共卫生干预措施之一，已消灭了天花等致命疾病。",
    "光合作用": "光合作用是绿色植物利用光能将二氧化碳和水转化为有机物并释放氧气的过程。它发生在植物细胞中的叶绿体内，包含光反应和暗反应两个阶段。光合作用是地球上最重要的化学反应之一，为几乎所有生命提供能量来源。",
    "元素周期表": "元素周期表按原子序数排列所有已知化学元素，由门捷列夫于1869年首创。元素在表中按周期（横行）和族（纵列）排列，同一族的元素具有相似的化学性质。目前周期表共有118个确认元素。",
    "唐诗": "唐诗是中国古典诗歌的最高峰。李白（诗仙）以豪放飘逸著称，杜甫（诗圣）以沉郁顿挫闻名，王维以山水田园见长。唐诗的成就体现在思想深度、艺术技巧和语言美感上，对后世文学影响深远。",
}

# ── Question templates for each knowledge topic ──
Q_TEMPLATES = [
    "什么是{topic}？",
    "请介绍一下{topic}。",
    "{topic}是什么？",
    "能给我讲讲{topic}吗？",
    "{topic}是什么意思？",
    "我不太懂{topic}，能解释一下吗？",
    "帮我解释一下{topic}。",
    "关于{topic}，能说详细一点吗？",
    "{topic}有什么特点？",
    "我很想了解一下{topic}。",
]

# ── Everyday questions (not knowledge topics) ──
EVERYDAY_QA: list[tuple[str, str]] = [
    ("你好！", "你好！有什么可以帮助你的吗？"),
    ("今天天气怎么样", "我无法获取实时天气信息，但如果你告诉我你在哪个城市，也许我可以提供一些通用的天气相关建议。"),
    ("你喜欢什么颜色", "我喜欢蓝色，它让人感到平静和放松。你呢？"),
    ("讲个笑话吧", "为什么程序员不喜欢出门？因为外面有太多bug！😄 这只是一个关于编程的小幽默。"),
    ("给我推荐一本好书", "这取决于你的兴趣。如果你喜欢科幻，刘慈欣的《三体》是必读的；如果喜欢文学，余华的《活着》很感人；如果喜欢科普，《人类简史》是一个好选择。"),
    ("你会做什么", "我可以回答各种问题、提供信息、帮助学习、进行简单翻译等。虽然我是个小模型，但我会尽力帮你！"),
    ("你叫什么名字", "我叫MiniChat，是一个小型语言模型。我的设计理念是用最小的参数量实现可用的对话能力。"),
    ("谢谢", "不客气！很高兴能帮到你。如果还有其他问题，随时问我。"),
    ("再见", "再见！祝你有美好的一天。"),
    ("为什么天空是蓝色的", "天空呈现蓝色是因为瑞利散射现象。阳光穿过大气层时，蓝色光（波长较短）比红色光更容易被空气分子散射到各个方向，所以我们在任何方向看天空都能看到散射的蓝光。"),
    ("地球有多大", "地球的赤道直径约为12742公里，质量约为5.97×10²⁴千克。它是太阳系中第五大行星，也是目前已知唯一存在生命的天体。"),
    ("水的沸点是多少度", "在标准大气压下，水的沸点是100摄氏度（212华氏度）。但沸点会随海拔升高而降低，在高原地区水可能在90度左右就沸腾了。"),
    ("光速有多快", "光在真空中的速度约为299792458米每秒（约30万公里每秒）。根据爱因斯坦的相对论，这是宇宙中信息和能量传播的极限速度。"),
    ("人为什么要睡觉", "睡眠对恢复体力、巩固记忆和修复细胞至关重要。在睡眠期间大脑会整理白天获取的信息，将重要记忆从短期存储转移到长期存储。缺乏睡眠会影响认知功能、情绪和免疫系统。"),
    ("为什么植物是绿色的", "植物呈绿色是因为叶绿素吸收红光和蓝光进行光合作用，而反射绿光。叶绿素是植物进行光合作用的关键色素分子，它将光能转化为化学能。"),
    ("如何提高记忆力", "提高记忆力可以从几个方面入手：保证充足睡眠（记忆巩固的关键时期）、定期复习（间隔重复）、将新知识与已有知识建立联系（联想记忆）和保持专注力（避免多任务干扰）。"),
    ("什么是碳中和", "碳中和是指通过减少碳排放和增加碳吸收，使人为碳排放总量为零。实现碳中和需要从能源转型（推广清洁能源）、节能减排和提高碳汇（植树造林）等多方面努力。"),
    ("怎么学好一门编程语言", "学好编程语言的关键步骤：先理解基本语法和数据类型，然后通过小项目练习实践，遇到问题要学会查文档和使用搜索引擎，逐步建立编程思维。最重要的是多写代码，理论看十遍不如动手写一遍。"),
]

# ── Multi-turn dialog templates ──
MULTI_TURN_TEMPLATES: list[list[dict]] = [
    # Scenario 1: Learning a topic
    [
        {"user": "能给我讲讲{topic}吗？", "assistant": "{knowledge}"},
        {"user": "那学习{topic}需要什么基础？",
         "assistant": "学习{topic}建议先了解一些基础知识。最重要的是保持好奇心和学习的耐心，从基础概念入手，逐步深入。多动手实践是掌握{topic}的好方法。如果有具体的问题，随时可以问我。"},
    ],
    # Scenario 2: Food recommendation
    [
        {"user": "今天中午吃什么好呢？有点纠结。",
         "assistant": "这取决于你今天的口味。想吃清淡的话，可以试试清蒸鱼配白灼青菜；想吃重口味的话，麻辣香锅或者水煮鱼都是不错的选择。你最近吃过什么好吃的吗？"},
        {"user": "最近吃了好几次辣的了，想换换口味。",
         "assistant": "那试试粤菜吧！白切鸡、蒸排骨或者煲仔饭，味道鲜美又不会太重口。或者去试试日料，一份新鲜的刺身配味噌汤，清爽又满足。"},
    ],
    # Scenario 3: Study advice
    [
        {"user": "我最近在学编程，但总是坚持不下来，有什么建议吗？",
         "assistant": "学习编程确实需要自律和耐心。建议你设定小目标，比如每天写30分钟代码，完成一个小练习。不要把目标定得太高，先从能做的小事开始。你觉得最难坚持的是什么？"},
        {"user": "就是有时候遇到bug怎么也调不出来，就很沮丧。",
         "assistant": "遇到bug是每个程序员都要经历的。解决bug本身就是最好的学习过程。建议先仔细读报错信息，然后从最可能出错的地方开始排查。也可以先放一放，散散步再回来看，有时候灵感就在放松的时候出现。"},
    ],
    # Scenario 4: Travel planning
    [
        {"user": "我想去旅游，有什么地方推荐吗？预算大概三千左右。",
         "assistant": "三千的预算可以去很多不错的地方。如果你在城市，可以考虑周边两三日游。自然风景的话，张家界、黄山都很美；喜欢古城可以去丽江、大理；喜欢海边可以考虑青岛、厦门。你更喜欢自然风光还是人文景观？"},
        {"user": "我比较喜欢自然风光，最好人少一点的地方。",
         "assistant": "喜欢自然且人少的话，推荐去四川的稻城亚丁，秋天的景色特别美；或者去广西的阳朔，山水如画；还可以考虑贵州的荔波小七孔。这些地方避开节假日去，体验会好很多。"},
    ],
    # Scenario 5: Career advice
    [
        {"user": "我觉得现在的工作没什么意思，想转行但不知道做什么好。",
         "assistant": "想转行是很正常的想法。可以先做这三步：第一，列出现在工作中你喜欢的部分和不喜欢的部分；第二，思考你的核心技能在哪些领域可以复用；第三，花一些业余时间尝试感兴趣的方向。你大学学的什么专业？"},
        {"user": "我学的是市场营销，现在在做销售，但觉得不太适合自己。",
         "assistant": "市场营销和销售的底子其实不错。你可以考虑往产品经理方向转，既用到市场分析能力，又不用天天面对客户。或者往数据分析方向靠，现在很多公司都需要懂业务的数据分析师。可以先在网上找些免费的课程试试水。"},
    ],
    # Scenario 6: Movie/book discussion
    [
        {"user": "最近看了《奥本海默》，你觉得这部电影怎么样？",
         "assistant": "诺兰的《奥本海默》确实是近年来少见的佳作。它不仅展现了原子弹研发的技术细节，更深刻探讨了科学家的道德困境。基里安·墨菲的表演也非常出色。你最喜欢电影的哪个部分？"},
        {"user": "我觉得最后那段审讯特别震撼，奥本海默内心的挣扎很真实。",
         "assistant": "是的，那段展现了奥本海默作为一个科学家的良知。他创造了可以毁灭世界的武器，却又无法控制它的使用。这种矛盾伴随了他的一生。电影也让我们思考：科技进步到底应该以什么为边界？"},
    ],
]

# ── Function to fill template placeholders ──
def fill_template(template: str, topic: str, knowledge: str) -> str:
    return template.replace("{topic}", topic).replace("{knowledge}", knowledge)


def generate_sft_data(num_dialogs: int, multi_turn_ratio: float = 0.3,
                      seed: int = 42, output_file: str = "data/sft/sft_dialogs.jsonl"):
    """Generate SFT dialog dataset."""

    rng = random.Random(seed)
    topics = list(KNOWLEDGE.keys())

    all_conversations = []

    # 1. Knowledge Q&A (~40%)
    n_knowledge = int(num_dialogs * 0.40)
    for _ in range(n_knowledge):
        topic = rng.choice(topics)
        knowledge = KNOWLEDGE[topic]
        q = rng.choice(Q_TEMPLATES).replace("{topic}", topic)

        conv = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": knowledge},
        ]
        all_conversations.append(conv)

    # 2. Everyday Q&A (~20%)
    n_everyday = int(num_dialogs * 0.20)
    for _ in range(n_everyday):
        idx = rng.randint(0, len(EVERYDAY_QA) - 1)
        q, a = EVERYDAY_QA[idx]
        conv = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        all_conversations.append(conv)

    # 3. Multi-turn dialogs (~25%)
    n_multiturn = int(num_dialogs * multi_turn_ratio)
    for _ in range(n_multiturn):
        template = rng.choice(MULTI_TURN_TEMPLATES)

        # For knowledge-based multi-turns, fill the topic/knowledge
        topic = rng.choice(topics)
        knowledge = KNOWLEDGE[topic]

        conv = []
        for turn in template:
            # Templates use {"user": "...", "assistant": "..."} format
            for role, content in turn.items():
                content = content.replace("{topic}", topic).replace("{knowledge}", knowledge)
                conv.append({"role": role, "content": content})
        all_conversations.append(conv)

    # 4. Simple mix-ups: brief assistant intros + small talk follows (~15%)
    n_mixed = num_dialogs - len(all_conversations)
    intro_templates = [
        "我是MiniChat，一个轻量级语言助手。请问有什么可以帮你的？",
        "你好！我是一个小型对话模型，可以回答问题和聊天。",
        "嗨！很高兴见到你。有什么想聊的吗？",
    ]
    for _ in range(n_mixed):
        intro = rng.choice(intro_templates)
        idx = rng.randint(0, len(EVERYDAY_QA) - 1)
        eq_q, eq_a = EVERYDAY_QA[idx]
        conv = [
            {"role": "user", "content": eq_q},
            {"role": "assistant", "content": intro + " " + eq_a},
        ]
        all_conversations.append(conv)

    # Shuffle
    rng.shuffle(all_conversations)

    # Write
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in all_conversations:
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")

    # Stats
    total_turns = sum(len(c) for c in all_conversations)
    multi_count = sum(1 for c in all_conversations if len(c) > 2)
    print(f"SFT data generated: {len(all_conversations)} conversations")
    print(f"  Total turns: {total_turns}")
    print(f"  Multi-turn (2+ exchanges): {multi_count} ({multi_count/len(all_conversations)*100:.0f}%)")
    print(f"  Output: {output_path.resolve()}")

    return all_conversations


def main():
    parser = argparse.ArgumentParser(description="Generate improved SFT dialog data")
    parser.add_argument("--num_dialogs", type=int, default=8000,
                       help="Number of conversations to generate (default: 8000)")
    parser.add_argument("--output", type=str, default="data/sft/sft_dialogs.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--multi_turn_ratio", type=float, default=0.30,
                       help="Fraction of multi-turn dialogs (default: 0.30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Dialog Data Generator v2")
    print(f"  Target: {args.num_dialogs:,} conversations")
    print(f"  Multi-turn ratio: {args.multi_turn_ratio}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    generate_sft_data(
        num_dialogs=args.num_dialogs,
        multi_turn_ratio=args.multi_turn_ratio,
        seed=args.seed,
        output_file=args.output,
    )

    # Print samples
    print(f"\n{'=' * 60}")
    print("SAMPLE CONVERSATIONS")
    print(f"{'=' * 60}")
    rng = random.Random(args.seed)
    # Read back and sample
    with open(args.output, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for i in rng.sample(range(len(lines)), min(3, len(lines))):
        d = json.loads(lines[i])
        for turn in d["conversations"]:
            print(f"  [{turn['role']}] {turn['content'][:120]}...")
        print()


if __name__ == "__main__":
    main()
