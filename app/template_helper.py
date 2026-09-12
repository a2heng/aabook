"""AuK 指令模板助手 - 离线标准指令生成器。"""

TEMPLATE_TASKS = {
    "文本合成": {
        "tpl": '请基于下面的描述: "{style_desc}",生成语音内容"{text}".',
        "defaults": {"style_desc": "年轻女生温柔体贴的语气，语速稍慢，句尾轻轻上扬", "text": "宝宝，欢迎回来"},
        "hint": "无需音频",
    },
    "音色克隆": {
        "tpl": 'Say the following with the same voice: "{text}"',
        "defaults": {"text": "大家好，欢迎收看今天的节目"},
        "hint": "需要参考音频",
    },
    "语速调整": {
        "tpl": "将语速调整为{speed}倍。",
        "defaults": {"speed": "0.75"},
        "options": {"speed": ["0.5", "0.75", "1.25", "1.5", "2.0"]},
        "hint": "需要音频",
    },
    "音量调高": {
        "tpl": "将音量升高{db}分贝。",
        "defaults": {"db": "5"},
        "options": {"db": ["5", "10", "15"]},
        "hint": "需要音频",
    },
    "音量调低": {
        "tpl": "将音量降低{db}分贝。",
        "defaults": {"db": "5"},
        "options": {"db": ["5", "10", "15"]},
        "hint": "需要音频",
    },
    "音调升高": {
        "tpl": "将音调升高{n}个半音。",
        "defaults": {"n": "1"},
        "options": {"n": ["1", "2", "3"]},
        "hint": "需要音频",
    },
    "音调降低": {
        "tpl": "将音调降低{n}个半音。",
        "defaults": {"n": "1"},
        "options": {"n": ["1", "2", "3"]},
        "hint": "需要音频",
    },
    "情感转换": {
        "tpl": "将情感转变为{emotion}。",
        "defaults": {"emotion": "happy"},
        "options": {"emotion": ["happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited"]},
        "hint": "需要音频",
    },
    "音色编辑": {
        "tpl": '请将这段音频的音色修改为符合以下描述的声音："{timbre_desc}"。',
        "defaults": {"timbre_desc": "音色清亮的年轻女性，语速平稳，吐字清晰"},
        "hint": "需要音频",
    },
    "内容替换": {
        "tpl": "把'{orig}'改成'{new}'",
        "defaults": {"orig": "今天下午开会", "new": "明天上午开会"},
        "hint": "需要音频",
    },
    "内容删除": {
        "tpl": "删掉'{target}'",
        "defaults": {"target": "那个"},
        "hint": "需要音频",
    },
    "内容插入": {
        "tpl": "在'{anchor}'后面加上'{text}'",
        "defaults": {"anchor": "你好", "text": "呀"},
        "hint": "需要音频",
    },
    "语音增强": {
        "tpl": "请清理这段输入语音，不做说话人删除，并去除房间混响，输出干净的人声结果。",
        "defaults": {},
        "hint": "需要音频",
    },
    "说话人分离": {
        "tpl": "这段音频中只保留第{n_zh}个开始说话的人对应的语音，去掉其余说话人。",
        "defaults": {"n_zh": "一"},
        "options": {"n_zh": ["一", "二", "三", "四", "五"]},
        "hint": "需要音频",
    },
    "提取人声": {
        "tpl": "请只保留歌声，其余声音都去掉。",
        "defaults": {},
        "hint": "需要音频",
    },
    "音质提升": {
        "tpl": "请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。",
        "defaults": {},
        "hint": "需要音频",
    },
    "耳语转正常": {
        "tpl": "把这段耳语转换成正常说话的声音。",
        "defaults": {},
        "hint": "需要音频",
    },
    "正常转耳语": {
        "tpl": "用小声耳语的方式把这段话说出来。",
        "defaults": {},
        "hint": "需要音频",
    },
    "歌词编辑": {
        "tpl": '把这段歌词中的"{orig}"改成"{new}"。',
        "defaults": {"orig": "明天你好", "new": "未来你好"},
        "hint": "需要音频",
    },
    "去口音": {
        "tpl": "请去掉这段语音里的方言口音，保持说话人音色一致。",
        "defaults": {},
        "hint": "需要音频",
    },
}


TEMPLATE_CATEGORIES = {
    "语音合成": ["文本合成", "音色克隆"],
    "内容编辑": ["内容替换", "内容删除", "内容插入", "歌词编辑"],
    "声学编辑": ["语速调整", "音量调高", "音量调低", "音调升高", "音调降低"],
    "副语言编辑": ["情感转换", "音色编辑", "去口音", "耳语转正常", "正常转耳语"],
    "增强与分离": ["语音增强", "说话人分离", "提取人声", "音质提升"],
}


def _render(task, overrides):
    fields = dict(TEMPLATE_TASKS[task]["defaults"])
    fields.update(overrides)
    try:
        return TEMPLATE_TASKS[task]["tpl"].format(**fields)
    except KeyError:
        return TEMPLATE_TASKS[task]["tpl"]


_OPTION_LABELS = {
    "语速调整": lambda v: f"语速{v}x",
    "音量调高": lambda v: f"音量+{v}dB",
    "音量调低": lambda v: f"音量-{v}dB",
    "音调升高": lambda v: f"音调+{v}",
    "音调降低": lambda v: f"音调-{v}",
    "情感转换": lambda v: f"情感{v}",
    "说话人分离": lambda v: f"分离第{'一二三四五'.index(v) + 1}人",
}


def tpl_instances():
    """Group concrete, ready-to-use instruction instances by category.

    Each item is ``(button_label, instruction_text)``.  Instances with a single
    tunable option expand into one button per option value so users can pick a
    concrete example with a single click.
    """
    groups = []
    for category, tasks in TEMPLATE_CATEGORIES.items():
        items = []
        for task in tasks:
            options = TEMPLATE_TASKS[task].get("options") or {}
            if len(options) == 1:
                key, values = next(iter(options.items()))
                label = _OPTION_LABELS.get(task)
                for value in values:
                    button_label = label(value) if label else f"{task}·{value}"
                    items.append((button_label, _render(task, {key: value})))
            else:
                items.append((task, _render(task, {})))
        groups.append((category, items))
    return groups


def tpl_select(task_name):
    t = TEMPLATE_TASKS[task_name]
    return t["tpl"], f"{task_name} ({t['hint']})"


def tpl_build(task_key, speed, db, n, emotion, n_zh, orig, new, anchor, target, style_desc, timbre_desc, text):
    t = TEMPLATE_TASKS[task_key]
    fields = dict(t["defaults"])
    params = {
        "speed": speed,
        "db": db,
        "n": n,
        "emotion": emotion,
        "n_zh": n_zh,
        "orig": orig,
        "new": new,
        "anchor": anchor,
        "target": target,
        "style_desc": style_desc,
        "timbre_desc": timbre_desc,
        "text": text,
    }
    for k, v in params.items():
        if v is not None and v != "":
            fields[k] = v
    try:
        return t["tpl"].format(**fields)
    except KeyError:
        return t["tpl"]
