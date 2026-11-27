"""Minimal query interpretation for the vector routes.

Scope note: this is deliberately not an LLM rewriter. It takes the raw request
and produces the parts the retrievers need — the text query, up to three short
audio descriptions, and any keyword groups — with a versioned, inspectable rule
set. The plan's ``interpretation`` field records ``rules`` so nobody mistakes it
for a model result.

The Chinese→English sound-description step exists because CLAP's text tower is
English-trained; BGE-M3 handles the original text unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from audiosearch.contracts.models import QueryPlan

RULES_VERSION = "query-rules-v2"

# Stable diagnostic codes for tests. User-facing warnings stay ordinary prose and
# are not a protocol the desktop is expected to parse.
CODE_RULE_CONVERTED = "RULE_CONVERTED"
CODE_PARTIAL_UNCONVERTED = "PARTIAL_UNCONVERTED"
CODE_EXCLUSION_NOT_ENFORCED = "EXCLUSION_NOT_ENFORCED"
CODE_EXCLUSION_SCOPE_UNCERTAIN = "EXCLUSION_SCOPE_UNCERTAIN"

# Small, explicit sound vocabulary. Extending it is a version bump.
_TERM_MAP: dict[str, str] = {
    "水": "water",
    "水滴": "water drop",
    "水滴声": "water drop",
    "水流": "water flow",
    "水花": "water splash",
    "海浪": "ocean waves",
    "水下": "underwater",
    "气泡": "bubbles",
    "火": "fire",
    "火焰": "flame",
    "火声": "fire burning",
    "燃烧": "burning",
    "噼啪": "crackling",
    "喷火器": "flamethrower",
    "喷射": "spray",
    "风": "wind",
    "风声": "wind",
    "风吹": "wind blowing",
    "呼啸": "whoosh",
    "掠过": "flyby",
    "魔法": "magic",
    "魔法阵": "magic circle",
    "施法": "spell casting",
    "咒语": "incantation",
    "能量球": "energy sphere",
    "蓄力": "charge up",
    "爆炸": "explosion",
    "冲击": "impact",
    "碰撞": "collision impact",
    "撞击": "impact hit",
    "敲击": "knock",
    "掉落": "falling",
    "落下": "falling",
    "金属": "metal",
    "钟": "gong",
    "玻璃": "glass",
    "玻璃杯": "glass cup",
    "玻璃碎裂": "glass shatter",
    "碎裂": "shatter",
    "碎裂声": "shatter",
    "破碎": "break",
    "轻碰": "soft clink",
    "陶瓷": "ceramic",
    "嗡嗡": "hum",
    "低频": "low frequency rumble",
    "轰鸣": "low rumble",
    "电流": "electric",
    "电击": "electric zap",
    "闪电": "lightning",
    "放电": "electric discharge",
    "电焊": "welding",
    "火花": "sparks",
    "划过": "cuts through",
    "空气": "air",
    "雨": "rain",
    "雨声": "rain",
    "雷": "thunder",
    "雷鸣": "thunder",
    "雷声": "thunder",
    "脚步": "footsteps",
    "脚步声": "footsteps",
    "皮革": "leather",
    "皮衣": "leather clothing",
    "塑料袋": "plastic bag",
    "布料": "cloth",
    "摩擦": "friction",
    "摩擦声": "friction",
    "刮擦": "scrape",
    "揉搓": "crumple",
    "木头": "wood",
    "石地": "stone ground",
    "石地上": "on stone ground",
    "石头": "stone",
    "鼓": "drum",
    "铃声": "chime",
    "钟声": "bell",
    "铃": "bell chime",
    "开门": "door opening",
    "开门声": "door opening",
    "关门": "door closing",
    "上升": "rising",
    "下降": "falling",
    "空灵": "ethereal",
    "低沉": "deep low",
    "尖锐": "sharp",
    "沉重": "heavy",
    "短促": "short",
    "持续": "continuous",
    "连续": "continuous",
    "缓慢": "slow",
    "快速": "quick",
    "潮湿": "damp",
    "远处": "distant",
    "太大": "large",
    "回声": "echo",
    "循环": "loop",
    "氛围": "ambience",
    "环境氛围": "ambience",
    "噪音": "noise",
    "狗叫": "dog bark",
    "狗叫声": "dog bark",
    "犬吠": "dog bark",
    "猫叫": "cat meow",
    "猫叫声": "cat meow",
    "鸟鸣": "bird chirp",
    "鸟叫": "bird chirp",
    "引擎": "engine",
    "引擎声": "engine",
    "发动机": "engine",
    "汽车引擎": "car engine",
    "汽车": "car",
    "启动": "start",
    "冰块": "ice",
    "冰": "ice",
    # UI / Casual Game SFX
    "金币": "gold coin",
    "硬币": "coin",
    "钱币": "coin",
    "星星": "star",
    "宝石": "gem crystal",
    "钻石": "diamond sparkle",
    "拾取": "pickup",
    "收集": "collect",
    "奖励": "reward bonus",
    "升级": "level up power-up",
    "经验": "experience gain",
    "解锁": "unlock",
    "宝箱": "treasure chest",
    "按钮": "button",
    "点击": "click",
    "按下": "press",
    "弹窗": "pop up",
    "弹出": "pop",
    "泡泡": "bubble pop",
    "气泡音": "bubble pop",
    "消除": "eliminate match",
    "合成": "merge combine",
    "连击": "combo",
    "暴击": "critical hit",
    "胜利": "victory win",
    "通关": "stage clear victory",
    "成功": "success",
    "失败": "fail lose",
    "错误": "error denied",
    "提示": "notification alert",
    "通知": "notification",
    "倒计时": "countdown tick",
    "滴答": "ticking clock",
    "滑过": "swipe whoosh",
    "滑动": "slide swipe",
    "翻页": "page turn",
    "清脆": "bright crisp",
    "柔和": "soft gentle",
    "轻快": "light playful",
    "叮": "ding chime",
    "闪烁": "twinkle shimmer",
    "闪耀": "sparkle",
    # Weapons / Combat / Sci-Fi
    "激光": "laser",
    "激光枪": "laser gun",
    "射线": "beam ray",
    "枪": "gun",
    "枪声": "gunshot",
    "手枪": "pistol",
    "步枪": "rifle",
    "机枪": "machine gun",
    "霰弹枪": "shotgun",
    "狙击": "sniper shot",
    "射击": "shooting shot",
    "开火": "fire shot",
    "连射": "burst fire",
    "子弹": "bullet",
    "弹壳": "shell casing",
    "换弹": "reload",
    "上膛": "cocking reload",
    "炮": "cannon",
    "炮弹": "artillery shell",
    "爆炸声": "explosion",
    "爆破": "blast detonation",
    "轰炸": "bombing blast",
    "冲击波": "shockwave blast",
    "刀": "blade knife",
    "剑": "sword",
    "拔刀": "sword draw",
    "收刀": "sword sheathe",
    "剑气": "sword energy slash",
    "挥砍": "sword slash",
    "劈砍": "heavy slash",
    "砍击": "slash hit",
    "挥舞": "swing whoosh",
    "武器": "weapon",
    "重金属": "heavy metal",
    "盔甲": "armor",
    "护甲": "armor",
    "盾牌": "shield",
    "盾击": "shield bash",
    "格挡": "block parry",
    "打击": "hit impact",
    "重击": "heavy impact punch",
    "轻击": "light hit tap",
    "拳击": "punch hit",
    "拳头": "punch",
    "弓": "bow",
    "射箭": "bow arrow release",
    "箭": "arrow",
    "科幻": "sci-fi futuristic",
    "太空": "space",
    "能量": "energy",
    "脉冲": "pulse",
    "光束": "energy beam",
    "等离子": "plasma",
    "巨大": "huge massive",
    # Creatures / Monsters / Vocals
    "怪物": "monster creature",
    "怪兽": "monster beast",
    "巨兽": "giant beast",
    "野兽": "wild beast",
    "恶魔": "demon",
    "巨龙": "dragon",
    "龙": "dragon",
    "僵尸": "zombie",
    "丧尸": "zombie",
    "幽灵": "ghost",
    "鬼魂": "ghost spirit",
    "咆哮": "roar",
    "怒吼": "fierce roar",
    "低吼": "deep growl",
    "嘶吼": "snarl hiss",
    "嚎叫": "howl",
    "哀嚎": "wail",
    "尖叫": "scream",
    "惨叫": "scream",
    "喘息": "panting breath",
    "呼吸": "breathing",
    "心跳": "heartbeat",
    "心跳声": "heartbeat",
    "翅膀": "wings flapping",
    "扇动": "flapping",
    "嘶嘶": "hiss",
    "低语": "whisper",
    "耳语": "whisper",
    "人声": "voice vocal",
    "笑声": "laugh",
    "哭声": "cry",
    "掌声": "applause",
    # Retro / 8-bit / Synth
    "复古": "retro vintage",
    "八位": "8-bit chiptune",
    "像素": "pixel chiptune",
    "游戏": "arcade game",
    "街机": "arcade",
    "芯片": "chiptune",
    "跳跃": "jump",
    "起跳": "jump",
    "落地": "land thud",
    "冲刺": "dash whoosh",
    "道具": "item power-up",
    "电子": "electronic synth",
    "合成器": "synthesizer",
    "故障": "glitch",
    "干扰": "static interference",
    "电流声": "electric static",
    "电弧": "electric arc",
    "雷电": "thunder lightning",
    # Machines / Robots / Vehicles / Alarms
    "机械": "mechanical",
    "机器人": "robot",
    "机甲": "mech",
    "伺服": "servo",
    "齿轮": "gear",
    "运转": "whirring operating",
    "转动": "spinning turning",
    "旋转": "rotating spin",
    "变形": "transform morph",
    "液压": "hydraulic",
    "活塞": "piston",
    "开关": "switch click",
    "按键": "key switch",
    "警报": "alarm siren",
    "警笛": "siren",
    "鸣笛": "horn honk",
    "汽笛": "whistle horn",
    "雷达": "radar sonar",
    "声呐": "sonar ping",
    "飞船": "spaceship",
    "宇宙飞船": "spaceship",
    "飞行器": "aircraft",
    "直升机": "helicopter",
    "飞机": "airplane",
    "无人机": "drone",
    "火车": "train",
    "刹车": "brake screech",
    "电梯": "elevator",
    "键盘": "keyboard typing",
    "打字": "typing",
    "鼠标": "mouse click",
    "相机": "camera",
    "快门": "camera shutter",
    # Magic / Elements / Foley / Ambience
    "冰冻": "freeze ice",
    "结冰": "freezing ice",
    "治疗": "heal chime",
    "回血": "heal",
    "传送": "teleport whoosh",
    "召唤": "summon",
    "护盾": "energy shield",
    "法术": "magic spell",
    "火球": "fireball",
    "火焰喷射": "flamethrower",
    "岩浆": "lava magma",
    "熔岩": "molten lava",
    "树叶": "leaves rustle",
    "落叶": "dry leaves",
    "草丛": "grass rustle",
    "踩雪": "snow footsteps",
    "雪": "snow",
    "泥泞": "mud squish",
    "岩石": "rock stone",
    "落石": "falling rocks",
    "坍塌": "collapse crumble",
    "木箱": "wooden crate",
    "木门": "wooden door",
    "铁门": "metal door",
    "链条": "metal chain",
    "铁链": "metal chain",
    "纸张": "paper",
    "撕纸": "paper tear",
    "拉链": "zipper",
    "钥匙": "keys jingle",
    "吱呀": "creak",
    "钟表": "clock",
    "暴雨": "heavy rain",
    "狂风": "strong wind",
    "微风": "gentle breeze",
    "溪流": "stream brook",
    "瀑布": "waterfall",
    "森林": "forest",
    "夜晚": "night ambience",
    "虫鸣": "insects crickets",
    "人群": "crowd",
    "城市": "city traffic",
    "车流": "traffic",
    "狼嚎": "wolf howl",
    "乌鸦": "crow caw",
    "青蛙": "frog croak",
    "马蹄": "horse hooves",
    "轻柔": "soft gentle",
    "明亮": "bright",
    "黑暗": "dark eerie",
    "诡异": "eerie creepy",
    "紧张": "tense",
    "厚重": "heavy deep",
    "金属声": "metallic clang",
    "水声": "water",
}

# Cues that start an exclusion object. None of these become a database hard filter.
_NEGATION_CUES: frozenset[str] = frozenset(
    {"不要", "别", "不要的", "没有", "去掉", "排除", "不含", "不是"}
)
_ENGLISH_NEGATION_CUES: frozenset[str] = frozenset({"without", "excluding", "except", "no", "not"})
# After these cues, a following 的 closes the excluded noun and returns to positive
# (「不含金属的木头敲击」). 不要 / without keep the rest of the clause excluded.
_RELATIVE_EXCLUSION_CUES: frozenset[str] = frozenset({"排除", "不含", "没有"})
_POSITIVE_ONLY: frozenset[str] = frozenset({"只要", "只有"})

_FILLERS = {
    "找",
    "搜索",
    "查",
    "查找",
    "一下",
    "一点",
    "一些",
    "的",
    "了",
    "和",
    "与",
    "或",
    "声音",
    "音效",
    "素材",
    "那种",
    "类似",
    "像",
    "请",
    "帮我",
    "想要",
    "需要",
    "有没有",
    "一个",
    "一段",
    "声",
    "后",
    "传来",
    "加上",
    "然后",
    "带",
}

_TOKEN = re.compile(
    r"(?P<filename>[A-Za-z][A-Za-z0-9'_-]*(?:\.[A-Za-z][A-Za-z0-9]*)+)"
    r"|(?P<english>[A-Za-z][A-Za-z0-9'_-]*)"
    r"|(?P<digits>[0-9]+)"
    r"|(?P<cjk>[\u4e00-\u9fff]+)"
    r"|(?P<sep>[,，。！!？?、；;：:]+)"
    r"|(?P<other>[^\s,，。！!？?、；;：:A-Za-z0-9\u4e00-\u9fff]+)"
)

# Longest-first so "玻璃碎裂" wins over "玻璃" and fillers are matched whole.
_LEXICON: tuple[str, ...] = tuple(
    sorted(
        {*_TERM_MAP, *_FILLERS, *_NEGATION_CUES, *_POSITIVE_ONLY},
        key=len,
        reverse=True,
    )
)

_SINGLE_CHAR_TERMS: frozenset[str] = frozenset(key for key in _TERM_MAP if len(key) == 1)


@dataclass(frozen=True, slots=True)
class InterpretedQuery:
    """What the rules extracted, kept inspectable for reporting."""

    keyword_groups: list[list[str]]
    audio_descriptions: list[str]
    text_query: str
    dropped_fillers: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    omitted_spans: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()
    exclusion_scope: str = "none"


@dataclass(frozen=True, slots=True)
class _Atom:
    kind: str
    raw: str
    value: str
    cue: str = ""


def _lexicon_kind(key: str) -> str:
    if key in _NEGATION_CUES:
        return "negation"
    if key in _POSITIVE_ONLY:
        return "only"
    if key in _FILLERS:
        return "filler"
    return "term"


def _match_at(term: str, index: int) -> tuple[str, str] | None:
    """Return (kind, key) for the longest lexicon match at index."""
    for key in _LEXICON:
        if term.startswith(key, index):
            return (_lexicon_kind(key), key)
    return None


def _cjk_segments(term: str) -> list[_Atom]:
    """Greedy longest-match segmentation with unknown spans kept intact.

    One-character source terms are accepted only when they are not glued to
    unmatched CJK. Unknown runs stay as a single ``unknown`` atom so they are
    never split into tags the rules claim to have understood.
    """
    atoms: list[_Atom] = []
    index = 0
    while index < len(term):
        match = _match_at(term, index)
        if match is not None:
            kind, key = match
            if kind == "term" and key in _SINGLE_CHAR_TERMS:
                end = index + len(key)
                if end < len(term) and _accepted_match(term, end) is None:
                    # Glued to following unmatched CJK: start an unknown span.
                    match = None
        if match is not None:
            kind, key = match
            if kind == "term":
                atoms.append(_Atom("term", key, _TERM_MAP[key]))
            elif kind == "negation":
                atoms.append(_Atom("negation", key, key, cue=key))
            elif kind == "only":
                atoms.append(_Atom("only", key, key))
            else:
                atoms.append(_Atom("filler", key, key))
            index += len(key)
            continue
        # Consume a maximal unknown span up to the next accepted match.
        start = index
        index += 1
        while index < len(term) and _accepted_match(term, index) is None:
            index += 1
        span = term[start:index]
        atoms.append(_Atom("unknown", span, span))
    return atoms


def _accepted_match(term: str, index: int) -> tuple[str, str] | None:
    match = _match_at(term, index)
    if match is None:
        return None
    kind, key = match
    if kind == "term" and key in _SINGLE_CHAR_TERMS:
        end = index + len(key)
        if end < len(term) and _match_at(term, end) is None:
            return None
    return match


def _tokenize(text: str) -> list[_Atom]:
    atoms: list[_Atom] = []
    pending_break = False

    def extend_content(new_atoms: list[_Atom]) -> None:
        nonlocal pending_break
        if pending_break and new_atoms:
            atoms.append(_Atom("boundary", "", ""))
        atoms.extend(new_atoms)
        pending_break = True

    for match in _TOKEN.finditer(text):
        if match.group("cjk"):
            extend_content(_cjk_segments(match.group("cjk")))
            continue
        token = match.group("filename") or match.group("english")
        if token:
            lowered = token.casefold()
            if lowered in _ENGLISH_NEGATION_CUES:
                if pending_break:
                    atoms.append(_Atom("boundary", "", ""))
                atoms.append(_Atom("negation", token, token, cue=lowered))
                pending_break = False
            else:
                # Ordinary English words, including articles and conjunctions,
                # stay as content. Dropping them here would rewrite the CLAP
                # description for queries that should keep the original wording.
                extend_content([_Atom("english", token, token)])
            continue
        if match.group("digits"):
            extend_content([_Atom("unknown", match.group("digits"), match.group("digits"))])
            continue
        if match.group("sep"):
            atoms.append(_Atom("separator", match.group("sep"), match.group("sep")))
            pending_break = False
            continue
        other = match.group("other")
        if other:
            extend_content([_Atom("unknown", other, other)])
    return atoms


def _user_warning(code: str, detail: str) -> str:
    if code == CODE_RULE_CONVERTED:
        return f"规则转换：{detail}"
    if code == CODE_PARTIAL_UNCONVERTED:
        return f"部分未转换：{detail}"
    if code == CODE_EXCLUSION_NOT_ENFORCED:
        return f"排除未强制执行：{detail}"
    if code == CODE_EXCLUSION_SCOPE_UNCERTAIN:
        return f"排除未强制执行：{detail}"
    return detail


def _is_plain_latin(text: str) -> bool:
    """True when the query has no Chinese characters to map."""
    return re.search(r"[\u4e00-\u9fff]", text) is None


def _join_unique(parts: list[str]) -> str:
    seen: list[str] = []
    for part in parts:
        text = part.strip()
        if text and text not in seen:
            seen.append(text)
    return " ".join(seen)


def interpret(text: str, *, max_descriptions: int = 3) -> InterpretedQuery:
    """Turn one request string into keywords plus concise English descriptions."""
    original = text.strip()
    if not original:
        return InterpretedQuery(
            keyword_groups=[],
            audio_descriptions=[],
            text_query="",
            dropped_fillers=(),
        )

    atoms = _tokenize(original)
    dropped: list[str] = []
    omitted: list[str] = []
    positives: list[str] = []
    excluded: list[str] = []
    groups: list[list[str]] = []
    current_group: list[str] = []

    def flush_group() -> None:
        nonlocal current_group
        if current_group:
            groups.append(current_group)
            current_group = []

    polarity_negative = False
    uncertain = False
    seen_cue = False
    cue_waiting_for_object = False
    active_cue = ""
    excluded_this_clause = 0

    def close_exclusion() -> None:
        nonlocal polarity_negative, cue_waiting_for_object, active_cue, excluded_this_clause
        polarity_negative = False
        cue_waiting_for_object = False
        active_cue = ""
        excluded_this_clause = 0

    for atom in atoms:
        if atom.kind in {"separator", "boundary"}:
            if atom.kind == "separator":
                if cue_waiting_for_object:
                    uncertain = True
                close_exclusion()
            flush_group()
            continue
        if atom.kind == "negation":
            seen_cue = True
            if polarity_negative or cue_waiting_for_object:
                uncertain = True
            polarity_negative = True
            cue_waiting_for_object = True
            active_cue = atom.cue or atom.raw
            excluded_this_clause = 0
            flush_group()
            continue
        if atom.kind == "only":
            if cue_waiting_for_object:
                uncertain = True
            close_exclusion()
            flush_group()
            continue
        if atom.kind == "filler":
            dropped.append(atom.raw)
            if (
                polarity_negative
                and atom.raw == "的"
                and active_cue in _RELATIVE_EXCLUSION_CUES
                and excluded_this_clause > 0
            ):
                close_exclusion()
                flush_group()
            continue

        # Content-bearing: term, english, or unknown span.
        if atom.kind == "unknown":
            omitted.append(atom.raw)

        if polarity_negative:
            excluded.append(atom.value)
            excluded_this_clause += 1
            cue_waiting_for_object = False
            continue

        positives.append(atom.value)
        current_group.append(atom.value)
        cue_waiting_for_object = False

    if cue_waiting_for_object:
        uncertain = True
    flush_group()

    codes: list[str] = [CODE_RULE_CONVERTED]
    messages: list[str] = [_user_warning(CODE_RULE_CONVERTED, "query-rules-v2，文本路保留原始输入")]

    if omitted:
        codes.append(CODE_PARTIAL_UNCONVERTED)
        joined = "、".join(omitted)
        messages.append(_user_warning(CODE_PARTIAL_UNCONVERTED, f"保留未识别片段「{joined}」"))

    exclusion_scope = "none"
    if uncertain:
        exclusion_scope = "uncertain"
        codes.append(CODE_EXCLUSION_SCOPE_UNCERTAIN)
        messages.append(
            _user_warning(
                CODE_EXCLUSION_SCOPE_UNCERTAIN,
                "排除范围无法可靠判定，已保留原文，未作为数据库硬过滤",
            )
        )
        # Conservative: do not invent a source and do not apply a guessed split.
        # The audio route gets the original wording; BGE already has the original.
        descriptions = [original]
        # Keep content words (including those that might have been excluded) so
        # we do not pretend a hard exclusion happened. Unknown spans stay whole.
        keyword_terms = [
            atom.value for atom in atoms if atom.kind in {"term", "english", "unknown"}
        ]
        groups = [keyword_terms] if keyword_terms else []
        excluded = []
    elif seen_cue:
        exclusion_scope = "recognized_not_enforced"
        codes.append(CODE_EXCLUSION_NOT_ENFORCED)
        if excluded:
            shown = "、".join(dict.fromkeys(excluded))
            messages.append(
                _user_warning(
                    CODE_EXCLUSION_NOT_ENFORCED,
                    f"已识别排除「{shown}」，但未作为数据库硬过滤",
                )
            )
        else:
            messages.append(
                _user_warning(
                    CODE_EXCLUSION_NOT_ENFORCED,
                    "已保留排除意图，但未作为数据库硬过滤",
                )
            )
        description = _join_unique(positives)
        descriptions = [description] if description else []
        # Only-exclusion: no positive source is invented. Do not fall back to
        # the raw string, which would restore excluded objects as a positive.
    elif _is_plain_latin(original):
        # Ordinary English / filenames with no negation: keep wording, order,
        # repeats and punctuation. Only leading/trailing whitespace is normalized.
        descriptions = [original]
    else:
        description = _join_unique(positives)
        descriptions = [description] if description else []
        if not descriptions:
            # All fillers or punctuation: keep the original so nothing is dropped.
            descriptions = [original]

    return InterpretedQuery(
        keyword_groups=groups,
        audio_descriptions=descriptions[:max_descriptions],
        text_query=original,
        dropped_fillers=tuple(dropped),
        warnings=tuple(messages),
        warning_codes=tuple(codes),
        omitted_spans=tuple(omitted),
        excluded_terms=tuple(dict.fromkeys(excluded)),
        exclusion_scope=exclusion_scope,
    )


def build_plan(text: str) -> QueryPlan:
    """A rules-based ``QueryPlan``; ``interpretation='rules'`` states the source."""
    interpreted = interpret(text)
    return plan_from_interpreted(text, interpreted)


def plan_from_interpreted(text: str, interpreted: InterpretedQuery) -> QueryPlan:
    """Build the public plan from an already-run interpretation."""
    return QueryPlan(
        original_text=text,
        mode="natural",
        keyword_groups=interpreted.keyword_groups,
        text_query=interpreted.text_query,
        audio_descriptions=interpreted.audio_descriptions,
        interpretation="rules",
        interpretation_version=RULES_VERSION,
    )


__all__ = [
    "CODE_EXCLUSION_NOT_ENFORCED",
    "CODE_EXCLUSION_SCOPE_UNCERTAIN",
    "CODE_PARTIAL_UNCONVERTED",
    "CODE_RULE_CONVERTED",
    "RULES_VERSION",
    "InterpretedQuery",
    "build_plan",
    "interpret",
    "plan_from_interpreted",
]
