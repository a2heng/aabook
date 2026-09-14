"""Numbered labelling extraction (plan-02).

The chapter is split by Python into *quote-aware minimal units* and numbered.
The agent only has to answer "who says each quoted unit" (``{"speakers": {...}}``);
everything else is derived structurally:

* a unit with no quote marks is always narration -- so prose can never be
  swallowed by the previous speaker (the dominant failure of the old approach);
* a quoted unit the agent did not label, or labelled with an unknown name,
  becomes dialogue with ``unresolved_role`` instead of silently disappearing;
* boundaries, merging and source alignment are pure Python, so the script text
  is always a verbatim slice of the novel.
"""

from __future__ import annotations

import difflib
import os
import sys
from dataclasses import dataclass, field

from .cleaning import Chapter
from .llm import LLMClient, prompt_hash
from .schema import Cast

PROMPT_ID = "extract.numbered.v1"

_QUOTE_MAP = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "‘": "'",
        "’": "'",
        "＂": '"',
        "﹃": '"',
        "﹄": '"',
    }
)

_QUOTE_CHARS = set("“”‘’「」『』\"'＂﹃﹄")

# Coarse boundaries: only strong sentence enders split units. Weak punctuation
# (，、；：) is kept inside a unit so quotes stay with their surrounding context
# instead of becoming bare fragments the model mislabels as dialogue.
_SENT_BOUNDARY = set("。！？…")
_MIN_OPEN = set('“『「"')
_MIN_CLOSE = set('”』」"')

NARRATOR_LABELS = {"旁白", "旁白君", "叙述", "叙述者", "画外音", "narrator", "narration", "neutral"}

NUMBERED_SYSTEM = """【角色表】（主要人物 + 固定路人；对白说话人按此对号入座）
__ROSTER__

下面按「章-句」编号（第X章-第Y句，例如 `3-2` 表示第3章第2句）；**每行是一个完整句子**，一次可能给你多章内容。
请对**含引号（“…”／「…」）的句子**，按引号**出现顺序**给出每个引号的说话人（一个数组），只输出 JSON：
{"speakers":{"3-1":["高文"],"3-2":["赫蒂"]}}

规则：
1. 只输出含引号的句子；纯叙述句（不含引号）不用列，代码自动算旁白。
2. 数组长度=该句引号个数，按出现顺序一一对应。**引号里不是当场说的话**（只是被提到的词/称呼/术语）写 "旁白"。
3. 是某人的话就标谁；结合上下文、称谓、身份推断。引号前的描述性称谓（如“贵族少女”“少年”“老者”）要结合上文对上表里对应的人物（用别名/描述判断），不要凭句子里恰好出现的另一个名字乱标。
4. 推断不出的**无名人物**（士兵、路人等）才按性别用固定路人：男声 `路人男1`／`路人男2`，女声 `路人女1`／`路人女2`。**不要留空、不要写“未知”、不要编造新名字**。
5. 人发出的声音都要标（说话、喊叫、惊呼、痛呼、嘟囔……哪怕一两个字）。

【示例】（第3章）
3-1 高文推开门，低声说道：“你来了。”
3-2 “好久不见。”赫蒂回答。
3-3 “拜伦！”瑞贝卡突然喊道。
3-4 瑞贝卡却不知道“老祖宗”脑海里都在想些什么，已经快哭出来了：“祖先大人……对不起……”
3-5 一个陌生士兵喝道：“站住！什么人！”
输出：
{"speakers":{"3-1":["高文"],"3-2":["赫蒂"],"3-3":["瑞贝卡"],"3-4":["旁白","瑞贝卡"],"3-5":["路人男1"]}}

说明：
- 3-3 由“瑞贝卡突然喊道”判断=瑞贝卡。
- 3-4 有两个引号：`“老祖宗”` 只是被提到的称呼、不是台词 → "旁白"；`“祖先大人……”` 是瑞贝卡说的 → "瑞贝卡"。
- 3-5 无名士兵 → `路人男1`。
- 键用原样「章-句」编号，值必须是数组。
"""


@dataclass
class Unit:
    kind: str
    role_id: str
    role_name: str
    raw_text: str
    tts_text: str
    paragraph: int = 0
    emotion: str = ""
    confidence: float = 1.0
    break_level: str = ""
    prompt_id: str = PROMPT_ID
    prompt_hash: str = ""
    flags: list[str] = field(default_factory=list)


def cast_block(cast: Cast) -> str:
    """Full role table (id/kind/aliases) used by the role-resolution agent."""
    lines = []
    for role in cast.roles.values():
        aliases = "/".join(role.aliases) if role.aliases else "-"
        lines.append(f"- {role.name} (id={role.role_id}, kind={role.kind}, aliases={aliases})")
    return "\n".join(lines)


def roster_block(cast: Cast) -> str:
    """Reader-facing roster for the prompt: name + aliases + a short description."""
    lines = []
    for role in cast.roles.values():
        if role.kind == "narrator":
            continue
        aliases = "、".join(role.aliases) if role.aliases else ""
        description = (role.description or "").strip()[:24]
        tags = "；".join(part for part in [f"别名：{aliases}" if aliases else "", description] if part)
        lines.append(f"- {role.name}（{tags}）" if tags else f"- {role.name}")
    lines.append("- 旁白（叙述与归属短语）")
    return "\n".join(lines)


def build_numbered_system(cast: Cast) -> str:
    return NUMBERED_SYSTEM.replace("__ROSTER__", roster_block(cast))


def _thinking_default() -> bool:
    return os.environ.get("AUDIOBOOK_EXTRACT_THINKING", "off").lower() in ("1", "on", "true", "yes")


EXTRACT_MAX_TOKENS = int(os.environ.get("AUDIOBOOK_EXTRACT_MAX_TOKENS", "2048"))


def _as_index(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-stripped, quote-normalized copy plus index back to ``text``."""
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        chars.append(char.translate(_QUOTE_MAP))
        positions.append(index)
    return "".join(chars), positions


def _align(norm: str, cursor: int, needle: str) -> tuple[int, int] | None:
    """Locate ``needle`` in ``norm`` at/after ``cursor``; exact first, fuzzy next.

    The fuzzy path maps the *full* matching span (first match block -> last match
    block) so a mid-segment paraphrase does not truncate the segment.
    """
    if not needle:
        return None
    found = norm.find(needle, cursor)
    if found != -1:
        return found, found + len(needle)
    window_end = min(len(norm), cursor + len(needle) * 2 + 64)
    window = norm[cursor:window_end]
    if not window:
        return None
    matcher = difflib.SequenceMatcher(None, needle, window, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return None
    matched = sum(block.size for block in blocks)
    if matched < max(4, int(len(needle) * 0.6)):
        return None
    start = cursor + blocks[0].b
    end = cursor + blocks[-1].b + blocks[-1].size
    return start, min(len(norm), end)


def strip_quotes(text: str | None) -> str:
    """Drop quote marks from the *spoken* text (role is carried by ``kind`` instead)."""
    return "".join(char for char in (text or "") if char not in _QUOTE_CHARS).strip()


def _has_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


_FEMALE_HINTS = "女她妈姐婆婶娘妃后姑姨妹"


def _passerby_for(label: str, span: str) -> tuple[str, str]:
    """Bucket an out-of-cast speaker into one of the fixed generic passerby roles."""
    text = f"{label}{span}"
    suffix = "f" if any(char in text for char in _FEMALE_HINTS) else "m"
    bucket = sum(ord(char) for char in label) % 2 + 1
    return f"passerby_{suffix}{bucket}", f"路人{'女' if suffix == 'f' else '男'}{bucket}"


def _append_unit(
    units: list[Unit],
    kind: str,
    role_id: str,
    role_name: str,
    text: str,
    *,
    flags: list[str] | None = None,
    confidence: float = 1.0,
    system_hash: str = "",
) -> None:
    text = text.strip()
    if not _has_content(text):
        return
    spoken = strip_quotes(text)
    flags = list(flags or [])
    if units and units[-1].kind == kind and units[-1].role_id == role_id:
        units[-1].raw_text += text
        units[-1].tts_text += spoken
        for flag in flags:
            if flag not in units[-1].flags:
                units[-1].flags.append(flag)
        return
    units.append(
        Unit(
            kind=kind,
            role_id=role_id,
            role_name=role_name,
            raw_text=text,
            tts_text=spoken,
            confidence=confidence,
            prompt_hash=system_hash,
            flags=flags,
        )
    )


def _quote_spans(text: str) -> list[dict]:
    """Quote-aware tiling *within* a sentence (used to split output, not the prompt)."""
    spans: list[dict] = []
    start = 0
    inside = False
    for index, char in enumerate(text):
        if char in _MIN_OPEN and not inside:
            if index > start:
                spans.append({"start": start, "end": index, "inside": False})
            start = index
            inside = True
        elif char in _MIN_CLOSE and inside:
            spans.append({"start": start, "end": index + 1, "inside": True})
            start = index + 1
            inside = False
    if start < len(text):
        spans.append({"start": start, "end": len(text), "inside": inside})
    return [span for span in spans if _has_content(text[span["start"] : span["end"]])]


def _sentence_units(text: str) -> list[dict]:
    """Split into *sentences* at strong enders only; quotes are never boundaries.

    A quoted sentence, its attribution and any trailing narration stay in one unit
    so the model sees the full context instead of a bare quote fragment.
    """
    units: list[dict] = []
    start = 0
    inside = False
    pending = False  # a strong ender was seen inside the current quote
    index = 0
    while index < len(text):
        char = text[index]
        if char in _MIN_OPEN:
            inside = True
        elif char in _MIN_CLOSE:
            if inside:
                inside = False
                if pending:  # the quote itself ended the sentence -> close the unit here
                    units.append({"start": start, "end": index + 1})
                    start = index + 1
                    pending = False
        elif char in _SENT_BOUNDARY:
            if inside:
                pending = True
            else:
                end = index + 1
                while end < len(text) and text[end] in _MIN_CLOSE:  # keep the closing quote
                    end += 1
                units.append({"start": start, "end": end})
                start = end
                index = end - 1
        index += 1
    if start < len(text):
        units.append({"start": start, "end": len(text)})
    result = []
    for unit in units:
        span = text[unit["start"] : unit["end"]]
        if not _has_content(span):
            continue
        result.append({"start": unit["start"], "end": unit["end"], "has_quote": any(s["inside"] for s in _quote_spans(span))})
    return result


def _parse_labels(payload) -> dict[str, str | list[str]] | None:
    if isinstance(payload, dict):
        raw = None
        for key in ("speakers", "labels", "roles"):
            if key in payload:
                raw = payload[key]
                break
    elif isinstance(payload, list):
        raw = payload
    else:
        return None
    if raw is None:
        return None
    labels: dict[str, str | list[str]] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            labels[str(key).strip()] = _normalize_label(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = item.get("id") or item.get("i") or item.get("index")
                if key is not None:
                    labels[str(key).strip()] = _normalize_label(item.get("role") or item.get("label") or "旁白")
    return labels


def _normalize_label(value) -> str | list[str]:
    """A sentence's label is either one role (str) or one role per quote (list)."""
    if isinstance(value, list):
        return [str(item).strip() for item in value]
    return str(value).strip()


def _looks_non_speech(span: str) -> bool:
    """A quoted span is treated as narration only if it does not look like an utterance.

    Exclamations/questions (ending in ！？… ) are always speech, so a model that
    labels them "旁白" cannot silence them into narration.
    """
    return not span.rstrip().rstrip("”』」\"'").endswith(("！", "？", "…"))


def _label_window(
    client: LLMClient, system: str, entries: list[dict], start: int, size: int, thinking: bool
) -> dict[str, str | list[str]]:
    """Label one window of numbered entries; ids are ``chapter-sentence`` (e.g. 3-2)."""
    window = entries[start : start + size]
    lines: list[str] = []
    for entry in window:
        if entry.get("header"):
            lines.append(entry["header"])
        lines.append(f"{entry['uid']} {entry['text']}")
    quoted_ids = [entry["uid"] for entry in window if entry["inside"]]
    user = (
        f"编号文本：\n{chr(10).join(lines)}\n\n"
        f"需要标注说话人的引号句编号（一个都不能漏）：{', '.join(quoted_ids)}\n\n只输出 JSON。"
    )
    last_error: Exception | None = None
    for attempt in (thinking, not thinking, thinking):
        try:
            payload = client.chat_json(system, user, max_tokens=EXTRACT_MAX_TOKENS, thinking=attempt)
        except Exception as error:  # noqa: BLE001 - retry on malformed replies
            last_error = error
            continue
        labels = _parse_labels(payload)
        if labels is not None:
            return labels
    print(
        f"[extract] WARN ids {window[0]['uid']}..{window[-1]['uid']} -> no labels: {last_error}",
        file=sys.stderr,
    )
    return {}


def _label_range(
    client: LLMClient, system: str, entries: list[dict], start: int, size: int, thinking: bool
) -> dict[str, str | list[str]]:
    """Label a range; on failure split it in half so only the bad sub-range degrades."""
    size = min(size, len(entries) - start)  # clamp: never recurse on a phantom length
    if size <= 0:
        return {}
    labels = _label_window(client, system, entries, start, size, thinking)
    if labels or size <= 24:
        return labels
    mid = start + size // 2
    left = _label_range(client, system, entries, start, mid - start, thinking)
    right = _label_range(client, system, entries, mid, start + size - mid, thinking)
    return {**left, **right}


def _label_entries(client: LLMClient, system: str, entries: list[dict], thinking: bool) -> dict[str, str | list[str]]:
    window_size = int(os.environ.get("AUDIOBOOK_EXTRACT_MAX_UNITS", "100"))
    labels: dict[str, str | list[str]] = {}
    for start in range(0, len(entries), window_size):
        labels.update(_label_range(client, system, entries, start, window_size, thinking))
    return labels


def _entry_for(span: str, inside: bool, uid: str, *, header: str = "") -> dict:
    return {"uid": uid, "text": span.strip(), "inside": inside, "header": header}


def _classify_units(
    chapter: Chapter, units: list[dict], labels: dict[str, str | list[str]], cast: Cast, system_hash: str
) -> list[Unit]:
    """Turn per-sentence labels into source-aligned ``Unit`` rows.

    A prompt unit is a whole sentence; the model labels the speaker of the quote
    inside it. Here the sentence is split back into narration vs quoted dialogue.
    """
    narrator = cast.narrator()
    result: list[Unit] = []
    for index, unit in enumerate(units, 1):
        label = labels.get(f"{chapter.chapter_id}-{index}")
        sentence = chapter.text[unit["start"] : unit["end"]]
        quote_index = 0
        for sub in _quote_spans(sentence):
            span = chapter.text[unit["start"] + sub["start"] : unit["start"] + sub["end"]]
            if not sub["inside"]:
                _append_unit(result, "narration", narrator.role_id, narrator.name, span, system_hash=system_hash)
                continue
            # per-quote label (list, in quote order) or a single label for the sentence
            if isinstance(label, list):
                sub_label = label[quote_index] if quote_index < len(label) else ""
            else:
                sub_label = label or ""
            quote_index += 1
            marked_narration = sub_label.strip() in NARRATOR_LABELS
            if marked_narration and _looks_non_speech(span):
                # genuine non-speech quote (term/title) -> narration
                _append_unit(result, "narration", narrator.role_id, narrator.name, span, system_hash=system_hash)
                continue
            role = None if marked_narration else cast.resolve(sub_label)
            if role is not None:
                _append_unit(result, "dialogue", role.role_id, role.name, span, system_hash=system_hash)
            else:
                # out of the main cast -> fixed generic passerby (never invent a name)
                role_id, role_name = _passerby_for(sub_label, span)
                _append_unit(
                    result,
                    "dialogue",
                    role_id,
                    role_name,
                    span,
                    flags=["passerby"],
                    confidence=0.6,
                    system_hash=system_hash,
                )
    return result


def _numbered_extract(
    client: LLMClient, chapter: Chapter, cast: Cast, system: str, system_hash: str, thinking: bool
) -> list[Unit]:
    units = _sentence_units(chapter.text)
    entries = [
        _entry_for(chapter.text[u["start"] : u["end"]], u["has_quote"], f"{chapter.chapter_id}-{index}")
        for index, u in enumerate(units, 1)
    ]
    labels = _label_entries(client, system, entries, thinking)
    return _classify_units(chapter, units, labels, cast, system_hash)


def extract_chapters(chapters: list[Chapter], cast: Cast, client: LLMClient | None) -> list[list[Unit]]:
    """Extract several chapters in one call (or a few windows), sharing one context."""
    if client is None:
        return [extract_chapter(chapter, cast, None) for chapter in chapters]
    system = build_numbered_system(cast)
    system_hash = prompt_hash(PROMPT_ID, system)
    thinking = _thinking_default()
    all_units: list[list[dict]] = []
    entries: list[dict] = []
    for chapter in chapters:
        units = _sentence_units(chapter.text)
        all_units.append(units)
        for unit_index, unit in enumerate(units, 1):
            entries.append(
                _entry_for(
                    chapter.text[unit["start"] : unit["end"]],
                    unit["has_quote"],
                    f"{chapter.chapter_id}-{unit_index}",
                    header=f"【第{chapter.chapter_id}章】" if unit_index == 1 else "",
                )
            )
    labels = _label_entries(client, system, entries, thinking)
    return [_classify_units(chapter, all_units[index], labels, cast, system_hash) for index, chapter in enumerate(chapters)]


def extract_chapter(chapter: Chapter, cast: Cast, client: LLMClient | None) -> list[Unit]:
    if client is None:
        narrator = cast.narrator()
        return [
            Unit(
                kind="narration",
                role_id=narrator.role_id,
                role_name=narrator.name,
                raw_text=chapter.text,
                tts_text=chapter.text,
                confidence=1.0,
            )
        ]
    system = build_numbered_system(cast)
    return _numbered_extract(client, chapter, cast, system, prompt_hash(PROMPT_ID, system), _thinking_default())
