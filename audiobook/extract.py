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

_SENT_BOUNDARY = set("。！？…；：，、,.!?;:")
_MIN_OPEN = set('“『「"')
_MIN_CLOSE = set('”』」"')

NARRATOR_LABELS = {"旁白", "旁白君", "叙述", "叙述者", "画外音", "narrator", "narration", "neutral"}

NUMBERED_SYSTEM = """【角色表】（只能从下表选说话人，对号入座；旁白不在此列）
__ROSTER__

下面原文已被代码切成最小句并逐句编号，其中**所有引号句（“…”／「…」）都必须出现在你的输出里**。只输出 JSON：
{"speakers":{"2":"高文","3":"赫蒂","7":"旁白"}}

规则：
1. **不能漏任何引号句**：凡是引号内的话都要给一个说话人，取上表名字；别名对齐（姑妈=赫蒂·塞西尔）。
2. 引号句若是**人在说话**（哪怕只有一两个字，如“拜伦！”“姑妈？”），必须写说话人。
3. 引号句若**不是人说话**（书名/术语/引文，如“第一王朝”），写 "旁白"。
4. 旁白叙述、归属/引述短语（“X说道”“X喊道，”）**不用列出**。
5. 说话人一变就分别标；相邻两句不同人也各标各的；无法确定 → 写 "旁白"。

【示例】
编号文本：
1 高文推开门，低声说道：
2 “你来了。”
3 “好久不见。”
4 赫蒂回答。
5 “拜伦！”瑞贝卡突然喊道。
6 他皱眉道：“你也在这儿？”
输出：
{"speakers":{"2":"高文","3":"赫蒂","5":"瑞贝卡","6":"高文"}}
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
    """Clean reader-facing roster for the prompt (name + aliases only)."""
    lines = []
    for role in cast.roles.values():
        if role.kind == "narrator":
            continue
        aliases = "、".join(role.aliases) if role.aliases else ""
        lines.append(f"- {role.name}（别名：{aliases}）" if aliases else f"- {role.name}")
    lines.append("- 旁白（叙述与归属短语）")
    return "\n".join(lines)


def build_numbered_system(cast: Cast) -> str:
    return NUMBERED_SYSTEM.replace("__ROSTER__", roster_block(cast))


def _thinking_default() -> bool:
    return os.environ.get("AUDIOBOOK_EXTRACT_THINKING", "off").lower() in ("1", "on", "true", "yes")


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
    if not text:
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


def _minimal_units(text: str) -> list[dict]:
    """Tile ``text`` into quote-aware minimal units (span + whether inside quotes)."""
    units: list[dict] = []
    start = 0
    inside = False
    for index, char in enumerate(text):
        if char in _MIN_OPEN and not inside:
            if index > start:
                units.append({"start": start, "end": index, "inside": inside})
            start = index
            inside = True
        elif char in _MIN_CLOSE and inside:
            units.append({"start": start, "end": index + 1, "inside": True})
            start = index + 1
            inside = False
        elif char in _SENT_BOUNDARY and not inside:
            units.append({"start": start, "end": index + 1, "inside": inside})
            start = index + 1
    if start < len(text):
        units.append({"start": start, "end": len(text), "inside": inside})
    return [unit for unit in units if text[unit["start"] : unit["end"]].strip()]


def _parse_labels(payload) -> dict[int, str] | None:
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
    labels: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            index = _as_index(key)
            if index is not None:
                labels[index] = str(value).strip()
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                index = _as_index(item.get("i") or item.get("id") or item.get("index"))
                if index is not None:
                    labels[index] = str(item.get("role") or item.get("label") or "旁白").strip()
            elif isinstance(item, str) and item.strip():
                labels[len(labels) + 1] = item.strip()
    return labels


def _numbered_extract(
    client: LLMClient, chapter: Chapter, cast: Cast, system: str, system_hash: str, thinking: bool
) -> list[Unit]:
    units = _minimal_units(chapter.text)
    listing = "\n".join(f"{index} {chapter.text[unit['start'] : unit['end']].strip()}" for index, unit in enumerate(units, 1))
    quoted_ids = [str(index) for index, unit in enumerate(units, 1) if unit["inside"]]
    user = f"编号文本：\n{listing}\n\n需要标注说话人的引号句编号（一个都不能漏）：{', '.join(quoted_ids)}\n\n只输出 JSON。"
    labels: dict[int, str] | None = None
    last_error: Exception | None = None
    for attempt in (thinking, not thinking, thinking):
        try:
            payload = client.chat_json(system, user, thinking=attempt)
        except Exception as error:  # noqa: BLE001 - retry on malformed replies
            last_error = error
            continue
        labels = _parse_labels(payload)
        if labels is not None:
            break
    if labels is None:
        raise RuntimeError(f"numbered extraction failed: {last_error}")
    narrator = cast.narrator()
    result: list[Unit] = []
    for index, unit in enumerate(units, 1):
        span = chapter.text[unit["start"] : unit["end"]]
        label = labels.get(index)
        if not unit["inside"] or (label is not None and label in NARRATOR_LABELS):
            # no quotes -> always narration; or the model marked a quote as non-speech
            _append_unit(result, "narration", narrator.role_id, narrator.name, span, system_hash=system_hash)
            continue
        role = cast.resolve(label) if label else None
        if role is None:
            _append_unit(
                result,
                "dialogue",
                narrator.role_id,
                narrator.name,
                span,
                flags=["unresolved_role"],
                confidence=0.3,
                system_hash=system_hash,
            )
        else:
            _append_unit(result, "dialogue", role.role_id, role.name, span, system_hash=system_hash)
    return result


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
