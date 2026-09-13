"""Single-pass structured extraction (Pass 1) per chapter.

Label-based (variant C): Python splits the chapter into numbered sentences, the
LLM only labels each sentence with ``kind``/``role``, and Python slices the
original text back out. This keeps the text bit-exact (no re-typing/truncation)
and makes the model's output tiny.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .cleaning import Chapter
from .llm import LLMClient, prompt_hash
from .schema import KINDS, Cast

PROMPT_ID = "extract.labels.v3"

_STRONG = set("。！？!?…")
_OPEN = set("“『「《【")
_CLOSE = set("”』」》】")
_PUNCT_ONLY = set("。！？!?…，、；：,;:—～~-·")
_INLINE_MAX = 20
_SPEECH_TAIL = ("：", ":", "说", "道", "问", "喊", "答", "叫")

SYSTEM_TEMPLATE = """你是中文小说角色标注器。正文已按"句段"编号：每个句段要么是引号外的叙述，要么是引号内的对白。
请为每个句段标注 kind 和 role，只输出 JSON：
{"labels":[{"i":1,"kind":"narration|dialogue|monologue","role":"角色名"}]}

判定规则：
- 引号内（“…”）的内容是 dialogue；引号外一般是 narration。
- 归属词是关键：对白紧邻"X说/道/问/喊/答/补充道"等时，role=X。归属词可能在对白前、对白后，或另一句段里：
  * 归属词在后一段（如 对白“我推测是某种恶魔的亚种，” / 下一段 赫蒂说道）：对白 role=赫蒂。
  * 归属词在同一段开头（如 赫蒂说道，“……”）：对白 role=赫蒂。
  * 整段只有引号内容时，依据上文最近出现的说话人归属。
- 无引号但明显是人物在说话（口语、称呼、语气词）也标 dialogue。
- 内心独白（心想/暗想/觉得/心理活动）标 monologue，role=该人物。
- 群像或无法确定说话人时，role=旁白。
- "X说道，"这类只含叙述和归属词的句段是 narration/旁白。
- 重要：凡 kind=narration 的句段，role 一律填"旁白"，不要填角色名（哪怕主语是某角色）。
- monologue 只用于第一人称内心活动；第三人称叙述用 narration。
- role 只能取角色表里的名字，旁白用"旁白"。
- 只输出编号标签，不要复述正文；每个编号都要有标签。

【角色表】
__CAST__
"""

USER_TEMPLATE = "章节标题：{title}\n\n编号正文：\n{numbered}\n"


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
    prompt_id: str = PROMPT_ID
    prompt_hash: str = ""
    flags: list[str] = field(default_factory=list)


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Return ``(paragraph_index, span)`` pairs, split on quote boundaries.

    Each span is either pure narration (quote depth 0) or pure dialogue (inside
    quotes), so voice assignment never mixes the two. Weak punctuation inside a
    quote never splits the quote.
    """
    sentences: list[tuple[int, str]] = []
    for para_index, paragraph in enumerate(text.split("\n")):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        buffer = ""
        depth = 0
        quote_at: int | None = None
        spans: list[str] = []

        def emit(text: str) -> None:
            span = text.strip()
            if span:
                spans.append(span)

        for char in paragraph:
            if char in _OPEN:
                if depth == 0:
                    quote_at = len(buffer)
                depth += 1
                buffer += char
            elif char in _CLOSE:
                buffer += char
                depth = max(0, depth - 1)
                if depth == 0 and quote_at is not None:
                    prefix, quoted = buffer[:quote_at], buffer[quote_at:]
                    core = quoted.strip("".join(_OPEN) + "".join(_CLOSE))
                    opening = prefix.strip()
                    inline = (
                        bool(opening)
                        and not opening.endswith(_SPEECH_TAIL)
                        and not opening.endswith("，")
                        and not opening.endswith(",")
                        and len(core) <= _INLINE_MAX
                        and (not core or core[-1] not in _STRONG)
                    )
                    if inline:
                        buffer = prefix + quoted
                    else:
                        emit(prefix)
                        emit(quoted)
                        buffer = ""
                    quote_at = None
            else:
                buffer += char
                if char in _STRONG and depth == 0:
                    emit(buffer)
                    buffer = ""
        emit(buffer)

        for span in spans:
            if span and all(ch in _PUNCT_ONLY for ch in span) and sentences and sentences[-1][0] == para_index:
                sentences[-1] = (para_index, sentences[-1][1] + span)
            else:
                sentences.append((para_index, span))
    return sentences


def cast_block(cast: Cast) -> str:
    lines = []
    for role in cast.roles.values():
        aliases = "/".join(role.aliases) if role.aliases else "-"
        lines.append(f"- {role.name} (id={role.role_id}, kind={role.kind}, aliases={aliases})")
    return "\n".join(lines)


def build_system(cast: Cast) -> str:
    return SYSTEM_TEMPLATE.replace("__CAST__", cast_block(cast))


def extract_prompt_hash(cast: Cast) -> str:
    return prompt_hash(PROMPT_ID, build_system(cast))


def _as_label_map(payload: dict | list) -> dict[int, dict]:
    items = payload if isinstance(payload, list) else payload.get("labels", []) if isinstance(payload, dict) else []
    labels: dict[int, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            labels[int(item["i"])] = item
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def _group_units(sentences: list[tuple[int, str]], labels: dict[int, dict], cast: Cast, system_hash: str) -> list[Unit]:
    units: list[Unit] = []
    current: Unit | None = None
    narrator = cast.narrator()
    for index, (paragraph, sentence) in enumerate(sentences, start=1):
        label = labels.get(index, {})
        kind = str(label.get("kind") or "narration").strip().lower()
        if kind not in KINDS:
            kind = "narration"
        label_role = str(label.get("role") or "").strip()
        role = cast.resolve(label_role)
        if kind == "narration":
            role = narrator
        role_id = role.role_id if role else ""
        role_name = role.name if role else label_role
        if current is not None and current.kind == kind and current.role_id == role_id and current.paragraph == paragraph:
            current.raw_text += sentence
            current.tts_text += sentence
            continue
        current = Unit(
            kind=kind,
            role_id=role_id,
            role_name=role_name,
            raw_text=sentence,
            tts_text=sentence,
            paragraph=paragraph,
            prompt_hash=system_hash,
        )
        if role is None:
            current.flags.append("unresolved_role")
            current.confidence = 0.2
        if index not in labels:
            current.flags.append("unlabeled")
            current.confidence = min(current.confidence, 0.3)
        units.append(current)
    return units


_ATTR_RE = re.compile(
    r"(?:说道|问道|喊道|答道|叫道|补充道|笑道|低声道|开口道|解释道|回应道|说|问|喊|答|(?<![知味通行街报大有难道])道)[，,：:。]?$"
)
_GROUP_RE = re.compile(
    r"(?:所有人|众人|大家|人们|他们|她们|士兵们|骑士们|战士们|将领们|两人|三人|四人|一群人|一众人)[：:,，。]?$"
)


def _earliest_role(text: str, cast: Cast):
    best = None
    best_pos = -1
    for role in cast.roles.values():
        for name in (role.name, *role.aliases):
            if name and (pos := text.find(name)) != -1 and (best_pos == -1 or pos < best_pos):
                best_pos, best = pos, role
    return best


def _speaker_signal(text: str, cast: Cast, *, colon: bool, comma: bool):
    """Return ``(role, strong)`` for an attribution clause, or ``None``.

    ``role`` is the narrator for group clauses ("众人"/"所有人…"). Strong
    signals (colon introduces speech, explicit 说/道 verb) override any existing
    label; weak signals (trailing comma) only fill in narrator rows.
    """
    text = text.strip()
    if not text:
        return None
    if colon and text.endswith(("：", ":")):
        strong = True
    elif _ATTR_RE.search(text):
        strong = True
    elif comma and text.endswith(("，", ",")):
        strong = False
    else:
        return None
    if _GROUP_RE.search(text):
        return cast.narrator(), strong
    role = _earliest_role(text, cast)
    return (role, strong) if role is not None else None


def _repair_speakers(units: list[Unit], cast: Cast) -> None:
    narrator = cast.narrator()
    for index, unit in enumerate(units):
        if unit.kind != "dialogue":
            continue
        neighbours = []
        if index > 0 and units[index - 1].kind == "narration":
            neighbours.append((units[index - 1].tts_text, True))
        if index + 1 < len(units) and units[index + 1].kind == "narration":
            neighbours.append((units[index + 1].tts_text, False))
        chosen = None
        for text, is_previous in neighbours:
            signal = _speaker_signal(text, cast, colon=is_previous, comma=is_previous)
            if signal is None:
                continue
            role, strong = signal
            if strong or unit.role_id == narrator.role_id:
                chosen = role
                break
        if chosen is not None:
            unit.role_id = chosen.role_id
            unit.role_name = chosen.name
            if "role_repaired" not in unit.flags:
                unit.flags.append("role_repaired")


def extract_chapter(chapter: Chapter, cast: Cast, client: LLMClient | None) -> list[Unit]:
    sentences = split_sentences(chapter.text)
    if not sentences:
        return []
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

    numbered = "\n".join(f"{index}. {sentence}" for index, (_para, sentence) in enumerate(sentences, start=1))
    system = build_system(cast)
    system_hash = extract_prompt_hash(cast)
    user = USER_TEMPLATE.format(title=chapter.title, numbered=numbered)
    thinking = os.environ.get("AUDIOBOOK_EXTRACT_THINKING", "off").lower() in ("1", "on", "true", "yes")
    last_error: Exception | None = None
    payload: dict | list | None = None
    for attempt_thinking in (thinking, not thinking, thinking):
        try:
            payload = client.chat_json(system, user, thinking=attempt_thinking)
            break
        except Exception as error:  # noqa: BLE001 - retry on any malformed reply
            last_error = error
    if payload is None:
        raise RuntimeError(f"extraction failed after retries: {last_error}")
    units = _group_units(sentences, _as_label_map(payload), cast, system_hash)
    _repair_speakers(units, cast)
    return units
