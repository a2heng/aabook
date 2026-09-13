"""LLM segmentation by explicit ``next`` anchors, plus declaration rewrite.

The agent only does two things (no reordering, no added content):
1. **断句** -- split the source into complete expressions.
2. **改声明** -- rewrite only the attribution/declaration narration (e.g.
   "低声说道" -> "低声自语"). Dialogue is kept verbatim; quotes are dropped.

Boundaries are not guessed from punctuation: every segment carries a ``next``
pointer = the first 6-10 characters of the *next* source segment. We align that
anchor back onto the source to place the cut exactly, so a segment can never
absorb the start of the next one, and the chain is easy to validate. ``kind`` and
``role`` are inherited from extraction (the adaptation agent never re-labels).
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

from .duration import estimate_text_duration
from .extract import Unit, _align, _normalize_with_map, strip_quotes
from .llm import LLMClient, prompt_hash
from .schema import Cast

PROMPT_ID = "prep.pointer.v1"
MAX_UNIT_SECONDS = float(os.environ.get("AUDIOBOOK_MAX_UNIT_SECONDS", "0"))
AGENT_WINDOW_SECONDS = float(os.environ.get("AUDIOBOOK_SEGMENT_WINDOW_SECONDS", "21"))
# Merge adjacent same-role narration up to this STANDARD length (unit 1 = gen_seconds).
NARRATION_MERGE_SECONDS = float(os.environ.get("AUDIOBOOK_NARRATION_MERGE_SECONDS", "24.5"))

_SENTENCE_RE = re.compile(r"[^。！？!?…\n]+[。！？!?…]*")
_CLAUSE_RE = re.compile(r"[^，、；：,;:]+[，、；：,;:]*")

SYSTEM_TEMPLATE = """你是中文有声书"断句 + 声明改编"agent。给你一段原文，你**只做两件事**，每段输出三个字段：

1. 断句：按**完整表达**把原文切成若干段。每段给出：
   - text：该段**原文逐字**（必须与原文完全一致，含标点；用于定位与校验）。
   - next：**下一段原文开头的前 6~10 个字**（用于定位切点；最后一段留空字符串）。
   - speech：该段的朗读文本。
2. 声明改编（**只改 speech**）：
   - 只允许在 speech 中改写"归属/引述"这类**声明**（如 "X说道"→"X低声自语"、"X很严肃地说道"→"X一脸严肃"、"X的声音从身后传来："→"身后传来X的声音："）；
   - 可在气口处加逗号；
   - **speech 的其余文字必须与 text 完全一致**；对白逐字保留；speech 里去掉引号。
   - **不得调序、不得加戏、不得改变人名/数字/剧情**。

只输出 JSON：
{{"segments":[{{"text":"原文逐字","next":"下一段原文开头","speech":"朗读文本"}}, ...]}}

"""

VERIFY_SYSTEM_TEMPLATE = """你是中文有声书校对器。给你原文和一份"断句+声明改编"结果，请检查并修正：
- 每段 text 是否与原文逐字一致；
- next 是否指向下一段原文的正确开头（前 6~10 字）；
- speech 是否只改了声明/气口（对白逐字、不带引号、无加戏/调序/改事实）。

若已很好就原样返回。只输出 JSON：{{"segments":[{{"text":"...","next":"...","speech":"..."}}]}}

"""


def build_system() -> str:
    return SYSTEM_TEMPLATE


def build_verify_system() -> str:
    return VERIFY_SYSTEM_TEMPLATE


def segment_prompt_hash() -> str:
    return prompt_hash(PROMPT_ID, build_system())


def _as_segments(payload: dict | list) -> list[dict]:
    if isinstance(payload, dict):
        items = payload.get("segments") or payload.get("labels") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    segments: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("source") or "").strip()
        speech = str(item.get("speech") or item.get("adapted") or "").strip()
        pointer = str(item.get("next") or item.get("next_start") or "").strip()
        if not text or text.isdigit():
            continue
        segments.append({"text": strip_quotes(text), "speech": strip_quotes(speech), "next": pointer})
    return segments


def _skeleton(text: str) -> list[tuple[str, int]]:
    return [(char, index) for index, char in enumerate(text) if char.isalnum()]


def _segment_spans(unit_text: str, segments: list[dict]) -> list[tuple[int, int]] | None:
    """Map verbatim phrase list back to spans in ``unit_text``; None if characters differ."""
    origin = _skeleton(unit_text)
    pieces = [_skeleton(str(segment.get("text", ""))) for segment in segments]
    if sum(len(piece) for piece in pieces) != len(origin):
        return None
    if "".join(char for piece in pieces for char, _ in piece) != "".join(char for char, _ in origin):
        return None
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            return None
        start = origin[cursor][1]
        next_cursor = cursor + len(piece)
        end = origin[next_cursor][1] if next_cursor < len(origin) else len(unit_text)
        spans.append((start, end))
        cursor = next_cursor
    return spans


def _windows(text: str, max_seconds: float) -> list[tuple[int, int]]:
    def pack(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        start: int | None = None
        end = 0
        for p_start, p_end in spans:
            if start is not None and estimate_text_duration(text[start:p_end]) > max_seconds:
                out.append((start, end))
                start, end = p_start, p_end
            else:
                start = p_start if start is None else start
                end = p_end
        if start is not None:
            out.append((start, end))
        return out

    sentences = [match.span() for match in _SENTENCE_RE.finditer(text) if match.group().strip()]
    if not sentences:
        return [(0, len(text))]
    result: list[tuple[int, int]] = []
    for start, end in pack(sentences):
        if estimate_text_duration(text[start:end]) <= max_seconds:
            result.append((start, end))
            continue
        clauses = [match.span() for match in _CLAUSE_RE.finditer(text[start:end]) if match.group().strip()]
        if not clauses:
            result.append((start, end))
            continue
        result.extend((start + c_start, start + c_end) for c_start, c_end in pack(clauses))
    return result


def _start_index(positions: list[int], from_orig: int) -> int:
    index = 0
    while index < len(positions) and positions[index] < from_orig:
        index += 1
    return index


def _anchor_start(norm: str, positions: list[int], from_orig: int, anchor: str) -> int | None:
    needle, _ = _normalize_with_map(anchor)
    if not needle:
        return None
    span = _align(norm, _start_index(positions, from_orig), needle)
    return positions[span[0]] if span is not None else None


def _segment_unit(unit: Unit, window: str, window_offset: int, segments: list[dict]) -> list[tuple[dict, str, bool]]:
    """Turn agent segments + next pointers into (segment, raw_source_span, anchor_failed)."""
    norm, positions = _normalize_with_map(window)
    boundaries = [0]
    failed = False
    for segment in segments[:-1]:
        start = _anchor_start(norm, positions, boundaries[-1], segment["next"])
        if start is None:
            # fall back to aligning the segment's own text
            own, _ = _normalize_with_map(segment["text"])
            span = _align(norm, _start_index(positions, boundaries[-1]), own)
            if span is not None:
                start = positions[span[1]] if span[1] < len(positions) else len(window)
            failed = True
        boundaries.append(max(start if start is not None else boundaries[-1], boundaries[-1]))
    boundaries.append(len(window))
    out: list[tuple[dict, str, bool]] = []
    for index, segment in enumerate(segments):
        source = unit.tts_text[window_offset + boundaries[index] : window_offset + boundaries[index + 1]]
        out.append((segment, source, failed))
    return out


_SENT_END = "。！？!?…"


def _merge_fragments(pieces: list[tuple[dict, str, bool]]) -> list[tuple[dict, str, bool]]:
    """Merge a fragment into the next piece so every unit is a complete expression.

    Pointer spans are contiguous, so the union of two adjacent raw slices is still
    an exact source span; we only ever widen a unit, never invent text.
    """

    def reading(segment: dict) -> str:
        return segment.get("speech") or segment["text"]

    merged: list[tuple[dict, str, bool]] = []
    for segment, raw, failed in pieces:
        if merged:
            previous, previous_raw, previous_failed = merged[-1]
            if reading(previous) and reading(previous)[-1] not in _SENT_END:
                merged[-1] = (
                    {
                        "text": previous["text"] + segment["text"],
                        "speech": reading(previous) + reading(segment),
                        "next": segment["next"],
                    },
                    previous_raw + raw,
                    previous_failed or failed,
                )
                continue
        merged.append((segment, raw, failed))
    if len(merged) > 1 and reading(merged[-1][0]) and reading(merged[-1][0])[-1] not in _SENT_END:
        last, last_raw, last_failed = merged.pop()
        previous, previous_raw, previous_failed = merged[-1]
        merged[-1] = (
            {
                "text": previous["text"] + last["text"],
                "speech": reading(previous) + reading(last),
                "next": last["next"],
            },
            previous_raw + last_raw,
            previous_failed or last_failed,
        )
    return merged


def _agent_adapt(client: LLMClient, system: str, text: str) -> list[dict]:
    payload = None
    for thinking in (False, True):
        try:
            payload = client.chat_json(system, f"原文：\n{text}\n", thinking=thinking)
            break
        except Exception:  # noqa: BLE001 - best-effort
            continue
    return _as_segments(payload) if payload is not None else []


def _agent_verify(client: LLMClient, system: str, text: str, segments: list[dict]) -> list[dict]:
    listing = "".join(f"{index}. next={segment['next']!r} {segment['text']}\n" for index, segment in enumerate(segments, start=1))
    user = f'原文：\n{text}\n\n结果：\n{listing}\n只输出 {{"segments":[...]}}。'
    for thinking in (False, True):
        try:
            payload = client.chat_json(system, user, thinking=thinking)
        except Exception:  # noqa: BLE001 - keep the first pass on failure
            continue
        verified = _as_segments(payload)
        if verified:
            return verified
    return segments


def _alnum(text: str) -> str:
    return "".join(char for char in text if char.isalnum())


def _build(unit: Unit, segment: dict, raw: str, narrator, system_hash: str, *, anchor_failed: bool = False) -> Unit:
    tts = strip_quotes(segment.get("speech") or segment["text"])
    raw = raw.strip()
    kind = unit.kind
    if kind == "narration":
        role_id, role_name = narrator.role_id, narrator.name
    else:
        role_id, role_name = unit.role_id, unit.role_name
    flags = [*unit.flags, "adapted"]
    if raw and _alnum(segment["text"]) != _alnum(raw):
        flags.append("source_mismatch")
    if raw and tts != raw:
        flags.append("rewritten")
    if kind != "narration" and raw and _alnum(tts) != _alnum(raw):
        flags.append("dialogue_edited")
    if anchor_failed:
        flags.append("anchor_failed")
    return replace(
        unit,
        kind=kind,
        role_id=role_id,
        role_name=role_name,
        raw_text=raw,
        tts_text=tts,
        break_level=segment.get("break", "sentence"),
        flags=flags,
        prompt_hash=unit.prompt_hash or system_hash,
    )


def _merge_narration(units: list[Unit]) -> list[Unit]:
    """Merge adjacent same-role narration up to NARRATION_MERGE_SECONDS (fewer, longer units)."""
    merged: list[Unit] = []
    for unit in units:
        if (
            merged
            and unit.kind == "narration"
            and merged[-1].kind == "narration"
            and merged[-1].role_id == unit.role_id
            and estimate_text_duration(merged[-1].tts_text + unit.tts_text) <= NARRATION_MERGE_SECONDS
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                raw_text=previous.raw_text + unit.raw_text,
                tts_text=previous.tts_text + unit.tts_text,
                break_level=unit.break_level,
            )
            continue
        merged.append(unit)
    return merged


def segment_units(
    units: list[Unit],
    cast: Cast,
    client: LLMClient | None,
    *,
    max_seconds: float = MAX_UNIT_SECONDS,
    verify: bool = False,
) -> list[Unit]:
    """One agent pass per unit: segment by ``next`` anchors and rewrite declarations only."""
    if client is None:
        return units
    system = build_system()
    verify_system = build_verify_system()
    system_hash = segment_prompt_hash()
    narrator = cast.narrator()
    result: list[Unit] = []
    for unit in units:
        if max_seconds > 0 and estimate_text_duration(unit.tts_text) <= max_seconds:
            result.append(unit)
            continue
        pieces: list[tuple[dict, str, bool]] = []
        for start, end in _windows(unit.tts_text, AGENT_WINDOW_SECONDS):
            window = unit.tts_text[start:end]
            segments = _agent_adapt(client, system, window)
            if verify and segments:
                segments = _agent_verify(client, verify_system, window, segments)
            if not segments:
                segments = [{"text": window, "next": ""}]
            pieces.extend(_segment_unit(unit, window, start, segments))
        result.extend(
            _build(unit, segment, raw, narrator, system_hash, anchor_failed=failed)
            for segment, raw, failed in _merge_fragments(pieces)
        )
    return _merge_narration(result)
