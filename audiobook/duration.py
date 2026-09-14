"""Duration estimation and punctuation-aware segmentation.

The estimator mirrors ``app/infer_gradio.py`` (``estimate_text_duration``) so the
front-end computes the exact ``gen_seconds`` the renderer will use. Keep the
constants in sync if that module ever changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SECONDS_PER_CJK_CHAR = 0.22
SECONDS_PER_LATIN_WORD = 0.40
SECONDS_PER_STRONG_PAUSE = 0.30
SECONDS_PER_WEAK_PAUSE = 0.12

# The renderer used to apply this scale (``duration_rate``) on top of the estimate.
# It is now internalised: ``estimate_text_duration`` returns the STANDARD time that
# goes straight to AuK as ``gen_seconds`` (unit 1, no further scaling anywhere).
STANDARD_RATE = 0.7

# Loosen the estimate so long/degraded rows are less likely to be cut:
# every character/word gets +5%, and short utterances get an extra +10%.
CHAR_BOOST = 1.05
SHORT_SENTENCE_BOOST = 1.10
SHORT_SENTENCE_MAX_CHARS = 12

# Tightened cap: rows longer than this (standard seconds ~= <=120 syllables) showed
# heavy ASR coverage loss, so split earlier.
MAX_SEGMENT_SECONDS = 20.0
# Derived caps (~20 s / (0.22 * 1.05 * 0.7) ~= 123 zh chars), with margin.
MAX_SEG_CHARS_ZH = 120
MAX_SEG_WORDS_EN = 70

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
_STRONG_PAUSE_RE = re.compile(r"[。！？!?…]+")
_WEAK_PAUSE_RE = re.compile(r"[，、；：,;:]+")
_STRONG_END_RE = re.compile(r"[。！？!?…][”』」\"'）》】]*$")
_WEAK_END_RE = re.compile(r"[，、；：,;:](?=[”』」\"'）》】]*$)")
_SENTENCE_RE = re.compile(r"[^。！？!?…\n]+[。！？!?…]*")
_CLAUSE_RE = re.compile(r"[^，、；：,;:]+[，、；：,;:]*")


@dataclass
class Segment:
    text: str
    punct_edited: bool = False


def estimate_text_duration(text: str | None) -> float:
    text = text or ""
    cjk = len(_CJK_CHAR_RE.findall(text))
    words = len(_LATIN_WORD_RE.findall(text))
    strong = len(_STRONG_PAUSE_RE.findall(text))
    weak = len(_WEAK_PAUSE_RE.findall(text))
    speech = (cjk * SECONDS_PER_CJK_CHAR + words * SECONDS_PER_LATIN_WORD) * CHAR_BOOST
    pauses = strong * SECONDS_PER_STRONG_PAUSE + weak * SECONDS_PER_WEAK_PAUSE
    duration = (speech + pauses) * STANDARD_RATE
    if cjk + words <= SHORT_SENTENCE_MAX_CHARS:
        duration *= SHORT_SENTENCE_BOOST
    return duration


def _greedy(pieces: list[str], max_seconds: float) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{piece}" if current else piece
        if current and estimate_text_duration(candidate) > max_seconds:
            chunks.append(current)
            current = piece
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_oversized(chunk: str, max_seconds: float) -> list[str]:
    clauses = [part for part in _CLAUSE_RE.findall(chunk) if part.strip()] or [chunk]
    pieces: list[str] = []
    for merged in _greedy(clauses, max_seconds):
        if estimate_text_duration(merged) <= max_seconds:
            pieces.append(merged)
        else:
            budget = max(8, int(max_seconds / (SECONDS_PER_CJK_CHAR * CHAR_BOOST * STANDARD_RATE)))
            pieces.extend(merged[index : index + budget] for index in range(0, len(merged), budget))
    return pieces


def _split_units(text: str, max_seconds: float) -> list[str]:
    sentences = [part for part in _SENTENCE_RE.findall(text) if part.strip()] or [text]
    units: list[str] = []
    for chunk in _greedy(sentences, max_seconds):
        if estimate_text_duration(chunk) <= max_seconds:
            units.append(chunk)
        else:
            units.extend(_split_oversized(chunk, max_seconds))
    return units


def segment_tts_text(text: str | None, max_seconds: float = MAX_SEGMENT_SECONDS) -> list[Segment]:
    """Split a unit of TTS text into <= ``max_seconds`` segments (split only, never merge)."""
    text = (text or "").strip()
    if not text:
        return []
    if estimate_text_duration(text) <= max_seconds:
        return [Segment(text, False)]

    segments: list[Segment] = []
    for unit in _split_units(text, max_seconds):
        unit = unit.strip()
        if not unit or not any(char.isalnum() for char in unit):
            continue  # drop punctuation-only fragments (e.g. a lone "。")
        if _WEAK_END_RE.search(unit):
            segments.append(Segment(_WEAK_END_RE.sub("。", unit), True))
        else:
            segments.append(Segment(unit, False))
    return segments or [Segment(text, False)]


def exceeds_cap(text: str | None) -> bool:
    """Fast gate using the derived char/word caps before the exact estimator."""
    text = text or ""
    cjk = len(_CJK_CHAR_RE.findall(text))
    words = len(_LATIN_WORD_RE.findall(text))
    return cjk > MAX_SEG_CHARS_ZH or words > MAX_SEG_WORDS_EN
