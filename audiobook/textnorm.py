"""Lightweight TTS-text normalisation before synthesis.

The renderer handles segments poorly when they end mid-clause, so use long dash runs or
repeated ellipses. We normalise those deterministically (the ``raw_text`` is left
untouched for audit).
"""

from __future__ import annotations

import re
import unicodedata

_ENDINGS = "。！？…"

# Letters, numbers, punctuation and marks in every script; whitespace is kept too.
# Everything else (emoji, box-drawing, math/currency/symbol glyphs, control chars)
# is not spoken by TTS and is dropped.
_KEEP_CATEGORIES = frozenset("LNPM")
# Decorative reference marks are punctuation by category but are never spoken.
_DROP_CHARS = frozenset("※§¶†‡•‣‧‰′″‹›")


def filter_tts_chars(text: str | None) -> str:
    """Keep only characters a TTS front-end can pronounce (letters/digits/punct)."""
    text = text or ""
    return "".join(
        char for char in text if char not in _DROP_CHARS and (char.isspace() or unicodedata.category(char)[0] in _KEEP_CATEGORIES)
    )


def normalize_tts(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace("——", "，").replace("—", "，")
    text = re.sub(r"\.{2,}", "……", text)  # ASCII dots -> Chinese ellipsis; ellipsis itself is kept
    text = re.sub(r"[，、；：]{2,}", "，", text)
    text = re.sub(r"([。！？…])[，、；：]+", r"\1", text)
    text = re.sub(r"[，、；：]+([。！？…])", r"\1", text)
    text = text.strip("，、；：")
    if text and text[-1] not in _ENDINGS:
        text += "。"
    return text


def clean_for_llm(text: str | None) -> str:
    """Persistent pre-LLM cleaning: drop non-speech glyphs, then normalise punctuation."""
    return normalize_tts(filter_tts_chars(text))


_UNWANTED_RE = re.compile(r"·{2,}|—{2,}|-{2,}|＊+|#{2,}")
_WS_RE = re.compile(r"[\s\u3000]+")


def one_paragraph(text: str | None) -> str:
    """Code-side preprocessing done ONCE, before the LLM: collapse the chapter to a
    single paragraph, turn ASCII ellipsis / dashes / decorative repeats into a comma
    (Chinese ellipsis ``……`` is kept), and drop unwanted whitespace."""
    text = re.sub(r"\.{2,}", "……", text or "")
    text = _UNWANTED_RE.sub("，", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"，{2,}", "，", text)
    return _WS_RE.sub("", text)
