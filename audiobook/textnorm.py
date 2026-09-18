"""Pre-LLM text cleaning.

The chapter text is displayed and marked 1:1, so cleaning must be lossless: paragraphs,
indentation and punctuation are kept. Only glyphs that cannot be spoken at all are dropped,
plus our own mark delimiters (``[`` ``]`` ``<`` ``>``).
"""

from __future__ import annotations

import re
import unicodedata

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


# Our own delimiters: square brackets (legacy vocal-event tags) and angle brackets (marks).
# Everything else's punctuation is kept, including decorative marks and dashes.
_DELIMITER_CHARS = frozenset("[]<>")


def keep_layout(text: str | None) -> str:
    """Pre-LLM cleaning that KEEPS paragraphs, indentation and **all punctuation**.

    Only our own delimiters (``[`` ``]`` ``<`` ``>``) are removed; dashes, quotes, ellipses
    and decorative punctuation stay as written, and paragraph/indentation is untouched (the
    chapters are displayed and marked 1:1; TTS-side normalisation happens at render time).
    ASCII ``...`` is folded into the Chinese ellipsis because it means the same thing.
    """
    out: list[str] = []
    for char in text or "":
        if char in _DELIMITER_CHARS:
            continue
        if char.isspace() or unicodedata.category(char)[0] in _KEEP_CATEGORIES:
            out.append(char)
    cleaned = re.sub(r"\.{2,}", "……", "".join(out))
    return re.sub(r"[ \t]+\n", "\n", cleaned)
