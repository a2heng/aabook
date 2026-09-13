"""Lightweight TTS-text normalisation before synthesis.

AuK handles segments poorly when they end mid-clause, use long dash runs or
repeated ellipses. We normalise those deterministically (the ``raw_text`` is left
untouched for audit).
"""

from __future__ import annotations

import re

_ENDINGS = "。！？…"


def normalize_tts(text: str | None) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.replace("……", "…").replace("——", "，").replace("—", "，")
    text = re.sub(r"…{2,}", "…", text)
    text = re.sub(r"[，、；：]{2,}", "，", text)
    text = re.sub(r"([。！？…])[，、；：]+", r"\1", text)
    text = re.sub(r"[，、；：]+([。！？…])", r"\1", text)
    text = text.strip("，、；：")
    if text and text[-1] not in _ENDINGS:
        text += "。"
    return text
