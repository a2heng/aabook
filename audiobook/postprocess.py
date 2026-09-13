"""Programmatic post-processing on built script rows.

Conservative narration merge: join adjacent same-role narration rows (a pure
narration run -- dialogue splits are left alone) when the synthesised span still
fits the TTS cap.

Time convention: ``target_duration_s`` is already the STANDARD time (the 0.7 is
internalised in ``estimate_text_duration``), so it is used directly and the cap is
28 standard seconds.
"""

from __future__ import annotations

import os

TTS_MAX_SECONDS = float(os.environ.get("AUDIOBOOK_TTS_MAX_SECONDS", "28"))

# Flags that mean "don't merge this row" (the text/attribution is uncertain).
BLOCK_MERGE_FLAGS = {"source_mismatch", "anchor_failed", "over_length", "unresolved_role"}


def standard_seconds(row: dict) -> float:
    """Standard timeline duration for a row (unit 1, already includes the 0.7)."""
    return float(row.get("target_duration_s") or 0)


def merge_adjacent_narration(
    rows: list[dict],
    *,
    tts_max_seconds: float = TTS_MAX_SECONDS,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Merge adjacent same-role narration rows while the standard span <= cap.

    Conservative on purpose: only narration, same chapter + role_id, only unflagged
    rows, and only when the whole merged span is short enough to synthesise.
    """
    out: list[dict] = []
    merged: list[tuple[str, str]] = []
    for row in rows:
        previous = out[-1] if out else None
        if previous is not None and _mergeable(previous, row, tts_max_seconds):
            _absorb(previous, row)
            merged.append((previous["seg_id"], row["seg_id"]))
            continue
        out.append(dict(row))
    return out, merged


def _flags(row: dict) -> set[str]:
    value = row.get("flags") or ""
    return {flag for flag in value.split(";") if flag} if isinstance(value, str) else set(value)


def _estimated(row: dict) -> float:
    return float(row.get("target_duration_s") or 0)


def _mergeable(previous: dict, row: dict, tts_max_seconds: float) -> bool:
    if previous.get("kind") != "narration" or row.get("kind") != "narration":
        return False
    if previous.get("chapter_id") != row.get("chapter_id") or previous.get("role_id") != row.get("role_id"):
        return False
    if _flags(previous) & BLOCK_MERGE_FLAGS or _flags(row) & BLOCK_MERGE_FLAGS:
        return False
    return _estimated(previous) + _estimated(row) <= tts_max_seconds


def _absorb(previous: dict, row: dict) -> None:
    previous["tts_text"] = (previous.get("tts_text") or "") + (row.get("tts_text") or "")
    previous["raw_text"] = (previous.get("raw_text") or "") + (row.get("raw_text") or "")
    previous["target_duration_s"] = round(_estimated(previous) + _estimated(row), 3)
    previous["flags"] = ";".join(sorted(_flags(previous) | _flags(row)))
