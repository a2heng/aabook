"""The marker convention for speech (rare symbols, so plain text never collides).

Speech is wrapped as ``⦃角色名␟朗读内容⦄``; everything outside the markers is narration.
- ``⦃`` U+2983 (open) / ``⦄`` U+2984 (close): rare "white curly brackets".
- ``␟`` U+241F (unit separator): role/content delimiter.

The agent (MCP ``mark_speaker``) produces this text; AuK parses it *mechanically* --
no LLM -- into narration/speech rows and maps each role to its fixed voice.
Override any symbol via env (``AUDIOBOOK_MARK_OPEN/CLOSE/SEP``).
"""

from __future__ import annotations

import os
import re

from .duration import estimate_text_duration
from .instructions import render_instruction
from .schema import Cast, ScriptRow

MARK_OPEN = os.environ.get("AUDIOBOOK_MARK_OPEN", "\u2983")  # ⦃
MARK_CLOSE = os.environ.get("AUDIOBOOK_MARK_CLOSE", "\u2984")  # ⦄
MARK_SEP = os.environ.get("AUDIOBOOK_MARK_SEP", "\u241f")  # ␟

MARKS_RE = re.compile(
    re.escape(MARK_OPEN)
    + r"([^"
    + re.escape(MARK_SEP + MARK_CLOSE)
    + r"]+)"
    + re.escape(MARK_SEP)
    + r"([^"
    + re.escape(MARK_CLOSE)
    + r"]*)"
    + re.escape(MARK_CLOSE)
)
NARRATOR = "旁白"


def _has_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


def parse_marks(text: str) -> list[dict]:
    """Split marked text into ``[{kind, role_name, text}]`` (narration = unmarked)."""
    segments: list[dict] = []
    pos = 0
    for match in MARKS_RE.finditer(text):
        narration = text[pos : match.start()].strip()
        if _has_content(narration):
            segments.append({"kind": "narration", "role_name": NARRATOR, "text": narration})
        content = match.group(2).strip()
        if _has_content(content):
            segments.append({"kind": "speech", "role_name": match.group(1).strip(), "text": content})
        pos = match.end()
    tail = text[pos:].strip()
    if _has_content(tail):
        segments.append({"kind": "narration", "role_name": NARRATOR, "text": tail})
    # Long speech is marked in several `speak` calls; stitch adjacent same-role pieces.
    merged: list[dict] = []
    for segment in segments:
        last = merged[-1] if merged else None
        if last and last["kind"] == "speech" and segment["kind"] == "speech" and last["role_name"] == segment["role_name"]:
            last["text"] += segment["text"]
        else:
            merged.append(dict(segment))
    return merged


_BARE_QUOTE_RE = re.compile(r"[“『「][^”』」\n]{1,80}?[”』」]")


def unmarked_quotes(marked: str) -> list[str]:
    """Quoted spans still *outside* any ⦃…⦄ mark (L3 leftovers / very short lines)."""
    stripped = MARKS_RE.sub("", marked)
    return [match.group(0) for match in _BARE_QUOTE_RE.finditer(stripped)]


def render_html(segments: list[dict], marked: str, path) -> None:
    """Write a two-part QA page: parsed segments table + the raw marked text."""
    import html

    rows = "".join(
        f"<tr class='{seg['kind']}'><td>{'speech' if seg['kind'] == 'speech' else 'narration'}</td>"
        f"<td>{html.escape(seg['role_name'])}</td><td>{html.escape(seg['text'])}</td></tr>"
        for seg in segments
    )
    path.write_text(
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>台本标注</title>"
        "<style>body{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:16px}"
        "table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:4px 8px;vertical-align:top}"
        "th{background:#f0f0f0}.speech{background:#e8f1ff}.narration{background:#fff}"
        "pre{white-space:pre-wrap;background:#fff;border:1px solid #eee;padding:8px}</style></head><body>"
        f"<h2>标注结果（{len(segments)} 段）</h2><table><thead><tr><th>类型</th><th>说话人</th><th>文本</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>标记文本（原始）</h2><pre>{html.escape(marked)}</pre></body></html>",
        encoding="utf-8",
    )


def live_fragments(original: str, current: str) -> list[dict]:
    """Inline display fragments for the live canvas (continuous text + overlays).

    Diff original vs current so every change is visible in place:
    - ``speech``    ⦃role␟…⦄ content -> role card (marker syntax hidden);
    - ``deleted``   characters gone from the original (struck through);
    - ``inserted``  new characters (e.g. a comma at a breath point);
    - ``narration`` untouched text.
    """
    import difflib

    # Strip the marker syntax from the current text first, remembering which regions are
    # speech. Diffing the *flattened* text keeps the alignment stable (otherwise difflib
    # can split ⦃ and ␟ away from the role and the marker shows up as literal characters).
    flat: list[str] = []
    spans: list[tuple[int, int, str]] = []
    role: str | None = None
    span_start = 0
    index = 0
    while index < len(current):
        if current.startswith(MARK_OPEN, index):
            sep = current.find(MARK_SEP, index)
            if sep >= 0:
                role = current[index + 1 : sep]
                span_start = len(flat)
                index = sep + 1
                continue
        if current.startswith(MARK_CLOSE, index):
            if role is not None:
                spans.append((span_start, len(flat), role))
            role = None
            index += 1
            continue
        flat.append(current[index])
        index += 1
    flattened = "".join(flat)

    fragments: list[dict] = []

    def push(kind: str, text: str, who: str | None = None) -> None:
        if not text:
            return
        last = fragments[-1] if fragments else None
        if last and last["kind"] == kind and last.get("role") == who:
            last["text"] += text
        else:
            item: dict = {"kind": kind, "text": text}
            if kind == "speech":
                item["role"] = who or ""
            fragments.append(item)

    def role_at(pos: int) -> str | None:
        for start, end, who in spans:
            if start <= pos < end:
                return who
        return None

    def emit(start: int, end: int, default: str) -> None:  # walk a flat range, splitting on speech spans
        while start < end:
            who = role_at(start)
            stop = end
            for span_start, span_end, _ in spans:
                if start < span_start < stop:
                    stop = span_start
                if start < span_end < stop:
                    stop = span_end
            push("speech" if who else default, flattened[start:stop], who)
            start = stop

    matcher = difflib.SequenceMatcher(None, original, flattened, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            emit(j1, j2, "narration")
        elif tag == "delete":
            push("deleted", original[i1:i2])
        else:  # insert / replace
            if tag == "replace":
                push("deleted", original[i1:i2])
            emit(j1, j2, "inserted")
    return fragments


def render_diff_html(original: str, marked: str, path, title: str = "改后 / 原文对照") -> None:
    """Char-level diff: deletions struck through, insertions highlighted."""
    import difflib
    import html

    parts: list[str] = []
    matcher = difflib.SequenceMatcher(None, original, marked, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(html.escape(marked[j1:j2]))
        elif tag == "delete":
            parts.append(f"<del>{html.escape(original[i1:i2])}</del>")
        elif tag == "insert":
            parts.append(f"<ins>{html.escape(marked[j1:j2])}</ins>")
        else:
            parts.append(f"<del>{html.escape(original[i1:i2])}</del><ins>{html.escape(marked[j1:j2])}</ins>")
    path.write_text(
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>" + html.escape(title) + "</title>"
        "<style>body{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:16px;line-height:1.9}"
        "pre{white-space:pre-wrap;word-break:break-all;background:#fff;border:1px solid #eee;padding:12px;font-size:14px}"
        "del{color:#b00;background:#ffe6e6;text-decoration:line-through}ins{color:#066;background:#e6f7f7;text-decoration:none}"
        "mark{background:#fff3bf;padding:0 1px}</style></head><body>"
        f"<h2>{html.escape(title)}</h2><p>灰字=保留旁白，红横线=删除，青底=新增标记/内容</p>"
        f"<pre>{''.join(parts)}</pre></body></html>",
        encoding="utf-8",
    )


def to_script_rows(
    text: str,
    cast: Cast,
    *,
    chapter_id: int = 0,
    chapter_title: str = "",
    start_order: int = 0,
) -> list[ScriptRow]:
    """Map marked text to source rows with the role's fixed ``voice_ref``."""
    rows: list[ScriptRow] = []
    for index, segment in enumerate(parse_marks(text), start=1):
        role = cast.resolve(segment["role_name"]) or cast.resolve(NARRATOR)
        rows.append(
            ScriptRow(
                order=start_order + index,
                chapter_id=chapter_id,
                chapter_title=chapter_title,
                seg_id=f"mk{chapter_id:03d}_s{index:04d}",
                kind="narration" if segment["kind"] == "narration" else "dialogue",
                role_id=role.role_id if role else "narrator",
                role_name=role.name if role else segment["role_name"],
                raw_text=segment["text"],
                tts_text=segment["text"],
                auk_task="zero_shot_tts",
                voice_ref=role.voice_ref if role else "",
                style_desc=role.style_desc if role else "",
                target_duration_s=round(estimate_text_duration(segment["text"]), 3),
                duration_source="est",
                pe_instruction=render_instruction("zero_shot_tts", segment["text"]),
            )
        )
    return rows
