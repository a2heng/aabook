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
    return segments


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
