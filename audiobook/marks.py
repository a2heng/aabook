"""The HTML-like marker convention for speech.

Speech is wrapped as ``<角色名>朗读内容</角色名>``; everything outside the tags is
narration. The source text is code-filtered, so angle brackets never collide with
real content. Override the tag pattern via env ``AUDIOBOOK_MARK_RE`` if needed.

Vocal events stay as square brackets inside the content, e.g. ``<高文>[叹气]好吧。</高文>``.
"""

from __future__ import annotations

import os
import re

from .schema import Cast, ScriptRow

MARK_RE = os.environ.get("AUDIOBOOK_MARK_RE", r"<([^<>\n]{1,24})>(.*?)</\1>")
MARKS_RE = re.compile(MARK_RE, re.DOTALL)
NARRATOR = "旁白"

# A short narration between two speech spans of the SAME role is a breath/beat, not a
# speaker change: drop it and merge the speech into one row so TTS is called once.
MAX_INTERRUPT_CHARS = int(os.environ.get("AUDIOBOOK_MERGE_INTERRUPT_CHARS", "12"))


def mark(role: str, text: str) -> str:
    return f"<{role}>{text}</{role}>"


def _has_content(text: str) -> bool:
    return any(char.isalnum() for char in text)


def parse_marks(text: str) -> list[dict]:
    """Split marked text into ``[{kind, role_name, text}]`` (narration = outside tags)."""
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

    # Merge so one person's turn becomes ONE row (fewer TTS calls): adjacent same-role
    # speech joins; a short narration between two spans of the same role is dropped;
    # adjacent narration joins.
    merged: list[dict] = []
    pending: dict | None = None
    for segment in segments:
        if pending is not None:
            last = merged[-1] if merged else None
            if (
                segment["kind"] == "speech"
                and last
                and last["kind"] == "speech"
                and last["role_name"] == segment["role_name"]
                and len(pending["text"]) <= MAX_INTERRUPT_CHARS
            ):
                last["text"] += segment["text"]
                pending = None
                continue
            if segment["kind"] == "narration":
                pending["text"] += segment["text"]
                continue
            merged.append(pending)
            pending = None
        if segment["kind"] == "narration":
            pending = dict(segment)
            continue
        last = merged[-1] if merged else None
        if last and last["kind"] == "speech" and last["role_name"] == segment["role_name"]:
            last["text"] += segment["text"]
            continue
        merged.append(dict(segment))
    if pending is not None:
        merged.append(pending)
    return merged


_BARE_QUOTE_RE = re.compile(r"[“『「][^”』」\n]{1,400}?[”』」]")


def unmarked_quotes(marked: str) -> list[str]:
    """Quoted spans still *outside* any tag (leftovers / very short lines)."""
    stripped = MARKS_RE.sub("", marked)
    return [match.group(0) for match in _BARE_QUOTE_RE.finditer(stripped)]


def quoted_spans(text: str) -> list[str]:
    """All quoted spans in a raw (unmarked) text -- used as a completeness checklist."""
    return [match.group(0) for match in _BARE_QUOTE_RE.finditer(text)]


_QUOTE_CHARS = '“”‘’「」『』"'


def strip_quotes(marked: str) -> tuple[str, int]:
    """Drop every leftover quote mark outside tags (chapter-end cleanup). Returns (text, count)."""
    stripped = MARKS_RE.sub("", marked)
    count = sum(stripped.count(char) for char in _QUOTE_CHARS)
    if not count:
        return marked, 0
    parts: list[str] = []
    last = 0
    for match in MARKS_RE.finditer(marked):
        parts.append(marked[last : match.start()].translate({ord(char): None for char in _QUOTE_CHARS}))
        parts.append(match.group(0))
        last = match.end()
    parts.append(marked[last:].translate({ord(char): None for char in _QUOTE_CHARS}))
    return "".join(parts), count


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
    - ``speech``    <role>…</role> content -> role card (tags hidden);
    - ``deleted``   characters gone from the original (struck through);
    - ``inserted``  new characters (e.g. a comma at a breath point);
    - ``narration`` untouched text.
    """
    import difflib

    flat: list[str] = []
    spans: list[tuple[int, int, str]] = []
    role: str | None = None
    span_start = 0
    index = 0
    while index < len(current):
        if current.startswith("<", index):
            close = current.find(">", index)
            if close > index:
                tag = current[index + 1 : close]
                if tag.startswith("/"):
                    if role is not None:
                        spans.append((span_start, len(flat), role))
                    role = None
                elif not tag.endswith("/"):
                    role = tag
                    span_start = len(flat)
                index = close + 1
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

    def emit(start: int, end: int, default: str) -> None:
        while start < end:
            who = role_at(start)
            stop = end
            for span_start_i, span_end, _ in spans:
                if start < span_start_i < stop:
                    stop = span_start_i
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
                voice_ref=role.voice_ref if role else "",
                style_desc=role.style_desc if role else "",
                target_duration_s=round(_estimate(segment["text"]), 3),
                duration_source="est",
            )
        )
    return rows


def _estimate(text: str) -> float:
    from .tts import estimate_text_duration

    return estimate_text_duration(text)
