"""Corpus cleaning and chapter splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "utf-16")

_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_PAGE_MARK_RE = re.compile(r"^\s*[-—=]{1,}\s*(第\s*\d+\s*页|Page\s*\d+)\s*[-—=]{1,}\s*$", re.IGNORECASE)

CHAPTER_PATTERNS = (
    re.compile(r"^\s*(第\s*[0-9零一二三四五六七八九十百千万两]+\s*[章节回卷篇部])\s*(.*)$"),
    re.compile(r"^\s*(Chapter\s+[0-9IVXLC]+)\b\s*(.*)$", re.IGNORECASE),
    re.compile(r"^\s*([序楔]章|前言|序言|后记|尾声|终章|番外)\s*(.*)$"),
    # fallback: short lines that embed a 第X卷/章/回 marker (e.g. "书名 第一卷 大厅(1)")
    re.compile(r"^\s*(.{0,30}?第\s*[0-9零一二三四五六七八九十百千万两]+\s*[章节回卷篇部]\s*.{0,20})$"),
)


@dataclass
class Chapter:
    chapter_id: int
    title: str
    text: str


def read_text(path: str | Path) -> str:
    data = Path(path).read_bytes()
    for encoding in ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def repair_quotes(text: str) -> str:
    """Balance and normalize quotes per paragraph (handles incomplete/ASCII quotes)."""
    fixed = []
    for paragraph in text.split("\n"):
        # ASCII quotes -> Chinese, alternating open/close within the paragraph.
        out = []
        double_open = True
        single_open = True
        for char in paragraph:
            if char == '"':
                out.append("“" if double_open else "”")
                double_open = not double_open
            elif char == "'":
                out.append("‘" if single_open else "’")
                single_open = not single_open
            else:
                out.append(char)
        paragraph = "".join(out)
        for opener, closer in (("“", "”"), ("‘", "’"), ("『", "』"), ("「", "」")):
            opens = paragraph.count(opener)
            closes = paragraph.count(closer)
            if opens > closes:
                paragraph = paragraph + closer * (opens - closes)
        fixed.append(paragraph)
    return "\n".join(fixed)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    lines = []
    for line in text.split("\n"):
        if _PAGE_MARK_RE.match(line):
            continue
        lines.append(_TRAILING_WS_RE.sub("\n", line).rstrip())
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text).strip() + "\n"
    return repair_quotes(text)


def split_chapters(text: str) -> list[Chapter]:
    """Split normalized text into chapters; falls back to a single chapter."""
    lines = text.split("\n")
    starts: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 40:
            continue
        for pattern in CHAPTER_PATTERNS:
            if pattern.match(stripped):
                starts.append((index, stripped))
                break

    if not starts:
        return [Chapter(chapter_id=1, title="全文", text=text.strip())]

    chapters: list[Chapter] = []
    preface = "\n".join(lines[: starts[0][0]]).strip()
    if preface:
        chapters.append(Chapter(chapter_id=1, title="前言", text=preface))
    for order, (start, title) in enumerate(starts):
        end = starts[order + 1][0] if order + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start + 1 : end]).strip()
        chapters.append(Chapter(chapter_id=len(chapters) + 1, title=title, text=body))
    kept = [chapter for chapter in chapters if chapter.text]
    for index, chapter in enumerate(kept, start=1):  # renumber so ids stay contiguous
        chapter.chapter_id = index
    return kept
