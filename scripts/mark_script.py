#!/usr/bin/env python3
"""The one way to turn a chapter into a stage-play script: one `edit` at a time.

Role of the model: a screenwriter adapting reading text into a stage-play script. It
only *flags* places through the single `edit` tool (speak / delete / replace); the MCP
server performs the mechanical change in code (drop quotes, remove a redundant
``名字：`` attribution, insert a comma at a breath point).

A one-to-many **character dictionary** (canonical name -> labels: formal name, forms of
address, nicknames) is extracted and maintained at the START of every chapter; the
marker always uses the canonical name. There are no passersby: minor people keep their
name and can be turned into passerby voices later, at the TTS stage.

    AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=qwen3.5-9b \
    AUDIOBOOK_LLM_PROFILE=qwen3.5-9b python scripts/mark_script.py 11 --count 5 --batch 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

THINK = os.environ.get("AUDIOBOOK_THINK", "1").lower() not in ("0", "off", "false", "no")

from audiobook.llm import LLMClient, config_from_env, raw_log, reasoning_of  # noqa: E402
from audiobook.marks import (  # noqa: E402
    NARRATOR,
    live_fragments,
    parse_marks,
    render_diff_html,
    render_html,
)
from audiobook.schema import Cast  # noqa: E402


_CLAUSE_RE = re.compile(r"[0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")
# Structural markers never count as sub-clauses: mark tags, chapter/section markers, labels.
_MARKER_RE = re.compile(r"<[^<>\n]*>|【[^】\n]*】|\[[^\]\n]*\]")


def _visible_view(line: str) -> tuple[str, list[int]]:
    """The line with markers/mark tags stripped, plus ``visible index -> original index``."""
    stripped: list[str] = []
    mapping: list[int] = []
    position = 0
    for match in _MARKER_RE.finditer(line):
        for index in range(position, match.start()):
            stripped.append(line[index])
            mapping.append(index)
        position = match.end()
    for index in range(position, len(line)):
        stripped.append(line[index])
        mapping.append(index)
    return "".join(stripped), mapping


def split_clauses(line: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every sub-clause in one line (markers are transparent).

    A sub-clause is a maximal run of Chinese characters / letters / digits: punctuation and
    spaces break it, so the model can address ANY piece of the text (quotes are not special).
    """
    plain, mapping = _visible_view(line)
    return [(mapping[m.start()], mapping[m.end() - 1] + 1) for m in _CLAUSE_RE.finditer(plain)]


def clause_cuts(line: str) -> list[int]:
    """Cut positions on the sentence axis: one before each sentence (binding its leading
    quotes/opening tags) plus a final cut after the last visible char (trailing tags excluded).
    Selecting sentence k means begin=k, end=k+1."""
    plain, mapping = _visible_view(line)
    cuts: list[int] = []
    for match in _CLAUSE_RE.finditer(plain):
        position = mapping[match.start()]
        while position > 0:
            if line[position - 1] in OPEN:
                position -= 1
                continue
            if line[position - 1] == ">":
                open_at = line.rfind("<", 0, position - 1)
                if open_at >= 0 and not line.startswith("</", open_at):
                    position = open_at
                    continue
            break
        cuts.append(position)
    cuts.append(mapping[-1] + 1 if mapping else len(line))
    return cuts


def clause_label(index: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA, ... (sub-clause labels as code-like letters)."""
    label = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def label_number(label: str) -> int:
    """A -> 1, Z -> 26, AA -> 27, ... ; plain digits are accepted too."""
    label = (label or "").strip()
    if label.isdigit():
        return int(label)
    value = 0
    for char in label.upper():
        if not ("A" <= char <= "Z"):
            return 0
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def number_text(text: str) -> str:
    """Render the sentence-axis cuts with the paragraph baked in: ``[3A]他叹道：[3B]“…”[3C]``.

    Each cut label is ``<paragraph number><letter>`` so a single label locates the place."""
    out: list[str] = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        cuts = clause_cuts(line)
        if len(cuts) <= 1:
            out.append(f"[{line_no}] {line}")
            continue
        pieces: list[str] = []
        cursor = 0
        for cut_index, position in enumerate(cuts[:-1], start=1):
            pieces.append(line[cursor:position])
            pieces.append(f"[{line_no}{clause_label(cut_index)}]")
            cursor = position
        pieces.append(line[cursor:])
        pieces.append(f"[{line_no}{clause_label(len(cuts))}]")
        out.append("".join(pieces))
    return "\n".join(out)


OPEN = "“『「"
CLOSE = "”』」"

# Search tolerance: straight and curly quotes are treated as equal (the model often
# types ASCII quotes while the text uses “ ”). Length is preserved, so indices map.
_CANON = {}
for _char in '“”"＂「」『』':
    _CANON[ord(_char)] = '"'
for _char in "‘’'":
    _CANON[ord(_char)] = "'"
_QUOTE_CANON = str.maketrans(_CANON)
_QUOTE_STRIP_CHARS = '"“”‘’「」『』'
_QUOTE_STRIP = str.maketrans("", "", _QUOTE_STRIP_CHARS)
_ALL_OPEN = OPEN + "‘"
_ALL_CLOSE = CLOSE + "’"


class ScriptServer:
    def __init__(self) -> None:
        self.text = ""
        self.edits: list[dict] = []
        self._refused_deletes: set[str] = set()

    # ---- tools ---------------------------------------------------------------
    def set_text(self, text: str) -> dict:
        self.text = text or ""
        self._refused_deletes = set()
        return {"chars": len(self.text)}

    def get_marked(self) -> dict:
        return {"text": self.text}

    def edit(
        self,
        op: str = "",
        text: str = "",
        role: str = "",
        find: str = "",
        replace: str = "",
        end: str | int = "",  # line mode: end split letter; no-line mode: legacy text anchor
        line: int = 0,
        begin=None,
    ) -> dict:
        """The one editing tool. ``op`` defaults to ``speak`` -- never to a text rewrite."""
        op = (op or "speak").strip() or "speak"
        if op == "speak":
            target = find if any(char in find for char in OPEN + CLOSE) else (text or find)
            return self._mark_speaker(target, role, end, line, begin)
        if op == "unmark":
            return self._unmark(text or find, end, line, begin)
        if op == "replace":
            return self._replace(find or text, replace)
        if op == "delete":
            return self._delete(text or find)
        return {"ok": False, "reason": f"未知操作 {op}；只能用 speak 或 unmark"}

    # ---- primitives ----------------------------------------------------------
    def _enclosing_quotes(self, start: int, end: int) -> tuple[int, int] | None:
        """Quoted span containing [start, end): scan left for the opening quote (a quoted span
        may contain several sentences, so only a closing quote / newline stops the scan) and
        right for its matching closing quote."""
        left = -1
        for i in range(start - 1, max(-1, start - 400), -1):
            char = self.text[i]
            if char in CLOSE or char == "\n":
                return None
            if char in OPEN:
                left = i
                break
        if left < 0:
            return None
        for j in range(end, min(len(self.text), end + 500)):
            char = self.text[j]
            if char in OPEN:
                return None
            if char in CLOSE:
                return left, j + 1
        return None

    def _find_short_in_quotes(self, needle: str) -> int:
        """Exact search for a very short needle (1-2 chars) INSIDE quoted spans only.

        Short needles are dangerous: ``text="不"`` would otherwise match the first 不 in the
        narration and mark the wrong place. Returns the quote content start or -1."""
        for opener, closer in zip(OPEN, CLOSE):
            position = 0
            while True:
                start = self.text.find(opener, position)
                if start < 0:
                    break
                end = self.text.find(closer, start + 1)
                if end < 0:
                    break
                content = self.text[start + 1 : end]
                found = content.find(needle)
                if found >= 0:
                    return start + 1 + found
                position = end + 1
        return -1

    def _find_quoted_fragment(self, needle: str) -> int:
        """Near-miss fallback: match the needle inside any quoted span with punctuation and
        quote marks ignored (e.g. model sends 「抱，抱歉」 for 「抱……抱歉……」). Returns the
        content start of the first matching span, or -1; the caller expands to the whole span."""
        plain_needle = "".join(char for char in needle if not unicodedata.category(char).startswith("P"))
        if len(plain_needle) < 2:
            return -1
        for opener, closer in zip(OPEN, CLOSE):
            position = 0
            while True:
                start = self.text.find(opener, position)
                if start < 0:
                    break
                end = self.text.find(closer, start + 1)
                if end < 0:
                    break
                content = self.text[start + 1 : end]
                plain_content = "".join(char for char in content if not unicodedata.category(char).startswith("P"))
                if plain_needle in plain_content:
                    return start + 1
                position = end + 1
        return -1

    def _line_span(self, line: int) -> tuple[int, int] | None:
        """``[start, end)`` of 1-based ``line`` in the current text (lines = natural paragraphs)."""
        if line < 1:
            return None
        position = 0
        for index, part in enumerate(self.text.split("\n"), start=1):
            if index == line:
                return position, position + len(part)
            position += len(part) + 1
        return None

    def _quoted_expand_in(self, start: int, stop: int, low: int, high: int) -> tuple[int, int] | None:
        """Expand ``[start, stop)`` to its enclosing quote pair, bounded to ``[low, high)``."""
        left = -1
        for i in range(start - 1, low - 1, -1):
            if self.text[i] in CLOSE:
                return None
            if self.text[i] in OPEN:
                left = i
                break
        if left < 0:
            return None
        for j in range(stop, high):
            if self.text[j] in OPEN:
                return None
            if self.text[j] in CLOSE:
                return left, j + 1
        return None

    def _clauses_in_line(self, low: int, high: int) -> list[tuple[int, int]]:
        """Absolute ``(start, end)`` of every sub-clause in ``[low, high)`` (same numbering as
        :func:`number_text`; existing marks are transparent, so numbers stay stable)."""
        return [(low + start, low + end) for start, end in split_clauses(self.text[low:high])]

    @staticmethod
    def _split_label(label) -> tuple[int, str]:
        """``"22D" -> (22, "D")``; bare ``"D"`` -> (0, "D"); pure digits stay the legacy cut index."""
        raw = str(label or "").strip().upper()
        raw = re.sub(r"[\[\]()（）【】<>《》\s_.\-—~·]+", "", raw)
        raw = raw.replace("第", "").replace("段", "").replace("句", "").replace("标记", "").replace("切割", "")
        digits = "".join(char for char in raw if char.isdigit())
        letters = "".join(char for char in raw if not char.isdigit())
        if digits and letters:
            return int(digits), letters
        return 0, (letters or digits)

    def _clause_span(self, begin, end, low: int, high: int) -> tuple[int, int] | None:
        """Resolve axis cuts: begin=k, end=k+1 selects sentence k; end=N+1 is the line end."""
        cuts = [low + position for position in clause_cuts(self.text[low:high])]
        first = label_number(self._split_label(begin)[1])
        end_letter = self._split_label(end)[1] if str(end or "").strip() else ""
        last = label_number(end_letter) if end_letter else first + 1
        if first < 1 or first >= len(cuts):
            return None
        if last <= first or last > len(cuts):
            last = len(cuts)  # lenient: a wild end still covers up to the line end
        return cuts[first - 1], cuts[last - 1]

    def _enclosing_tag(self, start: int, stop: int, strict_stop: bool = True):
        """The ``<role>…</role>`` tag whose content contains ``start`` (or ``None``).

        ``strict_stop`` also requires the requested span to stay within the tag; unmark
        only needs the start to fall inside it."""
        open_lt = self.text.rfind("<", 0, start + 1)
        if open_lt < 0:
            return None
        open_gt = self.text.find(">", open_lt)
        if open_gt < 0:
            return None
        role = self.text[open_lt + 1 : open_gt]
        if not role or role.startswith("/"):
            return None
        close_lt = self.text.find(f"</{role}>", open_gt + 1)
        if close_lt < 0 or not (open_gt <= start < close_lt or open_lt == start):
            return None
        if strict_stop and stop > close_lt + len(role) + 3:
            return None
        return open_lt, open_gt, close_lt, role

    _QUOTE_RUN_RE = re.compile(r"[“『「][^”』」]*[”』」]")

    def unmarked_quotes(self) -> list[dict]:
        """Quoted runs not covered by any mark tag (candidate misses), with axis labels.

        The check pass gets this list so the model only has to say WHO speaks each run."""
        found: list[dict] = []
        offset = 0
        for line_no, line in enumerate(self.text.split("\n"), start=1):
            if line:
                cuts = [offset + position for position in clause_cuts(line)]
                for match in self._QUOTE_RUN_RE.finditer(line):
                    run_start, run_end = offset + match.start(), offset + match.end()
                    if self._enclosing_tag(run_start, run_end, strict_stop=False) is not None:
                        continue
                    begin_label = end_label = ""
                    for cut_index in range(1, len(cuts)):
                        if cuts[cut_index - 1] <= run_start < cuts[cut_index]:
                            begin_label = f"{line_no}{clause_label(cut_index)}"
                        if cuts[cut_index - 1] < run_end <= cuts[cut_index]:
                            end_label = f"{line_no}{clause_label(cut_index + 1)}"
                            break
                    snippet = line[max(0, match.start() - 24) : match.end() + 24].strip()
                    found.append(
                        {
                            "line": line_no,
                            "begin": begin_label or f"{line_no}A",
                            "end": end_label or f"{line_no}{clause_label(max(1, len(cuts) - 1))}",
                            "text": match.group(0),
                            "snippet": snippet,
                        }
                    )
            offset += len(line) + 1
        return found

    def _unmark(self, text: str = "", end: str | int = "", line: int = 0, begin=None) -> dict:
        """Remove a wrong mark (unwrap its tag); original text is never touched."""
        resolved = self._resolve_span(text, end, line, begin)
        if isinstance(resolved, dict):
            return resolved
        start, stop, _low, _high, _exact = resolved
        tag = self._enclosing_tag(start, stop, strict_stop=False)
        if tag is None:
            return {"ok": False, "reason": "这一段没有标记，不需要取消"}
        open_lt, open_gt, close_lt, role = tag
        inner = self.text[open_gt + 1 : close_lt]
        self.text = self.text[:open_lt] + inner + self.text[close_lt + len(role) + 3 :]
        self.edits.append({"op": "unmark", "role": role, "text": text, "line": line, "begin": begin, "end": end})
        return {
            "ok": True,
            "unmarked": role,
            "text": inner[:24],
            "marked": inner[:400],
            "range": f"{begin}-{end or begin}" if begin else "",
        }

    def _resolve_span(self, text: str, end: str | int = "", line: int = 0, begin=None):
        """Resolve the target span; returns (start, stop, low, high, exact) or an error dict."""
        plain_needle = "".join(char for char in text if not unicodedata.category(char).startswith("P"))
        low, high = (0, len(self.text))
        exact = False
        start, stop = 0, 0
        begin_line = self._split_label(begin)[0] if begin not in (None, "") else 0
        if not line and begin_line:
            line = begin_line
        if line:
            span = self._line_span(line)
            if span is None:
                return {
                    "ok": False,
                    "reason": f"段落号 {line} 不存在；段落号写在切割标记里（形如 `[3A]`）",
                    "text": text[:24],
                }
            low, high = span
            if begin:
                span = self._clause_span(begin, end, low, high)
                if span is not None:
                    start, stop = span
                    exact = True
                elif not text:
                    count = len(self._clauses_in_line(low, high))
                    return {
                        "ok": False,
                        "reason": f"第 {line} 段没有这个分割号（该段共 {count} 段）",
                        "text": str(begin)[:24],
                    }
            if not exact and text:
                # text fallback: strip axis labels, locate inside the line, snap to its sentence
                cleaned = re.sub(r"\[[A-Za-z0-9]+\]", "", text).strip()
                index = self._find_index_in(self.text[low:high], cleaned)
                if index < 0:
                    index = self._find_index_in(self.text[low:high], cleaned.strip("“”「」‘’\"'"))
                if index >= 0:
                    position = low + index
                    cuts = [low + cut for cut in clause_cuts(self.text[low:high])]
                    for cut_index in range(1, len(cuts)):
                        if cuts[cut_index - 1] <= position < cuts[cut_index]:
                            start, stop = cuts[cut_index - 1], cuts[cut_index]
                            exact = True
                            break
                    if not exact:
                        start, stop = position, position + len(cleaned)
                        exact = True
            if not exact and not begin:
                cuts = [low + position for position in clause_cuts(self.text[low:high])]
                if len(cuts) == 2:
                    start, stop = cuts[0], cuts[1]
                    exact = True
                elif len(cuts) > 2:
                    return {
                        "ok": False,
                        "reason": f"第 {line} 段有 {len(cuts) - 1} 句，请给 begin（首句的起始切割）+ end（下一句的切割）",
                        "text": text[:24],
                    }
                else:
                    return {"ok": False, "reason": f"第 {line} 段没有可标的文字", "text": text[:24]}
            if not exact:
                return {"ok": False, "reason": f"第 {line} 段里找不到这段正文；请核对 text 或 begin/end", "text": text[:24]}
        elif len(plain_needle) <= 2:
            index = self._find_short_in_quotes(text)
            if index < 0:
                return {"ok": False, "reason": "这段太短、容易标错位置；请给这段对话的完整原文", "text": text}
            start, stop = index, index + len(text)
        else:
            index = self._find_index(text)
            if index < 0:
                index = self._find_quoted_fragment(text)
            if index < 0:
                return {"ok": False, "reason": "not found", "text": text}
            start, stop = index, index + len(text)
        return start, stop, low, high, exact

    def _mark_speaker(self, text: str, role: str, end: str | int = "", line: int = 0, begin=None) -> dict:
        resolved = self._resolve_span(text, end, line, begin)
        if isinstance(resolved, dict):
            return resolved
        start, stop, low, high, exact = resolved
        tag = self._enclosing_tag(start, stop)
        if tag is not None:
            open_lt, open_gt, close_lt, existing = tag
            new_role = self._canonical_role(role) or (role or "").strip()
            inner = self.text[open_gt + 1 : close_lt]
            close_gt = close_lt + len(existing) + 3
            rel_start = max(0, min(len(inner), start - (open_gt + 1)))
            rel_stop = max(rel_start, min(len(inner), stop - (open_gt + 1)))
            piece = inner[rel_start:rel_stop]
            if not new_role or (new_role == existing and rel_start == 0 and rel_stop == len(inner)):
                return {"ok": True, "already": True, "role": existing, "text": text[:24]}
            if rel_start == 0 and rel_stop == len(inner):
                replacement = f"<{new_role}>{inner}</{new_role}>"
            else:  # shrink/reshape: the rest of the old mark stays unmarked
                replacement = inner[:rel_start] + f"<{new_role}>{piece}</{new_role}>" + inner[rel_stop:]
            self.text = f"{self.text[:open_lt]}{replacement}{self.text[close_gt:]}"
            self.edits.append(
                {"op": "remark", "role": new_role, "from": existing, "text": text, "line": line, "begin": begin, "end": end}
            )
            return {
                "ok": True,
                "replaced": True,
                "role": new_role,
                "previous": existing,
                "text": piece[:24],
                "marked": piece[:400],
                "range": f"{begin}-{end or begin}" if begin else "",
            }
        anchored = False
        clamped = False
        expanded = False
        if not exact:
            if isinstance(end, str) and end and end != text and not line:
                search_from = start + 1
                while True:
                    tail = self._find_index_after(end, search_from)
                    if tail < 0:
                        break
                    if tail + len(end) >= stop:  # closing anchor must lie at/after the opening one
                        stop = tail + len(end)
                        anchored = True
                        break
                    search_from = tail + 1
            # One speech never covers two separate quoted spans: if the model's text spans a
            # closing quote, narration, then another opening quote, keep only the FIRST span.
            for opener, closer in (("“", "”"), ("「", "」"), ("『", "』")):
                close_at = self.text.find(closer, start, stop)
                if close_at >= 0 and self.text.find(opener, close_at + 1, stop) >= 0:
                    stop = close_at + 1
                    clamped = True
                    break
            if clamped:  # expand to the opening quote of THIS span (the next quote must stay out)
                for i in range(start - 1, max(low - 1, start - 400), -1):
                    if self.text[i] in CLOSE or self.text[i] == "\n":
                        break
                    if self.text[i] in OPEN:
                        start = i
                        break
                expanded = True
            else:
                enclosing = self._quoted_expand_in(start, stop, low, high) if line else self._enclosing_quotes(start, stop)
                if enclosing is not None:  # expand to the two quotes: quotes stay INSIDE the mark
                    start, stop = enclosing
                    expanded = True
        body = self.text[start:stop]
        # A mark only wraps the spoken words: narration/attribution stays OUTSIDE. Trim a wide
        # span to its quoted run; several quoted runs ("…" 他说，"…") become one mark each,
        # mechanically, so the narration between them stays unmarked.
        runs = [(start + match.start(), start + match.end()) for match in self._QUOTE_RUN_RE.finditer(body)]
        if not runs:
            # half a quote: pull the span onto the matching quote before judging
            if any(char in body for char in CLOSE) and not any(char in body for char in OPEN):
                left = start - 1
                while left > max(low - 1, start - 400):
                    if self.text[left] in OPEN:
                        start = left
                        break
                    if self.text[left] in CLOSE or self.text[left] == "\n":
                        break
                    left -= 1
            elif any(char in body for char in OPEN) and not any(char in body for char in CLOSE):
                right = stop
                while right < min(high, stop + 400):
                    if self.text[right] in CLOSE:
                        stop = right + 1
                        break
                    if self.text[right] in OPEN or self.text[right] == "\n":
                        break
                    right += 1
            body = self.text[start:stop]
            runs = [(start + match.start(), start + match.end()) for match in self._QUOTE_RUN_RE.finditer(body)]
        if len(runs) > 3 or (runs and runs[-1][1] - runs[0][0] > 200):
            return {
                "ok": False,
                "reason": "这一段跨了太多处引号/太长；请一处一处给（一段一个 speak）",
                "text": text[:24],
            }
        if runs:
            start, stop = runs[0][0], runs[-1][1]
            body = self.text[start:stop]
        if "<" in body:  # flatten inner marks that are fully inside the span (balanced pairs)
            body = re.sub(r"<([^<>/\n][^<>\n]*)>([^<>]*)</\1>", r"\2", body)
        if not runs and not any(char in body for char in OPEN + CLOSE):
            context = self.text[max(0, start - 20) : start]
            if not self._SPEECH_CUE_RE.search(context):
                return {
                    "ok": False,
                    "reason": "这看起来是旁白/描写，不是人物直接说的话；只标台词或明确标记的心声",
                    "text": text[:24],
                }
        cut = start
        role_override = ""
        if cut >= 1 and self.text[cut - 1] == "：":
            k = cut - 1
            while k > 0 and self.text[k - 1] not in "。！？\n“”‘’「」『』 \t，,；;：<>":
                k -= 1
            attribution = self.text[k : cut - 1]
            attr_canonical = self._name_index().get(attribution)
            if attr_canonical and attr_canonical != NARRATOR:
                if attr_canonical != self._canonical_role(role):
                    role_override = attr_canonical  # the quote says who is speaking: trust it
                cut = k  # a bare 「人名：」 is consumed either way
        if not any(char.isalnum() for char in body):
            return {"ok": False, "reason": "这段不是人物直接对话；只标人物直接说的话"}
        canonical = role_override or self._canonical_role(role) or (role or "").strip()
        if not canonical:
            return {"ok": False, "reason": "role 不能为空"}
        if len(runs) >= 2:  # several quoted runs in one request -> one mark per run
            pieces: list[str] = []
            cursor = cut
            texts: list[str] = []
            for run_start, run_end in runs:
                pieces.append(self.text[cursor:run_start])
                texts.append(self.text[run_start:run_end])
                pieces.append(f"<{canonical}>{texts[-1]}</{canonical}>")
                cursor = run_end
            pieces.append(self.text[cursor:stop])
            self.text = self.text[:cut] + "".join(pieces) + self.text[stop:]
            self.edits.append(
                {
                    "op": "speak",
                    "role": canonical,
                    "text": text,
                    "end": end,
                    "split": len(runs),
                    "attribution": cut < start,
                    "expanded": expanded,
                    "anchored": anchored,
                }
            )
            result = {
                "ok": True,
                "role": canonical,
                "split": True,
                "count": len(runs),
                "marked": " / ".join(texts)[:400],
            }
            if begin:
                result["begin"] = begin
                result["end"] = end or begin
                result["range"] = f"{begin}-{end or begin}"
            if role_override:
                result["role_note"] = f"按文中的归属「{role_override}」判定说话人"
                result["role_corrected_from"] = role
            return result
        wrapped = f"<{canonical}>{body}</{canonical}>"
        self.text = self.text[:cut] + wrapped + self.text[stop:]
        self.edits.append(
            {
                "op": "speak",
                "role": canonical,
                "text": text,
                "end": end,
                "attribution": cut < start,
                "expanded": expanded,
                "anchored": anchored,
            }
        )
        result = {"ok": True, "role": canonical, "text": body[:24], "marked": body[:400]}
        if begin:
            result["begin"] = begin
            result["end"] = end or begin
            result["range"] = f"{begin}-{end or begin}"
        if anchored:
            result["anchored"] = True
        if expanded:
            result["expanded"] = True
            result["note"] = "已按子句范围标记整段"
        if role_override:
            result["role_note"] = f"按文中的归属「{role_override}」判定说话人"
            result["role_corrected_from"] = role
        return result

    _SPEECH_TAIL_RE = re.compile(
        r"(说道|问道|答道|喊道|叫道|笑道|叹道|应道|喝道|骂道|吼道|念道|回答|低语|喃喃|嘟囔|传来|开口|说)[，,：:]?$"
    )
    _NON_SPEECH_COLON_RE = re.compile(r"(写着|写到|标着|刻着|印着|注着|列出|注明|写着|标注)$")
    # Unquoted speech/thought is only accepted right after a cue like 「他叹道：」「忽然想：」.
    _SPEECH_CUE_RE = re.compile(
        r"(说道|问道|答道|喊道|叫道|笑道|叹道|应道|喝道|骂道|吼道|念道|回答|低语|喃喃|嘟囔|开口|自语|"
        r"心说|暗想|寻思|嘀咕|心想|想道|说|道|想)[，,：:]?$"
    )

    def _looks_like_speech(self, start: int) -> bool:
        """Code guard: a quote right after speech cues is dialogue -- refuse to delete it."""
        ctx = self.text[max(0, start - 14) : start]
        if self._SPEECH_TAIL_RE.search(ctx):
            return True
        return ctx.endswith("：") and not self._NON_SPEECH_COLON_RE.search(ctx[:-1])

    def _empty_quote_pair(self, start: int, end: int, inside: bool) -> tuple[int, int] | None:
        """Empty / punctuation-only quote pair inside the target span, or starting right after it."""
        base = start if inside else end
        limit = end if inside else min(len(self.text), end + 1)
        open_at = next((i for i in range(base, limit) if self.text[i] in OPEN), -1)
        if open_at < 0:
            return None
        close_at = next((i for i in range(open_at + 1, len(self.text)) if self.text[i] in CLOSE), -1)
        if close_at < 0:
            return None
        content = self.text[open_at + 1 : close_at]
        if any(char.isalnum() for char in content) or len(content) > 40:
            return None
        return open_at, close_at + 1

    def _delete(self, target: str) -> dict:
        if not target:
            return {"ok": False, "reason": "empty"}
        span = None
        if any(char in target for char in OPEN + CLOSE):
            index = self._find_index(target)
            if index >= 0:
                span = (index, index + len(target))
        if span is None:
            span = self._locate(target, strip=True)
        if span is None:  # near-miss inside quotes -> let the enclosing-quote expansion handle it
            quoted_at = self._find_quoted_fragment(target)
            if quoted_at >= 0:
                span = (quoted_at, quoted_at + len(target))
        if span is not None:
            start, end = span
            if all(char not in _QUOTE_STRIP_CHARS for char in self.text[start:end]):
                if start >= 1 and self.text[start - 1] in "‘" and end < len(self.text) and self.text[end] in "’":
                    start -= 1  # single-quote wrapped (‘惊喜’): remove that pair too
                    end += 1
                else:
                    enclosing = self._enclosing_quotes(start, end)  # mid-quote fragment -> whole span
                    if enclosing is not None:
                        start, end = enclosing
                    else:
                        pair = self._empty_quote_pair(start, end, inside=False)  # 「他说道：」+ empty pair
                        if pair is not None:
                            start, end = pair
            else:
                pair = self._empty_quote_pair(start, end, inside=True)  # 「他说道：“”」 -> shrink to the pair
                if pair is not None:
                    start, end = pair
            inner = self.text[start:end]
            body = inner[1:-1] if (inner[:1] in _ALL_OPEN and inner[-1:] in _ALL_CLOSE) else inner
            keep = any(char.isalnum() for char in body)
            if keep and self._looks_like_speech(start) and target not in self._refused_deletes:
                self._refused_deletes.add(target)
                return {
                    "ok": False,
                    "reason": "这看起来是人物对话（前面有说话提示），请改用 edit(op=speak, role=规范名)；确认不是就再 delete 一次",
                    "text": target[:24],
                }
            cut = start
            if not keep and cut >= 1 and self.text[cut - 1] == "：":
                # empty / symbol-only quote (" " or "，"): check the 10 characters before it --
                # a dangling 「某某说道：」 inside that window goes together with the quote pair
                k = cut - 1
                floor = max(0, cut - 1 - 10)
                while k > floor and self.text[k - 1] not in "。！？\n“”‘’「」『』 \t，,；;：<>":
                    k -= 1
                attribution = self.text[k : cut - 1]
                if attribution and (self._SPEECH_TAIL_RE.search(attribution) or attribution in self._known_names()):
                    cut = k
            self.text = self.text[:cut] + (body if keep else "") + self.text[end:]
            self.edits.append({"op": "delete", "text": target, "kept": keep})
            return {"ok": True, "kept" if keep else "removed": body or inner}
        index = self._find_index(target)  # plain characters (stray punctuation)
        if index < 0:
            return {"ok": False, "reason": "not found", "text": target}
        self.text = self.text[:index] + self.text[index + len(target) :]
        self.edits.append({"op": "delete", "text": target})
        return {"ok": True, "removed": target}

    def _replace(self, find: str, replace: str) -> dict:
        index = self._find_index(find)
        if index < 0:
            return {"ok": False, "reason": "not found", "find": find}
        self.text = self.text[:index] + replace + self.text[index + len(find) :]
        self.edits.append({"op": "replace", "find": find, "replace": replace})
        return {"ok": True}

    # ---- helpers -------------------------------------------------------------
    def _name_index(self) -> dict[str, str]:
        """label/alias -> canonical name (from roles.json)."""
        book = os.environ.get("AUDIOBOOK_BOOK", "dawn")
        roles = APP_ROOT / "outputs" / book / "script" / "roles.json"
        cache_key = str(roles)
        if getattr(self, "_index_path", None) == cache_key:
            return self._index_cache
        index: dict[str, str] = {}
        if roles.is_file():
            for canonical, labels in json.loads(roles.read_text(encoding="utf-8")).items():
                index[canonical] = canonical
                for label in labels:
                    index.setdefault(str(label), canonical)
        self._index_cache, self._index_path = index, cache_key
        return index

    def _canonical_role(self, role: str) -> str | None:
        return self._name_index().get((role or "").strip())

    def set_name_index(self, index: dict[str, str]) -> None:
        """Test/offline hook: force the label -> canonical map."""
        book = os.environ.get("AUDIOBOOK_BOOK", "dawn")
        self._index_cache = dict(index)
        self._index_path = str(APP_ROOT / "outputs" / book / "script" / "roles.json")

    def _known_names(self) -> set[str]:
        """All canonical names + labels from the dictionary (roles.json, cast.json fallback)."""
        book = os.environ.get("AUDIOBOOK_BOOK", "dawn")
        roles = APP_ROOT / "outputs" / book / "script" / "roles.json"
        cache_key = str(roles)
        if getattr(self, "_names_path", None) == cache_key:
            return self._names_cache
        names: set[str] = set()
        if roles.is_file():
            for canonical, labels in json.loads(roles.read_text(encoding="utf-8")).items():
                names.add(canonical)
                names.update(str(label) for label in labels)
        else:
            cast_path = APP_ROOT / "outputs" / book / "cast.json"
            if cast_path.is_file():
                cast = Cast.load(str(cast_path))
                for item in cast.roles.values():
                    names.add(item.name)
                    names.update(item.aliases)
        self._names_cache, self._names_path = names, cache_key
        return names

    def _find_index(self, needle: str) -> int:
        return self._find_index_in(self.text, needle)

    def _find_index_after(self, needle: str, pos: int) -> int:
        """Same fuzzy finder, but only at/after ``pos`` (for the closing anchor)."""
        if pos < 0:
            pos = 0
        found = self._find_index_in(self.text[pos:], needle)
        return pos + found if found >= 0 else -1

    def _find_index_in(self, haystack: str, needle: str) -> int:
        if not needle:
            return -1
        index = haystack.find(needle)
        if index >= 0:
            return index
        index = haystack.translate(_QUOTE_CANON).find(needle.translate(_QUOTE_CANON))
        if index >= 0:
            return index
        # tolerate quote marks the model dropped (e.g. 「‘复活’」 given as 「复活」). Only when the
        # needle itself has no quotes: otherwise the matched length would not map back safely.
        squeezed_needle = needle.translate(_QUOTE_STRIP)
        if squeezed_needle and len(squeezed_needle) == len(needle):
            squeezed_text = haystack.translate(_QUOTE_STRIP)
            pos = squeezed_text.find(squeezed_needle)
            if pos >= 0:
                mapping = [i for i, char in enumerate(haystack) if char not in _QUOTE_STRIP_CHARS]
                return mapping[pos]
        # last resort: punctuation-insensitive (the model may drop or swap ，。！？); map back by
        # the letters themselves. Only for quote-free needles of a useful length.
        plain_needle = "".join(char for char in needle if not unicodedata.category(char).startswith("P"))
        if len(plain_needle) >= 4 and all(char not in _QUOTE_STRIP_CHARS for char in needle):
            plain_chars: list[str] = []
            plain_map: list[int] = []
            for position, char in enumerate(haystack):
                if not unicodedata.category(char).startswith("P"):
                    plain_chars.append(char)
                    plain_map.append(position)
            pos = "".join(plain_chars).find(plain_needle)
            if pos >= 0:
                return plain_map[pos]
        # tolerate a needle that accidentally includes mark syntax characters
        if "<" in needle or ">" in needle:
            plain = needle.replace("<", "").replace(">", "").replace("/", "")
            if plain:
                return haystack.find(plain)
        return -1

    def _locate(self, text: str, strip: bool = False) -> tuple[int, int] | None:
        text = (text or "").strip()
        if not text:
            return None
        for opener, closer in zip(OPEN, CLOSE):
            quoted = f"{opener}{text}{closer}"
            index = self._find_index(quoted)
            if index >= 0:
                return index, index + len(quoted)
        index = self._find_index(text)
        if index < 0:
            return None
        start, end = index, index + len(text)
        if self.text[start : start + 1] in OPEN:  # anchor starts at the opening quote: expand
            close_index = -1
            for char in CLOSE:
                pos = self.text.find(char, end)
                if pos >= 0 and (close_index < 0 or pos < close_index):
                    close_index = pos
            if close_index >= 0:
                return start, close_index + 1
        if strip and start > 0 and self.text[start - 1] in OPEN:
            close_index = -1
            for char in CLOSE:
                pos = self.text.find(char, end)
                if pos >= 0 and (close_index < 0 or pos < close_index):
                    close_index = pos
            if close_index >= 0:
                return start - 1, close_index + 1
        return start, end

    def call(self, name: str, args: dict) -> dict:
        tools = {"set_text": self.set_text, "edit": self.edit, "get_marked": self.get_marked}
        if name not in tools:
            raise ValueError(f"unknown tool: {name}")
        return tools[name](**args)

    def tool_specs(self) -> list[dict]:
        return [
            {
                "name": "edit",
                "description": (
                    "标注工具（一次一处，只标**人物直接说的话**）。正文里每一句前后都有唯一的切割标记，"
                    "形如 `[3A]`、`[22D]`（阿拉伯数字段落号+大写字母；相邻句共用中间的标记，行尾也有收尾标记）。"
                    "**铁律：begin、end、text、role 四个参数缺一不可**——begin/end 是这段发言前后两个标记"
                    "（选第 k 句 = 第 k 个标记到它后面一个标记，如 3A→3B；跨多句取起止标记），"
                    "text 是这段发言原文（照抄，可去掉切割标记），role 是人物表里的规范名。"
                    "标记自带段落号，不用另给 line。旁白/描写/叙述一律禁止标；标记只包说话内容"
                    "（「某某说道/淡淡地说道」留在标记外）。标错可改/删：同一处再 speak 会覆盖旧标记；"
                    "整段误标用 op=unmark 去掉（只去标记、不动原文）。"
                    '示例：{"op":"speak","begin":"3B","end":"3C","text":"你终于来了。","role":"陈默"}'
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["speak", "unmark"],
                            "description": "speak=标出说话人；unmark=去掉误标（只去标记，不改原文）",
                        },
                        "line": {"type": "integer", "description": "段落号（可省略：begin/end 标记自带段落号）"},
                        "begin": {"type": "string", "description": "起始切割标记，形如 3B / 22D（段落号+字母）"},
                        "end": {"type": "string", "description": "结束切割标记，形如 3C / 22E（单句 = begin 后一个标记）"},
                        "text": {"type": "string", "description": "这段发言的原文（必填；可省略切割标记）"},
                        "role": {"type": "string", "description": "说话人规范名（speak 必填）；判断不出时填「未知」"},
                    },
                    "required": ["op", "begin", "end", "text", "role"],
                },
            }
        ]


class MCPClient:
    """In-process adapter over :class:`ScriptServer` (no subprocess/IPC)."""

    def __init__(self, argv=None) -> None:
        self.server = ScriptServer()

    def initialize(self) -> None:
        return None

    def openai_tools(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]},
            }
            for t in self.server.tool_specs()
        ]

    def call(self, name: str, args: dict) -> str:
        return json.dumps(self.server.call(name, args), ensure_ascii=False)


HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mark_script · live</title>
<style>
:root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--panel2:#1b2230;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;
  --accent:#5aa9ff;--ok:#43c785;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;overflow:hidden}
header{height:46px;display:flex;gap:12px;align-items:center;padding:0 18px;border-bottom:1px solid var(--line)}
.brand{font-weight:700}.brand small{color:var(--dim);font-weight:400;margin-left:8px}
header .grow{flex:1}
.dot{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 8px var(--ok)}
#main{display:grid;grid-template-columns:1fr 340px;height:calc(100dvh - 46px)}
#left{display:flex;flex-direction:column;min-width:0;min-height:0;border-right:1px solid var(--line)}
#chbar{display:flex;gap:10px;align-items:center;padding:7px 16px;border-bottom:1px solid var(--line);font-size:12.5px;color:var(--dim)}
#chbar select{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:3px 8px;font-size:13px}
#chbar button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:3px 10px;cursor:pointer;font-size:12.5px}
#chbar button.on{background:var(--accent);color:#0b0f15;border-color:var(--accent);font-weight:700}
#article{overflow:auto;min-height:0;padding:20px 30px 60px;flex:1;font-size:15.5px;line-height:2.05;word-break:break-word}
#article p{margin:0 0 .55em;text-indent:2em}
#article p:empty{display:none}
.who{display:inline-block;font-size:11.5px;font-weight:700;color:#9fd0ff;background:#16314f;border-radius:6px;padding:0 7px;margin:0 3px 0 2px;vertical-align:1px;line-height:1.7}
.speech{background:#152238;border-radius:8px;padding:2px 6px;box-shadow:inset 0 0 0 1px #2b4a72}
del{color:#ff9d9d;background:#2a1414;text-decoration:line-through;border-radius:5px;padding:1px 3px}
ins{color:#9fe6c1;background:#122a1c;text-decoration:none;border-radius:5px;padding:1px 5px}
.new{animation:appear .65s cubic-bezier(.2,.9,.3,1.1) both}
@keyframes appear{0%{opacity:0;filter:blur(4px)}55%{opacity:1}100%{opacity:1;filter:blur(0)}}
#chat{overflow:hidden;min-height:0;padding:7px 9px;display:flex;flex-direction:column;justify-content:flex-end;gap:3px}
.ev{background:var(--panel);border:1px solid var(--line);border-left:3px solid #2b4a72;border-radius:8px;padding:3px 8px;font-size:12px;line-height:1.4;word-break:break-word}
.ico{display:inline-block;width:14px;margin-right:4px;text-align:center;opacity:.9}
.think{color:var(--dim);background:#12161d;border-left-color:#3a4252}.think summary{cursor:pointer;font-size:11.5px}
.call{border-color:#25415f;border-left-color:#3f74b0;background:#101a26}
.call.ok{border-left-color:var(--ok)}.call.bad{border-left-color:#ff7b7b}
.rst{margin-top:3px;padding-top:3px;border-top:1px dashed #2c3644}
.dict{border-color:#5c4620;border-left-color:#b8860b;background:#241d10}
.res{border-color:#234;border-left-color:#2e3d52;background:#111720;color:var(--dim);font-size:11.5px}
.res.ok{border-color:#1f4a33;border-left-color:var(--ok);color:#c9f0da;background:#111d16}
.res.bad{border-color:#5a2626;border-left-color:#ff7b7b;color:#ffc9c9;background:#1d1212}
.miss{border-color:#5c4620;border-left-color:#b8860b;background:#1d1a10;color:#ffdba8}
.sum{border-color:#3b3560;border-left-color:#8f7fe8;background:linear-gradient(180deg,#1b1830,#151a23)}
.ev code{font-size:11px;padding:0 3px}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:0 4px;border-radius:5px;margin-right:4px}
.b-speak{background:#17345a;color:#8fc4ff}.b-unmark{background:#4a2a12;color:#ffba75}.b-ok{background:#12351f;color:#7ee2a8}
.b-warn{background:#3a2f12;color:#ffd08a}.b-bad{background:#3a1616;color:#ff9d9d}.b-other{background:#232a36;color:#b8c4d4}
code{font-family:ui-monospace,Menlo,Consolas,monospace;background:#0d1219;border:1px solid var(--line);border-radius:5px;padding:0 4px;font-size:12px}
.role{color:var(--accent);font-weight:600}.sp{color:var(--dim)}.chap{color:var(--accent);font-weight:700;font-size:11.5px;margin-right:6px}
/* phone / narrow: stack vertically -- article on top, chat as a fixed-height stream below */
@media (max-width:820px){
  #main{grid-template-columns:1fr;grid-template-rows:1fr minmax(28dvh,38dvh)}
  #left{border-right:none;border-bottom:1px solid var(--line)}
  #article{padding:14px 16px 40px;font-size:16px;line-height:1.95}
  #chat{padding:8px 10px;gap:4px}
  .ev{font-size:12px;padding:4px 8px}
  #chbar{gap:8px;padding:6px 12px;font-size:12px;overflow-x:auto;white-space:nowrap}
  header{padding:0 12px}.brand{font-size:14px}
}
</style></head><body>
<header>
  <div class="brand">mark_script <small id="book">· live</small></div>
  <span class="grow"></span>
  <nav style="display:flex;gap:12px;align-items:center;font-size:13px">
    <a href="live.html" style="color:#e6ecf3;font-weight:700;text-decoration:none">台本实时</a>
    <a href="llm_raw.html" target="_blank" style="color:#5aa9ff;text-decoration:none">原始 I/O ↗</a>
    <a href="/dashboard" target="_blank" style="color:#5aa9ff;text-decoration:none">看板 ↗</a>
    <a href="/" target="_blank" style="color:#5aa9ff;text-decoration:none">文件库 ↗</a>
    <a id="breeze" href="#" target="_blank" style="color:#5aa9ff;text-decoration:none">Breeze ↗</a>
  </nav>
  <span class="dot"></span>
</header>
<div id="main">
  <div id="left">
    <div id="chbar">
      <select id="chs"></select>
      <button id="follow" class="on">跟随最新</button>
      <span id="finfo"></span>
    </div>
    <div id="article"><span class="sp">等待正文……</span></div>
  </div>
  <div id="chat"></div>
</div>
<script>
const BASE='__LIVE_BASE__', MAXCHAT=32;
document.getElementById('breeze').href='http://'+location.hostname+':8137/';
const article=document.getElementById('article'), chat=document.getElementById('chat'), chs=document.getElementById('chs');
const book=document.getElementById('book'), followBtn=document.getElementById('follow'), finfo=document.getElementById('finfo');
let offset=0, follow=true, selected=null, current=null, prevSig='', idxSig='', seen=new Set();
const wantCh=(new URLSearchParams(location.search).get('ch')||'').trim();  // ?ch=3 pins the picker
if(wantCh){selected=+wantCh;follow=false;followBtn.classList.remove('on');}
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
function fragHTML(f,isNew){const c=isNew?' new':'';
  if(f.kind==='speech')return '<span class="speech'+c+'"><span class="who">'+esc(f.role||'?')+'</span>'+esc(f.text)+'</span>';
  if(f.kind==='deleted')return '<del'+c+'>'+esc(f.text)+'</del>';
  if(f.kind==='inserted')return '<ins'+c+'>'+esc(f.text)+'</ins>';
  return '<span'+c+'>'+esc(f.text)+'</span>';}
function renderArticle(s){
  if(!s||!s.fragments||document.hidden)return;
  const sig=JSON.stringify(s.fragments); if(sig===prevSig)return;
  if(seen.chapter!==s.chapter){seen=new Set();seen.chapter=s.chapter;}
  const top=article.scrollTop; let html='<p>';
  for(const f of s.fragments){
    const parts=String(f.text).split(/\n+/);
    for(let i=0;i<parts.length;i++){
      if(i>0)html+='</p><p>';                       // newline -> real paragraph
      const txt=(i>0||html==='<p>')?parts[i].replace(/^[ \t\u3000]+/,''):parts[i];  // CSS handles the indent
      if(!txt)continue;
      const key=f.kind+'|'+(f.role||'')+'|'+parts[i];
      const isNew=!seen.has(parts[i])&&!seen.has(key); seen.add(parts[i]); seen.add(key);
      html+=fragHTML({...f,text:txt},isNew);
    }
  }
  html+='</p>';
  article.innerHTML=html; article.scrollTop=top; prevSig=sig;
  finfo.textContent=(s.chars||0)+' 字 · '+s.fragments.length+' 片段';
}
let pending=[], emptyTool=false;
function pushChat(e){
  const row=document.createElement('div'); row.className='ev '+e.cls;
  row.innerHTML=(e.icon?'<span class="ico">'+e.icon+'</span>':'')+(e.chapter?'<span class="chap">ch'+e.chapter+'</span>':'')+e.html;
  pending.push(row); return row;
}
function openCall(chapter){
  for(let i=pending.length-1;i>=0;i--){const r=pending[i];if(r.classList.contains('call')&&!r.querySelector('.rst'))return r;}
  for(let i=chat.childElementCount-1;i>=0;i--){const r=chat.children[i];if(r.classList.contains('call')&&!r.querySelector('.rst'))return r;}
  return null;
}
function flushChat(){
  if(!pending.length)return;
  const frag=document.createDocumentFragment();
  for(const row of pending)frag.appendChild(row);
  pending=[]; chat.appendChild(frag);
  let extra=chat.childElementCount-MAXCHAT;
  while(extra-->0)chat.removeChild(chat.firstChild);
}
function fmtRange(x){return (x&&(x.begin||x.end))?'<code>'+esc(x.begin||'?')+'~'+esc(x.end||'?')+'</code> ':'';}
function fmtCall(a){const x=a&&a.args||{};const keys=Object.keys(x);
  if(!keys.length)return '<span class="badge b-other">空调用</span><span class="sp">旧版记录没有参数</span>';
  const op=x.op||'speak';
  if(op==='speak')return '<span class="badge b-speak">标注</span>'+fmtRange(x)+'<span class="role">'+esc(x.role||'?')+'</span> <code>'+esc(x.text||'')+'</code>';
  if(op==='unmark')return '<span class="badge b-unmark">取消标记</span>'+fmtRange(x)+'<code>'+esc(x.text||'')+'</code>';
  if(op==='replace')return '<span class="badge b-other">替换</span><code>'+esc(x.find||'')+'</code> <span class="sp">→</span> <code>'+esc(x.replace||'')+'</code>';
  if(op==='delete')return '<span class="badge b-other">旧版·删除</span>'+(x.text||x.find?'<code>'+esc(x.text||x.find)+'</code>':'<span class="sp">（旧记录无参数）</span>');
  if(op==='reassign')return '<span class="badge b-other">旧版·改标</span>'+(x.role?'<span class="role">'+esc(x.role)+'</span> ':'')+'<code>'+esc(x.text||x.find||'')+'</code>';
  return '<span class="badge b-other">旧版·'+esc(op)+'</span><code>'+esc(JSON.stringify(x))+'</code>';}
function fmtResult(raw){let r=null;try{r=JSON.parse(raw);}catch(_){return esc(raw);}
  if(!r||typeof r!=='object')return esc(raw);
  if(r.ok===false)return '<span class="badge b-bad">✗ 拒绝</span>'+esc(r.reason||'')+(r.text?' · <code>'+esc(r.text)+'</code>':'');
  if(r.split)return '<span class="badge b-ok">✓ 拆成 '+r.count+' 处</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.unmarked)return '<span class="badge b-warn">已取消标记</span>'+fmtRange(r)+'<span class="role">'+esc(r.unmarked)+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.already)return '<span class="badge b-warn">已标过</span><span class="role">'+esc(r.role||'')+'</span>';
  if(r.replaced)return '<span class="badge b-ok">✓ 改标</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <span class="sp">←</span> <span class="role">'+esc(r.previous||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.ok)return '<span class="badge b-ok">✓ 标注</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  return esc(raw);}
function handle(e){
  if(e.type==='start'){book.textContent='· '+(e.book||'live');return;}
  if(e.type==='chapter'){current=e.chapter;if(follow)selected=e.chapter;return;}
  if(e.type==='roster'){pushChat({cls:'dict',icon:'✦',chapter:e.chapter,html:'词典 <b>+'+e.added+'</b> 标签 · 词条 '+e.roles});return;}
  if(e.type==='assistant'){const c=e.content||'';pushChat({cls:'think',icon:'🧠',chapter:e.chapter,html:'<details><summary>思考 '+c.length+' 字 · '+esc(c.slice(0,42))+'</summary>'+esc(c)+'</details>'});return;}
  if(e.type==='tool'){const x=e.args||{};
    if(!Object.keys(x).length){emptyTool=true;return;} emptyTool=false;
    pushChat({cls:'call',chapter:e.chapter,html:fmtCall(e.args)});return;}
  if(e.type==='result'){
    if(emptyTool){emptyTool=false;pushChat({cls:'res '+(e.ok?'ok':'bad'),icon:e.ok?'✓':'✗',chapter:e.chapter,html:fmtResult(e.result)});return;}
    const host=openCall(e.chapter);
    if(host){host.classList.add(e.ok?'ok':'bad');
      const line=document.createElement('div'); line.className='rst'; line.innerHTML=fmtResult(e.result); host.appendChild(line);}
    else pushChat({cls:'res '+(e.ok?'ok':'bad'),icon:e.ok?'✓':'✗',chapter:e.chapter,html:fmtResult(e.result)});
    return;}
  if(e.type==='missed'){pushChat({cls:'miss',icon:'🔍',chapter:e.chapter,html:'漏标扫描 <b>'+e.count+'</b> 处'+(e.samples&&e.samples.length?' <span class="sp">'+esc(e.samples.join(' / '))+'</span>':'')});return;}
  if(e.type==='done'){pushChat({cls:'sum',icon:'📝',chapter:e.chapter,html:'<b>本章完成</b> <span class="sp">'+esc(e.summary||'')+'</span>'});return;}}
function handleText(txt){
  for(const line of txt.split('\n')){if(!line.trim())continue;let e;try{e=JSON.parse(line);}catch(_){continue;}handle(e);}
}
async function tickChat(){
  try{const r=await fetch(BASE+'live.jsonl',{headers:{'Range':'bytes='+offset+'-'},cache:'no-store'});
    if(r.status===416){                                   // 到了 EOF：可能只是没新数据
      const h=await fetch(BASE+'live.jsonl',{method:'HEAD',cache:'no-store'});
      const size=+(h.headers.get('Content-Length')||0);
      if(size<offset){                                    // 文件变小 -> 被截断（新一次运行）
        const fb=await (await fetch(BASE+'live.jsonl',{cache:'no-store'})).arrayBuffer();
        offset=0;chat.innerHTML='';pending=[];
        handleText(new TextDecoder().decode(fb)); offset=fb.byteLength;
      } else { offset=size; }
    }
    else if(r.status===206||offset===0){const buf=await r.arrayBuffer();offset+=buf.byteLength;
      handleText(new TextDecoder().decode(buf));}
  }catch(err){}
  flushChat();
  setTimeout(tickChat,1200);
}
async function tickIndex(){
  try{const r=await fetch(BASE+'live_index.json?t='+Date.now(),{cache:'no-store'});
    if(!r.ok)return; const idx=await r.json();
    const cur=idx.current; if(cur&&follow)selected=cur;
    const ids=Object.keys(idx.chapters||{}).map(Number).sort((a,b)=>a-b);
    const listKey=ids.join(',');
    if(listKey!==idxSig){idxSig=listKey;                  // rebuild only when the set changes
      chs.innerHTML=ids.map(function(c){var d=idx.chapters[c]&&idx.chapters[c].done?' ✔':'';return '<option value="'+c+'">第 '+c+' 章'+d+'</option>';}).join('');}
    const want=String(selected||cur||''); if(chs.value!==want)chs.value=want;
  }catch(err){}
  setTimeout(tickIndex,1500);
}
async function tickState(){
  const cid=selected||current; if(!cid){setTimeout(tickState,800);return;}
  try{const r=await fetch(BASE+'ch'+String(cid).padStart(3,'0')+'.json?t='+Date.now(),{cache:'no-store'});
    if(r.ok)renderArticle(await r.json());}catch(err){}
  setTimeout(tickState,800);
}
chs.addEventListener('change',function(){selected=+chs.value;follow=false;followBtn.classList.remove('on');prevSig='';});
followBtn.addEventListener('click',function(){follow=true;followBtn.classList.add('on');if(current)selected=current;prevSig='';});
tickChat();tickIndex();tickState();
</script></body></html>
"""


RAW_HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mark_script · 原始 LLM I/O</title>
<style>
:root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;--accent:#5aa9ff;--bad:#ff7b7b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.65 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:6;display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 16px;background:var(--panel);border-bottom:1px solid var(--line)}
header b{font-size:14px}
nav{display:flex;gap:12px;font-size:13px}
nav a{color:var(--accent);text-decoration:none}nav a.on{color:var(--fg);font-weight:700}
header .grow{flex:1}.dim{color:var(--dim)}
header input{background:#0f141b;border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:4px 9px;font-size:12.5px}
header button{background:#1b2230;border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:4px 10px;font-size:12.5px;cursor:pointer}
header button.on{background:var(--accent);border-color:var(--accent);color:#0b0f15;font-weight:700}
#feed{padding:12px 16px 80px;display:flex;flex-direction:column;gap:10px;max-width:1080px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.card.fail{border-color:#6b2b2b;background:#1a1212}
.hd{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;color:#9fd0ff}
.tool{color:#9fe6c1;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;word-break:break-all}
.failmsg{color:var(--bad);font-size:12px;font-family:ui-monospace,Menlo,Consolas,monospace;word-break:break-all}
details{margin-top:4px}
summary{cursor:pointer;color:var(--dim)}
pre{white-space:pre-wrap;word-break:break-word;background:#0f141b;border:1px solid #1f2836;border-radius:8px;padding:8px 10px;margin:4px 0;font-size:12.5px}
.role{color:#ffb454}.reason{color:var(--dim)}.msg{border-left:2px solid #2b4a72;padding-left:8px;margin:4px 0}
</style></head><body>
<header>
  <b>原始 LLM I/O</b>
  <nav>
    <a href="live.html">台本实时</a>
    <a class="on" href="llm_raw.html">原始 I/O</a>
    <a href="/dashboard" target="_blank">看板</a>
    <a href="/" target="_blank">文件库</a>
    <a id="breeze" href="#" target="_blank">Breeze</a>
  </nav>
  <span class="grow"></span>
  <input id="q" placeholder="过滤…（ch/step/正文/失败）">
  <button id="follow" class="on">跟随最新</button>
  <span class="dim" id="meta"></span>
</header>
<div id="feed"></div>
<script>
const BASE='__LIVE_BASE__';
document.getElementById('breeze').href='http://'+location.hostname+':8137/';
const feed=document.getElementById('feed'), meta=document.getElementById('meta');
const q=document.getElementById('q'), followBtn=document.getElementById('follow');
let offset=0, pending=null, follow=true, shown=0, lastCards=[];
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function msgSummary(m){return '<span class="role">'+esc(m.role||'?')+'</span> <span class="dim">'+((m.content||'').length)+' 字'+(m.tool_calls?(' · 工具调用×'+m.tool_calls.length):'')+(m.tool_call_id?' · 工具结果':'')+'</span>';}
function pairCard(req,res){
  const card=document.createElement('div'); card.className='card';
  const where=(res.chapter!=null?'ch'+res.chapter+' · step'+res.step:(res.source||'llm'));
  const tc=(res.tool_calls||[]).map(function(t){return '<div class="tool">'+esc(t.name)+' '+esc(t.arguments)+'</div>';}).join('');
  card.innerHTML='<div class="hd"><b>'+esc(where)+'</b><span class="dim">'+esc(res.duration_s||'')+'s · finish='+esc(res.finish_reason||'')+(res.tool_calls?(' · 工具×'+res.tool_calls.length):'')+'</span></div>'
    +'<details class="out" open><summary>输出</summary>'
    +(res.reasoning?('<details class="reason"><summary>思考 '+res.reasoning.length+' 字</summary><pre class="rpre"></pre></details>'):'')
    +'<pre>'+esc(res.content||'')+'</pre>'+tc+'</details>'
    +'<details class="in"><summary>输入 · '+(req&&req.messages?req.messages.length:0)+' 条消息</summary><div class="msgs"></div></details>';
  const reason=card.querySelector('.reason');
  if(reason){const pre=reason.querySelector('pre');
    reason.addEventListener('toggle',function(){if(this.open&&!pre.textContent)pre.textContent=res.reasoning;});}
  card.dataset.text=(where+' '+(res.content||'')+' '+(res.reasoning||'')+' '+tc).toLowerCase();
  const msgs=card.querySelector('.msgs');
  card.querySelector('.in').addEventListener('toggle',function(){
    if(this.open&&!msgs.childElementCount&&req){
      req.messages.forEach(function(m){
        const d=document.createElement('details'); d.className='msg';
        d.innerHTML='<summary>'+msgSummary(m)+'</summary><pre>'+esc(m.content||'')+'</pre>';
        msgs.appendChild(d);});
    }
  });
  return card;
}
function tailResults(e,n){
  const out=[], msgs=e.messages||[];
  for(let i=msgs.length-1;i>=0&&out.length<n;i--){const m=msgs[i];
    if(m.role==='tool'||(m.role==='user'&&(m.content||'').indexOf('工具结果：')===0))out.push(m);}
  return out.reverse();
}
function markFail(card,message){
  let raw=(message.content||''); if(raw.indexOf('工具结果：')===0)raw=raw.slice('工具结果：'.length);
  try{const result=JSON.parse(raw||'{}'); if(result.ok!==false)return;
    card.classList.add('fail');
    const line=document.createElement('div'); line.className='failmsg';
    line.textContent='✗ '+(result.reason||'')+' '+(result.text||result.find||'');
    card.querySelector('.hd').appendChild(line);
    card.dataset.text+=' fail '+String(result.reason||'').toLowerCase();
  }catch(err){}
}
function prune(){while(feed.childElementCount>30)feed.removeChild(feed.firstChild);}
function applyFilter(){const v=q.value.trim().toLowerCase();feed.querySelectorAll('.card').forEach(function(c){c.style.display=(!v||c.dataset.text.indexOf(v)>=0)?'':'none';});}
function scrollBottom(){if(follow)window.scrollTo(0,document.body.scrollHeight);}
function handle(e){
  if(e.kind==='request'){
    if(lastCards.length){tailResults(e,lastCards.length).forEach(function(m,i){if(lastCards[i])markFail(lastCards[i],m);});lastCards=[];}
    pending=e; return;
  }
  if(e.kind==='response'){
    const card=pairCard(pending,e); pending=null; feed.appendChild(card); shown++;
    lastCards=(e.tool_calls||[]).map(function(){return card;});
    prune(); applyFilter(); scrollBottom();
    meta.textContent='已记录 '+shown+' 组 · 显示 '+feed.childElementCount;
  }
}
function handleText(txt){for(const line of txt.split('\n')){if(!line.trim())continue;let e;try{e=JSON.parse(line);}catch(_){continue;}handle(e);}}
async function tick(){
  try{const r=await fetch(BASE+'llm_raw.jsonl',{headers:{'Range':'bytes='+offset+'-'},cache:'no-store'});
    if(r.status===416){
      const h=await fetch(BASE+'llm_raw.jsonl',{method:'HEAD',cache:'no-store'});
      const size=+(h.headers.get('Content-Length')||0);
      if(size<offset){const fb=await (await fetch(BASE+'llm_raw.jsonl',{cache:'no-store'})).arrayBuffer();offset=0;feed.innerHTML='';shown=0;handleText(new TextDecoder().decode(fb));offset=fb.byteLength;}
      else offset=size;
    } else if(r.status===206||offset===0){const buf=await r.arrayBuffer();offset+=buf.byteLength;
      handleText(new TextDecoder().decode(buf));}
  }catch(err){}
  setTimeout(tick,1500);
}
q.addEventListener('input',applyFilter);
followBtn.addEventListener('click',function(){follow=!follow;followBtn.classList.toggle('on',follow);scrollBottom();});
window.addEventListener('scroll',function(){if(!follow)return;if(document.documentElement.scrollHeight-window.scrollY-window.innerHeight>160){follow=false;followBtn.classList.remove('on');}});
tick();
</script></body></html>
"""


def write_page(directory: Path) -> Path:
    """Write ``live.html`` for a script directory (callable without touching the log)."""
    app_root = Path(__file__).resolve().parent.parent
    base = "/" + str(directory.resolve().relative_to(app_root)) + "/"
    page = directory / "live.html"
    page.write_text(HTML.replace("__LIVE_BASE__", base), encoding="utf-8")
    return page


def write_raw_page(directory: Path) -> Path:
    """Write ``llm_raw.html`` (raw request/response viewer) for a script directory."""
    app_root = Path(__file__).resolve().parent.parent
    base = "/" + str(directory.resolve().relative_to(app_root)) + "/"
    page = directory / "llm_raw.html"
    page.write_text(RAW_HTML.replace("__LIVE_BASE__", base), encoding="utf-8")
    return page


class Live:
    """Event log + per-chapter fragments + a self-refreshing visual page."""

    def __init__(self, path: Path) -> None:
        self.dir = path.parent
        self.path = path
        self.index_path = self.dir / "live_index.json"
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        self.index: dict = {"current": 0, "chapters": {}}
        self._write_index()
        write_page(self.dir)
        write_raw_page(self.dir)

    def _write_index(self) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def emit(self, type: str, **event) -> None:  # noqa: A002 - 'type' matches the wire format
        event["type"] = type
        event["ts"] = round(time.time(), 3)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def set_state(self, chapter: int, chars: int, fragments: list[dict], done: bool = False) -> None:
        """Overwrite this chapter's canvas state (atomic) and refresh the index."""
        entry = self.index["chapters"].setdefault(str(chapter), {"chars": chars, "done": done})
        entry["chars"] = chars
        entry["done"] = entry.get("done", False) or done
        self.index["current"] = chapter
        payload = json.dumps({"chapter": chapter, "chars": chars, "fragments": fragments}, ensure_ascii=False)
        tmp = (self.dir / f"ch{chapter:03d}.json").with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.dir / f"ch{chapter:03d}.json")
        self._write_index()


LIVE_PORT = int(os.environ.get("AUDIOBOOK_LIVE_PORT", "8899"))


def ensure_live_server(port: int = LIVE_PORT) -> None:
    """Start the LAN/live file server if it is not already answering."""
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/live", timeout=2)
        return
    except Exception:  # noqa: BLE001 - not up yet
        pass
    log_path = APP_ROOT / ".cache" / "logs" / "serve_files.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            [sys.executable, "scripts/serve_files.py", "--dir", str(APP_ROOT), "--port", str(port)],
            cwd=APP_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"[live] serving http://127.0.0.1:{port}/live", flush=True)


LOCAL_SYSTEM = """你只做一件事：找出这段小说文字里**人物直接说的话**，为每一处标出说话人。
- 这只是一段连续的文字（没有章节概念），结合上下文判断谁在说话。
- **连续对话常常没有「某某说道」提示、两人交替**：必须逐句判断，
  谁说的算谁的，**正确识别双方转换**，不要把不同人的话拼到一起。
- **说一句 → 旁白/动作描写 → 同一个人接着说（可能不止两段）**：任意一段都要分别标全
  （一段一个 speak），不要合并成一段、也不要漏掉任何一段。
- **标记只包说话内容**：旁白、动作、描写、叙述都不是台词（没有引号、也没有「说道/忽然想」
  这类提示的就不要标）；「某某说道/淡淡地说道」这类旁白一律留在标记外。
- 正文里**每一句前后都有唯一的切割标记**，形如 `[3A]`、`[22D]`（阿拉伯数字段落号+大写字母；
  相邻句共用中间的标记，行尾也有收尾标记）。标一处就用一次
  edit(op="speak", begin=发言前的标记, end=发言后的标记, text=这段发言的原文, role=说话人规范名)：
  **位置和正文都要给**；一段完整的发言就取它**前面那个标记**和**后面那个标记**
  （只选一句 = 这句话前后的两个标记，例如 `[3B]` 到 `[3C]`），text 抄这段发言原文（可去掉标记）。
  标记自带段落号，不用再给 line；**每处都必须给 begin/end + text，一个都不能少**。
  说话人用人物表里的规范名（没有就按原文写法），判断不出填「未知」。
 只管抓对话；原文的文字和标点一个字都不要改。处理完停下，不要解释。"""

# Few-shot as REAL tool calls (the model learns the tool protocol directly); dialogue only.
FEW_SHOT_CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "正文（切割标记形如 `[3A]`）：\n[3A]他叹道：[3B]“你终于来了。”[3C]",
        [
            '{"op":"speak","begin":"3B","end":"3C","text":"你终于来了。","role":"陈默"}',
        ],
        [
            '{"ok": true, "role": "陈默", "text": "“你终于来了。”", "begin": "3B", "end": "3C", "range": "3B-3C"}',
        ],
    ),
    (
        "正文（切割标记形如 `[7A]`）：\n[7A]“这件事要从很久以前说起，[7B]中间经过了很多波折，[7C]最后我们还是在城南住下了。”[7D]",
        [
            '{"op":"speak","begin":"7A","end":"7D","text":"这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。","role":"陈默"}',
        ],
        [
            '{"ok": true, "role": "陈默", "text": "“这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。”", "begin": "7A", "end": "7D", "range": "7A-7D"}',
        ],
    ),
    (
        "正文（切割标记形如 `[9A]`）：\n[9A]“快走！”[9B]他高声提醒，[9C]“别管我！”[9D]",
        [
            '{"op":"speak","begin":"9A","end":"9B","text":"快走！","role":"陈默"}',
            '{"op":"speak","begin":"9C","end":"9D","text":"别管我！","role":"陈默"}',
        ],
        [
            '{"ok": true, "role": "陈默", "text": "“快走！”", "begin": "9A", "end": "9B", "range": "9A-9B"}',
            '{"ok": true, "role": "陈默", "text": "“别管我！”", "begin": "9C", "end": "9D", "range": "9C-9D"}',
        ],
    ),
    (
        "正文（切割标记形如 `[11C]`）：\n[11A]陈默坐下，[11B]忽然想：[11C]她又熬夜了吧。[11D]",
        [
            '{"op":"speak","begin":"11C","end":"11D","text":"她又熬夜了吧。","role":"陈默"}',
        ],
        [
            '{"ok": true, "role": "陈默", "text": "她又熬夜了吧。", "begin": "11C", "end": "11D", "range": "11C-11D"}',
        ],
    ),
]


def few_shot_messages(cases) -> list[dict]:
    """Build the few-shot message list (real tool_calls) from ``[text, calls, results]`` cases."""
    messages: list[dict] = []
    for case_index, case in enumerate(cases or [], start=1):
        if isinstance(case, dict):
            example = str(case.get("text") or "")
            calls = [str(item) for item in case.get("calls") or []]
            results = [str(item) for item in case.get("results") or []]
        elif isinstance(case, (list, tuple)) and len(case) == 3:
            example, calls, results = str(case[0]), [str(item) for item in case[1]], [str(item) for item in case[2]]
        else:
            continue
        if not example.strip() or not calls:
            continue
        messages.append({"role": "user", "content": example})
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"few{case_index}_{call_index}",
                        "type": "function",
                        "function": {"name": "edit", "arguments": call},
                    }
                    for call_index, call in enumerate(calls, start=1)
                ],
            }
        )
        messages += [
            {
                "role": "tool",
                "tool_call_id": f"few{case_index}_{call_index}",
                "content": results[call_index - 1] if call_index <= len(results) else "{}",
            }
            for call_index in range(1, len(calls) + 1)
        ]
    return messages


FEW_SHOT: list[dict] = few_shot_messages(FEW_SHOT_CASES)

STEP_MARK = (
    "现在只做一件事：从下面【要处理的正文】里抓出**人物直接说的话**。"
    '唯一允许的操作：edit(op="speak", begin=起始标记, end=结束标记, text=这段原文, role=说话人)。'
    "切割标记形如 `[3A]`、`[22D]`（阿拉伯数字段落号+大写字母，自带位置；相邻句共用、行尾也有）。"
    "**铁律**：每处必须同时给 begin、end、text、role 四个参数，缺任何一个都算失败；"
    "begin/end 是这段发言前后两个标记，text 是这段发言原文（照抄，可去掉切割标记），role 是人物表里的规范名。"
    "只标人物直接说的话：旁白、动作、描写、叙述一律禁止标；「某某说道/淡淡地说道」这类话留在标记外。"
    "连续对话必须逐句分清双方；说一句接一段旁白再接着说，必须分开标全。"
    "严禁没说完就结束、严禁把中间旁白包进标记。"
    "不要输出解释或 JSON 文本，只用工具，一次一处，标完立刻停下。"
)

CHECK_MARK = (
    "现在是检查环节：上面【目前的完整标记】里，`<角色>…</角色>` 包住的句子就是「角色」说的；"
    "`[3A]`、`[22D]` 是句子切割标记（阿拉伯数字段落号+大写字母，每句前后都有，相邻句共用）。\n"
    "**第一步：处理【机械扫描】列出的疑似漏标。**逐处判断：确实是人物直接说的话就必须 speak 补上"
    "（begin/end/text/role 缺一不可）；明显是术语/标语/书名/引用（不是人在说话）就跳过不标。\n"
    "**第二步：只允许改确实错的地方，禁止重标任何没问题的段落。**逐处核查以下三类错误：\n"
    "1) 漏标：人物直接说的话没标（连续对话的每一段、旁白隔开的续话、无「某某说道」的交替）；\n"
    "2) 多标/标错：旁白、描写、叙述被标进来，或说话人认错；\n"
    "3) 范围错：多裹了旁白/动作，或没说完就结束。\n"
    "**铁律**：旁白、描写、叙述不是台词，禁止标（尤其严禁把整段叙述标成「未知」）；"
    "标记只能包说话内容，「某某说道/淡淡地说道」这类旁白必须留在标记外。"
    "修正一律用 edit，并且每处必须同时给 begin、end、text、role 四个参数，缺一不可：\n"
    "- 说话人错 或 范围错 → op=speak 重标：给正确的 role 和正确的起止标记；role 没变也要重标收紧；\n"
    "- 漏标 → op=speak 补上；\n"
    "- 整段根本不是台词（全是旁白/描写/叙述）→ op=unmark 去掉这条标记。\n"
    "一次只改 1~2 处，只改真有问题的。改完立刻停下，不要解释。"
)

DEFAULT_CHECK_STEPS = 100  # tool steps for the check pass (0 = no check at all)
CHECK_ROUNDS = 1  # one marking pass, then one check pass (raise to iterate more)

ROSTER_SYSTEM = """你在维护一部小说的「人物词典」。输入是「已有词典」和「这段文本」。
把这段文本里出现、能指向具体人物的人整理进词典。

- 规范名：此人最完整/最正式的姓名（如"高文·塞西尔"）；同一人只留一条。
- **输出 JSON 的键必须全部是人物规范名**；禁止出现 aliases/name 这类字段名当键。
- **主词（规范名）必须是全名/全称**：简称/称谓/绰号都放进 aliases。就算先见到简称、后见到全名，
  也用全名当主词（同一人只留一条，旧的简称并进去）。
- aliases 见到多少收多少（模糊名称 + 能唯一定位的称谓/绰号），**不限数量**。
- **已有词典里能对上的就是同一人，必须沿用已有规范名**：已有「高文·塞西尔」就把「高文/老祖宗」并进去，
  禁止新增「高文」；已有「瑞贝卡·塞西尔」就不要新增「瑞贝卡」。
- **绝不收**代词/描述性短语/整句/泛称（大人/老爷/小姐/先生/骑士）；不收地名/组织/物品/种族/群体
  （混血精灵/士兵/众人这类一律不收，只收有名有姓的个人）。
- 已有条目只增补，不改名、不删除；**不要写声线/年龄/性别**（声线在造音色时按人物书重新设计）。

只输出 JSON：
{"规范名": {"aliases": ["高文", "老祖宗"]}}
"""

TOOLS = ["edit"]

SUMMARY_SYSTEM = """你在维护一部长篇小说的「前情摘要」，供后续章节判断「谁在说话」时做背景。
输入：上一版摘要、新增章节的出场角色与正文（可能多章）。输出新摘要（≤400 字），只保留判断说话人需要的信息：
- 人物关系/身份、当前地点与处境、正在发生的事件；新出现的称呼点明归属。
- 上一版里已经过时或无关的信息删掉；不要文学赏析，不要剧透后文。
只输出 JSON：{"summary": "……"}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter", type=int, help="first chapter id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--book", default="dawn")
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="rolling context window in chapters: first N chapters are fed in full, then a rolling summary takes over",
    )
    parser.add_argument("--max-steps", type=int, default=200, help="tool steps per chapter")
    parser.add_argument(
        "--check-steps",
        type=int,
        default=DEFAULT_CHECK_STEPS,
        help="tool steps for the end-of-chapter check pass (0 = off)",
    )
    parser.add_argument("--force", action="store_true", help="redo chapters that already have a marked file")
    parser.add_argument("--no-live", action="store_true", help="do not start the live page server")
    parser.add_argument("--live-port", type=int, default=LIVE_PORT)
    return parser.parse_args()


def load_roster(book: str) -> dict[str, list[str]]:
    """Resume the one-to-many dictionary from ``script/roles.json`` (empty on a cold start).

    No cast is preloaded: the dictionary is grown incrementally, chapter by chapter, so the
    model can only use names it actually met in the text.
    """
    path = APP_ROOT / "outputs" / book / "script" / "roles.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    clean: dict[str, list[str]] = {}
    # Longest names first: the full name becomes the canonical entry and absorbs its short forms.
    for name in sorted(data, key=len, reverse=True):
        labels = data[name]
        if name.lower() in ROSTER_STRUCT_KEYS or not isinstance(labels, list):
            continue
        _absorb_entry(clean, name, [str(label).strip() for label in labels if str(label).strip()])
    return clean


def load_summary(book: str) -> str:
    path = APP_ROOT / "outputs" / book / "script" / "summary.txt"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def save_summary(book: str, summary: str) -> None:
    path = APP_ROOT / "outputs" / book / "script" / "summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary.strip() + "\n", encoding="utf-8")


def load_summary_upto(book: str) -> int:
    path = APP_ROOT / "outputs" / book / "script" / "summary_upto.txt"
    try:
        return int(path.read_text(encoding="utf-8").strip()) if path.is_file() else 0
    except ValueError:
        return 0


def save_summary_upto(book: str, chapter: int) -> None:
    path = APP_ROOT / "outputs" / book / "script" / "summary_upto.txt"
    path.write_text(str(chapter), encoding="utf-8")


def compress(llm: LLMClient, chapter_text: str, roles: list[str], previous: str) -> str:
    """One small call per chapter: fold this chapter into the rolling summary (≤400 chars)."""
    try:
        result = llm.chat_json(
            SUMMARY_SYSTEM,
            f"【上一版摘要】\n{previous or '（无）'}\n\n【本章出场角色】\n{'、'.join(dict.fromkeys(roles)) or '（无）'}\n\n"
            f"【本章正文】\n{chapter_text}",
            max_tokens=1024,
            thinking=False,
        )
    except Exception as error:  # noqa: BLE001 - summary is best-effort, keep the old one
        print(f"  [summary] 压缩失败：{str(error)[:120]}", flush=True)
        return previous
    if isinstance(result, dict):
        summary = str(result.get("summary") or result.get("摘要") or "").strip()
    else:
        summary = str(result).strip()
    return summary or previous


# Keys the model sometimes emits as if they were character names; never roster entries.
ROSTER_STRUCT_KEYS = frozenset(
    {"aliases", "alias", "voice", "name", "age", "gender", "sample", "规范名", "别名", "声线", "人物", "词典"}
)


def _same_person(roster: dict[str, list[str]], existing: str, name: str, labels: list[str]) -> bool:
    """Entry ``existing`` and incoming ``(name, labels)`` look like one person."""
    known = roster.get(existing, [])
    if existing in labels or name in known:
        return True
    if len(existing) >= 2 and len(name) >= 2 and (existing in name or name in existing):
        return True
    return False


def _absorb_entry(roster: dict[str, list[str]], name: str, labels: list[str]) -> str:
    """Fold ``(name, labels)`` into the dictionary and return the canonical name.

    The full/formal name always wins as canonical -- even when only the short form was seen
    first, the later full name is promoted and the whole entry moves with it. Aliases have no
    cap: the dictionary may grow as the text flows."""
    bucket: list[str] = [name, *labels]
    canonical = name
    for existing in [key for key in roster if _same_person(roster, key, name, labels)]:
        canonical = existing if len(existing) > len(canonical) else canonical
        bucket.extend(roster.pop(existing))
    roster[canonical] = [canonical] + [item for item in dict.fromkeys(bucket) if item and item != canonical]
    return canonical


def maintain_roster(llm, roster: dict[str, list[str]], text: str) -> int:
    """Fold the people seen in this piece of text into the dictionary (no chapters, no voice)."""
    existing = "\n".join(f"{name}: {'、'.join(labels)}" for name, labels in roster.items())
    try:
        result = llm.chat_json(
            ROSTER_SYSTEM,
            f"【已有词典】\n{existing or '（空）'}\n\n【这段文本】\n{text}",
            thinking=False,  # extraction task: thinking only risks echoing the prompt / truncating JSON
        )
    except Exception as error:  # noqa: BLE001 - dictionary is best-effort
        print(f"  [roster] 维护失败：{str(error)[:120]}", flush=True)
        return 0
    if isinstance(result, dict):  # tolerate wrapper keys the model likes to add
        for key in ("人物词典", "roles", "人物", "characters", "roster"):
            if key in result and isinstance(result[key], (dict, list)):
                result = result[key]
                break
        items = list(result.items())
    elif isinstance(result, list):  # tolerate [{"规范名": ..., "aliases": [...]}, ...]
        items = [
            (item.get("规范名") or item.get("name") or item.get("canonical") or "", item)
            for item in result
            if isinstance(item, dict)
        ]
    else:
        return 0
    added = 0
    for name, entry in items:
        name = str(name).strip()
        if not name or name.lower() in ROSTER_STRUCT_KEYS:
            continue
        if isinstance(entry, dict):
            labels = [str(label).strip() for label in entry.get("aliases") or [] if str(label).strip()]
        elif isinstance(entry, list):  # tolerate the older {"name": [labels]} shape
            labels = [str(label).strip() for label in entry if str(label).strip()]
        else:
            labels = []
        before_labels = {item for bucket in roster.values() for item in bucket}
        _absorb_entry(roster, name, labels)
        added += len({item for bucket in roster.values() for item in bucket} - before_labels)
    if added:
        print(f"  [roster] +{added} 标签 · 词条 {len(roster)}", flush=True)
    return added


def resolve_label(roster: dict[str, list[str]], name: str) -> str:
    """Map a written label/name to its canonical entry (or register it as new)."""
    for canonical, labels in roster.items():
        if name == canonical or name in labels:
            return canonical
    return _absorb_entry(roster, name, [])


_THOUGHT_RE = re.compile(r"<\|?channel\|?>.*?<\|?channel\|>", re.DOTALL)


def _clean_assistant(content: str) -> str:
    """Strip gemma's leaked thought channel from assistant content before it enters history."""
    content = _THOUGHT_RE.sub("", content or "")
    for token in ("<|channel>", "<channel|>", "<|channel|>"):
        content = content.replace(token, "")
    return content.strip()


def _parse_args(raw: str | None) -> dict | None:
    try:
        data = json.loads(raw or "{}")
    except Exception:  # noqa: BLE001 - malformed tool-call JSON
        return None
    return data if isinstance(data, dict) else None


def _extract_edit(content: str) -> dict | None:
    """E4B often writes the edit as a JSON object in the text instead of a tool call."""
    from audiobook.llm import extract_json

    try:
        data = extract_json(content)
    except Exception:  # noqa: BLE001 - no JSON in the reply
        return None
    if isinstance(data, dict) and ("op" in data or "role" in data or "find" in data or "text" in data):
        return data
    return None


def run_turn(
    client, config, tools, mcp, messages, max_steps, counters, live: Live | None = None, chapter: int = 0, original: str = ""
) -> int:
    """Tool loop until the model stops calling tools. Guards against a failing retry loop.

    Returns the number of successful edits (0 = nothing changed)."""
    errors = 0
    edits_ok = 0
    last_sig, repeats, fail_total = "", 0, 0
    ok_sig, ok_repeats = "", 0
    for _step in range(1, max_steps + 1):
        started = time.perf_counter()
        raw_log(
            {
                "kind": "request",
                "source": "mark",
                "chapter": chapter,
                "step": _step,
                "model": config.model,
                "thinking": THINK,
                "messages": messages,
                "tools": [tool["function"]["name"] for tool in tools or []],
            }
        )
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=tools or None,
                tool_choice="auto" if tools else None,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=4096,
                extra_body={"top_k": config.top_k, "chat_template_kwargs": {"enable_thinking": THINK}},
            )
        except Exception as error:  # noqa: BLE001 - malformed tool-call JSON -> retry
            errors += 1
            print(f"  [llm] error {errors}: {str(error)[:140]}", flush=True)
            if errors > 6:
                return edits_ok
            messages.append({"role": "user", "content": "上一条工具调用参数 JSON 非法；请一次只改一处后重试。"})
            continue
        duration = time.perf_counter() - started
        counters["llm_s"] += duration
        counters["llm_calls"] += 1
        message = response.choices[0].message
        raw_log(
            {
                "kind": "response",
                "source": "mark",
                "chapter": chapter,
                "step": _step,
                "model": config.model,
                "duration_s": round(duration, 2),
                "content": message.content,
                "reasoning": reasoning_of(message),
                "finish_reason": response.choices[0].finish_reason,
                "tool_calls": [
                    {"name": call.function.name, "arguments": call.function.arguments} for call in (message.tool_calls or [])
                ],
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        content = _clean_assistant(message.content or "")
        finish_reason = response.choices[0].finish_reason
        # A response cut by the token limit may carry a half-written tool call; replaying it
        # makes llama.cpp reject the whole history ("Failed to parse tool call arguments").
        # Keep only calls whose arguments are valid JSON, and nudge the model when truncated.
        dropped = 0
        tool_calls = []
        for tc in message.tool_calls or []:
            arguments = tc.function.arguments or "{}"
            if _parse_args(arguments) is None:
                dropped += 1
                continue
            tool_calls.append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": arguments}})
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls or None})
        if dropped or (finish_reason == "length" and not tool_calls):
            messages.append(
                {"role": "user", "content": "上一条回复被截断或参数 JSON 不完整；一次只标一处，参数要完整（text 抄短一点）。"}
            )
            errors += 1
            if errors > 6:
                return edits_ok
            if not tool_calls:
                continue
        if live is not None and content:
            live.emit("assistant", chapter=chapter, content=content[:2000])
        if not tool_calls:
            edit = _extract_edit(content)  # E4B often writes the edit as text JSON
            if edit is None:
                return edits_ok
            if "role" in edit and "op" not in edit:
                edit["op"] = "speak"
            try:
                result = mcp.call("edit", edit)
            except Exception as error:  # noqa: BLE001
                result = json.dumps({"ok": False, "reason": f"tool error: {error}"}, ensure_ascii=False)
            counters["tool_calls"] += 1
            print(f"  edit(text) {str(edit)[:60]} -> {result[:70]}", flush=True)
            if live is not None:
                live.emit("tool", chapter=chapter, call="edit", args=edit)
                live.emit(
                    "result", chapter=chapter, ok=('"ok": false' not in result and "not found" not in result), result=result[:500]
                )
            messages.append({"role": "user", "content": f"工具结果：{result}"})
            sig = json.dumps(edit, ensure_ascii=False, sort_keys=True)
            if '"ok": false' in result or "not found" in result:
                fail_total += 1
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                messages.append(
                    {
                        "role": "user",
                        "content": "找不到目标。从【要处理的正文】里逐字复制一段重试（开头约 10 字）；"
                        "句子长时再给结尾约 10 字；一次一处，不是直接对话的不要标；没有就停下。",
                    }
                )
                if repeats >= 3 or fail_total >= 12:
                    return edits_ok
            elif '"already": true' in result:  # no change: do not count as progress, nudge forward
                ok_repeats = ok_repeats + 1 if sig == ok_sig else 1
                ok_sig = sig
                if ok_repeats >= 3:
                    return edits_ok
                messages.append({"role": "user", "content": "这处已经处理过，不要重复；继续找还没标出的对话。"})
            else:
                edits_ok += 1
                last_sig, repeats = "", 0
            continue
        for call in tool_calls:
            arguments = _parse_args(call["function"]["arguments"])
            if not arguments:  # malformed or empty: tell the model and move on (never crash)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": '{"ok": false, "reason": "arguments 不是合法 JSON，请一次只改一处，用 {"op":...} 重发"}',
                    }
                )
                messages.append({"role": "user", "content": "上一条工具调用参数不是合法 JSON；请一次只改一处后重试。"})
                errors += 1
                if errors > 12:
                    return edits_ok
                continue
            if "role" in arguments and "op" not in arguments:
                arguments["op"] = "speak"
            sig = call["function"]["name"] + call["function"]["arguments"]
            try:
                result = mcp.call(call["function"]["name"], arguments)
            except Exception as error:  # noqa: BLE001 - bad args must not kill the run
                result = json.dumps({"ok": False, "reason": f"tool error: {error}"}, ensure_ascii=False)
            counters["tool_calls"] += 1
            print(f"  {call['function']['name']} {str(arguments)[:60]} -> {result[:70]}", flush=True)
            if live is not None:
                live.emit("tool", chapter=chapter, call=call["function"]["name"], args=arguments)
                live.emit(
                    "result", chapter=chapter, ok=('"ok": false' not in result and "not found" not in result), result=result[:500]
                )
                current = json.loads(mcp.call("get_marked", {}))["text"]
                live.set_state(chapter, len(current), live_fragments(original, current))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            if '"ok": false' in result or "not found" in result:
                fail_total += 1
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                messages.append(
                    {
                        "role": "user",
                        "content": "找不到目标。从【要处理的正文】里逐字复制一段重试（开头约 10 字）；"
                        "句子长时再给结尾约 10 字；一次一处，不是直接对话的不要标；没有就结束。",
                    }
                )
                if fail_total >= 5:
                    return edits_ok
            elif '"already": true' in result:  # no change: do not count as progress, nudge forward
                ok_repeats = ok_repeats + 1 if sig == ok_sig else 1
                ok_sig = sig
                if ok_repeats >= 3:
                    return edits_ok
                messages.append({"role": "user", "content": "这处已经处理过，不要重复；继续找还没标出的对话。"})
            else:
                edits_ok += 1
    return edits_ok


def _stage_note(book: str, status: str, note: str) -> None:
    import json as _json
    import time as _time

    path = APP_ROOT / "outputs" / book / "pipeline.json"
    data = _json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"stages": {}}
    data.setdefault("stages", {})["script"] = {"status": status, "note": note, "ts": _time.time()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_workflow_overrides(args: argparse.Namespace) -> dict:
    """Overlay ``outputs/<book>/workflow.json`` (edited on the web) over prompts/params.

    The overlay wins over CLI values so the web page is the one place to iterate the flow.
    """
    path = APP_ROOT / "outputs" / args.book / "workflow.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    applied: dict = {}
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    for key, target in (("batch", "batch"), ("max_steps", "max_steps"), ("check_steps", "check_steps")):
        value = params.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            setattr(args, target, max(0 if target == "check_steps" else 1, value))
            applied[target] = getattr(args, target)
    if isinstance(params.get("think"), bool):
        globals()["THINK"] = params["think"]
        applied["THINK"] = params["think"]
    prompts = data.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {}
    for key, name in (
        ("local_system", "LOCAL_SYSTEM"),
        ("step_mark", "STEP_MARK"),
        ("roster_system", "ROSTER_SYSTEM"),
        ("check_mark", "CHECK_MARK"),
    ):
        value = prompts.get(key)
        if isinstance(value, str) and value.strip():
            globals()[name] = value
            applied[name] = "overlay"
    cases = data.get("few_shot")
    if isinstance(cases, list) and cases:
        built = few_shot_messages(cases)
        if built:
            globals()["FEW_SHOT"] = built
            applied["FEW_SHOT"] = len(cases)
    return applied


def _pid_alive(pid: int) -> bool:
    """True only for a live (non-zombie) process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    state = stat.rsplit(") ", 1)[-1][:1]
    return state not in ("Z", "")


def main() -> None:
    args = parse_args()
    overrides = apply_workflow_overrides(args)
    if overrides:
        print(f"[workflow] 覆盖: {json.dumps(overrides, ensure_ascii=False)}", flush=True)
    if not args.no_live:
        ensure_live_server(args.live_port)
    config = config_from_env()
    if config is None:
        raise SystemExit("no LLM configured")
    from openai import OpenAI  # type: ignore

    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=900)
    llm = LLMClient(config)
    mcp = MCPClient()
    mcp.initialize()
    all_tools = {t["function"]["name"]: t for t in mcp.openai_tools()}
    tools = [all_tools[name] for name in TOOLS]

    chapters = APP_ROOT / "outputs" / args.book / "chapters"
    out_dir = APP_ROOT / "outputs" / args.book / "script"
    out_dir.mkdir(parents=True, exist_ok=True)

    lock = out_dir / "run.pid"
    if lock.is_file():
        try:
            other = int(lock.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            other = 0
        if other and other != os.getpid() and not _pid_alive(other):
            other = 0
        if other:
            raise SystemExit(f"已有 run 在跑（pid {other}，同一本书）；不要并发跑两个，等它结束或先停掉")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    (out_dir / "llm_raw.jsonl").write_text("", encoding="utf-8")
    os.environ["AUDIOBOOK_LLM_RAW_LOG"] = str(out_dir / "llm_raw.jsonl")
    if args.count > 0:
        ids = [cid for cid in range(args.chapter, args.chapter + args.count) if (chapters / f"ch{cid:03d}.txt").is_file()]
    else:  # count<=0 -> every chapter from `chapter` to the end
        ids = [cid for cid in sorted(int(p.stem[2:]) for p in chapters.glob("ch*.txt")) if cid >= args.chapter]

    live = Live(out_dir / "live.jsonl")
    live.emit("start", total=len(ids), book=args.book, batch=args.batch)
    roster = load_roster(args.book)
    if roster:
        print(f"[resume] 已有词条 {len(roster)}", flush=True)
    counters = {"llm_s": 0.0, "llm_calls": 0, "tool_calls": 0}
    started = time.perf_counter()
    phases: list[dict] = []

    window = max(1, args.batch)  # first `window` chapters are fed in full, then the summary rolls
    summary = load_summary(args.book)
    summary_upto = load_summary_upto(args.book)
    if summary:
        print(f"[resume] 前情摘要 {len(summary)} 字（已含至第 {summary_upto} 章）", flush=True)

    def chapter_path(cid: int) -> Path:
        return chapters / f"ch{cid:03d}.txt"

    def roles_in(cid: int) -> list[str]:
        marked = out_dir / f"ch{cid:03d}.marked.txt"
        if not marked.is_file():
            return []
        return [seg["role_name"] for seg in parse_marks(marked.read_text(encoding="utf-8")) if seg["kind"] == "speech"]

    def seed_summary() -> None:
        """First summary, built from the whole accumulated window in one call."""
        nonlocal summary, summary_upto
        block = "\n\n".join(f"【第 {k} 章】\n{chapter_path(k).read_text(encoding='utf-8')}" for k in range(1, window + 1))
        summary = compress(llm, block, [role for k in range(1, window + 1) for role in roles_in(k)], "")
        summary_upto = window
        save_summary(args.book, summary)
        save_summary_upto(args.book, summary_upto)
        print(f"  [summary] 窗口前 {window} 章 -> {len(summary)} 字", flush=True)

    def fold_through(cid: int) -> None:
        """Make the summary cover chapters 1..cid; no-op while the window is still filling."""
        nonlocal summary, summary_upto
        if cid < window:
            return
        if summary_upto < window:
            seed_summary()
        while summary_upto < cid:
            k = summary_upto + 1
            summary = compress(llm, chapter_path(k).read_text(encoding="utf-8"), roles_in(k), summary)
            summary_upto = k
            save_summary(args.book, summary)
            save_summary_upto(args.book, summary_upto)
            print(f"  [summary] +第 {k} 章 -> {len(summary)} 字", flush=True)

    def prior_context(cid: int) -> str:
        """Full previous chapters while the window fills; the rolling summary afterwards."""
        if cid <= window:
            prior = "\n\n".join(f"【第 {k} 章】\n{chapter_path(k).read_text(encoding='utf-8')}" for k in range(1, cid))
            return (
                f"【前情（第 1..{cid - 1} 章原文，仅用于判断说话人；这些章节都已处理，不要处理）】\n{prior}"
                if prior
                else "（本章是开头）"
            )
        return f"【前情摘要】\n{summary or '（无）'}"

    for cid in ids:
        target = out_dir / f"ch{cid:03d}.marked.txt"
        raw_text = chapter_path(cid).read_text(encoding="utf-8")
        skipped = target.is_file() and not args.force
        if skipped:  # resumable: skip chapters already marked
            snapshot = target.read_text(encoding="utf-8")
            if not (out_dir / f"ch{cid:03d}.json").is_file():  # backfill canvas state for the picker
                live.set_state(cid, len(snapshot), live_fragments(raw_text, snapshot), done=True)
            print(f"[skip] ch{cid:03d} 已存在", flush=True)
        else:
            # Dictionary = side product of the text flowing through: fold this piece of text in,
            # then mark it. No chapters, no voice profiles here (voice is designed from the
            # dictionary at the voicebank stage).
            added = maintain_roster(llm, roster, raw_text)
            (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
            live.emit("roster", chapter=cid, added=added, roles=len(roster))
            names = "、".join(roster)  # consistency hint, not a gate: new names are registered as seen
            hint = f"已有人物（尽量沿用这些名字）：{names}\n\n" if names else ""
            live.emit("chapter", chapter=cid, chars=len(raw_text))
            live.set_state(cid, len(raw_text), live_fragments(raw_text, raw_text))
            mcp.call("set_text", {"text": raw_text})
            fold_through(cid - 1)  # summary must cover everything before this chapter
            think = config.thinking_system_token if THINK else ""  # profile-driven (Gemma: "<|think|>", Qwen: none)
            # Sentence-axis cuts carry the paragraph number (`[3A]`, `[22D]`): every cut label
            # locates its place by itself, and the MCP marks exactly the span between two labels.
            numbered = number_text(raw_text)
            # Few-shot goes AFTER the window and right BEFORE the task, so the examples sit next
            # to the instruction the model is about to execute (not buried above the big text).
            messages = [
                {"role": "system", "content": f"{think}{LOCAL_SYSTEM}\n\n{hint}"},
                *FEW_SHOT,
                {
                    "role": "user",
                    "content": f"{prior_context(cid)}\n\n【要处理的正文（每句前后有切割标记，形如 `[3A]`）】\n{numbered}",
                },
                {"role": "user", "content": STEP_MARK},
            ]
            run_turn(client, config, tools, mcp, messages, args.max_steps, counters, live, cid, raw_text)
            checks = 0
            for check_round in range(1, CHECK_ROUNDS + 1):
                if not args.check_steps:
                    break
                marked_view = number_text(json.loads(mcp.call("get_marked", {}))["text"])
                missing = mcp.server.unmarked_quotes()
                check_prior = f"【前情摘要（只作背景参考，不要处理）】\n{summary or '（无）'}"
                if missing:
                    listed = "\n".join(
                        f"- {item['begin']}~{item['end']}：{item['text']} ｜ 出处：{item['snippet']}" for item in missing[:60]
                    )
                    scan = (
                        f"【机械扫描：以下 {len(missing)} 处都在这章的正文里（标记号如 22F 都是本章的，"
                        f"与前情无关），引号内容尚未标记】\n{listed}"
                    )
                else:
                    scan = "【机械扫描：没有未标记的引号内容】"
                round_note = f"这是第 {check_round}/{CHECK_ROUNDS} 轮检查。"
                check_messages = [
                    {"role": "system", "content": f"{think}{LOCAL_SYSTEM}\n\n{hint}"},
                    {
                        "role": "user",
                        "content": (
                            f"{check_prior}\n\n【本章目前的完整标记（`<角色>…</角色>` 表示这段是「角色」说的）】\n{marked_view}"
                        ),
                    },
                    {"role": "user", "content": scan},
                    {"role": "user", "content": round_note + CHECK_MARK},
                ]
                run_turn(client, config, tools, mcp, check_messages, args.check_steps, counters, live, cid, raw_text)
                checks += 1
            snapshot = json.loads(mcp.call("get_marked", {}))["text"]
            # No end-of-chapter fill-in pass: whatever the edit loop produced stands; only the
            # mechanical cleanup below runs.
            for seg in parse_marks(snapshot):  # normalise any label/alias to the canonical name
                if seg["kind"] != "speech":
                    continue
                canonical = resolve_label(roster, seg["role_name"])
                if canonical != seg["role_name"]:
                    snapshot = snapshot.replace(f"<{seg['role_name']}>", f"<{canonical}>")
            # Quotes are part of the dialogue sentence and stay inside the marks: no cleanup.
            (out_dir / f"ch{cid:03d}.marked.txt").write_text(snapshot, encoding="utf-8")
            (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
            missed = mcp.server.unmarked_quotes()
            phases.append({"chapter": cid, "chars": len(snapshot), "roles": len(roster), "missed": len(missed), "checks": checks})
            print(f"[mark] ch{cid:03d} chars={len(snapshot)} 词条={len(roster)} 漏标引号={len(missed)}", flush=True)
            live.emit("missed", chapter=cid, count=len(missed), samples=[item["text"][:20] for item in missed[:5]])
            _stage_note(args.book, "run", f"{len(phases)}/{len(ids)}")
            live.set_state(cid, len(snapshot), live_fragments(raw_text, snapshot), done=True)
        fold_through(cid)  # skipped or marked: keep the rolling summary current
        live.emit("done", chapter=cid, roles=len(roster), summary=summary[:200])

    marked = "\n".join((out_dir / f"ch{cid:03d}.marked.txt").read_text(encoding="utf-8") for cid in ids)
    (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")

    stem = f"ch{ids[0]:03d}_x{len(ids)}"
    render_html(parse_marks(marked), marked, out_dir / f"{stem}.html")
    original = "\n".join((chapters / f"ch{cid:03d}.txt").read_text(encoding="utf-8") for cid in ids)
    render_diff_html(original, marked, out_dir / f"{stem}.diff.html")

    timing = {
        "start": args.chapter,
        "count": len(ids),
        "batch": args.batch,
        "wall_s": round(time.perf_counter() - started, 1),
        "llm_s": round(counters["llm_s"], 1),
        "llm_calls": counters["llm_calls"],
        "tool_calls": counters["tool_calls"],
        "roles": len(roster),
        "phases": phases,
    }
    print(f"\n[timing] {json.dumps(timing, ensure_ascii=False)}")
    print(f"[html] {out_dir / f'{stem}.html'}")


if __name__ == "__main__":
    main()
