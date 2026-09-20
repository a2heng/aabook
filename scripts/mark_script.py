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

THINK = True  # 标注/检查/词典均开思考
REASONING_BUDGET = int(os.environ.get("AUDIOBOOK_REASONING_BUDGET", "4096"))  # 思考上限
OUTPUT_TOKENS = int(os.environ.get("AUDIOBOOK_OUTPUT_TOKENS", "2048"))  # 思考之外的输出上限
MAX_TOKENS = REASONING_BUDGET + OUTPUT_TOKENS  # max_tokens = 思考 + 输出

from audiobook.llm import config_from_env, raw_log, reasoning_of  # noqa: E402
from audiobook.marks import (  # noqa: E402
    NARRATOR,
    live_fragments,
    parse_marks,
    render_diff_html,
    render_html,
)
from audiobook.schema import Cast  # noqa: E402


OPEN = "“『「"
CLOSE = "”』」"


def quote_spans(text: str) -> list[tuple[int, int]]:
    """Absolute ``(start, stop)`` of every quoted run, in reading order (``<role>`` tags
    transparent). A run is an opening quote up to its matching closing quote."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "<":  # skip a mark tag
            close = text.find(">", index)
            if close != -1 and "\n" not in text[index:close]:
                index = close + 1
                continue
        if char in OPEN:
            close = next((j for j in range(index + 1, len(text)) if text[j] in CLOSE), -1)
            if close != -1:
                spans.append((index, close + 1))
                index = close + 1
                continue
        index += 1
    return spans


def number_text(text: str) -> str:
    """The text as the model sees it: a number in front of every opening quote (``[1]“…”``).

    The number is the handle the model references; not every numbered quote is dialogue."""
    spans = quote_spans(text)
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for number, (start, _stop) in enumerate(spans, start=1):
        pieces.append(text[cursor:start])
        pieces.append(f"[{number}]")
        cursor = start
    pieces.append(text[cursor:])
    return "".join(pieces)


def label_number(label) -> int:
    """Parse a quote handle: ``1``, ``[1]``, ``q1``, ``第1处`` all -> 1; 0 when unparseable."""
    raw = str(label or "").strip()
    digits = "".join(char for char in raw if char.isdigit())
    return int(digits) if digits else 0


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
        self.roster: dict[str, list[str]] = {}
        self._roster_updates: list[tuple[str, int]] = []

    # ---- tools ---------------------------------------------------------------
    def set_text(self, text: str) -> dict:
        self.text = text or ""
        self.original = self.text  # pristine, never marked
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
        end: str | int = "",  # legacy text anchor (no-line mode)
        line: str | int = 0,  # quote number (the `[n]` in front of an opening quote)
        begin=None,  # alias of `line`
        n: str | int = 0,  # alias of `line`
        roster: dict | None = None,  # roster update for op="roster"
    ) -> dict:
        """The one editing tool. ``op`` defaults to ``speak`` -- never to a text rewrite."""
        op = (op or "speak").strip() or "speak"
        number = n or line or (begin if str(begin or "").strip() else 0)
        if op == "speak":
            return self._mark_speaker(text or find, role, end, number, begin)
        if op == "unmark":
            return self._unmark(text or find, end, number, begin)
        if op == "done":
            return {"ok": True, "done": True}
        if op == "replace":
            return self._replace(find or text, replace)
        if op == "delete":
            return self._delete(text or find)
        if op == "roster":
            return self._update_roster(roster or {})
        return {"ok": False, "reason": f"未知操作 {op}；只能用 speak/unmark/done/roster"}

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

    def _quote_span(self, number) -> tuple[int, int] | None:
        """Absolute ``(start, stop)`` of quote number ``number`` (1-based, in reading order)."""
        index = label_number(number)
        spans = quote_spans(self.text)
        if index < 1 or index > len(spans):
            return None
        return spans[index - 1]

    def _sentence_span(self, start: int, stop: int) -> tuple[int, int]:
        """Grow an UNQUOTED speech fragment to its sentence: stop left at a cue colon / sentence
        end, stop right at the next 。！？ (or newline). Already-terminated fragments stay as-is."""
        left = start
        while left > 0 and self.text[left - 1] not in "。！？\n：；;：":
            left -= 1
        right = stop
        if not (stop > start and self.text[stop - 1] in "。！？"):
            while right < len(self.text) and self.text[right] not in "。！？\n":
                right += 1
            if right < len(self.text) and self.text[right] in "。！？":
                right += 1  # include the closing punctuation
        return left, right

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

    def unmarked_quotes(self) -> list[dict]:
        """Every numbered quote not covered by any mark tag (candidate misses), in order.

        The check pass gets this list so the model only has to say WHO speaks each run."""
        found: list[dict] = []
        for number, (run_start, run_end) in enumerate(quote_spans(self.text), start=1):
            if self._enclosing_tag(run_start, run_end, strict_stop=False) is not None:
                continue
            line_no = self.text.count("\n", 0, run_start) + 1
            visible_text = self._visible_tags(self.text[run_start:run_end])[0]
            snippet = self._visible_tags(self.text[max(0, run_start - 24) : run_end + 24])[0].strip()
            found.append(
                {
                    "line": line_no,
                    "begin": str(number),
                    "end": str(number),
                    "number": number,
                    "text": visible_text,
                    "snippet": snippet,
                    "partial": "<" in self.text[run_start:run_end],
                }
            )
        return found

    def _unmark(self, text: str = "", end: str | int = "", line: str | int | None = 0, begin=None) -> dict:
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
        self.edits.append({"op": "unmark", "role": role, "text": text, "n": line, "begin": begin, "end": end})
        return {
            "ok": True,
            "unmarked": role,
            "text": inner[:24],
            "marked": inner[:400],
            "n": line,
        }

    def _resolve_span(self, text: str, end: str | int = "", line: str | int | None = 0, begin=None):
        """Resolve the target span; returns (start, stop, low, high, exact) or an error dict.

        Preferred handle: the quote number in ``line`` (``begin``/``end`` are aliases).
        ``text`` is the fallback (exact, then anchor matching for long needles)."""
        low, high = (0, len(self.text))
        number = label_number(line) or label_number(begin) or label_number(end)
        if number:
            span = self._quote_span(number)
            if span is not None:
                return span[0], span[1], low, high, True
            if not text:
                total = len(quote_spans(self.text))
                return {
                    "ok": False,
                    "reason": f"没有第 {number} 处引号（本章共 {total} 处）",
                    "text": str(number)[:24],
                }
        if text:
            located = self._locate_span(text)
            if located is not None:
                return located[0], located[1], low, high, False
            return {"ok": False, "reason": "找不到这段原文；请给引号序号（或逐字照抄完整发言）", "text": text[:24]}
        if not number:
            return {"ok": False, "reason": "请给引号序号（行首方括号里的数字）", "text": text[:24]}
        return {"ok": False, "reason": "定位失败", "text": text[:24]}

    @staticmethod
    def _visible_tags(region: str) -> tuple[str, list[int]]:
        """Strip ``<role>`` tags only (keep 【…】/[…] text), with visible->original mapping."""
        visible: list[str] = []
        mapping: list[int] = []
        index = 0
        while index < len(region):
            if region[index] == "<":
                close = region.find(">", index)
                if close != -1 and "\n" not in region[index:close]:
                    index = close + 1
                    continue
            visible.append(region[index])
            mapping.append(index)
            index += 1
        return "".join(visible), mapping

    def _overwrite(self, cut: int, spans: list[tuple[int, int]], role: str) -> list[str]:
        """Overwrite: drop every existing tag intersecting the region, then wrap each span
        (one tag per span). Returns the visible texts that got wrapped."""
        region_start = min([cut] + [start for start, _ in spans])
        region_end = max(stop for _, stop in spans)
        tag = self._enclosing_tag(spans[0][0], spans[-1][1], strict_stop=False)
        if tag is not None:
            open_lt, _open_gt, close_lt, existing = tag
            region_start = min(region_start, open_lt)
            region_end = max(region_end, close_lt + len(existing) + 3)
        region = self.text[region_start:region_end]
        visible, mapping = self._visible_tags(region)

        def to_visible(position: int) -> int:
            return sum(1 for original in mapping if region_start + original < position)

        pieces: list[str] = []
        texts: list[str] = []
        cursor = to_visible(spans[0][0]) if cut < spans[0][0] else 0  # drop a bare 「人名：」 attribution
        for span_start, span_stop in spans:
            visible_start, visible_stop = to_visible(span_start), to_visible(span_stop)
            texts.append(visible[visible_start:visible_stop])
            pieces.append(visible[cursor:visible_start])
            pieces.append(f"<{role}>{visible[visible_start:visible_stop]}</{role}>")
            cursor = visible_stop
        pieces.append(visible[cursor:])
        self.text = self.text[:region_start] + "".join(pieces) + self.text[region_end:]
        return texts

    def _mark_speaker(self, text: str, role: str, end: str | int = "", line: str | int | None = 0, begin=None) -> dict:
        resolved = self._resolve_span(text, end, line, begin)
        if isinstance(resolved, dict):
            return resolved
        start, stop, low, high, exact = resolved
        requested_role = self._canonical_role(role) or (role or "").strip()
        canonical = requested_role
        if not canonical:
            return {"ok": False, "reason": "role 不能为空"}
        # one or more quoted runs inside the requested span -> one mark per run (narration between
        # them stays outside). A bare quote number resolves to exactly one run.
        runs = quote_spans(self.text)
        inside = [(run_start, run_stop) for run_start, run_stop in runs if run_start >= start and run_stop <= stop]
        if not inside:
            enclosing = self._enclosing_quotes(start, stop)  # text landing inside a quote -> whole quote
            inside = [enclosing] if enclosing else [self._sentence_span(start, stop)]
        tag = self._enclosing_tag(inside[0][0], inside[-1][1], strict_stop=False)
        if tag is not None and tag[3] == canonical:
            open_gt, close_lt = tag[1], tag[2]
            if inside[0][0] <= open_gt + 1 and inside[-1][1] >= close_lt:
                return {"ok": True, "already": True, "role": tag[3], "text": text[:24]}
        body_start, body_stop = inside[0][0], inside[-1][1]
        if not any(char.isalnum() for char in self.text[body_start:body_stop]):
            return {"ok": False, "reason": "这段不是人物直接对话；只标人物直接说的话"}
        if not any(char in self.text[body_start:body_stop] for char in OPEN + CLOSE):
            context = self.text[max(0, body_start - 20) : body_start]
            if not self._SPEECH_CUE_RE.search(context):
                return {
                    "ok": False,
                    "reason": "这看起来是旁白/描写，不是人物直接说的话；只标台词或明确标记的心声",
                    "text": text[:24],
                }
        cut = body_start
        role_override = ""
        if cut >= 1 and self.text[cut - 1] == "：":
            k = cut - 1
            while k > 0 and self.text[k - 1] not in "。！？\n“”‘’「」『』 \t，,；;：<>":
                k -= 1
            attribution = self.text[k : cut - 1]
            attr_canonical = self._name_index().get(attribution)
            if attr_canonical and attr_canonical != NARRATOR:
                if attr_canonical != canonical:
                    role_override = attr_canonical  # the quote says who is speaking: trust it
                cut = k  # a bare 「人名：」 is consumed either way
        canonical = role_override or canonical
        previous = ""
        tag_before = self._enclosing_tag(inside[0][0], inside[-1][1], strict_stop=False)
        if tag_before is not None:
            previous = tag_before[3]
        texts = self._overwrite(cut, inside, canonical)
        body = " / ".join(texts)
        self.edits.append(
            {
                "op": "speak",
                "role": canonical,
                "text": text,
                "end": end,
                "n": line,
                "split": len(inside) if len(inside) > 1 else 0,
                "attribution": cut < body_start,
            }
        )
        result = {"ok": True, "role": canonical, "text": texts[0][:24], "marked": body[:400]}
        if line:
            result["n"] = line
        if len(inside) > 1:
            result["split"] = True
            result["count"] = len(inside)
        if previous and previous != canonical:
            result["replaced"] = True
            result["previous"] = previous
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

    def _update_roster(self, updates: dict) -> dict:
        """Merge character roster updates; stores for later retrieval by main()."""
        if not isinstance(updates, dict):
            return {"ok": False, "reason": "roster must be a dict"}
        added = 0
        for name, entry in updates.items():
            name = str(name).strip()
            if not name or name.lower() in ROSTER_STRUCT_KEYS:
                continue
            if isinstance(entry, dict):
                labels = [str(label).strip() for label in entry.get("aliases") or [] if str(label).strip()]
            elif isinstance(entry, list):
                labels = [str(label).strip() for label in entry if str(label).strip()]
            else:
                labels = []
            before = sum(len(v) for v in self.roster.values())
            _absorb_entry(self.roster, name, labels)
            added += sum(len(v) for v in self.roster.values()) - before
        self._roster_updates.append((str(updates), added))
        return {"ok": True, "added": added, "total": len(self.roster)}

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

    ANCHOR_CHARS = 10  # fuzzy locate = match this many leading chars, then this many trailing chars

    @staticmethod
    def _find_anchor_in(haystack: str, needle: str) -> int:
        """Tolerant anchor match: exact, then quote-insensitive, then punctuation-insensitive."""
        if not needle:
            return -1
        index = haystack.find(needle)
        if index >= 0:
            return index
        index = haystack.translate(_QUOTE_CANON).find(needle.translate(_QUOTE_CANON))
        if index >= 0:
            return index
        plain_chars: list[str] = []
        plain_map: list[int] = []
        for position, char in enumerate(haystack):
            if not unicodedata.category(char).startswith("P"):
                plain_chars.append(char)
                plain_map.append(position)
        plain_needle = "".join(char for char in needle if not unicodedata.category(char).startswith("P"))
        pos = "".join(plain_chars).find(plain_needle)
        return plain_map[pos] if pos >= 0 else -1

    def _locate_span(self, needle: str) -> tuple[int, int] | None:
        """Locate a speech by its text (mark tags are transparent): exact first; for long needles
        match the head and the tail separately (one match each) and take the span between them.
        Short needles (<= FUZZY_MIN_CHARS) are exact-only -- no fuzzy matching at all."""
        needle = (needle or "").strip()
        if not needle:
            return None
        visible, mapping = self._visible_tags(self.text)

        def to_original(visible_index: int) -> int:
            if visible_index < len(mapping):
                return mapping[visible_index]
            return mapping[-1] + 1 if mapping else len(self.text)

        plain_len = sum(1 for char in needle if not unicodedata.category(char).startswith("P"))
        if plain_len <= self.FUZZY_MIN_CHARS:
            # short: exact only, no fuzzy; an occurrence INSIDE a quoted run wins over narration
            for run_start, run_stop in quote_spans(visible):
                inside = visible.find(needle, run_start, run_stop)
                if inside >= 0:
                    return to_original(inside), to_original(inside + len(needle))
            index = visible.find(needle)
            return (to_original(index), to_original(index + len(needle))) if index >= 0 else None
        index = visible.find(needle)
        if index >= 0:
            return to_original(index), to_original(index + len(needle))
        found = self._find_anchor_in(visible, needle)  # whole sentence, tolerant
        if found >= 0:
            return to_original(found), to_original(found + len(needle))
        head, tail = needle[: self.ANCHOR_CHARS], needle[-self.ANCHOR_CHARS :]
        head_at = self._find_anchor_in(visible, head)
        tail_at = -1
        search_from = head_at + 1 if head_at >= 0 else 0
        if search_from < len(visible):
            tail_found = self._find_anchor_in(visible[search_from:], tail)
            if tail_found >= 0:
                tail_at = search_from + tail_found
        if head_at >= 0 and tail_at >= 0 and tail_at >= head_at:
            return to_original(head_at), to_original(tail_at + len(tail))
        if head_at >= 0:  # prefix alone is enough
            return to_original(head_at), to_original(min(len(visible), head_at + len(needle)))
        if tail_at >= 0:  # suffix alone is enough
            return to_original(max(0, tail_at - (len(needle) - len(tail)))), to_original(tail_at + len(tail))
        return None

    def _find_index(self, needle: str) -> int:
        return self._find_index_in(self.text, needle)

    FUZZY_MIN_CHARS = 10  # needles of <=10 chars must match exactly; fuzzy only above that

    def _find_index_in(self, haystack: str, needle: str) -> int:
        if not needle:
            return -1
        index = haystack.find(needle)
        if index >= 0:
            return index
        plain_len = sum(1 for char in needle if not unicodedata.category(char).startswith("P"))
        if plain_len <= self.FUZZY_MIN_CHARS:
            return -1  # short needle: no fuzzy matching (too easy to hit the wrong place)
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
                    "标注工具（一次一处，只标**人物直接说的话**）。"
                    "正文每个开引号前都有序号，形如 `[12]“…”`；**用序号定位**（n=第 n 处引号）。"
                    "**铁律：n、role 两个参数缺一不可**——n 是引号序号，role 是人物表里的规范名。"
                    "旁白/描写/叙述一律禁止标；引号里是术语/名词/标语/书名/强调（没有人在说话）时跳过不标。"
                    "修正只有一种方式：同一处再 speak 一次，覆盖旧标记（可改角色）。"
                    '检查完毕用 edit(op="done") 立刻结束。示例：{"op":"speak","n":12,"role":"角色名"}'
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["speak", "unmark", "done", "roster"],
                            "description": "speak=标出说话人（同一处再 speak 会覆盖旧标记）；"
                            "unmark=特殊含义的引号被误标时去掉标记（只去标记，不改原文）；"
                            "done=检查完毕、没有别的要改了，立刻结束；"
                            "roster=维护人物词典（注册新人物或添加别名）",
                        },
                        "n": {"type": "integer", "description": "引号序号（开引号前 `[n]` 的数字）"},
                        "role": {"type": "string", "description": "说话人规范名（speak 必填）；判断不出时填「未知」"},
                        "roster": {
                            "type": "object",
                            "description": '人物词典更新，格式 {"规范名": {"aliases": ["别名1", "别名2"]}}',
                        },
                    },
                    "required": ["op"],
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
<title>实时标记 · live</title>
<style>
:root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--panel2:#1b2230;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;
  --accent:#5aa9ff;--ok:#43c785;--warn:#ffb454}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif;overflow:hidden}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:#101319;z-index:5}
.brand{font-weight:700}.brand small{color:var(--dim);font-weight:400;margin-left:6px}
nav a{color:var(--accent);text-decoration:none;margin-right:12px;font-size:13px}nav a.on{color:var(--fg);font-weight:700}
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
.call{border-color:#25415f;border-left-color:#3f74b0;background:#101a26;font-size:12.5px}
.dict{border-color:#5c4620;border-left-color:#b8860b;background:#241d10;font-size:12px}
.res{border-color:#234;border-left-color:#2e3d52;background:#111720;color:var(--dim);font-size:11px}
.res.ok{border-color:#1f4a33;border-left-color:var(--ok);color:#c9f0da;background:#111d16}
.res.warn{border-color:#5c4620;border-left-color:#e0a545;color:#ffe1b0;background:#1d190e}   /* 改动留痕：覆盖/已标过/拆分 */
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
  <div class="brand">实时标记 <small id="book">· live</small></div>
  <span class="grow"></span>
  <nav>
    <a href="/dashboard">看板</a>
    <a id="promptlink" href="/prompt">提示词测试</a>
    <a id="selfpage" href="#" class="on">实时标记</a>
    <a id="rawpage" href="#">LLM原始IO</a>
    <a id="ttslink" href="/tts">TTS测试</a>
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
const BASE='__LIVE_BASE__', MAXCHAT=500;
function bookFromBase(){const m=BASE.match(/^\/outputs\/(.+)\/script\/$/);return m?m[1]:'';}
const BOOK=bookFromBase(), BOOKQ=BOOK?'?book='+encodeURIComponent(BOOK):'';
document.getElementById('promptlink').href='/prompt'+BOOKQ;
document.getElementById('selfpage').href='/live'+BOOKQ;
document.getElementById('rawpage').href='/raw'+BOOKQ;
document.getElementById('ttslink').href='/tts'+BOOKQ;
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
function resultClass(raw){
  let r=null;try{r=JSON.parse(raw);}catch(_){return 'bad';}
  if(!r||typeof r!=='object')return 'bad';
  if(r.ok===false)return 'bad';
  if(r.already||r.replaced||r.unmarked||r.split)return 'warn';
  return 'ok';
}
function resultIcon(raw){
  const c=resultClass(raw);
  return c==='ok'?'✓':c==='warn'?'⟳':'✗';
}
function flushChat(){
  if(!pending.length)return;
  const frag=document.createDocumentFragment();
  for(const row of pending)frag.appendChild(row);
  pending=[]; chat.appendChild(frag);
  let extra=chat.childElementCount-MAXCHAT;
  while(extra-->0)chat.removeChild(chat.firstChild);
}
function fmtRange(x){const n=x&&(x.n||x.line||x.begin);return n?'<code>['+esc(n)+']</code> ':'';}
function fmtCall(a, name){const x=a&&a.args||{};const keys=Object.keys(x);
  if(!keys.length)return '<span class="badge b-other">空调用</span><span class="sp">旧版记录没有参数</span>';
  const op=x.op||'speak';
  if(op==='speak')return '<span class="badge b-speak">标注</span>'+fmtRange(x)+'<span class="role">'+esc(x.role||'?')+'</span> <code>'+esc(x.text||'')+'</code>';
  if(op==='unmark')return '<span class="badge b-unmark">取消标记</span>'+fmtRange(x)+'<code>'+esc(x.text||'')+'</code>';
  if(op==='done')return '<span class="badge b-ok">✓ 完成</span><span class="sp">检查结束</span>';
  if(op==='replace')return '<span class="badge b-other">替换</span><code>'+esc(x.find||'')+'</code> <span class="sp">→</span> <code>'+esc(x.replace||'')+'</code>';
  if(op==='delete')return '<span class="badge b-other">旧版·删除</span>'+(x.text||x.find?'<code>'+esc(x.text||x.find)+'</code>':'<span class="sp">（旧记录无参数）</span>');
  if(op==='reassign')return '<span class="badge b-other">旧版·改标</span>'+(x.role?'<span class="role">'+esc(x.role)+'</span> ':'')+'<code>'+esc(x.text||x.find||'')+'</code>';
  return '<span class="badge b-other">旧版·'+esc(op)+'</span><code>'+esc(JSON.stringify(x))+'</code>';}
function fmtResult(raw){let r=null;try{r=JSON.parse(raw);}catch(_){return esc(raw);}
  if(!r||typeof r!=='object')return esc(raw);
  if(r.ok===false)return '<span class="badge b-bad">✗ 拒绝</span>'+esc(r.reason||'')+(r.text?' · <code>'+esc(r.text)+'</code>':'');
  if(r.split)return '<span class="badge b-ok">✓ 拆成 '+r.count+' 处</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.unmarked)return '<span class="badge b-warn">已取消标记</span>'+fmtRange(r)+'<span class="role">'+esc(r.unmarked)+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.already)return '<span class="badge b-warn">已标过</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <span class="sp">未改动</span>';
  if(r.replaced)return '<span class="badge b-warn">⟳ 覆盖</span>'+fmtRange(r)+'<span class="role">'+esc(r.previous||'')+'</span> <span class="sp">→</span> <span class="role">'+esc(r.role||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  if(r.ok)return '<span class="badge b-ok">✓ 标注</span>'+fmtRange(r)+'<span class="role">'+esc(r.role||'')+'</span> <code>'+esc(r.marked||'')+'</code>';
  return esc(raw);}
function handle(e){
  if(e.type==='start'){book.textContent='· '+(e.book||'live');return;}
  if(e.type==='chapter'){current=e.chapter;if(follow)selected=e.chapter;return;}
  if(e.type==='roster'){pushChat({cls:'dict',icon:'✦',chapter:e.chapter,html:'词典 <b>+'+e.added+'</b> 标签 · 词条 '+e.roles});return;}
  if(e.type==='assistant'){const c=e.content||'';pushChat({cls:'think',icon:'🧠',chapter:e.chapter,html:'<details><summary>思考 '+c.length+' 字 · '+esc(c.slice(0,42))+'</summary>'+esc(c)+'</details>'});return;}
  if(e.type==='tool'){const x=e.args||{};
    if(!Object.keys(x).length){emptyTool=true;return;} emptyTool=false;
    pushChat({cls:'call',chapter:e.chapter,html:fmtCall(e.args, e.call)});return;}
  if(e.type==='result'){  // every change keeps its own card: no merging, no erasing
    emptyTool=false;
    const verdict=resultClass(e.result);
    pushChat({cls:'res '+verdict,icon:resultIcon(e.result),chapter:e.chapter,html:fmtResult(e.result)});
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
let allChapters=null;  // full chapter list (chapters dir), fetched once
async function tickIndex(){
  try{
    if(allChapters===null&&BOOK){
      try{const c=await (await fetch('/chapters.json?book='+encodeURIComponent(BOOK),{cache:'no-store'})).json();
        allChapters=(c.chapters||[]);}catch(e){allChapters=[];}
    }
    const r=await fetch(BASE+'live_index.json?t='+Date.now(),{cache:'no-store'});
    if(!r.ok)return; const idx=await r.json();
    const cur=idx.current; if(cur&&follow)selected=cur;
    const done=new Set(Object.keys(idx.chapters||{}).filter(c=>idx.chapters[c]&&idx.chapters[c].done).map(Number));
    const ids=(allChapters&&allChapters.length?allChapters:Object.keys(idx.chapters||{}).map(Number)).sort((a,b)=>a-b);
    const listKey=ids.join(',')+'|'+[...done].sort((a,b)=>a-b).join(',');
    if(listKey!==idxSig){idxSig=listKey;                  // rebuild only when the set changes
      chs.innerHTML=ids.map(function(c){return '<option value="'+c+'">第 '+c+' 章'+(done.has(c)?' ✔':'')+'</option>';}).join('');}
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
<title>LLM原始IO · raw</title>
<style>
:root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;--accent:#5aa9ff;--bad:#ff7b7b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.65 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:6;display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 16px;background:#101319;border-bottom:1px solid var(--line)}
.brand{font-weight:700}.brand small{color:var(--dim);font-weight:400;margin-left:6px}
header .brand{font-size:14px}
nav{display:flex;gap:12px;font-size:13px}
nav a{color:var(--accent);text-decoration:none}nav a.on{color:var(--fg);font-weight:700}
header .grow{flex:1}.dim{color:var(--dim)}
header input{background:#0f141b;border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:4px 9px;font-size:12.5px}
header button{background:#1b2230;border:1px solid var(--line);border-radius:8px;color:var(--fg);padding:4px 10px;font-size:12.5px;cursor:pointer}
header button.on{background:var(--accent);border-color:var(--accent);color:#0b0f15;font-weight:700}
#feed{padding:12px 16px 80px;display:flex;flex-direction:column;gap:14px;max-width:1080px;margin:0 auto}
.chanhd{position:sticky;top:44px;z-index:4;color:var(--accent);font-weight:700;font-size:13px;padding:4px 2px;background:var(--bg);border-bottom:1px solid var(--line)}
.chan .cards{display:flex;flex-direction:column;gap:10px;padding-top:8px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.in{border-left:2px solid #2b4a72;padding-left:8px}
.reason{border-left:2px solid #3a4252;padding-left:8px}
.out{border-left:2px solid #1f4a33;padding-left:8px}
.toolbox{border-left:2px solid #5c4620;padding-left:8px}
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
  <div class="brand">LLM原始IO</div>
  <select id="chp"></select>
  <input id="q" placeholder="过滤…（step/正文/失败）">
  <button id="follow" class="on">跟随最新</button>
  <span class="dim" id="meta"></span>
  <span class="grow"></span>
  <nav>
    <a href="/dashboard">看板</a>
    <a id="promptlink" href="/prompt">提示词测试</a>
    <a id="livelink" href="/live">实时标记</a>
    <a class="on" href="/raw">LLM原始IO</a>
    <a id="ttslink" href="/tts">TTS测试</a>
  </nav>
</header>
<div id="feed"></div>
<script>
const BASE='__LIVE_BASE__';
function bookFromBase(){const m=BASE.match(/^\/outputs\/(.+)\/script\/$/);return m?m[1]:'';}
const BOOK=bookFromBase();
const bookq=BOOK?'?book='+encodeURIComponent(BOOK):'';
document.getElementById('promptlink').href='/prompt'+bookq;
document.getElementById('livelink').href='/live'+bookq;
document.getElementById('ttslink').href='/tts'+bookq;
const feed=document.getElementById('feed'), meta=document.getElementById('meta');
const q=document.getElementById('q'), followBtn=document.getElementById('follow'), chp=document.getElementById('chp');
let offset=0, pending=null, follow=true, shown=0, lastCards=[];
const chans=new Map();  // chapter -> section element (all chapters are kept)
let selectedCh='all';
function chanKey(res){return res.chapter!=null?String(res.chapter):'llm';}
function chanSection(key){
  let sec=chans.get(key);
  if(!sec){
    sec=document.createElement('section'); sec.className='chan';
    sec.innerHTML='<div class="chanhd">'+(key==='llm'?'LLM（词典/摘要）':'ch'+key)+' <span class="dim"></span></div><div class="cards"></div>';
    chans.set(key,sec); feed.appendChild(sec);
    refreshChanPicker();
  }
  return sec;
}
function refreshChanPicker(){
  const keys=[...chans.keys()].filter(k=>k!=='llm');
  keys.sort((a,b)=>Number(a)-Number(b));
  const cur=chp.value||'all';
  chp.innerHTML='<option value="all">全部章节</option>'+keys.map(k=>'<option value="'+esc(k)+'">ch'+esc(k)+'</option>').join('');
  chp.value=keys.includes(cur)?cur:(follow&&keys.length?keys[keys.length-1]:'all');
  selectedCh=chp.value||'all';
}
function applyChapter(){
  chans.forEach((sec,key)=>{sec.style.display=(selectedCh==='all'||selectedCh===key)?'':'none';});
}
chp.addEventListener('change',function(){selectedCh=chp.value;follow=false;followBtn.classList.remove('on');applyChapter();});
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function msgSummary(m){return '<span class="role">'+esc(m.role||'?')+'</span> <span class="dim">'+((m.content||'').length)+' 字'+(m.tool_calls?(' · 工具调用×'+m.tool_calls.length):'')+(m.tool_call_id?' · 工具结果':'')+'</span>';}
function pairCard(req,res){
  const card=document.createElement('div'); card.className='card';
  const where=(res.chapter!=null?'ch'+res.chapter+' · step'+res.step:(res.source||'llm'));
  const tc=(res.tool_calls||[]).map(function(t){return '<div class="tool">'+esc(t.name)+' '+esc(t.arguments)+'</div>';}).join('');
  card.innerHTML='<div class="hd"><b>'+esc(where)+'</b><span class="dim">'+esc(res.duration_s||'')+'s · finish='+esc(res.finish_reason||'')+(res.tool_calls?(' · 工具×'+res.tool_calls.length):'')+'</span></div>'
    +'<details class="in" open><summary>输入 · '+(req&&req.messages?req.messages.length:0)+' 条消息</summary><div class="msgs"></div></details>'
    +(res.reasoning?('<details class="reason" open><summary>思考 '+res.reasoning.length+' 字</summary><pre class="rpre">'+esc(res.reasoning)+'</pre></details>'):'')
    +'<details class="out" open><summary>输出</summary>'
    +'<pre>'+esc(res.content||'')+'</pre></details>'
    +(tc?'<details class="toolbox" open><summary>工具调用（打印）</summary>'+tc+'</details>':'');
  card.dataset.text=(where+' '+(res.content||'')+' '+(res.reasoning||'')+' '+tc).toLowerCase();
  const msgs=card.querySelector('.msgs');
  const fillMsgs=function(){
    if(!msgs.childElementCount&&req){
      req.messages.forEach(function(m){
        const d=document.createElement('details'); d.className='msg';
        d.innerHTML='<summary>'+msgSummary(m)+'</summary><pre>'+esc(m.content||'')+'</pre>';
        msgs.appendChild(d);});
    }
  };
  fillMsgs();  // input is open by default: populate right away
  card.querySelector('.in').addEventListener('toggle',function(){if(this.open)fillMsgs();});
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
function applyFilter(){const v=q.value.trim().toLowerCase();
  feed.querySelectorAll('.chan').forEach(function(sec){
    const inCh=selectedCh==='all'||sec===chans.get(selectedCh);
    sec.style.display=inCh?'':'none';
    if(!inCh)return;
    let visible=0;
    sec.querySelectorAll('.card').forEach(function(c){
      const ok=!v||c.dataset.text.indexOf(v)>=0; c.style.display=ok?'':'none'; if(ok)visible++;});
    sec.querySelector('.chanhd .dim').textContent='('+visible+'/'+sec.querySelectorAll('.card').length+')';
  });
}
function scrollBottom(){if(follow)window.scrollTo(0,document.body.scrollHeight);}
function handle(e){
  if(e.kind==='request'){
    if(lastCards.length){tailResults(e,lastCards.length).forEach(function(m,i){if(lastCards[i])markFail(lastCards[i],m);});lastCards=[];}
    pending=e; return;
  }
  if(e.kind==='response'){
    const card=pairCard(pending,e); pending=null; shown++;
    const key=chanKey(e), sec=chanSection(key);
    if(follow){chp.value=key;selectedCh=key;}
    sec.querySelector('.cards').appendChild(card);
    lastCards=(e.tool_calls||[]).map(function(){return card;});
    applyChapter(); applyFilter(); scrollBottom();
    meta.textContent='已记录 '+shown+' 组 · 章节 '+chans.size;
  }
}
function handleText(txt){for(const line of txt.split('\n')){if(!line.trim())continue;let e;try{e=JSON.parse(line);}catch(_){continue;}handle(e);}}
async function tick(){
  try{const r=await fetch(BASE+'llm_raw.jsonl',{headers:{'Range':'bytes='+offset+'-'},cache:'no-store'});
    if(r.status===416){
      const h=await fetch(BASE+'llm_raw.jsonl',{method:'HEAD',cache:'no-store'});
      const size=+(h.headers.get('Content-Length')||0);
      if(size<offset){const fb=await (await fetch(BASE+'llm_raw.jsonl',{cache:'no-store'})).arrayBuffer();offset=0;feed.innerHTML='';shown=0;chans.clear();handleText(new TextDecoder().decode(fb));offset=fb.byteLength;}
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


LOCAL_SYSTEM = """你的唯一任务：从这段小说正文里找出所有人物，维护人物词典。

## 输入
- 【人物词典全集】：已有的人物规范名和别名列表
- 【正文】：原始小说文本（未经任何处理）

## 你要做的
逐句通读正文，找出所有有姓名、有明确称呼的人物（连别名/绰号/称谓一起），
然后用 edit(op="roster", roster={"规范名": {"aliases": ["别名1", "别名2"]}})
注册新人物或给已有人物添加新别名。

## 规则
- **规范名**：此人最完整/最正式的姓名（如"高文·塞西尔"）；同一人只留一条。
- **已有词典里能对上的必须沿用已有规范名**，禁止新增重复条目。
- aliases 见到多少收多少（模糊名称 + 能唯一定位的称谓/绰号/特征指代），不限数量。
  例如「红发」「独臂将军」「那个瞎子」这种用特征指代一个人的，都可以收。
- **绝不收**代词/整句/泛称（大人/老爷/小姐/先生/骑士/众人）；不收地名/组织/物品/种族/群体。
- 已有条目只增补，不改名、不删除。

只用工具操作；处理完停下，不要解释。"""


# Few-shot as REAL tool calls (the model learns the tool protocol directly); dialogue only.
FEW_SHOT_CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "正文：\n他叹道：[1]“你终于来了。”",
        [
            '{"op":"speak","n":1,"role":"角色名"}',
        ],
        [
            '{"ok": true, "role": "角色名", "text": "“你终于来了。”", "marked": "“你终于来了。”", "n": 1}',
        ],
    ),
    (
        "正文：\n[1]“这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。”",
        [
            '{"op":"speak","n":1,"role":"角色名"}',
        ],
        [
            '{"ok": true, "role": "角色名", "text": "“这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。”"}',
        ],
    ),
    (
        "正文：\n[1]“快走！”他高声提醒，[2]“别管我！”",
        [
            '{"op":"speak","n":1,"role":"角色名"}',
            '{"op":"speak","n":2,"role":"角色名"}',
        ],
        [
            '{"ok": true, "role": "角色名", "text": "“快走！”", "marked": "“快走！”", "n": 1}',
            '{"ok": true, "role": "角色名", "text": "“别管我！”", "marked": "“别管我！”", "n": 2}',
        ],
    ),
    (
        "正文：\n[1]“传送法术”只是传说。他坐下，忽然想：她又熬夜了吧。",
        [
            '{"op":"speak","text":"她又熬夜了吧。","role":"角色名"}',
        ],
        [
            '{"ok": true, "role": "角色名", "text": "她又熬夜了吧。", "marked": "她又熬夜了吧。"}',
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
    "正文每个开引号前有序号 `[n]`；唯一允许的操作："
    'edit(op="speak", n=引号序号, role=说话人规范名)，n 就是那个数字。'
    "**铁律**：n、role 两个参数缺一不可；序号必须取自正文，不许自己编。"
    "**逐个序号都要过一遍**，用思考判断这一处是不是人物直接说的话："
    "术语、名词、标语/口号、书名/篇名、强调/反语（没有人在说话）就跳过不标；"
    "是台词就标，并判断谁说。"
    "**有些对话作者忘了打引号**（紧跟「某某说道/问道」等提示）：这种没有序号，改用 "
    'edit(op="speak", text=这段发言的完整原文, role=…) 标出。'
    "旁白、动作、描写、叙述一律禁止标；「某某说道/淡淡地说道」这类话留在标记外。"
    "连续对话必须逐句分清双方；说一句接一段旁白再接着说，必须分开标全。"
    "不要输出解释或 JSON 文本，只用工具，一次一处，标完立刻停下。"
)

CHECK_MARK = (
    "现在是检查环节：上面【本章完整标记】里，`[n]“…”` 的序号就是引号编号，"
    "`<角色>…</角色>` 包住的句子就是「角色」说的。\n"
    "**逐个编号过一遍**（这是硬要求）：每一处带序号的引号都要确认——是台词且标对了就跳过；"
    "漏标的补、标错的改、误标的删。\n"
    "1) 漏标：人物直接说的话没标（连续对话的每一段、旁白隔开的续话、无「某某说道」的交替；"
    "以及作者忘了打引号的直接发言，用 text 补标）；\n"
    "2) 多标/标错：旁白、描写、叙述被标进来，或说话人认错；\n"
    "3) 特殊含义的引号被误打：术语、名词、标语/口号、书名/篇名、强调/反语里没有人在说话，"
    "若被标成台词就是错的。\n"
    "**铁律**：旁白、描写、叙述不是台词，禁止标；标记只能包说话内容。"
    "修正一律用 edit：\n"
    "- 说话人错 → op=speak 重标：同一序号再 speak 一次，给正确的 role；\n"
    "- 漏标（有引号）→ op=speak 补上（给 n + role）；漏标（无引号）→ op=speak + text + role；\n"
    "- 特殊含义的引号被误标 → op=unmark 去掉这条标记（只去标记，不动原文）；\n"
    "一次只改 1~2 处，只改真有问题的，不要反复确认已经标对的地方。"
    '**全部编号都过完时，必须调用 edit(op="done") 立刻结束检查**；不要空转。'
)

DEFAULT_CHECK_STEPS = 100  # tool-step budget for the whole check pass (0 = no check at all)
CHECK_IDLE_STEPS = 2  # stop the check pass after N consecutive steps with no real change


TOOLS = ["edit"]
# 上下文只给「前后各一章」放在 system 里（当前章是任务文本，不再额外注入前情全文）：
#   AUDIOBOOK_INJECT_NEIGHBORS=0 关掉前文参考
INJECT_NEIGHBORS = os.environ.get("AUDIOBOOK_INJECT_NEIGHBORS", "1") == "1"
OPENING_CHAPTERS = 5  # chapters 1-5 are marked with the whole 1..5 block as context
PREV_CHAPTERS = 4  # from chapter 6 on: a rolling window of the previous 4 chapters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter", type=int, help="first chapter id")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--book", default="dawn")
    parser.add_argument(
        "--batch",
        type=int,
        default=10,
        help="context window: first N chapters are fed in full as prior context",
    )
    parser.add_argument("--max-steps", type=int, default=1000, help="tool steps per chapter")
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


# Keys the model sometimes emits as if they were character names; never roster entries.
ROSTER_STRUCT_KEYS = frozenset(
    {"aliases", "alias", "name", "规范名", "别名", "人物", "词典"}
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


def _result_changed(raw: str) -> bool:
    """True only when a tool result really changed the text (not ``already``/refused)."""
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or data.get("ok") is not True:
        return False
    return not data.get("already")


def run_turn(
    client,
    config,
    tools,
    mcp,
    messages,
    max_steps,
    counters,
    live: Live | None = None,
    chapter: int = 0,
    original: str = "",
    idle_limit: int = 0,
) -> int:
    """Tool loop until the model stops calling tools. Guards against a failing retry loop.

    ``idle_limit`` > 0 stops the pass after that many consecutive steps with no real change
    (the check pass uses it to bail out once it is only re-confirming existing marks).
    Returns the number of successful edits (0 = nothing changed)."""
    errors = 0
    edits_ok = 0
    last_sig, repeats, fail_total = "", 0, 0
    ok_sig, ok_repeats = "", 0
    idle_steps = 0

    def idle_now(changed: bool) -> bool:
        nonlocal idle_steps
        if not idle_limit:
            return False
        idle_steps = 0 if changed else idle_steps + 1
        return idle_steps >= idle_limit

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
                max_tokens=MAX_TOKENS,
                extra_body={
                    "top_k": config.top_k,
                    "chat_template_kwargs": {"enable_thinking": THINK},
                    "reasoning_budget_tokens": REASONING_BUDGET,
                },
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
                        "content": "找不到目标。从上面的正文/标记里照抄原文重试（开头约 10 字）；"
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
            if '"done": true' in result:
                return edits_ok
            if idle_now(_result_changed(result)):
                return edits_ok
            continue
        step_changed = False
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
            if '"done": true' in result:  # model says it is finished: stop right away
                return edits_ok
            if '"ok": false' in result or "not found" in result:
                fail_total += 1
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                messages.append(
                    {
                        "role": "user",
                        "content": "找不到目标。从上面的正文/标记里照抄原文重试（开头约 10 字）；"
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
                step_changed = True
        if idle_now(step_changed):
            return edits_ok
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
    prompts = data.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {}
    for key, name in (
        ("local_system", "LOCAL_SYSTEM"),
        ("step_mark", "STEP_MARK"),
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


def normalize_aliases(snapshot: str, roster: dict[str, list[str]]) -> str:
    """Rename every label/alias to the canonical role -- opening AND closing tags -- and
    collapse accidental duplicate nesting (``<X><X>…</X></X>``)."""
    for seg in parse_marks(snapshot):
        if seg["kind"] != "speech":
            continue
        canonical = resolve_label(roster, seg["role_name"])
        if canonical != seg["role_name"]:
            snapshot = snapshot.replace(f"<{seg['role_name']}>", f"<{canonical}>")
            snapshot = snapshot.replace(f"</{seg['role_name']}>", f"</{canonical}>")
    for _ in range(4):
        snapshot = re.sub(r"<([^<>/\n]+)><\1>", r"<\1>", snapshot)
        snapshot = re.sub(r"</([^<>/\n]+)></\1>", r"</\1>", snapshot)
    return snapshot


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

    def chapter_path(cid: int) -> Path:
        return chapters / f"ch{cid:03d}.txt"

    def roles_in(cid: int) -> list[str]:
        marked = out_dir / f"ch{cid:03d}.marked.txt"
        if not marked.is_file():
            return []
        return [seg["role_name"] for seg in parse_marks(marked.read_text(encoding="utf-8")) if seg["kind"] == "speech"]

    def neighbor_context(cid: int) -> str:
        """Reference chapters only (never the current chapter): chapters 1-5 see the whole opening
        block 1..5; from chapter 6 on a rolling window of the previous 4 (the oldest drops out)."""
        if not INJECT_NEIGHBORS:
            return ""
        if cid <= OPENING_CHAPTERS:
            others = [k for k in range(1, OPENING_CHAPTERS + 1) if k != cid]
        else:
            others = list(range(max(1, cid - PREV_CHAPTERS), cid))
        parts = []
        for other in others:
            path = chapter_path(other)
            if path.is_file():
                parts.append(f"【前文·第 {other} 章（仅供判断人物/称呼，不要标注）】\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(parts)

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
            names = "、".join(roster)  # consistency hint, not a gate: new names are registered as seen
            hint = f"已有人物（尽量沿用这些名字）：{names}\n\n" if names else ""
            live.emit("chapter", chapter=cid, chars=len(raw_text))
            live.set_state(cid, len(raw_text), live_fragments(raw_text, raw_text))
            mcp.call("set_text", {"text": raw_text})
            think = config.thinking_system_token if THINK else ""  # profile-driven (Gemma: "<|think|>", Qwen: none)
            # The model sees the plain chapter text and must quote the COMPLETE speech text;
            # the MCP locates it (fuzzy only for long needles) -- nothing about that is in the prompt.
            # Few-shot goes AFTER the window and right BEFORE the task, so the examples sit next
            # to the instruction the model is about to execute (not buried above the big text).
            neighbors = neighbor_context(cid)
            messages = [
                {"role": "system", "content": f"{think}{LOCAL_SYSTEM}\n\n{hint}" + (f"\n\n{neighbors}" if neighbors else "")},
                *FEW_SHOT,
                {"role": "user", "content": f"【要处理的正文】\n{number_text(raw_text)}"},
                {"role": "user", "content": STEP_MARK},
            ]
            run_turn(client, config, tools, mcp, messages, args.max_steps, counters, live, cid, raw_text)
            checks = 0
            if args.check_steps:  # one whole-chapter check pass
                current = json.loads(mcp.call("get_marked", {}))["text"]
                missing = mcp.server.unmarked_quotes()
                if missing:
                    listed = "\n".join(
                        f"- [{item['number']}] {item['text']}"
                        + ("（含旧标记，请整段重标）" if item.get("partial") else "")
                        + f" ｜ 出处：{item['snippet']}"
                        for item in missing[:120]
                    )
                    scan = f"【机械扫描：全章以下 {len(missing)} 处引号尚未标记】\n{listed}"
                else:
                    scan = "【机械扫描：全章没有未标记的引号内容】"
                check_messages = [
                    {
                        "role": "system",
                        "content": f"{think}{LOCAL_SYSTEM}\n\n{hint}" + (f"\n\n{neighbors}" if neighbors else ""),
                    },
                    {
                        "role": "user",
                        "content": f'【本章完整标记（`[n]"…"` 的 n 是引号编号，`<角色>…</角色>` 表示这段是「角色」说的）】\n{number_text(current)}',
                    },
                    {"role": "user", "content": scan},
                    {"role": "user", "content": f"这是全章检查。{CHECK_MARK}"},
                ]
                run_turn(
                    client,
                    config,
                    tools,
                    mcp,
                    check_messages,
                    args.check_steps,
                    counters,
                    live,
                    cid,
                    raw_text,
                    idle_limit=CHECK_IDLE_STEPS,  # fallback: stop when it only re-confirms
                )
                checks = 1
            snapshot = json.loads(mcp.call("get_marked", {}))["text"]
            # No end-of-chapter fill-in pass: whatever the edit loop produced stands; only the
            # mechanical cleanup below runs.
            snapshot = normalize_aliases(snapshot, roster)
            # Quotes are part of the dialogue sentence and stay inside the marks: no cleanup.
            (out_dir / f"ch{cid:03d}.marked.txt").write_text(snapshot, encoding="utf-8")
            (out_dir / "roles.json").write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
            missed = mcp.server.unmarked_quotes()
            phases.append({"chapter": cid, "chars": len(snapshot), "roles": len(roster), "missed": len(missed), "checks": checks})
            print(f"[mark] ch{cid:03d} chars={len(snapshot)} 词条={len(roster)} 漏标引号={len(missed)}", flush=True)
            live.emit("missed", chapter=cid, count=len(missed), samples=[item["text"][:20] for item in missed[:5]])
            _stage_note(args.book, "run", f"{len(phases)}/{len(ids)}")
            live.set_state(cid, len(snapshot), live_fragments(raw_text, snapshot), done=True)
        live.emit("done", chapter=cid, roles=len(roster))

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
