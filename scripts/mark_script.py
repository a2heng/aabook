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

    def edit(self, op: str = "", text: str = "", role: str = "", find: str = "", replace: str = "", end: str = "") -> dict:
        """The one editing tool. ``op`` is inferred when omitted."""
        op = op or ("speak" if role else "replace" if (find or replace) else "delete")
        if op == "speak":
            target = find if any(char in find for char in OPEN + CLOSE) else (text or find)
            return self._mark_speaker(target, role, end)
        if op == "replace":
            return self._replace(find or text, replace)
        return self._delete(text or find)

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

    def _mark_speaker(self, text: str, role: str, end: str = "") -> dict:
        # The model gives the dialogue's opening (~10 chars) and, for long speeches, its closing
        # (~10 chars); a short line is given whole. Punctuation is matched loosely by the finder.
        index = self._find_index(text)
        if index < 0:
            index = self._find_quoted_fragment(text)
        if index < 0:
            return {"ok": False, "reason": "not found", "text": text}
        start, stop = index, index + len(text)
        anchored = False
        if end and end != text:
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
        open_lt = self.text.rfind("<", 0, start)
        if open_lt >= 0:
            open_gt = self.text.find(">", open_lt)
            close_lt = self.text.find("<", stop)
            if open_gt != -1 and open_gt <= start and close_lt != -1 and self.text.startswith("</", close_lt):
                return {"ok": True, "already": True, "role": self.text[open_lt + 1 : open_gt], "text": text[:24]}
        # One speech never covers two separate quoted spans: if the model's text spans a closing
        # quote, narration, then another opening quote, keep only the FIRST span (each quote is
        # its own speech; never glue two pieces across the description). Only the same quote pair
        # counts, so inner quotes of one dialogue are not clipped.
        clamped = False
        for opener, closer in (("“", "”"), ("「", "」"), ("『", "』")):
            close_at = self.text.find(closer, start, stop)
            if close_at >= 0 and self.text.find(opener, close_at + 1, stop) >= 0:
                stop = close_at + 1
                clamped = True
                break
        expanded = False
        if clamped:  # expand to the opening quote of THIS span (the next quote must stay out)
            for i in range(start - 1, max(-1, start - 400), -1):
                if self.text[i] in CLOSE or self.text[i] == "\n":
                    break
                if self.text[i] in OPEN:
                    start = i
                    break
            expanded = True
        else:
            enclosing = self._enclosing_quotes(start, stop)
            if enclosing is not None:  # expand to the two quotes: quotes stay INSIDE the mark
                start, stop = enclosing
                expanded = True
        body = self.text[start:stop]
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
        result = {"ok": True, "role": canonical, "text": body[:24]}
        if anchored:
            result["anchored"] = True
        if expanded:
            result["expanded"] = True
            result["note"] = "已扩展到两端引号，整段记录为台词"
        if role_override:
            result["role_note"] = f"按引号前的归属「{role_override}」判定说话人"
            result["role_corrected_from"] = role
        return result

    _SPEECH_TAIL_RE = re.compile(
        r"(说道|问道|答道|喊道|叫道|笑道|叹道|应道|喝道|骂道|吼道|念道|回答|低语|喃喃|嘟囔|传来|开口|说)[，,：:]?$"
    )
    _NON_SPEECH_COLON_RE = re.compile(r"(写着|写到|标着|刻着|印着|注着|列出|注明|写着|标注)$")

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
                    "reason": "这看起来是对话引号（前面有说话提示），请改用 edit(op=speak, role=规范名)；确认不是人物对话就再 delete 一次",
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
                    "标注工具（一次一处）：抓人物**直接说的话**并标出说话人。"
                    "text 给这段对话的**完整原文**（逐字来自原文，一字不差；长台词也要整段给全）。"
                    "role 给说话人规范名（结合人物表和上下文判断，判断不出填「未知」）。"
                    '示例：{"op":"speak","text":"你终于来了。","role":"陈默"} / '
                    '{"op":"speak","text":"这件事要从很久以前说起，最后我们还是在城南住下了。","role":"陈默"}'
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["speak"]},
                        "text": {"type": "string", "description": "这段对话的完整原文（逐字，一字不差）"},
                        "role": {"type": "string", "description": "说话人规范名；判断不出时填「未知」"},
                    },
                    "required": ["text", "role"],
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
#main{display:grid;grid-template-columns:1fr 380px;height:calc(100dvh - 46px)}
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
#chat{overflow:hidden;min-height:0;padding:10px 12px;display:flex;flex-direction:column;justify-content:flex-end;gap:5px}
.ev{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:5px 9px;font-size:12.5px;line-height:1.55;word-break:break-word}
.ico{display:inline-block;width:15px;margin-right:5px;text-align:center;opacity:.9}
.think{color:var(--dim);background:#12161d}.think summary{cursor:pointer;font-size:12px}
.call{border-color:#25415f;background:#101a26}
.dict{border-color:#5c4620;background:#241d10}
.res{border-color:#234;background:#111720;color:var(--dim);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.res.ok{border-color:#1f4a33;color:#9fe6c1}.res.bad{border-color:#5a2626;color:#ffb3b3;background:#1d1212}
.sum{border-color:#3b3560;background:linear-gradient(180deg,#1b1830,#151a23)}
.ev code{font-size:11px;padding:0 3px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:0 5px;border-radius:6px;margin-right:5px}
.b-speak{background:#17345a;color:#8fc4ff}.b-delete{background:#4a2a12;color:#ffba75}.b-replace{background:#33234a;color:#c9a6ff}
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
let pending=[];
function pushChat(e){
  const row=document.createElement('div'); row.className='ev '+e.cls;
  row.innerHTML=(e.icon?'<span class="ico">'+e.icon+'</span>':'')+(e.chapter?'<span class="chap">ch'+e.chapter+'</span>':'')+e.html;
  pending.push(row);
}
function flushChat(){
  if(!pending.length)return;
  const frag=document.createDocumentFragment();
  for(const row of pending)frag.appendChild(row);
  pending=[]; chat.appendChild(frag);
  let extra=chat.childElementCount-MAXCHAT;
  while(extra-->0)chat.removeChild(chat.firstChild);
}
function fmtCall(a){const x=a&&a.args||{};
  const op=x.op||(x.role?'speak':(x.find!==undefined||x.replace!==undefined)?'replace':'delete');
  if(op==='speak')return '<span class="badge b-speak">speak</span><span class="role">'+esc(x.role||'?')+'</span> <code>'+esc(x.text||'')+'</code>';
  if(op==='replace')return '<span class="badge b-replace">replace</span><code>'+esc(x.find||'')+'</code> <span class="sp">→</span> <code>'+esc(x.replace||'')+'</code>';
  return '<span class="badge b-delete">delete</span><code>'+esc(x.text||'')+'</code>';}
function handle(e){
  if(e.type==='start'){book.textContent='· '+(e.book||'live');return;}
  if(e.type==='chapter'){current=e.chapter;if(follow)selected=e.chapter;return;}
  if(e.type==='roster'){pushChat({cls:'dict',icon:'✦',chapter:e.chapter,html:'词典 <b>+'+e.added+'</b> 标签 · 词条 '+e.roles});return;}
  if(e.type==='assistant'){pushChat({cls:'think',icon:'🧠',chapter:e.chapter,html:'<details><summary>思考</summary>'+esc(e.content)+'</details>'});return;}
  if(e.type==='tool'){pushChat({cls:'call',chapter:e.chapter,html:fmtCall(e.args)});return;}
  if(e.type==='result'){pushChat({cls:'res '+(e.ok?'ok':'bad'),icon:e.ok?'✓':'✗',chapter:e.chapter,html:esc(e.result)});return;}
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
- 这里没有章节、也没有“引号里的才是对话”这回事：眼前的文字就是全部材料；
  直接说的话不一定被引号包着，被引号包着的也不一定是直接说的话。
- **连续对话常常没有「某某说道」提示、两人交替**（甚至上引号直接接在下引号后面）：必须逐句判断，
  谁说的算谁的，**正确识别双方转换**，不要把不同人的话拼到一起。
- **标记以“一段话”为单位**：一个人的话中间被旁白/动作描写隔开时，就是两段（可能不止两段），
  要**分别标全（一段一个 speak）**；**绝不能把两段连成一段**，也不要漏掉任何一段。
- 找到一处就用一次 edit(op="speak", text=这段对话的完整原文, role=说话人规范名)：
  **text 给这一段对话的完整原文，逐字来自原文、一字不差**（长台词也要整段给全）。
  说话人用人物表里的规范名（没有就按原文写法），判断不出填「未知」。
只管抓对话；原文的文字、标点和引号一个字都不要改。处理完停下，不要解释。"""

# Few-shot as REAL tool calls (the model learns the tool protocol directly); dialogue only.
_FEW_SHOT_CASES: list[tuple[str, list[str], list[str]]] = [
    (
        "示例1（短句：≤20 字，text 给完整句子）：\n他叹道：“你终于来了。”",
        ['{"op":"speak","text":"你终于来了。","role":"陈默"}'],
        ['{"ok": true, "role": "陈默", "expanded": true}'],
    ),
    (
        "示例2（长句也要给完整原文）：\n“这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。”",
        ['{"op":"speak","text":"这件事要从很久以前说起，中间经过了很多波折，最后我们还是在城南住下了。","role":"陈默"}'],
        ['{"ok": true, "role": "陈默", "expanded": true}'],
    ),
    (
        "示例3（同一人被旁白隔成两段，要分两次 speak）：\n“快走！”他高声提醒，“别管我！”",
        ['{"op":"speak","text":"快走！","role":"陈默"}', '{"op":"speak","text":"别管我！","role":"陈默"}'],
        ['{"ok": true, "role": "陈默", "expanded": true}', '{"ok": true, "role": "陈默", "expanded": true}'],
    ),
]
FEW_SHOT: list[dict] = []
for _case_index, (_example, _calls, _results) in enumerate(_FEW_SHOT_CASES, start=1):
    FEW_SHOT.append({"role": "user", "content": _example})
    FEW_SHOT.append(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"few{_case_index}_{call_index}", "type": "function", "function": {"name": "edit", "arguments": call}}
                for call_index, call in enumerate(_calls, start=1)
            ],
        }
    )
    FEW_SHOT += [
        {"role": "tool", "tool_call_id": f"few{_case_index}_{call_index}", "content": result}
        for call_index, result in enumerate(_results, start=1)
    ]

STEP_MARK = (
    "现在只做一件事：从下面的正文里抓出**人物直接说的话**，用 "
    'edit(op="speak", text=这段对话的完整原文, role=说话人) 标出说话人；'
    "text 逐字来自原文、一字不差，长台词也要整段给全。不要输出解释或 JSON 文本，只用工具，一次一处，标完停下。"
    "连续对话的时候双方转换要能正确识别。标记按段来：说完接一段旁白/描写再接着说，就是两段，"
    "要分开标全，不能连成一段、也不能漏掉。"
)

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


def maintain_roster(llm: LLMClient, roster: dict[str, list[str]], text: str) -> int:
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
        tool_calls = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (message.tool_calls or [])
        ]
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls or None})
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
                messages.append({"role": "user", "content": "这处已经处理过，不要重复；继续处理还没处理的引号。"})
            else:
                edits_ok += 1
                last_sig, repeats = "", 0
            continue
        for call in tool_calls:
            arguments = _parse_args(call["function"]["arguments"])
            if arguments is None:  # malformed JSON: tell the model and move on (never crash)
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
                messages.append({"role": "user", "content": "这处已经处理过，不要重复；继续处理还没处理的引号。"})
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
    for key, target in (("batch", "batch"), ("max_steps", "max_steps")):
        value = params.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            setattr(args, target, max(1, value))
            applied[target] = getattr(args, target)
    if isinstance(params.get("think"), bool):
        globals()["THINK"] = params["think"]
        applied["THINK"] = params["think"]
    prompts = data.get("prompts")
    if not isinstance(prompts, dict):
        prompts = {}
    for key, name in (("local_system", "LOCAL_SYSTEM"), ("step_mark", "STEP_MARK"), ("roster_system", "ROSTER_SYSTEM")):
        value = prompts.get(key)
        if isinstance(value, str) and value.strip():
            globals()[name] = value
            applied[name] = "overlay"
    return applied


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
    from openai import OpenAI

    client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=900)
    llm = LLMClient(config)
    mcp = MCPClient()
    mcp.initialize()
    all_tools = {t["function"]["name"]: t for t in mcp.openai_tools()}
    tools = [all_tools[name] for name in TOOLS]

    chapters = APP_ROOT / "outputs" / args.book / "chapters"
    out_dir = APP_ROOT / "outputs" / args.book / "script"
    out_dir.mkdir(parents=True, exist_ok=True)
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
                f"【前情（第 1..{cid - 1} 章原文，仅用于判断说话人；这些章节的引号都已处理，不要处理）】\n{prior}"
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
            messages = [
                {"role": "system", "content": f"{think}{LOCAL_SYSTEM}\n\n{hint}"},
                *FEW_SHOT,
                {"role": "user", "content": f"{prior_context(cid)}\n\n【要处理的正文（只抓人物直接对话）】\n{raw_text}"},
                {"role": "user", "content": STEP_MARK},
            ]
            run_turn(client, config, tools, mcp, messages, args.max_steps, counters, live, cid, raw_text)
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
            phases.append({"chapter": cid, "chars": len(snapshot), "roles": len(roster)})
            print(f"[mark] ch{cid:03d} chars={len(snapshot)} 词条={len(roster)}", flush=True)
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
