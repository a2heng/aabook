#!/usr/bin/env python3
"""MCP server for script marking: the model only calls `edit`, code does the rest.

Three primitives over a working text (newline-delimited JSON-RPC on stdio):
- ``set_text``    load the chapter text to work on.
- ``edit``        the ONLY editing tool: one edit per call.
                    op=speak   wrap a quoted span as speech (role), drop its quotes, and
                               also remove a redundant ``名字：`` attribution right before it.
                    op=delete  unwrap a quoted term (drop quotes, keep the word); a
                               punctuation-only quote (“…”) is removed whole; lone
                               punctuation is removed as given.
                    op=replace  replace the first ``find`` with ``replace`` (comma at a
                               breath point, missing sentence end, ...).
- ``get_marked``  read the current marked text.

Speech is ``⦃角色名␟朗读内容⦄``; everything outside the markers is narration. The quotes
to look for are “ ” / 『』「」; straight ASCII quotes are tolerated.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.marks import MARK_CLOSE, MARK_OPEN, MARK_SEP  # noqa: E402
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


class ScriptServer:
    def __init__(self) -> None:
        self.text = ""
        self.edits: list[dict] = []

    # ---- tools ---------------------------------------------------------------
    def set_text(self, text: str) -> dict:
        self.text = text or ""
        return {"chars": len(self.text)}

    def get_marked(self) -> dict:
        return {"text": self.text}

    def edit(self, op: str = "", text: str = "", role: str = "", find: str = "", replace: str = "") -> dict:
        """The one editing tool. ``op`` is inferred when omitted."""
        op = op or ("speak" if role else "replace" if (find or replace) else "delete")
        if op == "speak":
            # Prefer whichever argument still carries the quotes, so the whole quoted
            # span is consumed (inner-only text would leave stray “ ” behind).
            target = find if any(char in find for char in OPEN + CLOSE) else (text or find)
            return self._mark_speaker(target, role)
        if op == "replace":
            return self._replace(find or text, replace)
        return self._delete(text or find)

    # ---- primitives ----------------------------------------------------------
    def _mark_speaker(self, text: str, role: str) -> dict:
        span = self._locate(text, strip=True)
        if span is None:
            return {"ok": False, "reason": "not found", "text": text}
        start, end = span
        inner = self.text[start:end]
        quoted = inner[:1] in OPEN and inner[-1:] in CLOSE
        if quoted:
            inner = inner[1:-1]
        cut = start
        if quoted and start >= 1 and self.text[start - 1] == "：":
            k = start - 1
            while k > 0 and self.text[k - 1] not in "。！？\n“”‘’「」『』 \t，,；;：":
                k -= 1
            name = self.text[k : start - 1]
            if 0 < len(name) <= 8 and self._is_known_name(name, role):
                cut = k
        wrapped = f"{MARK_OPEN}{role}{MARK_SEP}{inner}{MARK_CLOSE}"
        self.text = self.text[:cut] + wrapped + self.text[end:]
        self.edits.append({"op": "speak", "role": role, "text": text, "attribution": cut < start})
        return {"ok": True, "role": role, "spanned": wrapped}

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
        if span is not None:
            start, end = span
            inner = self.text[start:end]
            body = inner[1:-1] if (inner[:1] in OPEN and inner[-1:] in CLOSE) else inner
            keep = any(char.isalnum() for char in body)
            self.text = self.text[:start] + (body if keep else "") + self.text[end:]
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
    def _cast(self) -> "Cast | None":
        book = os.environ.get("AUDIOBOOK_BOOK", "dawn")
        path = APP_ROOT / "outputs" / book / "cast.json"
        if not path.is_file():
            return None
        if getattr(self, "_cast_path", None) != str(path):
            self._cast_obj = Cast.load(str(path))
            self._cast_path = str(path)
        return self._cast_obj

    def _is_known_name(self, name: str, role: str) -> bool:
        if name == role or name in role or role in name:
            return True
        cast = self._cast()
        if cast is None:
            return False
        return any(name in [item.name, *item.aliases] or name in item.name or item.name in name for item in cast.roles.values())

    def _find_index(self, needle: str) -> int:
        if not needle:
            return -1
        index = self.text.find(needle)
        if index >= 0:
            return index
        return self.text.translate(_QUOTE_CANON).find(needle.translate(_QUOTE_CANON))

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
                    "唯一改法：单条编辑。op=speak（台词，给 role，自动去引号并删掉前面多余的「人名：」）/ "
                    "delete（去掉引号留下词；引号内只有标点则整段删；单独标点则删该标点）/ "
                    "replace（把 find 换成 replace，如气口处补逗号）。"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": ["speak", "delete", "replace"]},
                        "text": {"type": "string", "description": "speak=引号连内容的整段；delete=要处理的词/标点"},
                        "role": {"type": "string", "description": "speak 时的规范名"},
                        "find": {"type": "string", "description": "replace 的定位片段（≤6 字）"},
                        "replace": {"type": "string", "description": "replace 的替换内容"},
                    },
                    "required": ["op"],
                },
            }
        ]


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    server = ScriptServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": message_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "novel-script", "version": "1.0.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": message_id, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": message_id, "result": {"tools": server.tool_specs()}})
        elif method == "tools/call":
            params = message.get("params", {})
            try:
                result = server.call(params.get("name"), params.get("arguments", {}) or {})
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]},
                    }
                )
            except Exception as error:  # noqa: BLE001 - surface tool errors to the agent
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message_id,
                        "result": {"content": [{"type": "text", "text": f"error: {error!r}"}], "isError": True},
                    }
                )
        elif message_id is not None:
            send({"jsonrpc": "2.0", "id": message_id, "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    main()
