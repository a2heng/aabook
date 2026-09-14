"""Client for the script MCP server (`scripts/script_mcp_server.py`).

The server is pure read/modify primitives over the working text; the model never emits
the whole text, it only calls tools. This client speaks newline-delimited JSON-RPC over
the server's stdio.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
SERVER = APP_ROOT / "scripts" / "script_mcp_server.py"


class MCPClient:
    def __init__(self, argv: list[str] | None = None) -> None:
        argv = argv or [str(APP_ROOT / ".venv" / "bin" / "python"), str(SERVER)]
        self.proc = subprocess.Popen(
            argv, cwd=APP_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        self._id = 0

    def _send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def _recv(self) -> dict:
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def request(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        while True:
            message = self._recv()
            if message.get("id") == self._id:
                return message

    def initialize(self) -> None:
        self.request(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "mark_script", "version": "1.0"}},
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def tools(self) -> list[dict]:
        return self.request("tools/list")["result"]["tools"]

    def openai_tools(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]}}
            for t in self.tools()
        ]

    def call(self, name: str, args: dict) -> str:
        result = self.request("tools/call", {"name": name, "arguments": args})
        return result["result"]["content"][0]["text"]
