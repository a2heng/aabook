#!/usr/bin/env python3
"""LAN file browser with inline HTML5 audio playback (stdlib only).

Serves a directory tree over HTTP so any device on the LAN can browse files and
play generated ``.wav`` output in the browser. Supports HTTP Range requests so
large files can be seeked.

Examples:
    python scripts/serve_files.py --dir . --port 8899
    python scripts/serve_files.py --dir outputs/mybook/render --port 8899
"""

from __future__ import annotations

import argparse
import html
import io
import mimetypes
import os
import re
import socket
from datetime import datetime
from http import HTTPStatus
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".yaml", ".yml", ".srt"}
TEXT_INLINE_MAX = 64 * 1024 * 1024
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


class _RangeFile(io.BufferedIOBase):
    """File-like wrapper that yields at most ``length`` bytes (for Range)."""

    def __init__(self, handle, length: int):
        self._handle = handle
        self._remaining = length

    def readable(self) -> bool:
        return True

    def read(self, size: int | None = -1) -> bytes:
        if size is None:
            size = -1
        if self._remaining <= 0:
            return b""
        if size < 0 or size > self._remaining:
            size = self._remaining
        data = self._handle.read(size)
        self._remaining -= len(data)
        return data

    def close(self) -> None:
        self._handle.close()


class FileBrowser(SimpleHTTPRequestHandler):
    server_version = "AukFiles/1.0"

    def send_head(self) -> io.BufferedIOBase | io.BytesIO | None:
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return self.list_directory(str(path))
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        if path.suffix.lower() in TEXT_EXTS and path.stat().st_size <= TEXT_INLINE_MAX:
            payload = decode_text(path.read_bytes()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return io.BytesIO(payload)

        try:
            handle = path.open("rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        stat = os.fstat(handle.fileno())
        size = stat.st_size
        content_type = self.guess_type(str(path))
        range_header = self.headers.get("Range")
        match = _RANGE_RE.match(range_header) if range_header else None
        if match:
            start = int(match.group(1) or 0)
            end = int(match.group(2)) if match.group(2) else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                handle.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            length = end - start + 1
            handle.seek(start)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return _RangeFile(handle, length)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        self.end_headers()
        return handle

    def guess_type(self, path) -> str:
        content_type = mimetypes.guess_type(path)[0]
        if not content_type:
            content_type = "text/plain" if Path(path).suffix.lower() in TEXT_EXTS else "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/json", "application/javascript"):
            return f"{content_type}; charset=utf-8"
        return content_type

    def list_directory(self, path):
        try:
            entries = list(os.scandir(path))
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Cannot list directory")
            return None
        entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))

        url_path = urlparse(self.path).path
        if not url_path.endswith("/"):
            self.send_response(HTTPStatus.MOVED_PERMANENTLY)
            self.send_header("Location", url_path + "/")
            self.end_headers()
            return None

        parent = None
        if Path(path).resolve() != Path(self.directory).resolve():
            parent = url_path.rstrip("/").rsplit("/", 1)[0] + "/" or "/"

        rows = []
        if parent is not None:
            rows.append(f'<tr><td colspan="3"><a class="dir" href="{html.escape(parent)}">⬆ 上级目录</a></td></tr>')
        for entry in entries:
            name = entry.name
            if name.startswith("."):
                continue
            is_dir = entry.is_dir()  # follow symlinks so linked output dirs are browsable
            try:
                stat = entry.stat()
            except OSError:
                continue
            href = url_path + quote(name) + ("/" if is_dir else "")
            size_text = "—" if is_dir else human_size(stat.st_size)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            label = html.escape(name) + ("/" if is_dir else "")
            cls = "dir" if is_dir else "file"
            link = f'<a class="{cls}" href="{href}">{label}</a>'
            audio = ""
            if not is_dir and Path(name).suffix.lower() in AUDIO_EXTS:
                audio = f'<audio controls preload="metadata" src="{href}"></audio>'
            rows.append(
                f'<tr data-name="{html.escape(name.lower())}">'
                f"<td>{link}{audio}</td><td class='size'>{size_text}</td><td class='mtime'>{modified}</td></tr>"
            )

        title = html.escape(unquote(url_path))
        body = _PAGE.format(title=title, rows="\n".join(rows))
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        import io

        return io.BytesIO(data)

    def log_message(self, format, *args):  # quieter logs
        return


_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AuK 文件库 {title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; font-family: system-ui, -apple-system, "Noto Sans CJK SC", sans-serif;
         background: #14161a; color: #e6e6e6; }}
  header {{ position: sticky; top: 0; background: #1d2026; padding: 12px 18px; border-bottom: 1px solid #2c313a; }}
  h1 {{ font-size: 15px; margin: 0 0 8px; font-weight: 600; color: #9ecbff; word-break: break-all; }}
  input {{ width: 100%; box-sizing: border-box; padding: 8px 10px; border-radius: 8px;
           border: 1px solid #333a45; background: #101216; color: #e6e6e6; }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 8px 18px; border-bottom: 1px solid #23272f; vertical-align: middle; }}
  tr:hover {{ background: #1a1e24; }}
  a {{ color: #7cc4ff; text-decoration: none; }}
  a.dir {{ font-weight: 600; }}
  a:hover {{ text-decoration: underline; }}
  .size, .mtime {{ color: #8a93a3; white-space: nowrap; width: 1%; font-size: 12px; }}
  audio {{ display: block; margin-top: 6px; height: 32px; width: 320px; max-width: 100%; }}
</style>
</head>
<body>
<header>
  <h1>📁 {title}</h1>
  <input id="q" placeholder="过滤文件名…" oninput="filterRows(this.value)">
</header>
<table id="t"><tbody>
{rows}
</tbody></table>
<script>
function filterRows(q) {{
  q = q.toLowerCase();
  document.querySelectorAll('#t tbody tr[data-name]').forEach(function (tr) {{
    tr.style.display = tr.dataset.name.indexOf(q) >= 0 ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""


def local_ip() -> str:
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        if sock is not None:
            sock.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LAN file browser with inline audio playback")
    parser.add_argument("--dir", default=".", help="root directory to serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = str(Path(args.dir).resolve())
    handler = lambda *a, **kw: FileBrowser(*a, directory=root, **kw)  # noqa: E731
    server = HTTPServer((args.host, args.port), handler)
    print(f"serving {root} at http://{local_ip()}:{args.port}/ (LAN)")
    server.serve_forever()


if __name__ == "__main__":
    main()
