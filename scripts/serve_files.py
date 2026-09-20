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
import fcntl
import io
import json
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (workflow_store lives here)

import workflow_store  # noqa: E402 - same directory (scripts/)

TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".jsonl", ".log", ".py", ".yaml", ".yml", ".srt"}
TEXT_INLINE_MAX = 64 * 1024 * 1024
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

# TTS 后台任务状态：active = 正在合成的请求数（Breeze 单并发，正常是 0/1）
_TTS_LOCK = threading.Lock()
_TTS_STATE: dict = {"active": 0, "last": ""}

# PureVox 降噪器（首次用到时加载，之后常驻；torch/CUDA 优先）
_DENOISER: Any = None
_DENOISER_LOCK = threading.Lock()


def _denoiser():
    global _DENOISER
    with _DENOISER_LOCK:
        if _DENOISER is None:
            from audiobook.denoise import Denoiser

            _DENOISER = Denoiser()
        return _DENOISER


def _breeze():
    from audiobook.tts import BreezeConfig, _health

    config = BreezeConfig()
    return config, _health(config.base_url)


LLM_BASE_URL = os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1")


def _llm_health() -> dict | None:
    """Check if the local LLM server is reachable (GET /models)."""
    try:
        import urllib.request

        req = urllib.request.Request(LLM_BASE_URL.rstrip("/") + "/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return {"status": "ok", "url": LLM_BASE_URL}
    except Exception:  # noqa: BLE001
        return None


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "big5"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


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


class RunLock:
    """File-lock guard for ``run.pid`` — prevents duplicate annotation tasks.

    Usage::

        with RunLock(pidfile) as lock:
            if lock.rejected:
                return {"ok": False, "message": lock.rejected}
            # ... start subprocess ...
            lock.commit(proc.pid)
    """

    def __init__(self, pidfile: Path):
        self._pidfile = pidfile
        self._lock_path = pidfile.with_suffix(".lock")
        self._fd: int = -1
        self.rejected: str | None = None

    def __enter__(self):
        self._pidfile.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.rejected = "后台已有标注任务运行中。请先重启 LLM 或点击「LLM原始IO」查看当前任务。"
            return self
        # Lock acquired — check for a genuinely running process
        if self._pidfile.is_file():
            try:
                old_pid = int(self._pidfile.read_text(encoding="utf-8").strip())
                stat_path = Path(f"/proc/{old_pid}/stat")
                if stat_path.is_file():
                    state = stat_path.read_text(encoding="utf-8").rsplit(") ", 1)[-1][:1]
                    if state != "Z":
                        self.rejected = f"后台已有标注任务运行中（PID {old_pid}）。请先重启 LLM 或点击「LLM原始IO」查看当前任务。"
                        return self
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *_exc):
        if self._fd >= 0:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass

    def commit(self, pid: int) -> None:
        """Write the PID file and release the lock (lock auto-releases on exit)."""
        self._pidfile.write_text(str(pid), encoding="utf-8")


class FileBrowser(SimpleHTTPRequestHandler):
    server_version = "BreezeFiles/1.0"

    def _send_file(self, target: Path, content_type: str) -> None:
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_books(self) -> None:
        """List book dirs under outputs/ (newest first) so the dashboard can auto-pick.

        Nested run dirs (``outputs/<book>/<run>``, used for previews) are included and named
        ``<book>/<run>``; ordering uses the newest mtime of the run dir and its ``script/``.
        """
        root = Path(self.directory) / "outputs"
        books = []
        if root.is_dir():
            candidates = list(root.iterdir())
            for item in list(candidates):  # one level deeper: outputs/<book>/<run>
                if not item.is_dir():
                    continue
                try:
                    candidates += [sub for sub in item.iterdir() if sub.is_dir()]
                except OSError:
                    continue
            for item in candidates:
                if not ((item / "chapters").is_dir() or (item / "script").is_dir()):
                    continue
                try:
                    mtime = item.stat().st_mtime
                    if (item / "script").is_dir():
                        mtime = max(mtime, (item / "script").stat().st_mtime)
                    books.append({"book": str(item.relative_to(root)), "mtime": mtime})
                except OSError:
                    continue
        books.sort(key=lambda x: x["mtime"], reverse=True)
        payload = json.dumps(books, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        html = (
            "<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>audiobook</title><style>"
            ":root{color-scheme:dark;--bg:#0d1017;--panel:#151a23;--line:#26303f;--fg:#e6ecf3;--dim:#8b97a8;--accent:#5aa9ff}"
            "*{box-sizing:border-box}"
            "body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 system-ui,-apple-system,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}"
            ".msg{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:16px}"
            ".msg h2{font-size:20px;color:var(--fg);margin:0}"
            ".msg p{color:var(--dim);font-size:15px;margin:0}"
            "a{color:var(--accent);text-decoration:none}"
            "</style></head><body>"
            f"<div class='msg'>{body}</div>"
            "</body></html>"
        )
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_raw_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        """Send a complete HTML file as-is (no wrapping)."""
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _inject_nav(self, html: str, active: str) -> str:
        """Inject nav bar into served HTML, marking *active* page."""
        nav = (
            '<header style="position:sticky;top:0;z-index:6;display:flex;gap:12px;align-items:center;'
            'padding:10px 16px;background:#101319;border-bottom:1px solid #26303f;font:14px/1.6 system-ui">'
            '<nav style="display:flex;gap:12px;font-size:13px">'
            '<a href="/dashboard" style="color:#5aa9ff;text-decoration:none">看板</a>'
            '<a href="/prompt" style="color:#5aa9ff;text-decoration:none">提示词测试</a>'
            '<a href="/live" style="color:#5aa9ff;text-decoration:none">实时标记</a>'
            '<a href="/raw" style="color:#5aa9ff;text-decoration:none">LLM原始IO</a>'
            '<a href="/tts" style="color:#5aa9ff;text-decoration:none">TTS测试</a>'
            "</nav></header>"
        )
        nav = nav.replace(f">{active}<", f' style="color:#e6ecf3;font-weight:700;text-decoration:none">{active}<')
        if "<body" in html:
            return html.replace("<body", f"<body>{nav}", 1)
        return nav + html

    def _fail(self, status: HTTPStatus, message: str) -> None:
        """JSON error response: http.server's send_error cannot carry non-latin1 text."""
        self._send_json({"ok": False, "error": str(message)}, status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8") or "{}")
        return data if isinstance(data, dict) else {}

    # ---- TTS test page API ---------------------------------------------------
    @staticmethod
    def _tts_chunks(text: str, limit: int = 500) -> list[str]:
        """Pack NARRATION into chunks of up to ``limit`` chars, cutting ONLY at 。 or paragraph
        breaks (never mid-sentence unless a single sentence exceeds the limit)."""
        units = re.split(r"(?<=。)|(?<=\n)", text)  # each unit ends with 。 or a newline
        chunks: list[str] = []
        current = ""
        for unit in units:
            if not unit:
                continue
            if len(current) + len(unit) <= limit:
                current += unit
                continue
            if current.strip():
                chunks.append(current.strip())
            while len(unit) > limit:  # one 。-less sentence longer than the limit: hard cut
                chunks.append(unit[:limit].strip())
                unit = unit[limit:]
            current = unit
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _tts_rows(self, book: str, chapter: int) -> dict:
        """Speech/narration rows of one chapter (marked text or script.csv)."""
        base = Path(self.directory) / "outputs" / book
        rows: list[dict] = []
        csv_path = base / "script.csv"
        if csv_path.is_file():
            import csv

            with csv_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if int(row.get("chapter_id") or 0) != chapter:
                        continue
                    rows.append(
                        {
                            "kind": row.get("kind") or "speech",
                            "role": row.get("role_name") or row.get("role_id") or "",
                            "text": row.get("tts_text") or row.get("raw_text") or "",
                            "style": row.get("style_desc") or "",
                            "emotion": row.get("emotion") or "",
                        }
                    )
        if not rows:
            marked = base / "script" / f"ch{chapter:03d}.marked.txt"
            if marked.is_file():
                from audiobook.marks import parse_marks

                for seg in parse_marks(marked.read_text(encoding="utf-8")):
                    if seg["kind"] == "speech":
                        rows.append({"kind": "speech", "role": seg.get("role_name") or "", "text": seg["text"]})
                    elif seg["text"].strip():
                        rows.append({"kind": "narration", "role": "", "text": seg["text"]})
        expanded: list[dict] = []  # only narration is split (speech stays whole)
        for row in rows:
            if row["kind"] != "narration":
                expanded.append(row)
                continue
            for chunk in self._tts_chunks(row["text"]):
                expanded.append({**row, "text": chunk})
        return {"book": book, "chapter": chapter, "rows": expanded}

    def _tts_library_paths(self) -> tuple[Path, Path]:
        """Global (book-independent) designed-voice library: voices/designs/*.wav + index."""
        return Path(self.directory) / "voices" / "designs", Path(self.directory) / "voices" / "designs.json"

    def _tts_voices(self, book: str = "") -> dict:
        """Global voice library: narrator presets + every designed voice (all books share it)."""
        voices: list[dict] = []
        narrator_path = Path(self.directory) / "voices" / "narrator.json"
        if narrator_path.is_file():
            data = json.loads(narrator_path.read_text(encoding="utf-8"))
            for label, item in (data.get("voices") or {}).items():
                ref = narrator_path.parent / item.get("file", "")
                voices.append(
                    {
                        "name": f"旁白·{label}",
                        "kind": "narrator",
                        "ref": str(ref.relative_to(self.directory)),
                        "ref_text": item.get("ref_text") or data.get("ref_text") or "",
                        "instruction": item.get("instruction") or "",
                        "cfg": data.get("cfg") or 4.0,
                        "seed": data.get("seed") or 58,
                    }
                )
        designs_dir, index_path = self._tts_library_paths()
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        for name, item in index.items():
            ref = designs_dir / str(item.get("file") or "")
            if not ref.is_file():
                continue
            voices.append(
                {
                    "name": name,
                    "kind": "design",
                    "ref": str(ref.relative_to(self.directory)),
                    "ref_text": item.get("ref_text") or "",
                    "instruction": item.get("instruction") or "",
                    "cfg": item.get("cfg") or 4.0,
                    "seed": item.get("seed") or 58,
                }
            )
        return {"book": book, "voices": voices}

    def _tts_say(self, payload: dict) -> dict:
        """Synthesize ONE line with the chosen reference voice (manual test)."""
        from audiobook.schema import ScriptRow
        from audiobook.tts import BreezeRenderer

        from audiobook.marks import tts_text_of

        book = str(payload.get("book") or "").strip()
        text = str(payload.get("text") or "").strip()
        ref = str(payload.get("ref") or "").strip()
        if not text:
            raise ValueError("text 不能为空")
        text = tts_text_of(text)  # 引号不读出来（显示里仍保留）
        config, health = _breeze()
        if health is None:
            raise ValueError("Breeze 服务没在跑：先 `python scripts/serve_breeze.py up`")
        config.cfg_scale = float(payload.get("cfg") or 1.0)
        config.seed = int(payload.get("seed") or 1234)
        config.direction = bool(str(payload.get("instruction") or "").strip())
        ref_path = str((Path(self.directory) / ref.lstrip("/")).resolve()) if ref else ""
        row = ScriptRow(
            kind="speech", role_name=str(payload.get("role") or "测试"), raw_text=text, tts_text=text, seg_id="tts-test"
        )
        renderer = BreezeRenderer(config)
        with _TTS_LOCK:
            _TTS_STATE["active"] += 1
            _TTS_STATE["last"] = f"合成 · {text[:16]}"
        try:
            samples, rate = renderer.synth(
                row, ref_path, str(payload.get("ref_text") or ""), str(payload.get("instruction") or "")
            )
        finally:
            with _TTS_LOCK:
                _TTS_STATE["active"] -= 1
        out_dir = Path(self.directory) / "outputs" / book / "tts_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = str(abs(hash((text, ref, config.cfg_scale, config.seed))) % (1 << 32))
        target = out_dir / f"say_{digest}.wav"
        import soundfile as sf

        sf.write(str(target), samples, rate)
        return {"ok": True, "url": "/" + str(target.relative_to(self.directory)), "seconds": round(len(samples) / rate, 2)}

    def _tts_design(self, payload: dict) -> dict:
        """Design a fresh voice and save it to the GLOBAL library (shared by every book)."""
        from audiobook.tts import BreezeRenderer

        from audiobook.marks import tts_text_of

        text = tts_text_of(str(payload.get("text") or "").strip())  # 引号不读出来
        instruction = str(payload.get("instruction") or "").strip()
        name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(payload.get("name") or "voice")).strip("_") or "voice"
        seed = int(payload.get("seed") or 58)
        cfg = float(payload.get("cfg") or 4.0)  # voice design default cfg = 4
        save = bool(payload.get("save", True))
        if not text or not instruction:
            raise ValueError("text / instruction 都不能为空")
        config, health = _breeze()
        if health is None:
            raise ValueError("Breeze 服务没在跑：先点「启动 Breeze 服务」")
        config.cfg_scale = cfg
        renderer = BreezeRenderer(config)
        with _TTS_LOCK:
            _TTS_STATE["active"] += 1
            _TTS_STATE["last"] = f"造声 · {instruction[:16]}"
        try:
            samples, rate = renderer.design_voice(text, instruction, seed=seed)
        finally:
            with _TTS_LOCK:
                _TTS_STATE["active"] -= 1
        import soundfile as sf

        if not save:  # 试听 only: throwaway file, library untouched
            out_dir = Path(self.directory) / ".cache" / "tts_test"
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"design_{abs(hash((text, instruction, seed, cfg))) % (1 << 32)}.wav"
            sf.write(str(target), samples, rate)
            return {"ok": True, "url": "/" + str(target.relative_to(self.directory)), "ref_text": text, "saved": False}
        designs_dir, index_path = self._tts_library_paths()
        designs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{name}.wav"
        target = designs_dir / filename
        sf.write(str(target), samples, rate)
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        index[name] = {"file": filename, "ref_text": text, "instruction": instruction, "seed": seed, "cfg": cfg}
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "url": "/" + str(target.relative_to(self.directory)),
            "ref": str(target.relative_to(self.directory)),
            "ref_text": text,
            "name": name,
            "saved": True,
        }

    def _tts_concat(self, payload: dict) -> dict:
        """Concatenate the synthesized chapter segments into one WAV (with small gaps)."""
        import numpy as np
        import soundfile as sf

        book = str(payload.get("book") or "").strip()
        try:
            chapter = int(payload.get("chapter") or 0)
        except (TypeError, ValueError):
            chapter = 0
        urls = [str(item) for item in (payload.get("urls") or []) if str(item).strip()]
        if not book or not urls:
            raise ValueError("book / urls 不能为空")
        parts: list[np.ndarray] = []
        rate = 0
        for url in urls:
            path = (Path(self.directory) / url.lstrip("/")).resolve()
            if not path.is_file():
                continue
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1).astype(np.float32)
            if not rate:
                rate = sr
            elif sr != rate:
                from math import gcd

                from scipy.signal import resample_poly

                divisor = gcd(sr, rate)
                mono = resample_poly(mono, rate // divisor, sr // divisor).astype(np.float32)
            parts.append(mono)
        if not parts:
            raise ValueError("段落音频都读不到，先合成")
        gap = np.zeros(int(0.35 * rate), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for index, part in enumerate(parts):
            if index:
                pieces.append(gap)
            pieces.append(part)
        combined = np.concatenate(pieces)
        fade = min(len(combined), int(0.05 * rate))
        if fade:
            combined[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        out_dir = Path(self.directory) / "outputs" / book / "tts_test"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"ch{chapter:03d}_full.wav"
        sf.write(str(target), combined, rate, subtype="PCM_16")
        return {
            "ok": True,
            "url": "/" + str(target.relative_to(self.directory)),
            "seconds": round(len(combined) / rate, 2),
            "segments": len(parts),
        }

    def _tts_denoise(self, payload: dict) -> dict:
        """PureVox denoise one chapter file (GPU; used automatically after concat)."""
        import soundfile as sf

        url = str(payload.get("url") or "").strip().lstrip("/")
        if not url:
            raise ValueError("url 不能为空")
        src = (Path(self.directory) / url).resolve()
        if not src.is_file():
            raise ValueError("音频不存在")
        data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
        denoiser = _denoiser()
        started = time.perf_counter()
        output = denoiser.process(data, sample_rate)  # back at the INPUT sample rate
        target = src.with_name(f"{src.stem}_denoised.wav")
        sf.write(str(target), output, sample_rate, subtype="PCM_16")
        from audiobook.tts import encode_lossy

        mp3 = encode_lossy(target, target.with_suffix(".mp3"), fmt="mp3", bitrate=str(payload.get("bitrate") or "64k"))
        return {
            "ok": True,
            "url": "/" + str(mp3.relative_to(self.directory)),
            "wav": "/" + str(target.relative_to(self.directory)),
            "seconds": round(len(output) / sample_rate, 2),
            "elapsed": round(time.perf_counter() - started, 2),
            "backend": denoiser.backend,
        }

    def _tts_keep(self, payload: dict) -> dict:
        """Save one already-generated candidate (temp wav) into the global library."""
        ref = str(payload.get("ref") or "").strip().lstrip("/")
        src = (Path(self.directory) / ref).resolve()
        if not ref or not src.is_file():
            raise ValueError("候选音频不存在，先生成")
        text = str(payload.get("ref_text") or "").strip()
        instruction = str(payload.get("instruction") or "").strip()
        seed = int(payload.get("seed") or 0)
        cfg = float(payload.get("cfg") or 4.0)
        name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(payload.get("name") or "voice")).strip("_") or "voice"
        designs_dir, index_path = self._tts_library_paths()
        designs_dir.mkdir(parents=True, exist_ok=True)
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        base, suffix = name, 2
        while name in index or (designs_dir / f"{name}.wav").exists():
            name = f"{base}-{suffix}"
            suffix += 1
        import shutil

        shutil.copyfile(src, designs_dir / f"{name}.wav")
        index[name] = {"file": f"{name}.wav", "ref_text": text, "instruction": instruction, "seed": seed, "cfg": cfg}
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "name": name, "ref": f"voices/designs/{name}.wav"}

    def _tts_rename(self, payload: dict) -> dict:
        """Rename one designed voice (file + library key). Parameters are not editable."""
        old = str(payload.get("name") or "").strip()
        new = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", str(payload.get("new_name") or "")).strip("_")
        if not old or not new:
            raise ValueError("name / new_name 不能为空")
        designs_dir, index_path = self._tts_library_paths()
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        if old not in index:
            raise ValueError("只能改全局音色库里的音色（旁白固定在 voices/ 下，不能改名）")
        item = index.pop(old)
        base, suffix = new, 2
        while new in index or (designs_dir / f"{new}.wav").exists():
            new = f"{base}-{suffix}"
            suffix += 1
        old_file = designs_dir / str(item.get("file") or "")
        new_file = designs_dir / f"{new}.wav"
        if old_file.is_file():
            old_file.rename(new_file)
        item["file"] = new_file.name
        index[new] = item
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "name": new, "ref": f"voices/designs/{new_file.name}"}

    def _tts_status(self) -> dict:
        _, health = _breeze()
        with _TTS_LOCK:
            active, last = _TTS_STATE["active"], _TTS_STATE["last"]
        return {"up": health is not None, "active": active, "last": last if active else "", "health": health or {}}

    def _tts_restart(self) -> dict:
        """Restart the owned breeze-server (kills whatever is being synthesized)."""
        pidfile = APP_ROOT / "build" / "breeze-cpp" / "server.pid"
        pid = 0
        if pidfile.is_file():
            try:
                pid = int(pidfile.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                pid = 0
            if pid:
                comm = Path(f"/proc/{pid}/comm")
                if comm.is_file() and comm.read_text(encoding="utf-8").strip() == "breeze-server":
                    os.kill(pid, 15)
                    for _ in range(20):
                        if not Path(f"/proc/{pid}").exists():
                            break
                        time.sleep(0.5)
            pidfile.unlink(missing_ok=True)
        result = self._tts_up()
        result["restarted"] = True
        result["message"] = "Breeze 已重启"
        return result

    def _tts_down(self) -> dict:
        """Stop the owned breeze-server and free GPU VRAM."""
        pidfile = APP_ROOT / "build" / "breeze-cpp" / "server.pid"
        pid = 0
        if pidfile.is_file():
            try:
                pid = int(pidfile.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                pid = 0
            if pid:
                comm = Path(f"/proc/{pid}/comm")
                if comm.is_file() and comm.read_text(encoding="utf-8").strip() == "breeze-server":
                    os.kill(pid, 15)
                    for _ in range(20):
                        if not Path(f"/proc/{pid}").exists():
                            break
                        time.sleep(0.5)
            pidfile.unlink(missing_ok=True)
        return {"ok": True, "message": "Breeze 已卸载，GPU 显存已释放"}

    # ── LLM (llama.cpp) management ────────────────────────────────────────────

    def _llm_status(self) -> dict:
        health = _llm_health()
        return {"up": health is not None, "health": health or {}, "url": LLM_BASE_URL}

    def _llm_up(self) -> dict:
        import subprocess

        if _llm_health():
            return {"ok": True, "message": "LLM 本来就在运行", "started": False}
        log_path = APP_ROOT / ".cache" / "logs" / "llm_serve.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "AUDIOBOOK_LLAMA_BIN": str(APP_ROOT / "build" / "llama-cpp" / "bin"),
            "AUDIOBOOK_LLM_GGUF": str(APP_ROOT / "ckpts" / "llm" / "Ternary-Bonsai-27B-PQ2_0.gguf"),
            "AUDIOBOOK_LLM_DRAFT": str(APP_ROOT / "ckpts" / "llm" / "Ternary-Bonsai-27B-dspark-dflash-Q4_1.gguf"),
            "AUDIOBOOK_LLM_SPEC": "draft-dspark",
            "AUDIOBOOK_LLM_SPEC_DRAFT_N_MAX": "4",
            "AUDIOBOOK_LLM_KV": "q4_0",
            "AUDIOBOOK_LLM_NGL": "99",
            "AUDIOBOOK_LLM_CTX": os.environ.get("AUDIOBOOK_LLM_CTX", "32768"),
            "NO_PROXY": "*",
        }
        with log_path.open("w") as log:
            subprocess.Popen(
                ["bash", "scripts/serve_llm_cuda.sh"],
                cwd=str(APP_ROOT),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # wait up to 30s for the server to come up
        for _ in range(60):
            time.sleep(0.5)
            if _llm_health():
                return {"ok": True, "message": "LLM 已启动", "started": True, "log": "/llm/serve.log"}
        return {"ok": False, "message": "LLM 启动超时（30s），查看日志 /llm/serve.log"}

    def _llm_down(self) -> dict:
        import signal as _signal
        import subprocess

        result = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
        pids = [int(p) for p in result.stdout.strip().split() if p.strip().isdigit()]
        if not pids:
            return {"ok": True, "message": "LLM 未在运行"}
        for pid in pids:
            try:
                os.kill(pid, _signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(20):
            time.sleep(0.5)
            alive = any(Path(f"/proc/{p}").exists() for p in pids if p)
            if not alive:
                return {"ok": True, "message": f"LLM 已停止（{len(pids)} 进程）"}
        return {"ok": True, "message": f"LLM 发送了停止信号（{len(pids)} 进程），可能仍在退出中"}

    def _prepare_book(self, payload: dict) -> dict:
        """Run the prepare stage (clean + split chapters) for a book."""
        import subprocess

        txt = str(payload.get("txt") or "").strip()
        book = str(payload.get("book") or "").strip()
        if not txt or not book:
            raise ValueError("txt 路径和 book 名不能为空")
        txt_path = (APP_ROOT / txt).resolve()
        if not txt_path.is_file():
            raise ValueError(f"文件不存在：{txt}")
        out_dir = APP_ROOT / "outputs" / book
        log_path = APP_ROOT / ".cache" / "logs" / f"prepare-{book}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            subprocess.run(
                [sys.executable, "-B", "scripts/build_book.py", str(txt_path), "--out", str(out_dir)],
                cwd=str(APP_ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        chapters = sorted(out_dir.glob("chapters/ch*.txt"))
        return {
            "ok": True,
            "message": f"准备完成：{len(chapters)} 章",
            "chapters": len(chapters),
            "out": str(out_dir.relative_to(APP_ROOT)),
        }

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _run_stage(self, book: str, stage: str, cmd: list[str], log_name: str) -> dict:
        """Run a pipeline stage in the background; return immediately with a log URL."""
        import subprocess

        base = APP_ROOT
        log = base / ".cache" / "logs" / f"{log_name}-{book}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "AUDIOBOOK_BOOK": book,
            "AUDIOBOOK_LLM_BASE_URL": os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
            "AUDIOBOOK_LLM_MODEL": os.environ.get("AUDIOBOOK_LLM_MODEL", "Ternary-Bonsai-27B-PQ2_0"),
            "AUDIOBOOK_LLM_PROFILE": os.environ.get("AUDIOBOOK_LLM_PROFILE", "Ternary-Bonsai-27B-PQ2_0"),
            "NO_PROXY": "*",
        }
        with log.open("w") as handle:
            subprocess.Popen(
                ["setsid", "--fork"] + cmd,
                cwd=str(base),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        return {"ok": True, "message": f"{stage} 已启动", "log": f"/stage/log?name={log_name}-{book}"}

    def _script_status(self, book: str) -> dict:
        """Check if an annotation task is running for this book, with progress."""
        if not book:
            return {"running": False, "paused": False}
        base = APP_ROOT / "outputs" / book / "script"
        pidfile = base / "run.pid"
        pause_file = base / ".paused"
        progress_file = base / "progress.json"
        result: dict[str, Any] = {"running": False, "paused": pause_file.is_file()}
        # read progress if available
        if progress_file.is_file():
            try:
                result["progress"] = json.loads(progress_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        if not pidfile.is_file():
            return result
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
            stat_path = Path(f"/proc/{pid}/stat")
            if stat_path.is_file():
                state = stat_path.read_text(encoding="utf-8").rsplit(") ", 1)[-1][:1]
                if state != "Z":
                    result["running"] = True
                    result["pid"] = pid
        except (ValueError, OSError):
            pass
        return result

    def _script_pause(self, book: str) -> dict:
        """Pause the running annotation task for this book."""
        if not book:
            raise ValueError("book 不能为空")
        base = APP_ROOT / "outputs" / book / "script"
        pidfile = base / "run.pid"
        if not pidfile.is_file():
            return {"ok": False, "message": "标注未在运行"}
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
            stat_path = Path(f"/proc/{pid}/stat")
            if not stat_path.is_file():
                return {"ok": False, "message": "标注进程已结束"}
        except (ValueError, OSError):
            return {"ok": False, "message": "标注未在运行"}
        pause_file = base / ".paused"
        pause_file.write_text(str(pid), encoding="utf-8")
        return {"ok": True, "message": "标注已暂停", "paused": True}

    def _script_resume(self, book: str) -> dict:
        """Resume the paused annotation task for this book."""
        if not book:
            raise ValueError("book 不能为空")
        base = APP_ROOT / "outputs" / book / "script"
        pause_file = base / ".paused"
        if not pause_file.is_file():
            return {"ok": True, "message": "标注未暂停", "paused": False}
        pause_file.unlink()
        return {"ok": True, "message": "标注已恢复", "paused": False}

    def _script_stop(self, book: str) -> dict:
        """Stop the running annotation task for this book."""
        if not book:
            raise ValueError("book 不能为空")
        base = APP_ROOT / "outputs" / book / "script"
        pidfile = base / "run.pid"
        # clean up pause file
        pause_file = base / ".paused"
        if pause_file.is_file():
            pause_file.unlink()
        if not pidfile.is_file():
            return {"ok": True, "message": "标注未在运行"}
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pidfile.unlink(missing_ok=True)
            return {"ok": True, "message": "标注未在运行"}
        import signal as _signal

        try:
            os.kill(pid, _signal.SIGTERM)
        except ProcessLookupError:
            pass
        pidfile.unlink(missing_ok=True)
        return {"ok": True, "message": f"已停止标注（PID {pid}）"}

    def _stage_script(self, payload: dict) -> dict:
        """Run mark_script for all chapters (background)."""
        import subprocess

        book = str(payload.get("book") or "").strip()
        if not book:
            raise ValueError("book 不能为空")
        if not _llm_health():
            return {"ok": False, "message": "LLM 未启动，请先点击「启动 LLM」"}
        start = int(payload.get("start") or 0)
        end = int(payload.get("end") or 0)
        count = int(payload.get("count") or 0)
        base = APP_ROOT
        out = base / "outputs" / book
        chapter_ids = sorted(int(p.stem[2:]) for p in (out / "chapters").glob("ch*.txt"))
        if not chapter_ids:
            raise ValueError("无章节，请先准备书籍")
        if not start:
            start = chapter_ids[0]
        if not end:
            end = chapter_ids[-1]
        if count:
            end = start + count - 1
        log = base / ".cache" / "logs" / f"script-{book}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        pidfile = out / "script" / "run.pid"
        with RunLock(pidfile) as lock:
            if lock.rejected:
                return {"ok": False, "message": lock.rejected}
            env = {
                **os.environ,
                "AUDIOBOOK_BOOK": book,
                "AUDIOBOOK_LLM_BASE_URL": os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                "AUDIOBOOK_LLM_MODEL": os.environ.get("AUDIOBOOK_LLM_MODEL", "Ternary-Bonsai-27B-PQ2_0"),
                "AUDIOBOOK_LLM_PROFILE": os.environ.get("AUDIOBOOK_LLM_PROFILE", "Ternary-Bonsai-27B-PQ2_0"),
                "NO_PROXY": "*",
            }
            python = base / ".venv" / "bin" / "python"
            py = str(python if python.is_file() else Path(sys.executable))
            with log.open("w") as handle:
                proc = subprocess.Popen(
                    [
                        "setsid",
                        "--fork",
                        py,
                        "-B",
                        "scripts/mark_script.py",
                        str(start),
                        "--count",
                        str(end - start + 1),
                        "--book",
                        book,
                        "--batch",
                        "5",
                    ],
                    cwd=str(base),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            lock.commit(proc.pid)
        return {
            "ok": True,
            "message": f"标注已启动（ch{start:03d}-ch{end:03d}，共 {end - start + 1} 章）",
            "log": f"/stage/log?name=script-{book}",
            "pid": proc.pid,
        }

    def _stage_convert(self, payload: dict) -> dict:
        """Run marks_to_script (sync)."""
        import subprocess

        book = str(payload.get("book") or "").strip()
        if not book:
            raise ValueError("book 不能为空")
        out = APP_ROOT / "outputs" / book
        log = APP_ROOT / ".cache" / "logs" / f"convert-{book}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/marks_to_script.py",
                    "--marked-dir",
                    str(out / "script"),
                    "--cast",
                    str(out / "cast.json"),
                    "--out",
                    str(out),
                ],
                cwd=str(APP_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return {"ok": True, "message": "convert 完成", "log": f"/stage/log?name=convert-{book}"}

    def _stage_voicebank(self, payload: dict) -> dict:
        """Run build_voicebank_breeze (sync, needs Breeze running)."""
        import subprocess

        book = str(payload.get("book") or "").strip()
        if not book:
            raise ValueError("book 不能为空")
        out = APP_ROOT / "outputs" / book
        log = APP_ROOT / ".cache" / "logs" / f"voicebank-{book}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/build_voicebank_breeze.py",
                    "--book",
                    book,
                    "--out",
                    str(out),
                ],
                cwd=str(APP_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return {"ok": True, "message": "voicebank 完成", "log": f"/stage/log?name=voicebank-{book}"}

    def _stage_render(self, payload: dict) -> dict:
        """Run render_book (sync, needs Breeze running)."""
        import subprocess

        book = str(payload.get("book") or "").strip()
        if not book:
            raise ValueError("book 不能为空")
        out = APP_ROOT / "outputs" / book
        script_csv = out / "script.csv"
        if not script_csv.is_file():
            raise ValueError(f"缺少 {script_csv}，请先跑 convert")
        log = APP_ROOT / ".cache" / "logs" / f"render-{book}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle:
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/render_book.py",
                    "--script",
                    str(script_csv),
                    "--out",
                    str(out / "render"),
                    "--cast",
                    str(out / "cast.json"),
                ],
                cwd=str(APP_ROOT),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return {"ok": True, "message": "render 完成", "log": f"/stage/log?name=render-{book}"}

    def _stage_log(self, payload: dict) -> dict:
        """Return the tail of a stage log."""
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("name 不能为空")
        log = APP_ROOT / ".cache" / "logs" / f"{name}.log"
        if not log.is_file():
            return {"ok": True, "content": "", "exists": False}
        content = log.read_text(encoding="utf-8", errors="replace")
        return {"ok": True, "content": content[-8192:], "exists": True}

    def _tts_up(self) -> dict:
        from audiobook.tts import BreezeConfig, start_server

        config = BreezeConfig()
        # paths in BreezeConfig are project-relative: anchor them to the served root
        config.server_bin = str(APP_ROOT / config.server_bin)
        config.model_path = str(APP_ROOT / config.model_path)
        config.voices_dir = str(APP_ROOT / config.voices_dir)
        process = start_server(config, log=lambda message: None)
        if process is not None:  # record the pid so `scripts/serve_breeze.py down` can stop it
            pidfile = APP_ROOT / "build" / "breeze-cpp" / "server.pid"
            pidfile.parent.mkdir(parents=True, exist_ok=True)
            pidfile.write_text(str(process.pid), encoding="utf-8")
            message = "Breeze 已启动"
        else:
            message = "Breeze 本来就在运行"
        return {"ok": True, "url": config.base_url, "started": process is not None, "message": message}

    def _chapter_state(self, book: str, chapter: int) -> dict:
        """Raw + marked text of one example chapter, plus whether a run is alive."""
        base = Path(self.directory) / "outputs" / book
        raw_path = base / "chapters" / f"ch{chapter:03d}.txt"
        marked_path = base / "script" / f"ch{chapter:03d}.marked.txt"
        pidfile = base / "script" / "run.pid"
        running = False
        pid = 0
        if pidfile.is_file():
            try:
                pid = int(pidfile.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)
                running = True
            except (OSError, ValueError):
                running = False
        return {
            "book": book,
            "chapter": chapter,
            "raw": raw_path.read_text(encoding="utf-8") if raw_path.is_file() else "",
            "marked": marked_path.read_text(encoding="utf-8") if marked_path.is_file() else "",
            "raw_mtime": raw_path.stat().st_mtime if raw_path.is_file() else 0,
            "marked_mtime": marked_path.stat().st_mtime if marked_path.is_file() else 0,
            "running": running,
            "pid": pid if running else 0,
        }

    def _run_chapter(self, payload: dict) -> dict:
        """Run mark_script for ONE example chapter (reads the persisted overlay)."""
        book = str(payload.get("book") or "").strip()
        try:
            chapter = int(payload.get("chapter") or 0)
        except (TypeError, ValueError):
            chapter = 0
        if not book or chapter <= 0:
            raise ValueError("book / chapter 不能为空")
        base = Path(self.directory)
        if not (base / "outputs" / book / "chapters" / f"ch{chapter:03d}.txt").is_file():
            raise ValueError(f"章节不存在：{book} ch{chapter:03d}")
        python = base / ".venv" / "bin" / "python"
        log = base / ".cache" / "logs" / f"run-{book.replace('/', '_')}-ch{chapter:03d}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        pidfile = base / "outputs" / book / "script" / "run.pid"
        with RunLock(pidfile) as lock:
            if lock.rejected:
                return {"ok": False, "message": lock.rejected}
            env = {
                **os.environ,
                "AUDIOBOOK_BOOK": book,
                "AUDIOBOOK_LLM_BASE_URL": os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
                "AUDIOBOOK_LLM_MODEL": os.environ.get("AUDIOBOOK_LLM_MODEL", "Ternary-Bonsai-27B-PQ2_0"),
                "AUDIOBOOK_LLM_PROFILE": os.environ.get("AUDIOBOOK_LLM_PROFILE", "Ternary-Bonsai-27B-PQ2_0"),
                "NO_PROXY": "*",
            }
            with log.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    [
                        str(python if python.is_file() else Path(sys.executable)),
                        "-B",
                        "scripts/mark_script.py",
                        str(chapter),
                        "--count",
                        "1",
                        "--book",
                        book,
                        "--force",
                        "--no-live",
                    ],
                    cwd=base,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            lock.commit(proc.pid)
        return {"pid": proc.pid, "log": str(log.relative_to(base))}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/books.json":
            self._send_books()
            return
        if route == "/chapter.json":
            query = parse_qs(parsed.query)
            book = (query.get("book") or [""])[0]
            try:
                chapter = int((query.get("chapter") or ["0"])[0])
            except ValueError:
                chapter = 0
            try:
                if not book or chapter <= 0:
                    raise ValueError("book / chapter 不能为空")
                self._send_json(self._chapter_state(book, chapter))
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/chapters.json":
            book = (parse_qs(parsed.query).get("book") or [""])[0]
            try:
                if not book:
                    raise ValueError("book 不能为空")
                chapters_dir = Path(self.directory) / "outputs" / book / "chapters"
                ids = sorted(int(path.stem[2:]) for path in chapters_dir.glob("ch*.txt")) if chapters_dir.is_dir() else []
                self._send_json({"book": book, "chapters": ids})
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route.startswith("/api/"):
            # /api/<book>/<filepath> — serve raw data from outputs/<book>/script/
            parts = route[len("/api/") :].split("/", 1)
            book = unquote(parts[0]) if parts else ""
            subpath = unquote(parts[1]) if len(parts) > 1 else ""
            if not book or not subpath:
                self._fail(HTTPStatus.BAD_REQUEST, "用法: /api/<book>/<file>")
                return
            target = Path(self.directory) / "outputs" / book / "script" / subpath
            if not target.is_file():
                self._fail(HTTPStatus.NOT_FOUND, f"{book}/script/{subpath} 不存在")
                return
            ct = "application/json" if subpath.endswith(".json") else "text/plain; charset=utf-8"
            if subpath.endswith(".jsonl"):
                ct = "application/x-ndjson; charset=utf-8"
            self._send_file(target, ct)
            return
        # ── Prompt test API (book-independent) ────────────────────────────────────
        if route == "/prompt/defaults":
            try:
                self._send_json(self._prompt_defaults())
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prompt/versions":
            try:
                self._send_json(self._prompt_versions())
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route.startswith("/prompt/version/"):
            vid = route[len("/prompt/version/") :]
            try:
                self._send_json(self._prompt_version_get(vid))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prompt/history":
            try:
                self._send_json(self._prompt_history())
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route.startswith("/prompt/run/"):
            rid = route[len("/prompt/run/") :]
            try:
                self._send_json(self._prompt_run_get(rid))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        # ── Static pages (all from static/) ──────────────────────────────────────
        _pages = {
            "/": "dashboard.html",
            "/dashboard": "dashboard.html",
            "/prompt": "prompt.html",
            "/workflow": "prompt.html",
            "/tts": "tts.html",
            "/live": "live.html",
            "/raw": "raw.html",
        }
        if route in _pages:
            target = Path(self.directory) / "static" / _pages[route]
            if target.is_file():
                self._send_file(target, "text/html; charset=utf-8")
                return
            self._fail(HTTPStatus.NOT_FOUND, f"static/{_pages[route]} not found")
            return
        if route == "/tts/rows.json":
            query = parse_qs(parsed.query)
            book = (query.get("book") or [""])[0]
            try:
                chapter = int((query.get("chapter") or ["0"])[0])
            except ValueError:
                chapter = 0
            try:
                if not book or chapter <= 0:
                    raise ValueError("book / chapter 不能为空")
                self._send_json(self._tts_rows(book, chapter))
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/tts/voices.json":
            book = (parse_qs(parsed.query).get("book") or [""])[0]  # book optional: library is global
            try:
                self._send_json(self._tts_voices(book))
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route in ("/tts/health.json", "/tts/status.json"):
            self._send_json(self._tts_status())
            return
        if route == "/llm/status.json":
            self._send_json(self._llm_status())
            return
        if route == "/script/status.json":
            query = parse_qs(parsed.query)
            book = (query.get("book") or [""])[0]
            self._send_json(self._script_status(book))
            return
        if route == "/status.json":
            query = parse_qs(parsed.query)
            book = (query.get("book") or [""])[0]
            llm = self._llm_status()
            _, breeze_health = _breeze()
            script = self._script_status(book) if book else {"running": False, "paused": False}
            self._send_json(
                {
                    "llm": llm,
                    "breeze": {"up": breeze_health is not None, "health": breeze_health or {}},
                    "script": script,
                }
            )
            return
        if route == "/llm/serve.log":
            log = APP_ROOT / ".cache" / "logs" / "llm_serve.log"
            if log.is_file():
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(log.read_bytes()[-8192:])
            else:
                self._send_html("<h2>日志不存在</h2>")
            return
        if route == "/txt/list":
            txt_dir = APP_ROOT / "assets" / "txt"
            files = sorted(f.name for f in txt_dir.glob("*.txt")) if txt_dir.is_dir() else []
            self._send_json({"ok": True, "files": files, "dir": str(txt_dir.relative_to(APP_ROOT))})
            return
        if route == "/workflow.json":
            book = parse_qs(parsed.query).get("book", [""])[0]
            try:
                if not book:
                    raise ValueError("book 不能为空")
                self._send_json(workflow_store.merged(book))
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/workflow/run":
            try:
                self._send_json(self._run_chapter(self._read_json()))
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route in ("/llm/up", "/llm/down"):
            try:
                self._read_json()  # consume body
                if route == "/llm/up":
                    self._send_json(self._llm_up())
                else:
                    self._send_json(self._llm_down())
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/script/stop":
            try:
                payload = self._read_json()
                book = str(payload.get("book") or "").strip()
                self._send_json(self._script_stop(book))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/script/pause":
            try:
                payload = self._read_json()
                book = str(payload.get("book") or "").strip()
                self._send_json(self._script_pause(book))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/script/resume":
            try:
                payload = self._read_json()
                book = str(payload.get("book") or "").strip()
                self._send_json(self._script_resume(book))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prompt/version":
            try:
                self._send_json(self._prompt_version_save(self._read_json()))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prompt/restore":
            try:
                self._send_json(self._prompt_restore(self._read_json()))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prompt/run":
            try:
                self._send_json(self._prompt_run(self._read_json()))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route == "/prepare":
            try:
                self._send_json(self._prepare_book(self._read_json()))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route in ("/script", "/convert", "/voicebank", "/render", "/stage/log"):
            try:
                payload = self._read_json()
                if route == "/script":
                    self._send_json(self._stage_script(payload))
                elif route == "/convert":
                    self._send_json(self._stage_convert(payload))
                elif route == "/voicebank":
                    self._send_json(self._stage_voicebank(payload))
                elif route == "/render":
                    self._send_json(self._stage_render(payload))
                else:
                    self._send_json(self._stage_log(payload))
            except Exception as error:  # noqa: BLE001
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        tts_routes = (
            "/tts/say",
            "/tts/design",
            "/tts/keep",
            "/tts/rename",
            "/tts/concat",
            "/tts/denoise",
            "/tts/up",
            "/tts/down",
            "/tts/restart",
        )
        if route in tts_routes:
            try:
                payload = self._read_json()
                if route == "/tts/say":
                    self._send_json(self._tts_say(payload))
                elif route == "/tts/design":
                    self._send_json(self._tts_design(payload))
                elif route == "/tts/keep":
                    self._send_json(self._tts_keep(payload))
                elif route == "/tts/rename":
                    self._send_json(self._tts_rename(payload))
                elif route == "/tts/concat":
                    self._send_json(self._tts_concat(payload))
                elif route == "/tts/denoise":
                    self._send_json(self._tts_denoise(payload))
                elif route == "/tts/down":
                    self._send_json(self._tts_down())
                elif route == "/tts/restart":
                    self._send_json(self._tts_restart())
                else:
                    self._send_json(self._tts_up())
            except Exception as error:  # noqa: BLE001 - report to the page
                self._fail(HTTPStatus.BAD_REQUEST, str(error))
            return
        if route != "/workflow.json":
            self._send_html("<h2>not found</h2>")
            return
        try:
            payload = self._read_json()
            book = str(payload.get("book") or parse_qs(parsed.query).get("book", [""])[0]).strip()
            if not book:
                raise ValueError("book 不能为空")
            if payload.get("action") == "clear":
                state = workflow_store.clear(book, actor="web")
            else:
                state = workflow_store.save(book, payload, actor="web")
            self._send_json(state)
        except Exception as error:  # noqa: BLE001 - report to the page
            self._fail(HTTPStatus.BAD_REQUEST, str(error))

    def send_head(self) -> io.BufferedIOBase | io.BytesIO | None:  # type: ignore
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return self.list_directory(str(path))
        if not path.is_file():
            self._send_html("<h2>File not found</h2>")
            return None

        range_header = self.headers.get("Range")

        if not range_header and path.suffix.lower() in TEXT_EXTS and path.stat().st_size <= TEXT_INLINE_MAX:
            payload = decode_text(path.read_bytes()).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", self.guess_type(str(path)))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            return io.BytesIO(payload)

        try:
            handle = path.open("rb")
        except OSError:
            self._send_html("<h2>File not found</h2>")
            return None

        stat = os.fstat(handle.fileno())
        size = stat.st_size
        content_type = self.guess_type(str(path))
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
        # The file browser is gone: directory URLs just go back to the dashboard.
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", "/dashboard")
        self.end_headers()
        return None

    # ── Prompt test helpers ─────────────────────────────────────────────────
    def _prompt_test_dir(self) -> Path:
        return Path(self.directory) / "prompt_test"

    def _prompt_defaults(self) -> dict:
        """Return Python code's default prompts + default input text."""
        # Import mark_script to get its constants
        sys.path.insert(0, str(Path(self.directory)))
        try:
            from scripts.mark_script import LOCAL_SYSTEM, STEP_MARK, CHECK_MARK  # type: ignore[import-not-found]
        except ImportError:
            LOCAL_SYSTEM = STEP_MARK = CHECK_MARK = ""
        default_input = ""
        inp = self._prompt_test_dir() / "default_input.txt"
        if inp.is_file():
            default_input = inp.read_text(encoding="utf-8")
        return {
            "ok": True,
            "prompts": {"LOCAL_SYSTEM": LOCAL_SYSTEM, "STEP_MARK": STEP_MARK, "CHECK_MARK": CHECK_MARK},
            "input": default_input,
        }

    def _prompt_versions(self) -> dict:
        """List saved prompt versions."""
        vdir = self._prompt_test_dir() / "versions"
        if not vdir.is_dir():
            return {"ok": True, "versions": []}
        versions = []
        for p in sorted(vdir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["id"] = p.stem
                versions.append(data)
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "versions": versions}

    def _prompt_version_get(self, vid: str) -> dict:
        """Get a specific prompt version."""
        vfile = self._prompt_test_dir() / "versions" / f"{vid}.json"
        if not vfile.is_file():
            raise ValueError(f"版本 {vid} 不存在")
        data = json.loads(vfile.read_text(encoding="utf-8"))
        data["id"] = vid
        return {"ok": True, "version": data}

    def _prompt_version_save(self, payload: dict) -> dict:
        """Save a new prompt version."""
        vdir = self._prompt_test_dir() / "versions"
        vdir.mkdir(parents=True, exist_ok=True)
        vid = payload.get("id") or f"v{int(time.time())}"
        data = {
            "label": payload.get("label", vid),
            "prompts": payload.get("prompts", {}),
            "input": payload.get("input", ""),
            "params": payload.get("params", {}),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        vfile = vdir / f"{vid}.json"
        vfile.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "id": vid, "message": f"版本 {vid} 已保存"}

    def _prompt_restore(self, payload: dict) -> dict:
        """Restore prompts to defaults or a specific version."""
        vid = payload.get("version_id")
        if vid:
            return self._prompt_version_get(vid)
        return self._prompt_defaults()

    def _prompt_run(self, payload: dict) -> dict:
        """Run a test: save snapshot + execute marking in background."""
        import subprocess

        prompts = payload.get("prompts", {})
        input_text = payload.get("input", "")
        params = payload.get("params", {})
        version_id = payload.get("version_id", f"v{int(time.time())}")

        # Save run snapshot
        rdir = self._prompt_test_dir() / "runs"
        rdir.mkdir(parents=True, exist_ok=True)
        run_id = f"run_{int(time.time())}"
        run_data = {
            "version_id": version_id,
            "prompts": prompts,
            "input": input_text,
            "params": params,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        run_file = rdir / f"{run_id}.json"
        run_file.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Write temporary workflow overlay for this test
        test_book = "__prompt_test__"
        test_out = Path(self.directory) / "outputs" / test_book
        test_out.mkdir(parents=True, exist_ok=True)
        test_chapters = test_out / "chapters"
        test_chapters.mkdir(exist_ok=True)
        # Write the input text as chapter 1
        (test_chapters / "ch001.txt").write_text(input_text, encoding="utf-8")
        # Write workflow overlay
        wf = test_out / "workflow.json"
        wf_data = {
            "params": params,
            "prompts": prompts,
            "few_shot": payload.get("few_shot", []),
        }
        wf.write_text(json.dumps(wf_data, ensure_ascii=False, indent=2), encoding="utf-8")
        # Write empty roles.json
        (test_out / "script" / "roles.json").write_text("{}", encoding="utf-8")
        (test_out / "script").mkdir(exist_ok=True)

        if not _llm_health():
            return {"ok": False, "message": "LLM 未启动"}

        base = Path(self.directory)
        py = str(base / ".venv" / "bin" / "python")
        log = base / ".cache" / "logs" / f"prompt-run-{run_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "AUDIOBOOK_WORKFLOW_DIR": str(test_out)}
        with log.open("w") as handle:
            proc = subprocess.Popen(
                [py, "-B", "scripts/mark_script.py", "1", "--count", "1", "--book", test_book, "--force"],
                cwd=str(base),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        # Save pid to run record
        run_data["pid"] = proc.pid
        run_data["log"] = f"prompt-run-{run_id}.log"
        run_file.write_text(json.dumps(run_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "run_id": run_id, "pid": proc.pid, "message": "测试已启动"}

    def _prompt_history(self) -> dict:
        """List all test run records."""
        rdir = self._prompt_test_dir() / "runs"
        if not rdir.is_dir():
            return {"ok": True, "runs": []}
        runs = []
        for p in sorted(rdir.glob("run_*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["id"] = p.stem
                runs.append(data)
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "runs": runs[:50]}

    def _prompt_run_get(self, rid: str) -> dict:
        """Get a specific test run record."""
        rfile = self._prompt_test_dir() / "runs" / f"{rid}.json"
        if not rfile.is_file():
            raise ValueError(f"记录 {rid} 不存在")
        data = json.loads(rfile.read_text(encoding="utf-8"))
        data["id"] = rid
        # Try to read output from the test book
        test_out = Path(self.directory) / "outputs" / "__prompt_test__" / "script"
        out_file = test_out / "ch001.marked.txt"
        if out_file.is_file():
            data["output"] = out_file.read_text(encoding="utf-8")
        return {"ok": True, "run": data}

    def log_message(self, format, *args):  # quieter logs
        return


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
    import atexit
    import signal as _signal

    def _cleanup_llm():
        """Kill llama-server when serve_files exits."""
        import subprocess as _sp

        try:
            for line in _sp.check_output(["pgrep", "-x", "llama-server"], text=True).splitlines():
                pid = int(line.strip())
                os.kill(pid, _signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass

    atexit.register(_cleanup_llm)

    def _sigterm(signum, frame):
        _cleanup_llm()
        raise SystemExit(0)

    _signal.signal(_signal.SIGTERM, _sigterm)
    _signal.signal(_signal.SIGINT, _sigterm)

    args = parse_args()
    root = str(Path(args.dir).resolve())
    handler = lambda *a, **kw: FileBrowser(*a, directory=root, **kw)  # noqa: E731
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"serving {root} at http://{local_ip()}:{args.port}/ (LAN)")
    server.serve_forever()


if __name__ == "__main__":
    main()
