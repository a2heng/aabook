"""Breeze TTS 2 backend, single file: synth + assemble + encode + events.

Sections (top to bottom): loudness/assembly, speed (atempo), length estimate,
vocal events, then the HTTP renderer and server lifecycle. Import from here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .schema import ScriptRow

"""Assemble per-row wavs into chapter tracks, then (optionally) the whole book.

Loudness is normalized to ``TARGET_LUFS`` (EBU-ish) with a hard peak ceiling, so
chapters can be concatenated without level jumps between speakers.
"""


TARGET_LUFS = -16.0
PEAK_CEILING = 0.95
SAME_SPEAKER_GAP = 0.25
SPEAKER_CHANGE_GAP = 0.40
KIND_CHANGE_GAP = 0.50

# Final deliverables default to MP3 64k: the most portable lossy format. Opus
# and AAC are available and smaller/better at equal bitrate if compatibility allows.
LOSSY_FORMATS = {
    "mp3": ("libmp3lame", "mp3"),
    "aac": ("aac", "m4a"),
    "opus": ("libopus", "opus"),
}


def encode_lossy(src: str | Path, dst: str | Path, *, fmt: str = "mp3", bitrate: str = "64k") -> Path:
    """Transcode a WAV to a small lossy file with ffmpeg; return the new path."""
    if fmt not in LOSSY_FORMATS:
        raise ValueError(f"unknown format: {fmt!r} (choose from {', '.join(LOSSY_FORMATS)})")
    codec, _ext = LOSSY_FORMATS[fmt]
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-c:a", codec, "-b:a", bitrate]
    if fmt == "opus":
        command += ["-vbr", "on", "-application", "audio"]
    command.append(str(dst))
    subprocess.run(command, check=True)
    return dst


def atempo(audio: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Pitch-preserving speed change via ffmpeg ``atempo`` (speed 1.0 = unchanged).

    Breeze has no duration knob, so this is the deterministic way to hit a target
    pace; use direction instructions for expressive pacing instead.
    """
    if abs(speed - 1.0) < 0.01 or audio.size == 0:
        return audio
    speed = min(4.0, max(0.25, float(speed)))
    filters = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-filter:a",
        ",".join(filters),
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(command, input=pcm, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target: float = TARGET_LUFS, peak: float = PEAK_CEILING
) -> np.ndarray:
    output = audio.astype(np.float64)
    try:
        import pyloudnorm as pyln

        measured = float(pyln.Meter(sample_rate).integrated_loudness(output))
        if np.isfinite(measured) and measured > -70.0:
            output *= 10.0 ** ((target - measured) / 20.0)
    except Exception:  # noqa: BLE001 - loudness is best-effort
        pass
    ceiling = float(np.max(np.abs(output))) if output.size else 0.0
    if ceiling > peak:
        output *= peak / ceiling
    return output.astype(np.float32)


def _gap_seconds(previous, current) -> float:
    if previous is None:
        return 0.0
    if previous.kind != current.kind:
        return KIND_CHANGE_GAP
    if previous.role_id != current.role_id:
        return SPEAKER_CHANGE_GAP
    return SAME_SPEAKER_GAP


def assemble_rows(
    items: list[tuple[Path, object]],
    out_path: str | Path,
    *,
    sample_rate: int = 24000,
    normalize: bool = True,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    """Concatenate ``(wav_path, row)`` in order with speaker-aware gaps."""
    pieces: list[np.ndarray] = []
    previous = None
    for path, row in items:
        audio, _ = load_mono(path)
        gap = _gap_seconds(previous, row)
        if gap > 0:
            pieces.append(np.zeros(int(sample_rate * gap), dtype=np.float32))
        pieces.append(audio)
        previous = row
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if normalize and merged.size:
        merged = normalize_loudness(merged, sample_rate, target_lufs)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), merged, sample_rate, subtype="PCM_16")
    return merged


def assemble_chapters(
    chapter_paths: list[tuple[int, Path]],
    out_path: str | Path,
    *,
    sample_rate: int = 24000,
    gap: float = 0.8,
    normalize: bool = True,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for index, (_, path) in enumerate(chapter_paths):
        audio, _ = load_mono(path)
        if index:
            pieces.append(np.zeros(int(sample_rate * gap), dtype=np.float32))
        pieces.append(audio)
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if normalize and merged.size:
        merged = normalize_loudness(merged, sample_rate, target_lufs)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), merged, sample_rate, subtype="PCM_16")
    return merged


"""Reading-time estimate for one segment (no sentence splitting).

The estimator reports a standard duration used for bookkeeping/stats only; the
Breeze renderer decides its own generation length.
"""


SECONDS_PER_CJK_CHAR = 0.22
SECONDS_PER_LATIN_WORD = 0.40
SECONDS_PER_STRONG_PAUSE = 0.30
SECONDS_PER_WEAK_PAUSE = 0.12

STANDARD_RATE = 0.7
CHAR_BOOST = 1.05
SHORT_SENTENCE_BOOST = 1.10
SHORT_SENTENCE_MAX_CHARS = 12

_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
_STRONG_PAUSE_RE = re.compile(r"[。！？!?…]+")
_WEAK_PAUSE_RE = re.compile(r"[，、；：,;:]+")


def estimate_text_duration(text: str | None) -> float:
    text = text or ""
    cjk = len(_CJK_CHAR_RE.findall(text))
    words = len(_LATIN_WORD_RE.findall(text))
    strong = len(_STRONG_PAUSE_RE.findall(text))
    weak = len(_WEAK_PAUSE_RE.findall(text))
    speech = (cjk * SECONDS_PER_CJK_CHAR + words * SECONDS_PER_LATIN_WORD) * CHAR_BOOST
    pauses = strong * SECONDS_PER_STRONG_PAUSE + weak * SECONDS_PER_WEAK_PAUSE
    duration = (speech + pauses) * STANDARD_RATE
    if cjk + words <= SHORT_SENTENCE_MAX_CHARS:
        duration *= SHORT_SENTENCE_BOOST
    return duration


"""Breeze TTS 2 inline vocal events (Chinese square-bracket tags).

The model's tag vocabulary is **free form**, not a fixed token list: the official
examples are ``[笑]`` ``[咳嗽]`` ``[清嗓子]`` ``[叹气]`` (English side: ``(laugh)``
``(cough)`` ``(clears throat)`` ``(sigh)``), and descriptive tags like
``[无奈的冷笑]`` generally work too. Events only fire under guidance (cfg 2~3).

This module is the single source of truth: the marking MCP server uses
:data:`VOCAL_EVENTS` for its tool enum, and the renderer uses :func:`find_tags`
to detect any tag and raise cfg. Because cleaning strips all source square
brackets first, any remaining ``[...]`` in the script is one of ours.
"""


# Curated set for consistency (grouped by category), not an exhaustive whitelist.
VOCAL_EVENTS: tuple[str, ...] = (
    # 笑
    "笑",
    "轻笑",
    "大笑",
    "冷笑",
    "苦笑",
    "嗤笑",
    "干笑",
    "哼笑",
    "坏笑",
    # 哭
    "哭",
    "抽泣",
    "哽咽",
    "啜泣",
    "呜咽",
    # 呼吸
    "叹气",
    "长叹",
    "喘气",
    "喘息",
    "深吸一口气",
    "呼出一口气",
    "倒吸一口气",
    "屏息",
    # 喉咙
    "咳嗽",
    "干咳",
    "清嗓子",
    "咳",
    # 口腔
    "咂嘴",
    "啧",
    "吞咽",
    "咽口水",
    "舔唇",
    "打哈欠",
    # 其他
    "冷哼",
    "闷哼",
    "呻吟",
    "惊呼",
    "打喷嚏",
    "鼻音",
)

TAG_RE = re.compile(r"\[([^\[\]]{1,8})\]")
_CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]{1,8}$")


def normalize_tag(tag: str | None) -> str:
    return (tag or "").strip().strip("[] ")


def is_event_tag(tag: str | None) -> bool:
    """Accept the curated set or any short Chinese tag (free-form vocabulary)."""
    cleaned = normalize_tag(tag)
    return bool(cleaned) and (cleaned in VOCAL_EVENTS or bool(_CJK_RE.match(cleaned)))


def find_tags(text: str | None) -> list[str]:
    """All inline tags in a line (empty when there are none)."""
    return TAG_RE.findall(text or "")


"""Breeze TTS 2 (C++/GGUF) HTTP renderer: ``ScriptRow`` -> per-row wav.

Alternative TTS backend that drives ``breeze-server`` over HTTP, which keeps the
GGUF model resident and exposes both the streaming synthesis API and the built-in
WebUI. Cloning needs the reference clip *and* its exact transcript, so per-role
reference text is resolved from ``voicebank_meta.json`` (or ASR as a fallback).

Cloning is fail-closed: a reference without a transcript would silently degrade to
voice design (see ``third_party/Breeze-TTS-2.cpp/src/generation.cpp``), so it
raises instead. Design is exposed separately via :meth:`BreezeRenderer.design_voice`
for the voice-manufacturing step.
"""


DEFAULT_SERVER_BIN = "build/breeze-cpp/breeze-server"
DEFAULT_MODEL = "ckpts/Breeze-TTS-2.cpp/breeze-tts-2-q8_0.gguf"
DEFAULT_URL = "http://127.0.0.1:8137"
DEFAULT_VOICES_DIR = "build/breeze-cpp/voices"
DEFAULT_LOG = "build/breeze-cpp/server.log"

EMOTION_ZH = {
    "happy": "开心",
    "angry": "愤怒",
    "sad": "悲伤",
    "fearful": "恐惧",
    "surprised": "惊讶",
    "disgusted": "厌恶",
    "calm": "平静",
    "excited": "兴奋",
}

# Inline vocal events only fire under guidance; raise cfg for tagged lines.
CFG_WITH_EVENTS = 2.5


@dataclass
class BreezeConfig:
    base_url: str = DEFAULT_URL
    model_path: str = DEFAULT_MODEL
    server_bin: str = DEFAULT_SERVER_BIN
    voices_dir: str = DEFAULT_VOICES_DIR
    server_log: str = DEFAULT_LOG
    host: str = "127.0.0.1"
    port: int = 8137
    webui: bool = True
    cfg_scale: float = 1.0
    seed: int = 1234
    timeout: float = 600.0
    direction: bool = False
    speed: float = 1.0
    target_lufs: float = TARGET_LUFS
    peak_ceiling: float = PEAK_CEILING
    normalize_rows: bool = True


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _health(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with _OPENER.open(f"{url.rstrip('/')}/health", timeout=timeout) as response:
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - not up yet
        return None
    if data.get("status") != "ok":
        return None
    return data


def start_server(
    config: BreezeConfig | None = None,
    *,
    log_path: str | Path | None = None,
    wait: float = 300.0,
    log=print,
) -> subprocess.Popen | None:
    """Start ``breeze-server`` unless one is already healthy.

    Returns the process handle if this call started it, else ``None`` (so a
    caller never kills a server it did not launch).
    """
    config = config or BreezeConfig()
    if _health(config.base_url) is not None:
        return None

    binary = Path(config.server_bin)
    model = Path(config.model_path)
    if not binary.is_file():
        raise FileNotFoundError(f"breeze-server not found: {binary} (build Breeze-TTS-2.cpp first)")
    if not model.is_file():
        raise FileNotFoundError(f"Breeze GGUF not found: {model}")

    log_path = Path(log_path or config.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(config.voices_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        str(model),
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--voices-dir",
        config.voices_dir,
    ]
    if config.webui:
        cmd.append("--webui")
    log(f"[breeze] starting server -> {log_path}")
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        deadline = time.time() + wait
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"breeze-server exited early (code {process.returncode}); see {log_path}")
            if _health(config.base_url) is not None:
                log(f"[breeze] up at {config.base_url}")
                return process
            time.sleep(1.0)
        raise RuntimeError(f"breeze-server did not become healthy in {wait:.0f}s; see {log_path}")
    except BaseException:
        stop_server(process)
        raise


def stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes]] | None = None) -> tuple[bytes, str]:
    boundary = f"----breeze{os.urandom(8).hex()}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")
    for name, (filename, payload) in (files or {}).items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
        chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
        chunks.append(payload)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def wav_bytes(path: str | Path) -> bytes:
    """Read any audio file and return mono PCM16 WAV bytes (what the server expects)."""
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    buffer = io.BytesIO()
    sf.write(buffer, mono, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _content_digest(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


class Asr:
    """Thin faster-whisper wrapper used to recover missing reference transcripts."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, path: str | Path) -> str:
        segments, _info = self._model.transcribe(str(path), language="zh", beam_size=1, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()


def voice_map_from_args(spec: list[str] | None) -> dict[str, str]:
    """Parse ``--voice ROLE=PATH`` overrides."""
    voices: dict[str, str] = {}
    for item in spec or []:
        if "=" not in item:
            continue
        role, path = item.split("=", 1)
        if role.strip() and Path(path).is_file():
            voices[role.strip()] = path.strip()
    return voices


class BreezeRenderer:
    def __init__(self, config: BreezeConfig | None = None, *, voice_meta: dict | None = None, log=print):
        self.config = config or BreezeConfig()
        self.voice_meta = voice_meta or {}
        self.log = log
        self._process: subprocess.Popen | None = None
        self._ref_texts: dict[str, str] = {}
        self._digests: dict[str, str] = {}
        self._asr = None
        self._sample_rate = 24000

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._process = start_server(self.config, log=self.log)
        health = _health(self.config.base_url) or {}
        self._sample_rate = int(health.get("sample_rate", 24000))

    def close(self) -> None:
        stop_server(self._process)
        self._process = None

    def __enter__(self) -> BreezeRenderer:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- voices ------------------------------------------------------------
    def voice_for(self, row: ScriptRow, voices: dict[str, str] | None = None) -> str:
        voices = voices or {}
        return row.voice_ref or voices.get(row.role_name) or voices.get(row.role_id) or ""

    def ref_text_for(self, ref_path: str, role_name: str = "", role_id: str = "") -> str:
        if not ref_path:
            return ""
        for key in (role_name, role_id):
            meta = self.voice_meta.get(key) if key else None
            text = (meta or {}).get("ref_text", "") if isinstance(meta, dict) else ""
            if text.strip():
                return text.strip()
        if ref_path in self._ref_texts:
            return self._ref_texts[ref_path]
        try:
            text = self._transcribe(ref_path)
        except Exception as error:  # noqa: BLE001 - fail closed below
            raise RuntimeError(f"no transcript for reference {ref_path} and ASR failed: {error}") from error
        if not text.strip():
            raise RuntimeError(f"no transcript for reference {ref_path}; add it to voicebank_meta.json")
        self._ref_texts[ref_path] = text
        return text

    def _transcribe(self, ref_path: str) -> str:
        if self._asr is None:
            self._asr = Asr(
                model=os.environ.get("AUK_ASR_MODEL", "small"),
                device=os.environ.get("AUK_ASR_DEVICE", "cpu"),
            )
        text = self._asr.transcribe(ref_path)
        self.log(f"[breeze] ASR {Path(ref_path).name} -> {text[:40]!r}")
        return text

    # -- synthesis ---------------------------------------------------------
    def instruction_for(self, row: ScriptRow, role_style: str = "") -> str:
        parts: list[str] = []
        style = row.style_desc or role_style
        if style:
            parts.append(style)
        if row.emotion:
            label = EMOTION_ZH.get(row.emotion, row.emotion)
            parts.append(f"情绪{label}")
        return "，".join(parts)

    def _post(self, fields: dict[str, str], files: dict[str, tuple[str, bytes]] | None, context: str):
        body, content_type = _multipart(fields, files)
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/v1/audio/speech",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        try:
            with _OPENER.open(request, timeout=self.config.timeout) as response:
                rate = int(response.headers.get("X-Sample-Rate", self._sample_rate))
                payload = response.read()
        except Exception as error:  # noqa: BLE001 - add request context
            raise RuntimeError(f"breeze request failed ({context}): {error}") from error
        if len(payload) < 2 or len(payload) % 2:
            raise RuntimeError(f"breeze returned invalid PCM ({context}): {len(payload)} bytes")
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0, rate

    def design_voice(self, text: str, instruction: str, *, seed: int | None = None):
        """Synthesize a fresh voice from a description (no reference audio)."""
        if not instruction.strip():
            raise ValueError("voice design requires an instruction")
        fields = {
            "text": text,
            "instruction": instruction,
            "cfg_scale": str(self.config.cfg_scale),
            "seed": str(self.config.seed if seed is None else seed),
        }
        return self._post(fields, None, "design")

    def synth(self, row: ScriptRow, voice_ref: str, ref_text: str = "", role_style: str = ""):
        cfg = self.cfg_for(row.tts_text)
        fields: dict[str, str] = {
            "text": row.tts_text,
            "cfg_scale": str(cfg),
            "seed": str(self.config.seed),
        }
        if not voice_ref:
            raise RuntimeError(f"no reference voice for {row.role_name or row.role_id!r} (build a voicebank first)")
        if not ref_text.strip():
            raise RuntimeError(f"clone without transcript for {voice_ref}; refusing to fall back to design")
        fields["ref_text"] = ref_text
        if self.config.direction:
            instruction = self.instruction_for(row, role_style)
            if instruction:
                fields["instruction"] = instruction
        files = {"ref_audio": ("ref.wav", wav_bytes(voice_ref))}
        samples, rate = self._post(fields, files, row.seg_id)
        speed = self.speed_for(row)
        if abs(speed - 1.0) > 0.01:
            samples = atempo(samples, rate, speed)
        return samples, rate

    def cfg_for(self, text: str) -> float:
        """Vocal events only fire under guidance, so raise cfg for tagged lines."""
        if find_tags(text):
            return max(self.config.cfg_scale, CFG_WITH_EVENTS)
        return self.config.cfg_scale

    def speed_for(self, row: ScriptRow) -> float:
        return row.speed if row.speed and abs(row.speed - 1.0) > 0.01 else self.config.speed

    def _key(self, row: ScriptRow, voice_ref: str, ref_text: str) -> str:
        digest = ""
        if voice_ref:
            if voice_ref not in self._digests:
                self._digests[voice_ref] = _content_digest(voice_ref)
            digest = self._digests[voice_ref]
        instruction = self.instruction_for(row) if self.config.direction else ""
        payload = "|".join(
            [
                "breeze",
                Path(self.config.model_path).name,
                row.tts_text,
                digest,
                ref_text,
                instruction,
                f"{self.cfg_for(row.tts_text):.2f}",
                f"{self.speed_for(row):.3f}",
                str(self.config.seed),
                str(self._sample_rate),
                f"{self.config.target_lufs:.1f}" if self.config.normalize_rows else "raw",
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def render_rows(
        self,
        rows: list[ScriptRow],
        out_dir: str | Path,
        *,
        voices: dict[str, str] | None = None,
        limit: int | None = None,
    ) -> dict[int, Path]:
        row_dir = Path(out_dir) / "rows"
        row_dir.mkdir(parents=True, exist_ok=True)
        rendered: dict[int, Path] = {}
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            voice = self.voice_for(row, voices)
            ref_text = self.ref_text_for(voice, row.role_name, row.role_id)
            key = self._key(row, voice, ref_text)
            path = row_dir / f"{row.seg_id}__{key}.wav"
            if path.exists() and path.stat().st_size > 0:
                rendered[row.order] = path
                continue
            array, sample_rate = self.synth(row, voice, ref_text)
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite audio for {row.seg_id}")
            if self.config.normalize_rows:
                array = normalize_loudness(array, sample_rate, self.config.target_lufs, self.config.peak_ceiling)
            temp = path.with_suffix(".tmp.wav")
            sf.write(str(temp), np.clip(array, -1.0, 1.0), sample_rate, subtype="FLOAT")
            os.replace(temp, path)  # atomic: a crash never leaves a half-written cache hit
            rendered[row.order] = path
        return rendered

    def normalize_track(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self.config.normalize_rows:
            return audio
        return normalize_loudness(audio, sample_rate, self.config.target_lufs, self.config.peak_ceiling)


__all__ = [
    "Asr",
    "BreezeConfig",
    "BreezeRenderer",
    "start_server",
    "stop_server",
    "voice_map_from_args",
    "wav_bytes",
]
