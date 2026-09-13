"""AuK-backed renderer: ``ScriptRow`` -> per-segment wav (resumable, cached).

Torch/AuK are imported lazily so the front-end (``audiobook.pipeline``) never
needs a GPU. Rows are rendered independently so a long book can be resumed and
parallelised later.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .assembler import PEAK_CEILING, TARGET_LUFS, normalize_loudness
from .duration import estimate_text_duration
from .instructions import render_instruction
from .schema import ScriptRow

DEFAULT_FLASH_CKPT = "ckpts/AuK-Flash/auk_flash.safetensors"
DEFAULT_FLASH_CONFIG = "ckpts/AuK-Flash/config.yaml"
DEFAULT_BASE_CKPT = "ckpts/AuK/auk_base.safetensors"
DEFAULT_BASE_CONFIG = "ckpts/AuK/config.yaml"
DEFAULT_VOICE_REF = "assets/voice-reference/不同情绪音色/男-中音，平静，柔和.wav"


@dataclass
class RenderConfig:
    variant: str = "flash"
    ckpt_path: str = ""
    config_path: str = ""
    device: str | None = None
    dtype: str = "bf16"
    seed: int = 1234
    duration_rate: float = 1.0
    min_seconds: float = 0.6
    max_seconds: float = 28.0
    nfe: int = 4
    cfg: float = 0.0
    target_lufs: float = TARGET_LUFS
    peak_ceiling: float = PEAK_CEILING
    normalize_rows: bool = True
    default_voice_ref: str = DEFAULT_VOICE_REF

    def resolved_paths(self) -> tuple[str, str]:
        if self.variant == "base":
            return self.ckpt_path or DEFAULT_BASE_CKPT, self.config_path or DEFAULT_BASE_CONFIG
        return self.ckpt_path or DEFAULT_FLASH_CKPT, self.config_path or DEFAULT_FLASH_CONFIG

    def seconds_for(self, row: ScriptRow) -> float:
        base = row.target_duration_s or estimate_text_duration(row.tts_text)
        return min(self.max_seconds, max(self.min_seconds, base * self.duration_rate))


def _row_key(row: ScriptRow, voice_ref: str, seconds: float, config: RenderConfig, sample_rate: int) -> str:
    payload = "|".join(
        [
            row.pe_instruction or row.tts_text,
            voice_ref or "<none>",
            f"{seconds:.3f}",
            str(config.seed),
            config.variant,
            str(sample_rate),
            f"{config.target_lufs:.1f}" if config.normalize_rows else "raw",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


class Renderer:
    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self._engine = None
        self._sample_rate = 24000

    def load(self):
        from auk.infer.infer_auk import AukInfer

        ckpt, config = self.config.resolved_paths()
        self._engine = AukInfer(
            config_path=config,
            ckpt_path=ckpt,
            device=self.config.device,
            dtype=self.config.dtype,
        )
        self._sample_rate = int(getattr(self._engine, "target_sample_rate", 24000))
        return self._engine

    @property
    def engine(self):
        return self._engine or self.load()

    def synth(self, row: ScriptRow, voice_ref: str):
        instruction = row.pe_instruction or render_instruction(row.auk_task, row.tts_text)
        seconds = self.config.seconds_for(row)
        content = [{"type": "text", "text": instruction}]
        audio = voice_ref or None
        if audio:
            content.append({"type": "audio", "audio": audio})
        messages = [{"role": "user", "content": content}]
        return self.engine.generate(
            messages,
            audio=audio,
            gen_seconds=seconds,
            nfe=self.config.nfe,
            cfg_strength=self.config.cfg,
            seed=self.config.seed,
        )

    def voice_for(self, row: ScriptRow, voices: dict[str, str] | None = None) -> str:
        voices = voices or {}
        return row.voice_ref or voices.get(row.role_name) or voices.get(row.role_id) or self.config.default_voice_ref

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
            seconds = self.config.seconds_for(row)
            key = _row_key(row, voice, seconds, self.config, self._sample_rate)
            path = row_dir / f"{row.seg_id}__{key}.wav"
            if path.exists() and path.stat().st_size > 0:
                rendered[row.order] = path
                continue
            audio, sample_rate = self.synth(row, voice)
            array = audio.detach().to("cpu", dtype=None).float().squeeze(0).numpy()
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
        """Apply the configured loudness target to an assembled track."""
        if not self.config.normalize_rows:
            return audio
        return normalize_loudness(audio, sample_rate, self.config.target_lufs, self.config.peak_ceiling)


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


def estimate_total_seconds(rows: list[ScriptRow], config: RenderConfig | None = None) -> float:
    config = config or RenderConfig()
    return sum(config.seconds_for(row) for row in rows)


def ceil_segments(seconds: float, segment_seconds: float = 20.0) -> int:
    return max(1, int(math.ceil(seconds / segment_seconds)))


__all__ = [
    "DEFAULT_VOICE_REF",
    "RenderConfig",
    "Renderer",
    "estimate_total_seconds",
    "voice_map_from_args",
]
