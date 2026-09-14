#!/usr/bin/env python3
"""Prepare per-role reference audio for AuK cloning.

Trims leading/trailing silence, caps length, converts to mono at the model's
sample rate (24 kHz), and writes ``voices.json`` (role -> wav). Reference audio
is prepended to the target latents, so shorter is cheaper/steadier; 5-12 s is a
good range.

Examples:
    python scripts/prepare_refs.py --out outputs/book/refs --voices outputs/book/voices.json \\
        --map "旁白=assets/voice-reference/…/白马留声.wav" --map "高文=…/中年男声45.wav"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

TARGET_SAMPLE_RATE = 24000
MAX_SECONDS = 12.0
SILENCE_THRESHOLD = 0.01
FADE_SECONDS = 0.02
VAD_SAMPLE_RATE = 16000
VAD_MIN_GAP = 0.03
VAD_FADE = 0.005


def vad_speech_spans(mono: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """Silero VAD speech spans (in samples at ``sr``); VAD runs at 16 kHz."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    target = VAD_SAMPLE_RATE
    if sr != target:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(sr, target)
        audio = resample_poly(mono.astype(np.float32), target // divisor, sr // divisor)
        scale = sr / target
    else:
        audio = mono.astype(np.float32)
        scale = 1.0
    stamps = get_speech_timestamps(audio, VadOptions(), sampling_rate=target)
    return [(int(stamp["start"] * scale), int(stamp["end"] * scale)) for stamp in stamps]


def remove_silence(mono: np.ndarray, sr: int, spans: list[tuple[int, int]]) -> np.ndarray:
    """Concatenate VAD speech spans, dropping internal silence (light fade + tiny gap)."""
    if not spans:
        return mono
    fade = int(VAD_FADE * sr)
    gap = np.zeros(int(VAD_MIN_GAP * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for start, end in spans:
        start, end = max(0, start), min(len(mono), end)
        if end - start <= 0:
            continue
        segment = mono[start:end].astype(np.float32).copy()
        edge = min(fade, len(segment) // 2)
        if edge > 0:
            segment[:edge] *= np.linspace(0.0, 1.0, edge, dtype=np.float32)
            segment[-edge:] *= np.linspace(1.0, 0.0, edge, dtype=np.float32)
        if parts and gap.size:
            parts.append(gap)
        parts.append(segment)
    return np.concatenate(parts) if parts else mono


def prepare(src: Path, max_seconds: float, target_sr: int, *, vad: bool = True) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if vad and mono.size:
        mono = remove_silence(mono, sr, vad_speech_spans(mono, sr))
    if mono.size:
        loud = np.where(np.abs(mono) > SILENCE_THRESHOLD)[0]
        if loud.size:
            mono = mono[loud[0] : loud[-1] + 1]
    if len(mono) > max_seconds * sr:
        mono = mono[: int(max_seconds * sr)]
    fade = min(len(mono), int(FADE_SECONDS * sr))
    if fade:
        mono[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    if sr != target_sr and mono.size:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // divisor, sr // divisor).astype(np.float32)
        sr = target_sr
    return mono, sr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 24 kHz mono reference audio")
    parser.add_argument("--out", required=True, help="directory for prepared reference wavs")
    parser.add_argument("--voices", required=True, help="path to write voices.json")
    parser.add_argument("--map", action="append", required=True, help="ROLE=PATH (repeatable)")
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--no-vad", action="store_true", help="disable VAD silence removal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voices: dict[str, str] = {}
    for item in args.map:
        if "=" not in item:
            continue
        role, source = item.split("=", 1)
        src = Path(source.strip())
        if not src.is_file():
            print(f"[skip] missing: {src}")
            continue
        audio, sr = prepare(src, args.max_seconds, args.sample_rate, vad=not args.no_vad)
        destination = out / f"{role.strip()}.wav"
        temp = destination.with_suffix(".tmp.wav")
        sf.write(str(temp), audio, sr, subtype="FLOAT")
        os.replace(temp, destination)
        voices[role.strip()] = str(destination)
        print(f"[ok] {role.strip():<16} {len(audio) / sr:4.1f}s @ {sr}  <- {src.name}")
    Path(args.voices).write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.voices}: {len(voices)} roles")


if __name__ == "__main__":
    main()
