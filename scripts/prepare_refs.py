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


def prepare(src: Path, max_seconds: float, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
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
        from scipy.signal import resample_poly

        from math import gcd

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
        audio, sr = prepare(src, args.max_seconds, args.sample_rate)
        destination = out / f"{role.strip()}.wav"
        temp = destination.with_suffix(".tmp.wav")
        sf.write(str(temp), audio, sr, subtype="FLOAT")
        os.replace(temp, destination)
        voices[role.strip()] = str(destination)
        print(f"[ok] {role.strip():<16} {len(audio)/sr:4.1f}s @ {sr}  <- {src.name}")
    Path(args.voices).write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.voices}: {len(voices)} roles")


if __name__ == "__main__":
    main()
