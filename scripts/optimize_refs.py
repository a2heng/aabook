#!/usr/bin/env python3
"""One-shot reference-audio optimization for AuK cloning (bandwidth extension).

This is a *preprocessing* step, run once per voice. It emits a normal 24 kHz mono
wav, so the downstream generator is unchanged -- it just receives a better
reference. Deliberately decoupled from the text pipeline: the text prompt is
preprocessed by the LLM, the reference wav by AuK; the two never interact.

Method: AuK ``improve_quality / bandwidth_extension`` (the only reference-editing
task that survives testing; denoise barely helps and AuK dereverb is broken --
see docs/audiobook-workflow.md 6.3).

Examples:
    python scripts/optimize_refs.py --in "assets/voice-reference/逗哥音色整理合集/角色扮演" \\
        --out outputs/refs_bwe --voices outputs/refs_bwe/voices.json
    python scripts/optimize_refs.py --map "高文=path/a.wav" --map "旁白=path/b.wav" \\
        --out outputs/refs_bwe --voices outputs/refs_bwe/voices.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
for _path in (APP_ROOT / "vendor", APP_ROOT / "third_party" / "AuK" / "src", APP_ROOT, APP_ROOT / "scripts"):
    sys.path.insert(0, str(_path))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(APP_ROOT)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audiobook.renderer import RenderConfig, Renderer  # noqa: E402
from audiobook.schema import ScriptRow  # noqa: E402
from prepare_refs import prepare  # noqa: E402

BWE_INSTRUCTION = "请对这段语音做超分辨率/带宽扩展处理，恢复被削掉的高频成分，输出宽带纯净人声。"
TARGET_SAMPLE_RATE = 24000
MAX_SECONDS = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize reference audio once with AuK bandwidth extension")
    parser.add_argument("--in", dest="src_dir", default=None, help="directory of reference wavs")
    parser.add_argument("--pattern", default="*.wav")
    parser.add_argument("--map", action="append", default=None, help="ROLE=PATH (repeatable)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--voices", default=None, help="write voices.json (name -> optimized wav)")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--no-vad", action="store_true", help="disable VAD silence removal before bwe")
    return parser.parse_args()


def collect(args: argparse.Namespace) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    if args.src_dir:
        for path in sorted(Path(args.src_dir).glob(args.pattern)):
            items.append((path.stem, path))
    for raw in args.map or []:
        if "=" not in raw:
            continue
        name, source = raw.split("=", 1)
        path = Path(source.strip())
        if path.is_file():
            items.append((name.strip(), path))
    return items


def main() -> None:
    args = parse_args()
    items = collect(args)
    if not items:
        print("[error] nothing to process (use --in or --map)", file=sys.stderr)
        raise SystemExit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.patches import apply_patches

    apply_patches()
    renderer = Renderer(RenderConfig(variant=args.variant, seed=args.seed))

    voices: dict[str, str] = {}
    for index, (name, source) in enumerate(items, start=1):
        prepared, prep_sr = prepare(source, args.max_seconds, args.sample_rate, vad=not args.no_vad)
        staging = out_dir / f".{name}.prep.wav"
        sf.write(str(staging), prepared, prep_sr, subtype="FLOAT")
        seconds = len(prepared) / prep_sr
        row = ScriptRow(
            seg_id=name,
            tts_text="",
            pe_instruction=BWE_INSTRUCTION,
            target_duration_s=seconds,
            auk_task="improve_quality",
        )
        audio, sample_rate = renderer.synth(row, str(staging))
        array = audio.detach().to("cpu", dtype=None).float().squeeze(0).numpy()
        if not np.isfinite(array).all():
            print(f"[skip] {name}: non-finite output", file=sys.stderr)
            staging.unlink(missing_ok=True)
            continue
        destination = out_dir / f"{name}.wav"
        temp = destination.with_suffix(".tmp.wav")
        sf.write(str(temp), np.clip(array, -1.0, 1.0), sample_rate, subtype="FLOAT")
        os.replace(temp, destination)
        cleaned, sr = prepare(destination, args.max_seconds, args.sample_rate, vad=False)
        sf.write(str(destination), cleaned, sr, subtype="FLOAT")
        staging.unlink(missing_ok=True)
        voices[name] = str(destination)
        print(f"[ok] {index}/{len(items)} {name:<16} {seconds:4.1f}s -> {len(cleaned) / sr:4.1f}s", flush=True)

    if args.voices:
        Path(args.voices).parent.mkdir(parents=True, exist_ok=True)
        Path(args.voices).write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.voices}: {len(voices)} voices")


if __name__ == "__main__":
    main()
