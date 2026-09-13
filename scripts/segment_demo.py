#!/usr/bin/env python3
"""Compare AuK prosody on a long passage: no split vs split at several lengths.

Long single generations degrade in intonation. This renders the same passage
under different ``segment_tts_text`` caps (each segment gets its own
``gen_seconds``), concatenates the segments, and lets you listen side by side.

Example:
    python scripts/segment_demo.py --ref outputs/refs_bwe/俏皮公主.wav \\
        --text '…long passage…' --out outputs/segment_demo \\
        --max-seconds 20 --max-seconds 10 --max-seconds 6 --max-seconds 4 --duration-rate 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parent.parent
for _path in (APP_ROOT / "vendor", APP_ROOT / "third_party" / "AuK" / "src", APP_ROOT):
    sys.path.insert(0, str(_path))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(APP_ROOT)

from audiobook.assembler import TARGET_LUFS, normalize_loudness  # noqa: E402
from audiobook.duration import estimate_text_duration, segment_tts_text  # noqa: E402
from audiobook.renderer import RenderConfig, Renderer  # noqa: E402
from audiobook.schema import ScriptRow  # noqa: E402

CLONE_TEMPLATE = 'Say the following with the same voice: "{text}"'
SEGMENT_GAP = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-passage segmentation A/B for AuK")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="segment")
    parser.add_argument("--max-seconds", type=float, action="append", required=True)
    parser.add_argument("--duration-rate", type=float, default=1.0)
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    return parser.parse_args()


def render_segment(renderer: Renderer, text: str, ref: str, rate: float) -> tuple[np.ndarray, int]:
    row = ScriptRow(
        tts_text=text,
        pe_instruction=CLONE_TEMPLATE.format(text=text),
        target_duration_s=0.0,
        auk_task="zero_shot_tts",
    )
    audio, sample_rate = renderer.synth(row, ref)
    array = audio.detach().to("cpu", dtype=None).float().squeeze(0).numpy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite audio for segment: {text[:20]}")
    return array, sample_rate


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.patches import apply_patches

    apply_patches()
    renderer = Renderer(RenderConfig(variant=args.variant, seed=args.seed, duration_rate=args.duration_rate))

    results: list[tuple[str, np.ndarray, int]] = []
    compare_pieces: list[np.ndarray] = []
    manifest: dict = {"text": args.text, "duration_rate": args.duration_rate, "ref": args.ref, "settings": []}

    for max_seconds in args.max_seconds:
        segments = segment_tts_text(args.text, max_seconds)
        pieces: list[np.ndarray] = []
        sample_rate = 24000
        gap = np.zeros(int(SEGMENT_GAP * sample_rate), dtype=np.float32)
        for index, segment in enumerate(segments):
            array, sample_rate = render_segment(renderer, segment.text, args.ref, args.duration_rate)
            if index:
                pieces.append(gap)
            pieces.append(array)
        combined = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
        combined = normalize_loudness(combined, sample_rate, args.target_lufs)
        label = f"max{max_seconds:g}"
        path = out_dir / f"{args.name}_{label}.wav"
        sf.write(str(path), np.clip(combined, -1.0, 1.0), sample_rate, subtype="PCM_16")
        results.append((label, combined, sample_rate))
        if compare_pieces:
            compare_pieces.append(np.zeros(int(0.7 * sample_rate), dtype=np.float32))
        compare_pieces.append(combined)
        manifest["settings"].append(
            {
                "max_seconds": max_seconds,
                "segments": [segment.text for segment in segments],
                "seconds": round(len(combined) / sample_rate, 2),
                "path": str(path),
            }
        )
        print(
            f"[ok] {label}: {len(segments)} segs, total {len(combined) / sample_rate:.2f}s "
            f"(est {estimate_text_duration(args.text):.1f}s) -> {path}",
            flush=True,
        )

    if results:
        sample_rate = results[0][2]
        combined = np.concatenate(compare_pieces)
        compare = out_dir / f"{args.name}_compare.wav"
        sf.write(str(compare), np.clip(combined, -1.0, 1.0), sample_rate, subtype="PCM_16")
        manifest["compare"] = str(compare)
        manifest["order"] = [label for label, _, _ in results]
    (out_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] compare -> {out_dir / f'{args.name}_compare.wav'}")


if __name__ == "__main__":
    main()
