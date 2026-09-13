#!/usr/bin/env python3
"""Render one line across several (bwe-optimized) reference voices for listening.

Same text, same seed, same (tightened) gen_seconds; only the reference voice
changes. Clips are loudness-normalized so the comparison is about timbre/prosody,
not level.

Example:
    python scripts/voice_demo.py --voices-json outputs/refs_bwe/voices.json \\
        --voice 俏皮公主 --voice 傲娇女王 --voice 冰山女王 --voice 青春男大 \\
        --text '别……先别杀我啊！比起这个你们老祖宗的棺材板要压不住了啊！' \\
        --out outputs/voice_demo --duration-rate 0.9
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
from audiobook.renderer import RenderConfig, Renderer  # noqa: E402
from audiobook.schema import ScriptRow  # noqa: E402

CLONE_TEMPLATE = 'Say the following with the same voice: "{text}"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one line across several reference voices")
    parser.add_argument("--text", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="voice_demo")
    parser.add_argument("--voice", action="append", required=True, help="voice name (repeatable)")
    parser.add_argument("--voices-json", default=None, help="mapping name -> reference wav")
    parser.add_argument("--ref", action="append", default=None, help="NAME=REF override (repeatable)")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--duration-rate", type=float, default=1.0)
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS)
    return parser.parse_args()


def resolve_refs(args: argparse.Namespace) -> dict[str, str]:
    refs: dict[str, str] = {}
    if args.voices_json:
        refs.update(json.loads(Path(args.voices_json).read_text(encoding="utf-8")))
    for item in args.ref or []:
        if "=" in item:
            name, path = item.split("=", 1)
            refs[name.strip()] = path.strip()
    return refs


def main() -> None:
    args = parse_args()
    refs = resolve_refs(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.patches import apply_patches

    apply_patches()
    renderer = Renderer(RenderConfig(variant=args.variant, seed=args.seed, duration_rate=args.duration_rate))

    clips: list[tuple[str, np.ndarray, int]] = []
    for index, voice in enumerate(args.voice, start=1):
        ref = refs.get(voice)
        if not ref or not Path(ref).is_file():
            print(f"[skip] {voice}: no reference", file=sys.stderr)
            continue
        row = ScriptRow(
            seg_id=f"{args.name}_{index:02d}_{voice}",
            kind="dialogue",
            tts_text=args.text,
            pe_instruction=CLONE_TEMPLATE.format(text=args.text),
            target_duration_s=0,  # use the estimator, then duration_rate
            auk_task="zero_shot_tts",
        )
        audio, sample_rate = renderer.synth(row, ref)
        array = audio.detach().to("cpu", dtype=None).float().squeeze(0).numpy()
        if not np.isfinite(array).all():
            print(f"[skip] {voice}: non-finite", file=sys.stderr)
            continue
        array = normalize_loudness(array, sample_rate, args.target_lufs)
        path = out_dir / f"{index:02d}_{voice}.wav"
        sf.write(str(path), np.clip(array, -1.0, 1.0), sample_rate, subtype="PCM_16")
        clips.append((voice, array, sample_rate))
        print(f"[ok] {index}/{len(args.voice)} {voice:<10} {len(array) / sample_rate:.2f}s -> {path}", flush=True)

    if not clips:
        raise SystemExit("nothing rendered")
    sample_rate = clips[0][2]
    gap = np.zeros(int(0.7 * sample_rate), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for _, array, _ in clips:
        pieces.extend([array, gap])
    combined = np.concatenate(pieces[:-1]) if pieces else np.zeros(0, dtype=np.float32)
    compare = out_dir / f"{args.name}_compare.wav"
    sf.write(str(compare), np.clip(combined, -1.0, 1.0), sample_rate, subtype="PCM_16")
    manifest = {
        "text": args.text,
        "duration_rate": args.duration_rate,
        "seed": args.seed,
        "target_lufs": args.target_lufs,
        "compare": str(compare),
        "order": [voice for voice, _, _ in clips],
    }
    (out_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[demo] compare -> {compare}")


if __name__ == "__main__":
    main()
