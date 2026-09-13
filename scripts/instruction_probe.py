#!/usr/bin/env python3
"""A/B probe: does AuK obey free-form additions to the zero-shot instruction?

PE never emits these (the canonical clone template is fixed), so this probes the
model itself: same reference audio, same text, same seed and gen_seconds, only
the instruction string changes. Each instruction may contain ``{text}``.

Example:
    python scripts/instruction_probe.py \
        --ref outputs/book20_out/refs/混血精灵少女.wav \
        --text '别……先别杀我啊！比起这个你们老祖宗的棺材板要压不住了啊！' \
        --out outputs/emotion_probe --name panic
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
from audiobook.duration import estimate_text_duration  # noqa: E402
from audiobook.renderer import RenderConfig, Renderer  # noqa: E402
from audiobook.schema import ScriptRow  # noqa: E402

DEFAULT_INSTRUCTIONS = [
    ("baseline", 'Say the following with the same voice: "{text}"'),
    (
        "fearful",
        'Say the following with the same voice. The speaker is terrified, on the verge of tears, pleading desperately: "{text}"',
    ),
    ("angry", 'Say the following with the same voice, shouting in furious anger: "{text}"'),
    ("excited", 'Say the following with the same voice, in a bright, excited and eager tone: "{text}"'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe instruction control on AuK")
    parser.add_argument("--ref", required=True, help="reference audio (voice to clone)")
    parser.add_argument("--text", required=True, help="target text")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--name", default="probe", help="file prefix")
    parser.add_argument("--instruction", action="append", default=None, help="LABEL=INSTRUCTION (repeatable, {text} allowed)")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seconds", type=float, default=0.0, help="fix gen_seconds (0 = estimate from text)")
    parser.add_argument("--duration-rate", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=20.0, help="renderer clamp for gen_seconds")
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS, help="loudness target for all clips (fair A/B)")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def build_instructions(spec: list[str] | None) -> list[tuple[str, str, float]]:
    """Parse ``LABEL[@mult]=INSTRUCTION``; a literal ``-`` instruction passes the source through."""
    if not spec:
        return [(label, instruction, 1.0) for label, instruction in DEFAULT_INSTRUCTIONS]
    items = []
    for raw in spec:
        if "=" in raw:
            label, instruction = raw.split("=", 1)
        else:
            label, instruction = f"v{len(items) + 1}", raw
        mult = 1.0
        label = label.strip()
        if "@" in label:
            label, _, mult_text = label.rpartition("@")
            mult = float(mult_text)
        items.append((label, instruction, mult))
    return items


def main() -> None:
    args = parse_args()
    seconds = args.seconds or estimate_text_duration(args.text)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from app.patches import apply_patches

    apply_patches()

    config = RenderConfig(variant=args.variant, seed=args.seed, duration_rate=args.duration_rate, max_seconds=args.max_seconds)
    renderer = Renderer(config)
    print(f"[probe] ref={args.ref} seconds={seconds:.2f} -> {out_dir}", flush=True)

    instructions = build_instructions(args.instruction)
    clips: list[tuple[str, str, np.ndarray, int]] = []
    for index, (label, instruction, mult) in enumerate(instructions, start=1):
        rendered = instruction.format(text=args.text)
        if rendered.strip() == "-":
            array, sample_rate = sf.read(args.ref, dtype="float32")
            if array.ndim > 1:
                array = array.mean(axis=1)
        else:
            row = ScriptRow(
                seg_id=f"{args.name}_{index:02d}_{label}",
                kind="dialogue",
                tts_text=args.text,
                pe_instruction=rendered,
                target_duration_s=seconds * mult,
                auk_task="zero_shot_tts",
            )
            audio, sample_rate = renderer.synth(row, args.ref)
            array = audio.detach().to("cpu", dtype=None).float().squeeze(0).numpy()
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite audio for {label}")
        if not args.no_normalize:
            array = normalize_loudness(np.asarray(array, dtype=np.float32), sample_rate, args.target_lufs)
        path = out_dir / f"{index:02d}_{label}.wav"
        sf.write(str(path), np.clip(array, -1.0, 1.0), sample_rate, subtype="PCM_16")
        clips.append((label, rendered, array, sample_rate))
        print(f"[probe] {index}/{len(instructions)} {label}: {len(array) / sample_rate:.2f}s -> {path}", flush=True)

    sample_rate = clips[0][3]
    gap = np.zeros(int(0.7 * sample_rate), dtype=np.float32)
    combined = []
    for _, _, array, _ in clips:
        combined.extend([array, gap])
    combined = np.concatenate(combined[:-1]) if combined else np.zeros(0, dtype=np.float32)
    compare = out_dir / f"{args.name}_compare.wav"
    sf.write(str(compare), np.clip(combined, -1.0, 1.0), sample_rate, subtype="PCM_16")

    manifest = {
        "ref": args.ref,
        "text": args.text,
        "seconds": round(seconds, 3),
        "seed": args.seed,
        "variant": args.variant,
        "compare": str(compare),
        "items": [{"label": label, "instruction": instruction} for label, instruction, _, _ in clips],
    }
    (out_dir / f"{args.name}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] compare -> {compare}", flush=True)


if __name__ == "__main__":
    main()
