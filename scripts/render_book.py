#!/usr/bin/env python3
"""Render a built script.csv into audio (per-row wavs -> per-chapter -> book).

Rendering is resumable: existing per-row wavs (matching the same hash) are reused.

Examples:
    python scripts/render_book.py --script outputs/mybook/script.csv --out outputs/mybook/render
    python scripts/render_book.py --script outputs/mybook/script.csv --out outputs/mybook/render \\
        --voices outputs/mybook/voices.json --limit-chapters 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
for _path in (APP_ROOT / "vendor", APP_ROOT / "third_party" / "AuK" / "src", APP_ROOT):
    sys.path.insert(0, str(_path))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(APP_ROOT)

from audiobook.assembler import TARGET_LUFS, assemble_chapters, assemble_rows  # noqa: E402
from audiobook.canonical import canonicalize_rows  # noqa: E402
from audiobook.renderer import RenderConfig, Renderer, voice_map_from_args  # noqa: E402
from audiobook.schema import Cast, read_script  # noqa: E402

VOICE_POOL = [
    "assets/voice-reference/不同情绪音色/男-温暖、智勇双全、正直.wav",
    "assets/voice-reference/不同情绪音色/女-明亮、坚定自信、师姐.wav",
    "assets/voice-reference/不同情绪音色/男-骄纵傲气，小正经.wav",
    "assets/voice-reference/不同情绪音色/女-温柔、姐姐.wav",
    "assets/voice-reference/不同情绪音色/灵犀-女-俏皮，活泼.wav",
    "assets/voice-reference/不同情绪音色/男-中音，慢速，柔和.wav",
    "assets/voice-reference/不同情绪音色/女-冷艳、师姐、妩媚.wav",
    "assets/voice-reference/不同情绪音色/男-傲慢、狂妄.wav",
]
NARRATOR_VOICE = "assets/voice-reference/不同情绪音色/男-中音，平静，柔和.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a script.csv with AuK")
    parser.add_argument("--script", required=True, help="path to script.csv")
    parser.add_argument("--out", required=True, help="output directory for audio")
    parser.add_argument("--cast", default=None, help="cast.json (enables role canonicalization)")
    parser.add_argument("--voices", default=None, help="JSON mapping role name/alias -> reference wav")
    parser.add_argument("--voice", action="append", default=None, help="role=path override (repeatable)")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--duration-rate", type=float, default=1.0, help="extra scale on standard duration (1.0 = as-is)")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="clamp on gen_seconds (tightened; long rows degraded)")
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS, help="loudness target for rows and masters")
    parser.add_argument("--no-normalize", action="store_true", help="write raw AuK levels (no loudness normalization)")
    parser.add_argument("--limit-chapters", type=int, default=0, help="only the first N chapters")
    parser.add_argument("--limit-rows", type=int, default=0, help="only the first N rows (smoke test)")
    parser.add_argument("--no-assemble", action="store_true")
    return parser.parse_args()


def build_voice_map(rows, explicit: dict[str, str]) -> dict[str, str]:
    voices = dict(explicit)
    for row in rows:
        name = row.role_name or row.role_id
        if not name or name in voices:
            continue
        if row.role_id == "narrator" or name in ("旁白", "叙述"):
            voices[name] = NARRATOR_VOICE
        else:
            index = int.from_bytes(name.encode("utf-8"), "little") % len(VOICE_POOL)
            voices[name] = VOICE_POOL[index]
    return voices


def main() -> None:
    args = parse_args()
    rows = read_script(args.script)
    rows.sort(key=lambda row: row.order)
    cast = Cast.load(args.cast) if args.cast else None
    canonicalize_rows(rows, cast)
    if cast is not None:
        # rows whose speaker was unknown at build time (flagged new_role) can get a
        # voice now that the merged cast contains them
        for row in rows:
            if row.voice_ref:
                continue
            role = cast.roles.get(row.role_id) or cast.resolve(row.role_name)
            if role and role.voice_ref:
                row.voice_ref = role.voice_ref

    chapters: OrderedDict[int, list] = OrderedDict()
    for row in rows:
        chapters.setdefault(row.chapter_id, []).append(row)
    if args.limit_chapters:
        chapters = OrderedDict(list(chapters.items())[: args.limit_chapters])
    selected = [row for chapter_rows in chapters.values() for row in chapter_rows]
    if args.limit_rows:
        selected = selected[: args.limit_rows]

    explicit = voice_map_from_args(args.voice)
    if args.voices:
        explicit.update(json.loads(Path(args.voices).read_text(encoding="utf-8")))
    voices = build_voice_map(selected, explicit)

    config = RenderConfig(
        variant=args.variant,
        ckpt_path=args.ckpt,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        duration_rate=args.duration_rate,
        max_seconds=args.max_seconds,
        target_lufs=args.target_lufs,
        normalize_rows=not args.no_normalize,
    )
    from app.patches import apply_patches

    apply_patches()

    out_dir = Path(args.out)
    renderer = Renderer(config)
    print(f"[render] variant={args.variant} rows={len(selected)} chapters={len(chapters)} out={out_dir}")

    (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rendered: dict[int, Path] = {}
    chapter_paths: list[tuple[int, Path]] = []
    audio_seconds = 0.0
    seen = 0
    for chapter_id, chapter_rows in chapters.items():
        limit = (args.limit_rows - seen) if args.limit_rows else None
        if limit is not None and limit <= 0:
            break
        chapter_rendered = renderer.render_rows(chapter_rows, out_dir, voices=voices, limit=limit)
        rendered.update(chapter_rendered)
        seen += len(chapter_rendered)
        audio_seconds += sum(config.seconds_for(row) for row in chapter_rows if row.order in chapter_rendered)
        if args.no_assemble:
            continue
        items = [(chapter_rendered[row.order], row) for row in chapter_rows if row.order in chapter_rendered]
        if not items:
            continue
        chapter_path = out_dir / "chapters" / f"ch{chapter_id:04d}.wav"
        assemble_rows(items, chapter_path, normalize=not args.no_normalize, target_lufs=args.target_lufs)
        chapter_paths.append((chapter_id, chapter_path))
        print(f"[assemble] ch{chapter_id:04d}: {len(items)} rows -> {chapter_path}", flush=True)

    elapsed = time.perf_counter() - started
    rtf = elapsed / audio_seconds if audio_seconds else 0.0
    print(f"[render] done {len(rendered)} rows in {elapsed:.1f}s | audio {audio_seconds / 60:.1f} min | RTF {rtf:.2f}x")

    manifest = {
        "rows": len(rendered),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(elapsed, 1),
        "rtf": round(rtf, 3),
    }
    if chapter_paths:
        book_path = out_dir / "book.wav"
        assemble_chapters(chapter_paths, book_path, normalize=not args.no_normalize, target_lufs=args.target_lufs)
        manifest["chapters"] = len(chapter_paths)
        manifest["book"] = str(book_path)
        print(f"[assemble] book -> {book_path}")
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
