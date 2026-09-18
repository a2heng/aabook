#!/usr/bin/env python3
"""Render a built script.csv into audio with Breeze TTS 2 (rows -> chapters -> book).

Rendering is resumable: existing per-row wavs (matching the same hash) are reused.
Cloning needs a reference wav per role plus its exact transcript, so pass
``--voices`` (voicebank.json) and optionally ``--voice-meta`` (voicebank_meta.json);
the meta file is auto-detected next to ``--out`` if omitted.

Examples:
    python scripts/render_book.py --script outputs/mybook/script.csv --out outputs/mybook/render \\
        --voices outputs/mybook/voicebank.json --voice-meta outputs/mybook/voicebank_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path

import soundfile as sf

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.chdir(APP_ROOT)

from audiobook.tts import (  # noqa: E402
    TARGET_LUFS,
    BreezeConfig,
    BreezeRenderer,
    assemble_chapters,
    assemble_rows,
    voice_map_from_args,
)
from audiobook.canonical import canonicalize_rows  # noqa: E402
from audiobook.schema import Cast, read_script  # noqa: E402

_EXT = {"opus": "opus", "aac": "m4a", "mp3": "mp3", "wav": "wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a script.csv with Breeze TTS 2")
    parser.add_argument("--script", required=True, help="path to script.csv")
    parser.add_argument("--out", required=True, help="output directory for audio")
    parser.add_argument("--cast", default=None, help="cast.json (enables role canonicalization)")
    parser.add_argument("--voices", default=None, help="JSON mapping role name/alias -> reference wav")
    parser.add_argument(
        "--voice-meta",
        default=None,
        help="voicebank_meta.json (role -> ref_text) for cloning; auto-detected from --out/..",
    )
    parser.add_argument("--voice", action="append", default=None, help="role=path override (repeatable)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-lufs", type=float, default=TARGET_LUFS, help="loudness target for rows and masters")
    parser.add_argument("--no-normalize", action="store_true", help="write raw levels (no loudness normalization)")
    parser.add_argument(
        "--audio-format",
        choices=["mp3", "aac", "opus", "wav"],
        default="mp3",
        help="final chapter/book format (default mp3; per-row cache stays wav)",
    )
    parser.add_argument("--bitrate", default="64k", help="lossy bitrate (default 128k for mp3)")
    parser.add_argument("--keep-wav", action="store_true", help="keep the intermediate chapter/book wavs")
    parser.add_argument("--limit-chapters", type=int, default=0, help="only the first N chapters")
    parser.add_argument("--start-chapter", type=int, default=0, help="first chapter_id to render (inclusive)")
    parser.add_argument("--end-chapter", type=int, default=0, help="last chapter_id to render (inclusive)")
    parser.add_argument("--limit-rows", type=int, default=0, help="only the first N rows (smoke test)")
    parser.add_argument("--no-assemble", action="store_true")
    parser.add_argument("--breeze-url", default=None, help="breeze-server base URL (default 127.0.0.1:8137)")
    parser.add_argument("--breeze-model", default=None, help="Breeze GGUF path")
    parser.add_argument("--breeze-bin", default=None, help="breeze-server binary path")
    parser.add_argument("--breeze-cfg", type=float, default=1.0, help="Breeze cfg_scale (1.0 disables guidance)")
    parser.add_argument("--speed", type=float, default=1.0, help="global speed change via ffmpeg atempo (pitch-preserving)")
    parser.add_argument(
        "--breeze-direction",
        action="store_true",
        help="send per-row style/emotion as a voice-direction instruction",
    )
    parser.add_argument(
        "--breeze-no-start",
        action="store_true",
        help="never start breeze-server; require one already running",
    )
    return parser.parse_args()


def load_voice_meta(args: argparse.Namespace) -> dict:
    path = args.voice_meta
    if not path:
        candidate = Path(args.out).resolve().parent / "voicebank_meta.json"
        if candidate.is_file():
            path = str(candidate)
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wav_seconds(path: str | Path) -> float:
    try:
        info = sf.info(str(path))
        return info.frames / float(info.samplerate or 24000)
    except Exception:  # noqa: BLE001 - accounting only
        return 0.0


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
    if args.start_chapter:
        chapters = OrderedDict((cid, chs) for cid, chs in chapters.items() if cid >= args.start_chapter)
    if args.end_chapter:
        chapters = OrderedDict((cid, chs) for cid, chs in chapters.items() if cid <= args.end_chapter)
    if args.limit_chapters:
        chapters = OrderedDict(list(chapters.items())[: args.limit_chapters])
    selected = [row for chapter_rows in chapters.values() for row in chapter_rows]
    if args.limit_rows:
        selected = selected[: args.limit_rows]

    explicit = voice_map_from_args(args.voice)
    if args.voices:
        explicit.update(json.loads(Path(args.voices).read_text(encoding="utf-8")))
    voices = explicit

    out_dir = Path(args.out)
    voice_meta = load_voice_meta(args)

    breeze = BreezeConfig(
        cfg_scale=args.breeze_cfg,
        seed=args.seed,
        target_lufs=args.target_lufs,
        normalize_rows=not args.no_normalize,
        direction=args.breeze_direction,
    )
    if args.breeze_url:
        breeze.base_url = args.breeze_url.rstrip("/")
        parsed = urllib.parse.urlparse(breeze.base_url)
        if parsed.hostname:
            breeze.host = parsed.hostname
        if parsed.port:
            breeze.port = parsed.port
    if args.breeze_model:
        breeze.model_path = args.breeze_model
    if args.breeze_bin:
        breeze.server_bin = args.breeze_bin

    renderer = BreezeRenderer(breeze, voice_meta=voice_meta)
    if not args.breeze_no_start:
        renderer.start()
    print(
        f"[render] backend=breeze url={breeze.base_url} cfg={breeze.cfg_scale} "
        f"direction={breeze.direction} rows={len(selected)} chapters={len(chapters)} out={out_dir}"
    )

    (out_dir / "chapters").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rendered: dict[int, Path] = {}
    chapter_paths: list[tuple[int, Path]] = []
    audio_seconds = 0.0
    seen = 0
    try:
        for chapter_id, chapter_rows in chapters.items():
            limit = (args.limit_rows - seen) if args.limit_rows else None
            if limit is not None and limit <= 0:
                break
            chapter_rendered = renderer.render_rows(chapter_rows, out_dir, voices=voices, limit=limit)
            rendered.update(chapter_rendered)
            seen += len(chapter_rendered)
            audio_seconds += sum(
                wav_seconds(chapter_rendered[row.order]) for row in chapter_rows if row.order in chapter_rendered
            )
            if args.no_assemble:
                continue
            items = [(chapter_rendered[row.order], row) for row in chapter_rows if row.order in chapter_rendered]
            if not items:
                continue
            chapter_path = out_dir / "chapters" / f"ch{chapter_id:04d}.wav"
            assemble_rows(items, chapter_path, normalize=not args.no_normalize, target_lufs=args.target_lufs)
            chapter_paths.append((chapter_id, chapter_path))
            print(f"[assemble] ch{chapter_id:04d}: {len(items)} rows -> {chapter_path}", flush=True)
    finally:
        renderer.close()

    elapsed = time.perf_counter() - started
    rtf = elapsed / audio_seconds if audio_seconds else 0.0
    print(f"[render] done {len(rendered)} rows in {elapsed:.1f}s | audio {audio_seconds / 60:.1f} min | RTF {rtf:.2f}x")

    manifest = {
        "rows": len(rendered),
        "audio_seconds": round(audio_seconds, 1),
        "wall_seconds": round(elapsed, 1),
        "rtf": round(rtf, 3),
    }
    book_wav: Path | None = None
    if chapter_paths:
        book_wav = out_dir / "book.wav"
        assemble_chapters(chapter_paths, book_wav, normalize=not args.no_normalize, target_lufs=args.target_lufs)
        manifest["chapters"] = len(chapter_paths)
        print(f"[assemble] book -> {book_wav}")

    if book_wav is not None and args.audio_format != "wav":
        from audiobook.tts import encode_lossy

        for _chapter_id, wav_path in chapter_paths:
            final = wav_path.with_suffix(f".{_EXT[args.audio_format]}")
            encode_lossy(wav_path, final, fmt=args.audio_format, bitrate=args.bitrate)
            print(f"[encode] {wav_path.name} -> {final.name} ({args.bitrate})", flush=True)
        book_final = book_wav.with_suffix(f".{_EXT[args.audio_format]}")
        encode_lossy(book_wav, book_final, fmt=args.audio_format, bitrate=args.bitrate)
        manifest["book"] = str(book_final)
        manifest["format"] = args.audio_format
        manifest["bitrate"] = args.bitrate
        print(
            f"[encode] book -> {book_final} ({book_final.stat().st_size / 1e6:.1f} MB, was {book_wav.stat().st_size / 1e6:.1f} MB)",
            flush=True,
        )
        if not args.keep_wav:
            for wav_path in [book_wav, *[path for _, path in chapter_paths]]:
                wav_path.unlink(missing_ok=True)
    elif book_wav is not None:
        manifest["book"] = str(book_wav)

    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
