#!/usr/bin/env python3
"""Build a whole book's script chapter by chapter, resumable.

1. prepare: clean + split the source txt -> ``outputs/<book>/{source,clean}.txt`` + ``chapters/``
   (plus ``chapters.json`` with the real chapter titles).
2. build: one chapter at a time -> ``outputs/<book>/chNNN/script.{csv,json,sqlite}``.
   A chapter whose ``script.csv`` already exists is skipped, so the job can be
   stopped and restarted at any time (LLM calls are cached in ``.cache/llm`` too).
3. merge: fuse the per-chapter scripts into ``outputs/<book>/script.{csv,json,sqlite}``
   with a global ``order``, plus ``qa_report.json`` / ``role_stats.json``.

    python scripts/build_book.py assets/txt/novel.txt --out outputs/mybook \\
        --cast outputs/mybook/cast.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.canonical import canonicalize_rows  # noqa: E402
from audiobook.cleaning import Chapter, normalize_text, read_text, split_chapters  # noqa: E402
from audiobook.llm import LLMClient  # noqa: E402
from audiobook.pipeline import _qa_report, build_rows  # noqa: E402
from audiobook.postprocess import merge_adjacent_narration  # noqa: E402
from audiobook.schema import Cast, ScriptRow, read_script, write_script, write_script_json, write_script_sqlite  # noqa: E402
from audiobook.stats import distribution_report  # noqa: E402
from audiobook.textnorm import clean_for_llm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-book resumable script builder")
    parser.add_argument("input", help="source .txt (already stripped of site boilerplate)")
    parser.add_argument("--out", required=True, help="output root, e.g. outputs/mybook")
    parser.add_argument("--cast", default=None, help="cast.json (required unless --prepare-only)")
    parser.add_argument("--prepare-only", action="store_true", help="only write source/clean/chapters")
    parser.add_argument("--start", type=int, default=0, help="first chapter id (inclusive)")
    parser.add_argument("--end", type=int, default=0, help="last chapter id (inclusive)")
    parser.add_argument("--limit", type=int, default=0, help="only N chapters (smoke test)")
    parser.add_argument("--force", action="store_true", help="rebuild chapters that already exist")
    parser.add_argument("--no-merge-narration", action="store_true", help="skip adjacent narration merge")
    return parser.parse_args()


def _write_chapters(chapters: list[Chapter], out: Path) -> None:
    chapter_dir = out / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        (chapter_dir / f"ch{chapter.chapter_id:03d}.txt").write_text(chapter.text + "\n", encoding="utf-8")
    manifest = [{"chapter_id": c.chapter_id, "title": c.title} for c in chapters]
    (out / "chapters.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare(input_path: str, out: Path) -> list[Chapter]:
    raw = read_text(input_path)
    source = normalize_text(raw)
    clean = clean_for_llm(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text(source, encoding="utf-8")
    (out / "clean.txt").write_text(clean, encoding="utf-8")
    chapters = split_chapters(clean)
    _write_chapters(chapters, out)
    print(f"[prepare] chapters={len(chapters)} -> {out}", flush=True)
    return chapters


def _merge_narration(rows: list[ScriptRow], enabled: bool) -> list[ScriptRow]:
    if not enabled:
        return rows
    merged, pairs = merge_adjacent_narration([asdict(row) for row in rows])
    result = [ScriptRow(**item) for item in merged]
    if pairs:
        for index, row in enumerate(result, start=1):
            row.order = index
    return result


def _build_chapter(chapter: Chapter, out: Path, cast: Cast, client: LLMClient, *, merge: bool) -> tuple[int, int]:
    chapter_dir = out / f"ch{chapter.chapter_id:03d}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(chapter, cast, client, client.model_id, 0)
    canonicalize_rows(rows, cast)
    rows = _merge_narration(rows, merge)
    for index, row in enumerate(rows, start=1):
        row.order = index
    write_script(rows, chapter_dir / "script.csv")
    write_script_json(rows, chapter_dir / "script.json")
    write_script_sqlite(rows, chapter_dir / "script.sqlite")
    new_roles = sorted({row.role_name for row in rows if "new_role" in (row.flags or "").split(";")})
    if new_roles:
        (chapter_dir / "new_roles.json").write_text(json.dumps(new_roles, ensure_ascii=False, indent=2), encoding="utf-8")
    issues = _qa_report(rows)["issues"]
    return len(rows), issues


def _merge_book(chapters: list[Chapter], out: Path, cast: Cast) -> None:
    rows: list[ScriptRow] = []
    for chapter in chapters:
        script = out / f"ch{chapter.chapter_id:03d}" / "script.csv"
        if script.is_file():
            rows.extend(read_script(script))
    rows.sort(key=lambda row: (row.chapter_id, row.order))
    for index, row in enumerate(rows, start=1):
        row.order = index
    write_script(rows, out / "script.csv")
    write_script_json(rows, out / "script.json")
    write_script_sqlite(rows, out / "script.sqlite")
    report = _qa_report(rows)
    (out / "qa_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = distribution_report(rows, cast)
    (out / "role_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[merge] rows={len(rows)} issues={report['issues']} -> {out}/script.csv", flush=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    chapters = _prepare(args.input, out)
    if args.prepare_only:
        return

    if not args.cast:
        raise SystemExit("--cast is required to build (run finalize_roster.py first)")
    cast = Cast.load(args.cast)
    client = LLMClient.from_env()
    if client is None:
        raise SystemExit("no LLM configured (AUDIOBOOK_LLM_BASE_URL / AUDIOBOOK_LLM_MODEL)")

    merge = not args.no_merge_narration
    selected = [
        chapter
        for chapter in chapters
        if (not args.start or chapter.chapter_id >= args.start) and (not args.end or chapter.chapter_id <= args.end)
    ]
    if args.limit:
        selected = selected[: args.limit]

    built = skipped = failed = 0
    started = time.perf_counter()
    for index, chapter in enumerate(selected, start=1):
        script = out / f"ch{chapter.chapter_id:03d}" / "script.csv"
        if script.is_file() and script.stat().st_size > 0 and not args.force:
            skipped += 1
            continue
        started_chapter = time.perf_counter()
        try:
            rows, issues = _build_chapter(chapter, out, cast, client, merge=merge)
        except Exception as error:  # noqa: BLE001 - one bad chapter must not kill the run
            failed += 1
            print(f"[build] ERROR ch{chapter.chapter_id:03d}: {error!r} (skipped)", flush=True)
            traceback.print_exc()
            continue
        built += 1
        print(
            f"[build] {index}/{len(selected)} ch{chapter.chapter_id:03d} {chapter.title[:20]} "
            f"rows={rows} issues={issues} {time.perf_counter() - started_chapter:.1f}s",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    print(f"[build] done built={built} skipped={skipped} failed={failed} in {elapsed / 60:.1f} min", flush=True)

    _merge_book(chapters, out, cast)


if __name__ == "__main__":
    main()
