#!/usr/bin/env python3
"""Prepare a book: clean + normalise + split the source txt into chapters.

The script itself is produced later by ``scripts/mark_script.py``.

    python scripts/build_book.py assets/txt/novel.txt --out outputs/mybook
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.cleaning import Chapter, normalize_text, read_text, split_chapters  # noqa: E402
from audiobook.textnorm import clean_for_llm, one_paragraph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean + split a novel into chapters")
    parser.add_argument("input", help="source .txt (already stripped of site boilerplate)")
    parser.add_argument("--out", required=True, help="output root, e.g. outputs/mybook")
    return parser.parse_args()


def prepare(input_path: str, out: Path) -> list[Chapter]:
    raw = read_text(input_path)
    source = normalize_text(raw)
    clean = clean_for_llm(source)
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text(source, encoding="utf-8")
    (out / "clean.txt").write_text(clean, encoding="utf-8")
    chapters = split_chapters(clean)
    chapter_dir = out / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        text = one_paragraph(chapter.text)
        (chapter_dir / f"ch{chapter.chapter_id:03d}.txt").write_text(text + "\n", encoding="utf-8")
    manifest = [{"chapter_id": c.chapter_id, "title": c.title} for c in chapters]
    (out / "chapters.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare] chapters={len(chapters)} -> {out}", flush=True)
    return chapters


def main() -> None:
    args = parse_args()
    prepare(args.input, Path(args.out))


if __name__ == "__main__":
    main()
