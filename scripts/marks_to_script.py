#!/usr/bin/env python3
"""Marked text (⦃角色␟内容⦄) -> script.csv, mechanically (no LLM).

Single file:
    python scripts/marks_to_script.py outputs/dawn/script/ch003.marked.txt \
        --cast outputs/dawn/cast.json --out outputs/dawn/ch003 --chapter-id 3

Whole book (all chNNN.marked.txt in a directory -> one combined script):
    python scripts/marks_to_script.py --marked-dir outputs/dawn/script \
        --cast outputs/dawn/cast.json --out outputs/dawn
    # then: python scripts/render_book.py --script outputs/dawn/script.csv ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.marks import parse_marks, to_script_rows  # noqa: E402
from audiobook.schema import Cast, Role, write_script, write_script_json  # noqa: E402


def cast_from_roles(roles_path: Path) -> Cast:
    """Build a cast from the incremental one-to-many dictionary (voices assigned later)."""
    data = json.loads(roles_path.read_text(encoding="utf-8"))
    cast = Cast()
    for name, labels in data.items():
        cast.add(Role(role_id=name, name=name, aliases=[label for label in labels if label and label != name]))
    cast.narrator()
    return cast


CHAPTER_RE = re.compile(r"ch(\d+)")


def chapter_of(path: Path) -> int:
    match = CHAPTER_RE.search(path.name)
    return int(match.group(1)) if match else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert marked text to a script")
    parser.add_argument("marked", nargs="?", help="a single marked text file")
    parser.add_argument("--marked-dir", default=None, help="directory of chNNN.marked.txt (whole book)")
    parser.add_argument("--cast", default="outputs/dawn/cast.json", help="cast.json; falls back to --roles")
    parser.add_argument("--roles", default=None, help="roles.json (the incremental dictionary) when cast.json is absent")
    parser.add_argument("--out", required=True, help="output dir (script.csv/json)")
    parser.add_argument("--chapter-id", type=int, default=0)
    parser.add_argument("--chapter-title", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    cast_path = Path(args.cast)
    roles_path = Path(args.roles) if args.roles else (Path(args.marked_dir) / "roles.json" if args.marked_dir else None)
    if cast_path.is_file():
        cast = Cast.load(cast_path)
    elif roles_path and roles_path.is_file():
        cast = cast_from_roles(roles_path)
    else:
        raise SystemExit(f"missing {cast_path} and no roles.json fallback")
    out.mkdir(parents=True, exist_ok=True)

    if args.marked_dir:
        files = sorted(Path(args.marked_dir).glob("ch*.marked.txt"), key=chapter_of)
        rows = []
        for path in files:
            chapter_id = chapter_of(path)
            rows += to_script_rows(path.read_text(encoding="utf-8"), cast, chapter_id=chapter_id, start_order=len(rows))
        speech = sum(1 for row in rows if row.kind == "dialogue")
        print(f"chapters={len(files)} rows={len(rows)} (speech={speech}) -> {out / 'script.csv'}")
    else:
        if not args.marked:
            raise SystemExit("need <marked> or --marked-dir")
        text = Path(args.marked).read_text(encoding="utf-8")
        rows = to_script_rows(text, cast, chapter_id=args.chapter_id, chapter_title=args.chapter_title)
        segments = parse_marks(text)
        narration = sum(1 for seg in segments if seg["kind"] == "narration")
        print(f"rows={len(rows)} (speech={len(segments) - narration} narration={narration}) -> {out / 'script.csv'}")

    write_script(rows, out / "script.csv")
    write_script_json(rows, out / "script.json")


if __name__ == "__main__":
    main()
