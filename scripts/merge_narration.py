#!/usr/bin/env python3
"""Conservatively merge adjacent narration rows in a built script (post-process).

    python scripts/merge_narration.py --script outputs/prep_num2/script.json \
        --max-seconds 40 --out outputs/report.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.postprocess import merge_adjacent_narration  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conservative narration merge (post-process)")
    parser.add_argument("--script", required=True, help="script.json")
    parser.add_argument("--out-json", default=None, help="where to write the merged script.json")
    parser.add_argument("--report", default="outputs/report.html", help="report html (fixed default)")
    parser.add_argument("--tts-max", type=float, default=28.0, help="TTS cap in standard seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = json.loads(Path(args.script).read_text(encoding="utf-8"))
    merged_rows, merged = merge_adjacent_narration(rows, tts_max_seconds=args.tts_max)
    before = sum(1 for row in rows if row.get("kind") == "narration")
    after = sum(1 for row in merged_rows if row.get("kind") == "narration")
    print(f"rows {len(rows)} -> {len(merged_rows)}  (narration {before} -> {after}, merged {len(merged)})")
    for first, second in merged:
        print(f"  merged {first} + {second}")

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out_json}")

    if args.report:
        print(f"report: skipped (visualize_script 已退役) {args.report}")


if __name__ == "__main__":
    main()
