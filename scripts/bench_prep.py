#!/usr/bin/env python3
"""Time the pure-LLM preprocessing (extract + prepare) per chapter.

The LLM server is external, so wall time here is generation time only (model is
already loaded). Reports per-phase seconds and LLM call counts.

Usage:
    AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=spark-4b \
    AUDIOBOOK_LLM_PROFILE=spark-4b \
    python scripts/bench_prep.py --input outputs/book20_out/clean.txt \
        --cast outputs/book20_out/cast.json --chapters 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.agent import RoleAgent  # noqa: E402
from audiobook.cleaning import normalize_text, read_text, split_chapters  # noqa: E402
from audiobook.extract import extract_chapter  # noqa: E402
from audiobook.llm import LLMClient  # noqa: E402
from audiobook.schema import Cast  # noqa: E402
from audiobook.segment import segment_units  # noqa: E402


class CountingClient:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def chat_json(self, *args, **kwargs):
        self.calls += 1
        return self.inner.chat_json(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Time LLM preprocessing per chapter")
    parser.add_argument("--input", required=True, help="novel txt (split into chapters) or a single chapter txt")
    parser.add_argument("--cast", required=True)
    parser.add_argument("--chapters", type=int, default=1, help="number of chapters to process")
    parser.add_argument("--out", default=None, help="optional json report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = LLMClient.from_env()
    if client is None:
        raise SystemExit("no LLM configured")
    cast = Cast.load(args.cast)
    chapters = split_chapters(normalize_text(read_text(args.input)))[: args.chapters]

    report = {"model": client.model_id, "chapters": []}
    totals = {"extract": 0.0, "prep": 0.0, "calls": 0, "units": 0, "rows": 0, "audio_seconds": 0.0}
    for index, chapter in enumerate(chapters, start=1):
        counter = CountingClient(client)
        start = time.perf_counter()
        units = extract_chapter(chapter, cast, counter)  # type: ignore[arg-type]
        RoleAgent(cast, counter).run(units)  # type: ignore[arg-type]
        extract_seconds = time.perf_counter() - start
        extract_calls = counter.calls

        start = time.perf_counter()
        prepared = segment_units(units, cast, counter)  # type: ignore[arg-type]
        prep_seconds = time.perf_counter() - start
        prep_calls = counter.calls - extract_calls

        from audiobook.duration import estimate_text_duration

        audio_seconds = sum(estimate_text_duration(unit.tts_text) for unit in prepared)
        report["chapters"].append(
            {
                "chapter_id": chapter.chapter_id,
                "title": chapter.title[:20],
                "chars": len(chapter.text),
                "units": len(units),
                "segments": len(prepared),
                "extract_seconds": round(extract_seconds, 2),
                "prep_seconds": round(prep_seconds, 2),
                "extract_calls": extract_calls,
                "prep_calls": prep_calls,
                "audio_seconds": round(audio_seconds, 1),
            }
        )
        totals["extract"] += extract_seconds
        totals["prep"] += prep_seconds
        totals["calls"] += counter.calls
        totals["units"] += len(units)
        totals["rows"] += len(prepared)
        totals["audio_seconds"] += audio_seconds
        print(
            f"[bench] ch{chapter.chapter_id:03d} {chapter.title[:16]:16} chars={len(chapter.text):5d} "
            f"extract={extract_seconds:6.1f}s({extract_calls:3d}) prep={prep_seconds:6.1f}s({prep_calls:3d}) "
            f"segs={len(prepared)}",
            flush=True,
        )

    report["totals"] = {key: round(value, 2) if isinstance(value, float) else value for key, value in totals.items()}
    report["totals"]["rtf"] = (
        round((totals["extract"] + totals["prep"]) / totals["audio_seconds"], 3) if totals["audio_seconds"] else 0
    )
    print(
        f"[bench] TOTAL extract={totals['extract']:.1f}s prep={totals['prep']:.1f}s calls={totals['calls']} "
        f"segments={totals['rows']} audio={totals['audio_seconds'] / 60:.1f}min "
        f"RTF={report['totals']['rtf']}x"
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
