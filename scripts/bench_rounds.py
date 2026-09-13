#!/usr/bin/env python3
"""Benchmark 1-round vs 2-round (self-check) LLM text preparation.

Measures wall time, LLM call count and output shape for the same extracted units,
then writes a per-unit side-by-side HTML. This is a *pure-LLM* experiment: no
mechanical splitting is involved anywhere.

Usage:
    AUDIOBOOK_LLM_BASE_URL=... AUDIOBOOK_LLM_MODEL=... \
    python scripts/bench_rounds.py --chapter outputs/book20_out/chapters/ch003.txt \
        --cast outputs/book20_out/cast.json --out outputs/bench_rounds
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.cleaning import read_text, split_chapters  # noqa: E402
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
    parser = argparse.ArgumentParser(description="1-round vs 2-round prep benchmark")
    parser.add_argument("--chapter", required=True)
    parser.add_argument("--cast", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def run_phase(units, cast, client, cache_dir: str, verify: bool):
    os.environ["AUDIOBOOK_LLM_CACHE"] = cache_dir
    counter = CountingClient(client)
    started = time.perf_counter()
    prepared = segment_units(units, cast, counter, verify=verify)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started
    return prepared, elapsed, counter.calls


def render_html(units, first, second) -> str:
    by_raw_1: dict[str, list] = {}
    for unit in first:
        by_raw_1.setdefault(unit.raw_text, []).append(unit)
    by_raw_2: dict[str, list] = {}
    for unit in second:
        by_raw_2.setdefault(unit.raw_text, []).append(unit)

    rows = []
    for raw in by_raw_1:
        one = by_raw_1[raw]
        two = by_raw_2.get(raw, one)
        changed = [u.tts_text for u in one] != [u.tts_text for u in two]
        rows.append(
            f"""
<article class="card {"changed" if changed else ""}">
  <div class="raw">{html.escape(raw[:200])}</div>
  <div class="cols">
    <div class="col"><h4>1 轮 · {len(one)} 段</h4>{"".join(f'<div class="seg">{html.escape(u.tts_text)}</div>' for u in one)}</div>
    <div class="col"><h4>2 轮 · {len(two)} 段</h4>{"".join(f'<div class="seg">{html.escape(u.tts_text)}</div>' for u in two)}</div>
  </div>
</article>"""
        )
    return f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>1轮 vs 2轮</title>
<style>
body{{margin:0;padding:24px;background:#11151c;color:#dfe6f0;font:14px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}}
h2{{font-size:18px}} .card{{background:#161b26;border:1px solid #2a3345;border-radius:10px;padding:12px;margin:10px 0}}
.card.changed{{border-color:#e58f4e}} .raw{{color:#8794a8;font-size:12px;margin-bottom:8px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:12px}} .col h4{{margin:0 0 6px;font-size:12px;color:#8b97a8}}
.seg{{background:#1b2130;border-radius:6px;padding:3px 8px;margin:3px 0}}
</style></head><body>
<h2>1 轮 vs 2 轮（橙色=有变化）</h2>{"".join(rows)}</body></html>"""


def main() -> None:
    args = parse_args()
    client = LLMClient.from_env()
    if client is None:
        raise SystemExit("no LLM configured (set AUDIOBOOK_LLM_BASE_URL/MODEL)")
    cast = Cast.load(args.cast)
    os.environ["AUDIOBOOK_LLM_CACHE"] = ".cache/llm"  # extraction may be cached
    chapters = split_chapters(read_text(args.chapter))
    chapter = chapters[0]
    units = extract_chapter(chapter, cast, client)
    print(f"units extracted: {len(units)}")

    first, t1, c1 = run_phase(units, cast, client, ".cache/bench_r1", verify=False)
    second, t2, c2 = run_phase(units, cast, client, ".cache/bench_r2", verify=True)

    def air(prepared):
        return sum(1 for u in prepared if "air_added" in u.flags)

    print(f"1 round: {t1:6.1f}s  calls={c1:4d}  segments={len(first):4d}  air={air(first)}")
    print(f"2 round: {t2:6.1f}s  calls={c2:4d}  segments={len(second):4d}  air={air(second)}")
    print(f"delta  : {t2 - t1:6.1f}s  calls={c2 - c1:4d}  segments={len(second) - len(first):4d}  air={air(second) - air(first)}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.html").write_text(render_html(units, first, second), encoding="utf-8")
    (out / "bench.json").write_text(
        json.dumps(
            {
                "chapters": 1,
                "units": len(units),
                "round1": {"seconds": round(t1, 2), "calls": c1, "segments": len(first), "air": air(first)},
                "round2": {"seconds": round(t2, 2), "calls": c2, "segments": len(second), "air": air(second)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out / 'compare.html'}")


if __name__ == "__main__":
    main()
