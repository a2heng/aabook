#!/usr/bin/env python3
"""Probe how long an input/output Spark-X2.5-4B can handle in one JSON call.

Feeds prefixes of a chapter of increasing length and asks the model to segment
them into ``{"segments":[{"head":..,"role":..,"text":..}]}``. Reports token
usage, finish_reason, output size and whether the JSON parses.

Usage:
    python scripts/probe_llm_len.py --input outputs/book20_out/chapters/ch003.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

import openai  # noqa: E402

from audiobook.cleaning import normalize_text, read_text  # noqa: E402

SYSTEM = """你是中文小说剧本标注器。把原文切成句段，只输出 JSON：
{"segments":[{"head":"该段前6-10字","role":"旁白或角色名","text":"该段朗读文本"}]}
规则：归属短语（“X说道：”）算旁白；引号内对白单独成段并标说话人；极短对白不能漏；长句在气口加逗号；顺序覆盖、逐字照抄。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[400, 800, 1200, 1600, 2200, 3000, 4000, 6000, 8000])
    parser.add_argument("--max-tokens", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = openai.OpenAI(
        base_url=os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
        api_key="sk-local",
        timeout=600,
    )
    model = os.environ.get("AUDIOBOOK_LLM_MODEL", "spark-4b")
    text = normalize_text(read_text(args.input))
    print(f"chapter chars={len(text)} model={model}")
    print(f"{'in_chars':>8} {'prompt_tok':>10} {'out_chars':>9} {'out_tok':>7} {'finish':>8} {'secs':>6} {'json':>5} {'segs':>5}")
    for n in args.lengths:
        chunk = text[:n]
        user = f"原文：\n{chunk}\n\n只输出 JSON。"
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                temperature=0.3,
                top_p=0.9,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{n:>8} ERROR {exc}")
            continue
        elapsed = time.time() - start
        choice = response.choices[0]
        out = choice.message.content or ""
        usage = response.usage
        parsed = None
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            pass
        segs = len(parsed.get("segments", [])) if isinstance(parsed, dict) else -1
        print(
            f"{n:>8} {usage.prompt_tokens:>10} {len(out):>9} {usage.completion_tokens:>7} "
            f"{choice.finish_reason:>8} {elapsed:>6.1f} {('ok' if parsed else 'BAD'):>5} {segs:>5}",
            flush=True,
        )


if __name__ == "__main__":
    main()
