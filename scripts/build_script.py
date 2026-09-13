#!/usr/bin/env python3
"""Build a ``script.csv`` (剧本) from a raw novel ``.txt``.

Examples:
    python scripts/build_script.py novel.txt --out outputs/mybook --no-llm
    AUDIOBOOK_LLM_BASE_URL=http://127.0.0.1:8080/v1 AUDIOBOOK_LLM_MODEL=qwen2.5-7b-instruct \\
        python scripts/build_script.py novel.txt --out outputs/mybook --voices voices.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.llm import LLMClient, LLMConfig  # noqa: E402
from audiobook.models import get_profile  # noqa: E402
from audiobook.pipeline import build_script  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Novel -> script.csv (AuK audiobook front-end)")
    parser.add_argument("input", help="raw novel .txt")
    parser.add_argument("--out", required=True, help="output directory (e.g. outputs/mybook)")
    parser.add_argument("--voices", default=None, help="JSON mapping role name/alias -> reference audio path")
    parser.add_argument("--cast", default=None, help="existing cast.json to reuse (hand-editable)")
    parser.add_argument("--refresh-cast", action="store_true", help="ignore any saved cast and re-discover")
    parser.add_argument("--no-llm", action="store_true", help="skip the LLM (single-narrator fallback smoke test)")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base url")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None, help="model name served by the endpoint")
    parser.add_argument("--model-id", default="", help="label recorded in the script for reproducibility")
    return parser.parse_args()


def make_client(args: argparse.Namespace) -> LLMClient | None:
    if args.no_llm:
        return None
    env = os.environ
    base_url = args.base_url or env.get("AUDIOBOOK_LLM_BASE_URL") or env.get("LLM_BASE_URL")
    api_key = args.api_key or env.get("AUDIOBOOK_LLM_API_KEY") or env.get("LLM_API_KEY") or "sk-local"
    model = args.model or env.get("AUDIOBOOK_LLM_MODEL") or env.get("LLM_MODEL_NAME")
    if not base_url or not model:
        return None
    profile = get_profile(env.get("AUDIOBOOK_LLM_PROFILE"))
    return LLMClient(
        LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            min_p=profile.min_p,
            presence_penalty=profile.presence_penalty,
            repetition_penalty=profile.repetition_penalty,
            max_tokens=int(env.get("AUDIOBOOK_LLM_MAX_TOKENS", profile.max_tokens)),
        )
    )


def main() -> None:
    args = parse_args()
    voices = json.loads(Path(args.voices).read_text(encoding="utf-8")) if args.voices else None
    client = make_client(args)
    if client is None and not args.no_llm:
        print("[warn] no LLM endpoint configured -> falling back to single-narrator mode", file=sys.stderr)

    result = build_script(
        args.input,
        args.out,
        client=client,
        voices=voices,
        model_id=args.model_id,
        cast_source=args.cast,
        refresh_cast=args.refresh_cast,
    )
    print(f"chapters : {result.chapters}")
    print(f"rows     : {result.rows}")
    print(f"issues   : {result.issues}")
    print(f"script   : {result.script_path}")
    print(f"cast     : {result.cast_path}")
    print(f"qa       : {result.qa_path}")
    print(f"stats    : {result.stats_path}")

    report = json.loads(Path(result.qa_path).read_text(encoding="utf-8"))
    for flag, seg_ids in sorted(report["flags"].items(), key=lambda item: -len(item[1])):
        preview = ", ".join(seg_ids[:5])
        print(f"  [{flag}] {len(seg_ids)} -> {preview}")


if __name__ == "__main__":
    main()
