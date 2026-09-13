#!/usr/bin/env python3
"""ASR-check rendered rows against the script, comparing *pinyin*.

For every row wav we transcribe with faster-whisper and compare the pinyin of the
ASR text with the pinyin of the expected ``tts_text``. Pinyin comparison tolerates
homophones, so the rows that still differ are the ones worth listening to.

    python scripts/asr_check.py --script outputs/ep003/script.json \
        --rows outputs/ep003/audio/rows --out outputs/ep003/asr_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from pypinyin import Style, lazy_pinyin  # noqa: E402

_KEEP = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")
_DIGITS = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def pinyin(text: str | None) -> list[str]:
    text = _KEEP.sub("", text or "")
    text = "".join(_DIGITS.get(char, char) for char in text)
    return lazy_pinyin(text, style=Style.NORMAL)


def ratio(a: list[str], b: list[str]) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASR + pinyin check of rendered rows")
    parser.add_argument("--script", required=True)
    parser.add_argument("--rows", required=True, help="row wav directory")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from faster_whisper import WhisperModel

    rows = json.loads(Path(args.script).read_text(encoding="utf-8"))
    by_seg = {}
    for wav in Path(args.rows).glob("*.wav"):
        by_seg[wav.name.split("__")[0]] = wav

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    results = []
    for row in sorted(rows, key=lambda r: r.get("order") or 0):
        seg_id = row.get("seg_id", "")
        wav = by_seg.get(seg_id)
        if wav is None:
            continue
        segments, _info = model.transcribe(str(wav), language="zh", beam_size=args.beam_size, vad_filter=False)
        heard = "".join(segment.text for segment in segments).strip()
        expected = row.get("tts_text") or ""
        exp_py, heard_py = pinyin(expected), pinyin(heard)
        score = ratio(exp_py, heard_py)
        results.append(
            {
                "order": row.get("order"),
                "seg_id": seg_id,
                "role": row.get("role_name"),
                "expected": expected,
                "asr": heard,
                "score": round(score, 3),
                "exp_len": len(exp_py),
                "asr_len": len(heard_py),
            }
        )
        mark = "  " if score >= args.threshold else "!!"
        print(f"{mark} #{row.get('order'):>3} {seg_id} score={score:.2f} | {expected[:22]} || {heard[:22]}", flush=True)

    flagged = [r for r in results if r["score"] < args.threshold]
    print(f"\nchecked {len(results)} rows, {len(flagged)} below {args.threshold}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
