#!/usr/bin/env python3
"""Voice audition grid: short design texts x gender x seed.

Filename carries every parameter, e.g.
``cfg4_seed11_青年男_你好很高兴见到你.wav``. Browse ``index.html`` via serve_files.

    python scripts/audition_voices.py --out outputs/voice_seedroll
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.chdir(APP_ROOT)

from audiobook.tts import BreezeConfig, BreezeRenderer  # noqa: E402

CFG = 4.0
AGE = "青年"
GENDERS = ("男", "女")
SEEDS = (11, 27, 58, 93, 140, 206)
# Short is better for the design reference; neutral declaratives only.
TEXTS = (
    "你好，很高兴见到你。",
    "今天天气真好。",
    "我们出发吧。",
    "故事就从这里开始。",
)


def slug(text: str, limit: int = 14) -> str:
    body = re.sub(r"[\s，。、！？；：,.!?;:\"'（）()【】\[\]/·]+", "", text)
    return body[:limit]


def build_cases() -> list[dict]:
    cases = []
    for gender in GENDERS:
        instr = f"一位{AGE}{gender}性，日常说话，语气自然。"
        for seed in SEEDS:
            for text in TEXTS:
                name = f"cfg{CFG:g}_seed{seed}_{AGE}{gender}_{slug(text)}"
                cases.append(
                    {
                        "name": name,
                        "group": f"{AGE}{gender}（cfg{CFG:g}）",
                        "cfg": CFG,
                        "seed": seed,
                        "instr": instr,
                        "text": text,
                    }
                )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice audition grid")
    parser.add_argument("--out", default="outputs/voice_seedroll")
    parser.add_argument("--breeze-url", default=None)
    parser.add_argument("--breeze-no-start", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases()

    config = BreezeConfig(cfg_scale=CFG, seed=0)
    if args.breeze_url:
        config.base_url = args.breeze_url.rstrip("/")
    renderer = BreezeRenderer(config)
    if not args.breeze_no_start:
        renderer.start()

    rows = []
    try:
        for case in cases:
            path = out / f"{case['name']}.wav"
            if path.is_file() and not args.force:
                rows.append({**case, "wav": path.name, "sec": round(path.stat().st_size / 2 / 24000, 1)})
                continue
            renderer.config.cfg_scale = case["cfg"]
            renderer.config.seed = case["seed"]
            audio, rate = renderer.design_voice(case["text"], case["instr"])
            sf.write(str(path), np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16")
            rows.append({**case, "wav": path.name, "sec": round(len(audio) / rate, 1)})
            print(f"[design] {path.name}  ({len(audio) / rate:.1f}s)", flush=True)
    finally:
        renderer.close()

    (out / "candidates.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    blocks = []
    for group in dict.fromkeys(r["group"] for r in rows):
        items = []
        for r in [x for x in rows if x["group"] == group]:
            items.append(
                f"<div class='cand'><div class='meta'><b>{html.escape(r['name'])}</b>"
                f" <span class='sec'>{r['sec']}s</span></div>"
                f"<audio controls preload='metadata' src='{r['wav']}'></audio></div>"
            )
        blocks.append(f"<h2>{html.escape(group)}</h2>" + "".join(items))
    page = (
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>声线试听</title>"
        "<style>body{font-family:system-ui,'Noto Sans CJK SC',sans-serif;background:#14161a;color:#e6e6e6;margin:18px}"
        "h2{font-size:15px;color:#9ecbff;margin:22px 0 8px}.cand{display:flex;align-items:center;gap:12px;"
        "background:#1b1f27;border:1px solid #2a313c;border-radius:10px;padding:8px 12px;margin:6px 0}"
        ".meta{flex:1;font-size:12.5px}.sec{color:#8a93a3}</style></head><body>"
        f"<h1>声线试听（{len(rows)}）</h1>" + "".join(blocks) + "</body></html>"
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"[audition] {len(rows)} candidates -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
