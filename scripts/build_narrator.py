#!/usr/bin/env python3
"""Frozen narrator voices: regenerate ``voices/narrator_{m,f}.wav`` deterministically.

The narrator is part of the deliverable and must be reproducible, so its parameters
live in ``voices/narrator.json`` (git-tracked) and this script is the only thing that
writes the reference WAVs. Re-running with the same JSON reproduces the same audio.

    python scripts/build_narrator.py            # build the missing wavs
    python scripts/build_narrator.py --force    # rebuild all
"""

from __future__ import annotations

import argparse
import json
import os
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

SPEC = APP_ROOT / "voices" / "narrator.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen narrator references")
    parser.add_argument("--spec", default=str(SPEC))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--breeze-url", default=None)
    parser.add_argument("--breeze-no-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    voices = spec.get("voices") or {"男": {"instruction": "", "file": "narrator_m.wav"}}
    out_dir = Path(args.spec).parent

    config = BreezeConfig(cfg_scale=float(spec["cfg"]), seed=int(spec["seed"]))
    if args.breeze_url:
        config.base_url = args.breeze_url.rstrip("/")
    renderer = BreezeRenderer(config)
    if not args.breeze_no_start:
        renderer.start()
    try:
        for gender, voice in voices.items():
            out = out_dir / voice["file"]
            if out.is_file() and not args.force:
                print(f"[narrator] exists: {out} (use --force to rebuild)")
                continue
            audio, rate = renderer.design_voice(spec["text"], voice["instruction"])
            if not np.isfinite(audio).all():
                raise RuntimeError(f"non-finite narrator audio for {gender}")
            sf.write(str(out), np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16")
            print(f"[narrator] {out.name} ({gender})  cfg={spec['cfg']} seed={spec['seed']}  {len(audio) / rate:.2f}s")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
