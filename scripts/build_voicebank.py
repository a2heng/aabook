#!/usr/bin/env python3
"""Bootstrap a per-role voice bank.

For each role: ``instruct_tts`` synthesizes a sample from the role's personality
plus one of its lines, ASR transcribes it, and the result becomes the cloning
reference used by ``render_book.py``.

Examples:
    python scripts/build_voicebank.py --cast outputs/mybook/cast.json --out outputs/mybook
    python scripts/build_voicebank.py --cast outputs/mybook/cast.json --out outputs/mybook --no-asr
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
for _path in (APP_ROOT / "vendor", APP_ROOT / "third_party" / "AuK" / "src", APP_ROOT):
    sys.path.insert(0, str(_path))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(APP_ROOT)

from audiobook.canonical import canonicalize_rows  # noqa: E402
from audiobook.renderer import RenderConfig, Renderer  # noqa: E402
from audiobook.schema import Cast, read_script  # noqa: E402
from audiobook.stats import role_distribution  # noqa: E402
from audiobook.voicebank import Asr, build_voice_bank  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-role voice bank with AuK")
    parser.add_argument("--cast", required=True, help="path to cast.json")
    parser.add_argument("--out", required=True, help="output dir (writes voicebank.json + references/)")
    parser.add_argument("--script", default=None, help="script.csv: only build references for roles that speak")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--asr-model", default=os.environ.get("AUK_ASR_MODEL", "small"))
    parser.add_argument("--asr-device", default=os.environ.get("AUK_ASR_DEVICE", "cpu"))
    parser.add_argument("--no-asr", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from app.patches import apply_patches

    apply_patches()

    cast = Cast.load(args.cast)
    if args.script:
        rows = read_script(args.script)
        canonicalize_rows(rows, cast)
        stats = role_distribution(rows, cast)
        speaking = {stat.role_id for stat in stats.values() if stat.speaking}
        cast.roles = {rid: role for rid, role in cast.roles.items() if rid in speaking or role.kind == "narrator"}
        print(f"[voicebank] speaking roles ({len(cast.roles)}): {[role.name for role in cast.roles.values()]}")

    config = RenderConfig(
        variant=args.variant,
        ckpt_path=args.ckpt,
        config_path=args.config,
        device=args.device,
        seed=args.seed,
    )
    renderer = Renderer(config)
    asr = None
    if not args.no_asr:
        try:
            asr = Asr(model=args.asr_model, device=args.asr_device)
        except Exception as error:  # noqa: BLE001 - ASR optional
            print(f"[voicebank] ASR unavailable ({error}); reference text will be empty")

    out = Path(args.out)
    bank = build_voice_bank(renderer, cast, out / "references", asr=asr, log=print)
    print(f"[voicebank] {len(bank)} references -> {out / 'voicebank.json'}")


if __name__ == "__main__":
    main()
