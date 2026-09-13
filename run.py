#!/usr/bin/env python3
"""AuK wrapper launcher.

The AuK project is a pristine git submodule under ``third_party/AuK``. Our code
lives in ``app/`` and ``vendor/`` (shims). Nothing inside the submodule is edited.

Only one engine is registered per launch so a 16 GB GPU never holds two copies
of the Qwen text encoder:

    python run.py          # AuK-Flash (4-step distilled), default
    python run.py base     # full AuK (Base), 32-step
    python run.py both     # register both variants (needs ~24 GB VRAM)

Path order:
  1. ``vendor/``       -> ``torchaudio`` / ``silero_vad`` / ``funasr`` shims
  2. ``third_party/AuK/src`` -> the ``auk`` package
  3. ``APP_ROOT``      -> our ``app`` package
"""

from __future__ import annotations

import argparse
import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Keep these in sync with the layout under ``ckpts/``.
CKPT_PATHS = {
    "flash": os.path.join(APP_ROOT, "ckpts", "AuK-Flash", "auk_flash.safetensors"),
    "base": os.path.join(APP_ROOT, "ckpts", "AuK", "auk_base.safetensors"),
}


def build_argv(argv: list[str]) -> list[str]:
    """Turn launcher args into an ``app.infer_gradio.main`` argv list."""
    parser = argparse.ArgumentParser(description="AuK wrapper launcher")
    parser.add_argument(
        "variant",
        nargs="?",
        choices=["flash", "base", "full", "both"],
        default="flash",
        help="which model to load: flash (default), base/full, or both",
    )
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device", default=None, help="e.g. cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true", help="load the engine at startup")
    parser.add_argument("--qwen_path", default=None)
    args = parser.parse_args(argv)

    out = [sys.argv[0], "--dtype", args.dtype, "--host", args.host, "--port", str(args.port)]
    if args.device:
        out += ["--device", args.device]
    if args.qwen_path:
        out += ["--qwen_path", args.qwen_path]
    if args.share:
        out += ["--share"]
    if args.preload:
        out += ["--preload"]

    # Passing any *_ckpt flag makes infer_gradio show only the supplied variants.
    if args.variant in ("base", "full", "both"):
        out += ["--base_ckpt", CKPT_PATHS["base"]]
    if args.variant in ("flash", "both"):
        out += ["--flash_ckpt", CKPT_PATHS["flash"]]
    return out


def main() -> None:
    os.chdir(APP_ROOT)
    # Clear proxy env vars that break httpx/gradio
    for key in list(os.environ):
        if "proxy" in key.lower():
            del os.environ[key]
    # Reduce CUDA allocator fragmentation (must be set before torch is imported)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sys.path.insert(0, os.path.join(APP_ROOT, "third_party", "AuK", "src"))
    sys.path.insert(0, os.path.join(APP_ROOT, "vendor"))
    sys.path.insert(0, APP_ROOT)

    sys.argv = build_argv(sys.argv[1:])

    from app.patches import apply_patches  # noqa: E402

    apply_patches()

    from app.infer_gradio import main as gradio_main  # noqa: E402

    gradio_main()


if __name__ == "__main__":
    main()
