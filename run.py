#!/usr/bin/env python3
"""AuK wrapper launcher.

The AuK project is a pristine git submodule under ``third_party/AuK``. Our code
lives in ``app/`` and ``vendor/`` (shims). Nothing inside the submodule is edited.

Path order:
  1. ``vendor/``       -> ``torchaudio`` / ``silero_vad`` / ``funasr`` shims
  2. ``third_party/AuK/src`` -> the ``auk`` package
  3. ``APP_ROOT``      -> our ``app`` package
"""

import os
import sys

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# config.yaml uses paths relative to the wrapper root (ckpts/...)
os.chdir(APP_ROOT)

# Clear proxy env vars that break httpx/gradio
for key in list(os.environ):
    if "proxy" in key.lower():
        del os.environ[key]

sys.path.insert(0, os.path.join(APP_ROOT, "third_party", "AuK", "src"))
sys.path.insert(0, os.path.join(APP_ROOT, "vendor"))
sys.path.insert(0, APP_ROOT)

sys.argv = [sys.argv[0], "--dtype", "bf16", "--port", "7860"]

from app.patches import apply_patches  # noqa: E402

apply_patches()

from app.infer_gradio import main  # noqa: E402

main()
