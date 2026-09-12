#!/usr/bin/env python3
"""Download AuK model checkpoints. Default source: ModelScope / 魔搭社区.

`auk-flash` is the 4-step distilled model (fast inference); `auk` is the base model.

Usage:
    # Download all models (AuK + AuK-Flash 4-step + Qwen2.5-Omni-3B) from ModelScope
    python scripts/download_models.py

    # Download only the 4-step distilled model
    python scripts/download_models.py --model auk-flash

    # Download only AuK base model
    python scripts/download_models.py --model auk

    # Download to a custom directory
    python scripts/download_models.py --output /data/ckpts

    # List available models
    python scripts/download_models.py --list
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ModelScope model IDs and their local directory names
MODELS = {
    "auk": {
        "model_id": "Tencent-Hunyuan/AuK",
        "local_dir": "AuK",
        "description": "AuK base model (speech generation & editing)",
    },
    "auk-flash": {
        "model_id": "Tencent-Hunyuan/AuK-Flash",
        "local_dir": "AuK-Flash",
        "description": "AuK-Flash distilled model (fast 4-step inference)",
    },
    "qwen": {
        "model_id": "Qwen/Qwen2.5-Omni-3B",
        "local_dir": "Qwen2.5-Omni-3B",
        "description": "Qwen2.5-Omni-3B text encoder",
    },
}

# Default output directory (relative to project root)
DEFAULT_OUTPUT_DIR = "ckpts"


def get_project_root() -> Path:
    """Return the project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def download_with_modelscope(model_id: str, local_dir: str) -> None:
    """Download a model using modelscope Python SDK."""
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(model_id=model_id, local_dir=local_dir)
    print(f"  [OK] Downloaded {model_id} -> {local_dir}")


def download_with_git(model_id: str, local_dir: str) -> None:
    """Download a model using modelscope git clone (fallback)."""
    url = f"https://www.modelscope.cn/models/{model_id}.git"
    cmd = ["git", "clone", url, local_dir]
    print(f"  Cloning from {url} ...")
    subprocess.check_call(cmd)


def download_model(model_key: str, output_dir: Path) -> bool:
    """Download a single model. Returns True on success."""
    info = MODELS[model_key]
    model_id = info["model_id"]
    local_dir = output_dir / info["local_dir"]

    print(f"\n{'=' * 60}")
    print(f"Model:   {model_key}")
    print(f"ID:      {model_id}")
    print(f"Dest:    {local_dir}")
    print(f"Info:    {info['description']}")
    print(f"{'=' * 60}")

    incomplete = list(local_dir.glob("**/*.incomplete")) if local_dir.exists() else []
    if local_dir.exists() and any(local_dir.iterdir()) and not incomplete:
        print(f"  [SKIP] {local_dir} already exists and is not empty.")
        return True
    if incomplete:
        print(f"  [RESUME] found {len(incomplete)} incomplete file(s), resuming download...")

    local_dir.mkdir(parents=True, exist_ok=True)

    # Try modelscope SDK first, fall back to git clone
    try:
        download_with_modelscope(model_id, str(local_dir))
        return True
    except ImportError:
        print("  [WARN] modelscope SDK not installed, trying git clone...")
    except Exception as e:
        print(f"  [WARN] modelscope SDK failed ({e}), trying git clone...")

    try:
        download_with_git(model_id, str(local_dir))
        return True
    except Exception as e:
        print(f"  [ERROR] Git clone failed: {e}")
        return False


def list_models() -> None:
    """Print available models."""
    print("Available models for download:\n")
    for key, info in MODELS.items():
        print(f"  {key:12s}  {info['model_id']:35s}  {info['description']}")
    print(f"\nUsage: python {__file__} --model <name> [<name> ...]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AuK models from ModelScope (魔搭社区)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help=f"Output directory for checkpoints (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = get_project_root() / DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Expand "all"
    models_to_download = []
    for m in args.model:
        if m == "all":
            models_to_download = list(MODELS.keys())
            break
        models_to_download.append(m)

    # Download each model
    results = {}
    for model_key in models_to_download:
        ok = download_model(model_key, output_dir)
        results[model_key] = ok

    # Summary
    print(f"\n{'=' * 60}")
    print("Download Summary")
    print(f"{'=' * 60}")
    for model_key, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {model_key:12s}  [{status}]")

    if all(results.values()):
        print("\nAll models downloaded successfully!")
        print("You can now run: auk-gradio  or  auk-infer")
    else:
        print("\nSome downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
