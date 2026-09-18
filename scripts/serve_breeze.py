#!/usr/bin/env python3
"""Start / stop the Breeze TTS 2 C++ server (HTTP API + WebUI).

python scripts/serve_breeze.py up            # start detached (WebUI on)
python scripts/serve_breeze.py status
python scripts/serve_breeze.py down
python scripts/serve_breeze.py up --host 0.0.0.0 --port 8137
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

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]

from audiobook.tts import (  # noqa: E402
    BreezeConfig,
    _health,
    start_server,
)

PID_FILE = APP_ROOT / "build" / "breeze-cpp" / "server.pid"


def _owned_pid() -> int | None:
    if not PID_FILE.is_file():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    comm = Path(f"/proc/{pid}/comm")
    if not comm.is_file() or comm.read_text(encoding="utf-8").strip() != "breeze-server":
        return None
    return pid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage the Breeze TTS 2 server")
    parser.add_argument("action", choices=["up", "down", "status"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8137)
    parser.add_argument("--model", default=None)
    parser.add_argument("--bin", default=None, help="path to breeze-server")
    parser.add_argument("--no-webui", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BreezeConfig(host=args.host, port=args.port, webui=not args.no_webui)
    if args.model:
        config.model_path = args.model
    if args.bin:
        config.server_bin = args.bin
    config.base_url = f"http://{args.host}:{args.port}"

    if args.action == "status":
        health = _health(config.base_url)
        if health is None:
            print(f"[breeze] not running at {config.base_url}")
            sys.exit(1)
        print(json.dumps(health, ensure_ascii=False))
        return

    if args.action == "down":
        pid = _owned_pid()
        if pid is None:
            PID_FILE.unlink(missing_ok=True)
            print("[breeze] no owned server pid; nothing stopped")
            return
        os.kill(pid, 15)
        for _ in range(15):
            if not Path(f"/proc/{pid}").exists():
                break
            time.sleep(1.0)
        PID_FILE.unlink(missing_ok=True)
        print(f"[breeze] stopped pid {pid}")
        return

    process = start_server(config)
    if process is not None:
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(process.pid), encoding="utf-8")
        print(f"[breeze] pid {process.pid}, WebUI http://{config.host}:{config.port}/")
    else:
        print(f"[breeze] already running at {config.base_url}")


if __name__ == "__main__":
    main()
