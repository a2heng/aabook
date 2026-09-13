#!/usr/bin/env python3
"""Download the GGUF preprocessing LLM from the HuggingFace mirror.

Two backends:
  * ``--connections 1``: ``hf_hub_download`` single stream, auto-resume.
  * ``--connections N``: N parallel HTTP range connections over a fixed-size
    chunk queue, with a ``.state.json`` sidecar that records finished chunks so
    resumes are exact (no reliance on file size, which is sparse for parallel
    writes) and safe across connection-count changes.

Reuses an existing ``hf_hub_download`` ``.incomplete`` prefix when present.

Only the weights GGUF is needed by llama.cpp. Auxiliary files in the repo:
  * ``imatrix_*.gguf``  - importance matrix, used only when *quantizing*, not inference.
  * ``mmproj-*.gguf``   - multimodal (vision) projector, unused for text-only.
  * ``config.json`` / ``README.md`` - Hub metadata, not read by llama.cpp.

Examples:
    python scripts/download_llm.py --list
    python scripts/download_llm.py --connections 8
    python scripts/download_llm.py --connections 8 --file Qwen3.8-27B-UD-IQ3_S.gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "unsloth/Qwen3.8-27B-GGUF"
DEFAULT_FILE = "Qwen3.8-27B-UD-IQ4_XS.gguf"
DEFAULT_ENDPOINT = "https://hf-mirror.com"
DEFAULT_LOCAL_DIR = "ckpts/llm"
CHUNK_SIZE = 64 << 20  # 64 MiB
MAX_RETRIES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a GGUF LLM from the mirror")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--file", default=DEFAULT_FILE)
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="relative to the repo root")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--connections", type=int, default=8, help="parallel range connections (1 = hf_hub_download)")
    parser.add_argument("--retries", type=int, default=10, help="auto-resume attempts on failure")
    parser.add_argument("--list", action="store_true", help="list files in the repo and exit")
    return parser.parse_args()


def _auth_headers(token: str | None) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _resolve_url(endpoint: str, repo: str, filename: str) -> str:
    return f"{endpoint.rstrip('/')}/{repo}/resolve/main/{filename}"


def _remote_size(url: str, token: str | None) -> int:
    import requests

    with requests.get(url, headers=_auth_headers(token), stream=True, allow_redirects=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        return int(response.headers["Content-Length"])


def _adopt_incomplete(local_dir: Path, out: Path) -> None:
    if out.exists():
        return
    cache_dir = local_dir / ".cache" / "huggingface" / "download"
    candidates = []
    if cache_dir.exists():
        candidates = sorted(cache_dir.glob("*.incomplete"), key=lambda path: path.stat().st_size, reverse=True)
    if candidates:
        os.replace(candidates[0], out)
        print(f"resume   : adopted {candidates[0].name} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


def _prepare_state(local_dir: Path, out: Path, total: int) -> tuple[Path, dict]:
    """Load/create the chunk state. Without a state file the current file is
    treated as a valid contiguous prefix (true after ``hf_hub_download`` or a
    truncate); a truncated tail is dropped so no sparse hole is counted."""
    state_path = Path(str(out) + ".state.json")
    _adopt_incomplete(local_dir, out)
    state = {"total": total, "chunk_size": CHUNK_SIZE, "done": []}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if loaded.get("total") == total and loaded.get("chunk_size") == CHUNK_SIZE:
                state = loaded
        except (json.JSONDecodeError, OSError):
            pass
    if out.exists() and not state["done"]:
        size = out.stat().st_size
        complete = size // CHUNK_SIZE
        state["done"] = list(range(complete))
        if size % CHUNK_SIZE:
            os.truncate(out, complete * CHUNK_SIZE)
    return state_path, state


def _write_state(state_path: Path, state: dict) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, state_path)


def _download_chunk(url: str, out: Path, start: int, end: int, token: str | None) -> None:
    import requests

    position = start
    attempt = 0
    while position < end:
        try:
            headers = _auth_headers(token)
            headers["Range"] = f"bytes={position}-{end - 1}"
            with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=(15, 120)) as response:
                if response.status_code == 200:
                    raise RuntimeError("server ignored Range")
                response.raise_for_status()
                handle = os.open(out, os.O_WRONLY)
                try:
                    for chunk in response.iter_content(4 << 20):
                        if chunk:
                            os.pwrite(handle, chunk, position)
                            position += len(chunk)
                finally:
                    os.close(handle)
            attempt = 0
        except Exception as exc:  # noqa: BLE001 - transient network failures
            attempt += 1
            if attempt > MAX_RETRIES:
                raise
            wait = min(30, 2**attempt)
            print(f"retry    : bytes {position}-{end} attempt {attempt}: {exc!r} ({wait}s)", flush=True)
            time.sleep(wait)


def parallel_download(local_dir: Path, url: str, out: Path, connections: int, token: str | None) -> None:
    total = _remote_size(url, token)
    state_path, state = _prepare_state(local_dir, out, total)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        out.touch()

    total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    done = set(state["done"])
    pending = [index for index in range(total_chunks) if index not in done]
    if not pending:
        print(f"already  : {out} ({total / 1e9:.2f} GB)", flush=True)
        return

    print(
        f"size     : {total / 1e9:.2f} GB, {len(done)}/{total_chunks} chunks done, "
        f"{connections} connections, {len(pending)} to go",
        flush=True,
    )

    queue = list(pending)
    lock = threading.Lock()
    stop = threading.Event()
    counter = {"chunks": 0}

    def worker() -> None:
        while not stop.is_set():
            with lock:
                if not queue:
                    return
                index = queue.pop()
            start = index * CHUNK_SIZE
            end = min(total, start + CHUNK_SIZE)
            try:
                _download_chunk(url, out, start, end, token)
            except Exception as exc:  # noqa: BLE001
                stop.set()
                raise exc
            with lock:
                done.add(index)
                counter["chunks"] += 1
                state["done"] = sorted(done)
                _write_state(state_path, state)

    def monitor() -> None:
        while not stop.is_set():
            finished = len(done)
            print(f"progress : {finished}/{total_chunks} chunks ({finished / total_chunks * 100:.1f}%)", flush=True)
            if finished >= total_chunks:
                return
            time.sleep(5)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            for _ in range(connections):
                pool.submit(worker)
    finally:
        stop.set()
        monitor_thread.join(timeout=1)

    if sorted(done) != list(range(total_chunks)):
        raise SystemExit("incomplete: some chunks are still missing")
    os.truncate(out, total)
    with out.open("rb") as handle:
        if out.suffix == ".gguf" and handle.read(4) != b"GGUF":
            raise SystemExit("invalid GGUF magic")
    state_path.unlink(missing_ok=True)


def hf_download(repo: str, filename: str, local_dir: Path, token: str | None, retries: int) -> Path:
    from huggingface_hub import hf_hub_download

    for attempt in range(1, retries + 1):
        try:
            return Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(local_dir), token=token))
        except Exception as exc:  # noqa: BLE001 - transient network failures
            wait = min(60, 2**attempt)
            print(f"attempt {attempt}/{retries} failed: {exc!r}; resume in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"download failed after {retries} attempts")


def main() -> None:
    args = parse_args()
    os.chdir(APP_ROOT)
    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    if args.list:
        from huggingface_hub import HfApi

        info = HfApi(endpoint=args.endpoint).model_info(args.repo, files_metadata=True)
        for sibling in sorted(info.siblings, key=lambda item: item.size or 0):
            print(f"{(sibling.size or 0) / 1e9:8.2f} GB  {sibling.rfilename}")
        return

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    url = _resolve_url(args.endpoint, args.repo, args.file)
    out = local_dir / args.file
    print(f"repo     : {args.repo}", flush=True)
    print(f"file     : {args.file}", flush=True)
    print(f"endpoint : {args.endpoint}", flush=True)
    print(f"local    : {local_dir}", flush=True)

    if args.connections > 1:
        parallel_download(local_dir, url, out, args.connections, args.token)
    else:
        out = hf_download(args.repo, args.file, local_dir, args.token, args.retries)

    print(f"done     : {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
