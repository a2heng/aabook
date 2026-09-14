#!/usr/bin/env python3
"""End-to-end audiobook pipeline: txt -> script.csv -> rendered audio.

One resumable entry point over every stage; each stage can be run alone:

    prepare   clean + split the source txt          -> outputs/<book>/{source,clean}.txt + chapters/
    script    one-edit-at-a-time stage-play marking  -> outputs/<book>/script/chNNN.marked.txt (+roles.json)
    convert   marked text -> script.csv (no LLM)     -> outputs/<book>/script.csv
    render    AuK zero-shot TTS per row + assembly   -> outputs/<book>/render/{rows,chapters,book.wav}

The LLM stages need the local llama.cpp server; this script starts it if it is not
already up and stops it before rendering (rendering needs the whole GPU).

    python scripts/run_book.py assets/txt/novel.txt --book mybook
    python scripts/run_book.py assets/txt/novel.txt --book mybook --stages script
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PY = str(APP_ROOT / ".venv" / "bin" / "python")
STAGES = ("prepare", "script", "convert", "render")
LLM_STAGES = {"script"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Whole-book audiobook pipeline")
    parser.add_argument("input", help="source .txt")
    parser.add_argument("--book", required=True, help="book name (outputs/<book>)")
    parser.add_argument("--out", default=None, help="override output root")
    parser.add_argument(
        "--stages",
        default="prepare,script,convert,render",
        help=f"comma list of {STAGES}",
    )
    parser.add_argument("--llm-base-url", default=os.environ.get("AUDIOBOOK_LLM_BASE_URL", "http://127.0.0.1:8080/v1"))
    parser.add_argument("--llm-model", default=os.environ.get("AUDIOBOOK_LLM_MODEL", "spark-4b"))
    parser.add_argument("--llm-profile", default=os.environ.get("AUDIOBOOK_LLM_PROFILE", "spark-4b"))
    parser.add_argument("--llm-gguf", default="ckpts/llm/Spark-X2.5-4B-Q8_0.gguf")
    parser.add_argument("--no-serve-llm", action="store_true", help="assume the LLM server is already running")
    parser.add_argument("--keep-llm", action="store_true", help="do not stop the LLM before rendering")
    parser.add_argument("--variant", choices=["flash", "base"], default="flash")
    parser.add_argument("--duration-rate", type=float, default=0.8)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--target-lufs", type=float, default=-16.0)
    parser.add_argument("--start", type=int, default=0, help="first chapter id (script stage)")
    parser.add_argument("--end", type=int, default=0, help="last chapter id (script stage)")
    parser.add_argument("--limit", type=int, default=0, help="only N chapters (script stage)")
    parser.add_argument("--force", action="store_true", help="rebuild completed chapters")
    return parser.parse_args()


def _run(cmd: list[str], env: dict | None = None, log_path: Path | None = None) -> None:
    """Run a stage, tee-ing combined stdout/stderr to ``log_path`` (under outputs/)."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=APP_ROOT,
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log is not None:
            log.write(line)
            log.flush()
    code = process.wait()
    if log is not None:
        log.close()
    if code != 0:
        raise SystemExit(f"stage failed (exit {code}): {' '.join(cmd)}")


def _llm_env(args: argparse.Namespace) -> dict:
    return {
        "AUDIOBOOK_LLM_BASE_URL": args.llm_base_url,
        "AUDIOBOOK_LLM_MODEL": args.llm_model,
        "AUDIOBOOK_LLM_PROFILE": args.llm_profile,
        "NO_PROXY": "*",
    }


def _llm_up(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=3):
            return True
    except Exception:  # noqa: BLE001 - not up yet
        return False


def _start_llm(args: argparse.Namespace, log_path: Path) -> None:
    if _llm_up(args.llm_base_url):
        print("[llm] already running", flush=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "AUDIOBOOK_LLM_GGUF": args.llm_gguf,
        "AUDIOBOOK_LLM_SPEC": "",
        "AUDIOBOOK_LLM_NGL": "99",
        "AUDIOBOOK_LLM_CTX": os.environ.get("AUDIOBOOK_LLM_CTX", "16384"),
        "NO_PROXY": "*",
    }
    with log_path.open("w") as log:
        subprocess.Popen(
            ["bash", "scripts/serve_llm_cuda.sh"],
            cwd=APP_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"[llm] starting, log -> {log_path}", flush=True)
    for _ in range(60):
        if _llm_up(args.llm_base_url):
            print("[llm] up", flush=True)
            return
        time.sleep(2)
    raise SystemExit("LLM server did not come up in 120s")


def _stop_llm() -> None:
    if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0:
        subprocess.run(["pkill", "-x", "llama-server"])
        for _ in range(30):
            if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode != 0:
                break
            time.sleep(1)
        print("[llm] stopped (GPU freed)", flush=True)


def main() -> None:
    args = parse_args()
    out = Path(args.out) if args.out else APP_ROOT / "outputs" / args.book
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    for stage in stages:
        if stage not in STAGES:
            raise SystemExit(f"unknown stage: {stage}")

    logs_dir = out / "logs"
    if set(stages) & LLM_STAGES and not args.no_serve_llm:
        _start_llm(args, logs_dir / "llm_server.log")
    llm_env = _llm_env(args)

    if "prepare" in stages:
        _run(
            [PY, "scripts/build_book.py", args.input, "--out", str(out), "--prepare-only"],
            log_path=logs_dir / "prepare.log",
        )

    if "script" in stages:
        chapter_ids = sorted(int(path.stem[2:]) for path in (out / "chapters").glob("ch*.txt"))
        if not chapter_ids:
            raise SystemExit(f"no chapters under {out / 'chapters'}; run the prepare stage first")
        start = args.start or chapter_ids[0]
        end = args.end or chapter_ids[-1]
        if args.limit:
            end = start + args.limit - 1
        _run(
            [
                PY,
                "scripts/mark_script.py",
                str(start),
                "--count",
                str(max(0, end - start + 1)),
                "--book",
                args.book,
                "--mode",
                "seq",
            ],
            env={"AUDIOBOOK_BOOK": args.book, **llm_env},
            log_path=logs_dir / "script.log",
        )

    if "convert" in stages:
        _run(
            [
                PY,
                "scripts/marks_to_script.py",
                "--marked-dir",
                str(out / "script"),
                "--cast",
                str(out / "cast.json"),
                "--out",
                str(out),
            ],
            log_path=logs_dir / "convert.log",
        )

    if "render" in stages:
        if set(stages) & LLM_STAGES and not args.keep_llm:
            _stop_llm()
        script = out / "script.csv"
        if not script.is_file():
            raise SystemExit(f"missing {script}; run the script stage first")
        _run(
            [
                PY,
                "scripts/render_book.py",
                "--script",
                str(script),
                "--out",
                str(out / "render"),
                "--cast",
                str(out / "cast.json"),
                "--variant",
                args.variant,
                "--duration-rate",
                str(args.duration_rate),
                "--max-seconds",
                str(args.max_seconds),
                "--target-lufs",
                str(args.target_lufs),
                *(["--start-chapter", str(args.start)] if args.start else []),
                *(["--end-chapter", str(args.end)] if args.end else []),
            ],
            log_path=logs_dir / "render.log",
        )

    print("\n[done] stage(s): " + ", ".join(stages), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        sys.exit(130)
