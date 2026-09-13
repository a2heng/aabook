#!/usr/bin/env python3
"""Web pipeline panel: run every step of the audiobook flow from the browser.

Steps (each button also runnable on its own):
  1. 导入/切章   txt -> outputs/<book>/chapters/*.txt (+ source.txt/clean.txt)
  2. 人物字典    statistical discovery + LLM -> outputs/<book>/roster.json
  3. 生成台本    extraction + prep for one chapter -> script.csv/json
  4. 渲染音频    AuK zero-shot TTS per row + assemble
  5. ASR 体检    faster-whisper pinyin coverage/score
  6. 报告        四列对照 report.html

    python scripts/serve_pipeline.py --port 7860
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.chdir(APP_ROOT)

import gradio as gr  # noqa: E402

from audiobook.cleaning import normalize_text, read_text, split_chapters  # noqa: E402
from audiobook.roster import to_cast  # noqa: E402
from audiobook.textnorm import clean_for_llm, normalize_tts  # noqa: E402

PY = str(APP_ROOT / ".venv" / "bin" / "python")
CACHE_LIB = (
    f"{APP_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cublas/lib:"
    f"{APP_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib"
)


def _llm_env(base_url: str, model: str, profile: str) -> dict:
    return {
        "AUDIOBOOK_LLM_BASE_URL": base_url,
        "AUDIOBOOK_LLM_MODEL": model,
        "AUDIOBOOK_LLM_PROFILE": profile,
        "NO_PROXY": "*",
    }


def _stream(cmd: list[str], env: dict | None = None) -> Iterator[str]:
    process = subprocess.Popen(
        cmd, cwd=APP_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env={**os.environ, **(env or {})}
    )
    log = ""
    assert process.stdout is not None
    for line in process.stdout:
        log += line
        yield log
    process.wait()
    yield log + f"\n[exit {process.returncode}]"


def step_llm_start() -> str:
    if subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True).returncode == 0:
        return "LLM 已在运行"
    log_path = APP_ROOT / "outputs" / "llm_server_web.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    env = {
        **os.environ,
        "AUDIOBOOK_LLM_GGUF": "ckpts/llm/Spark-X2.5-4B-Q8_0.gguf",
        "AUDIOBOOK_LLM_SPEC": "",
        "AUDIOBOOK_LLM_NGL": "99",
        "AUDIOBOOK_LLM_CTX": "16384",
        "NO_PROXY": "*",
    }
    subprocess.Popen(
        ["bash", "scripts/serve_llm_cuda.sh"],
        cwd=APP_ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return "LLM 启动中（约 20s 后就绪）…"


def step_llm_stop() -> str:
    subprocess.run(["pkill", "-x", "llama-server"])
    return "LLM 已停止（显存已释放）"


def step_import(txt_path: str, book: str) -> Iterator[tuple[str, list[str]]]:
    yield "导入/切章…\n", gr.update()
    src = Path(txt_path)
    if not src.is_file():
        yield f"找不到文件：{src}\n", gr.update()
        return
    source = normalize_text(read_text(src))
    clean = clean_for_llm(source)
    out = APP_ROOT / "outputs" / book
    (out / "chapters").mkdir(parents=True, exist_ok=True)
    (out / "source.txt").write_text(source, encoding="utf-8")
    (out / "clean.txt").write_text(clean, encoding="utf-8")
    chapters = split_chapters(clean)
    labels = []
    for chapter in chapters:
        name = f"ch{chapter.chapter_id:03d}.txt"
        (out / "chapters" / name).write_text(normalize_tts(chapter.text), encoding="utf-8")
        labels.append(f"{chapter.chapter_id:03d} · {chapter.title[:28]}")
    yield f"切出 {len(chapters)} 章 -> outputs/{book}/chapters\n", gr.update(choices=labels, value=labels[0] if labels else None)


def step_roster(book: str, base_url: str, model: str, profile: str) -> Iterator[tuple[str, object]]:
    out = APP_ROOT / "outputs" / book
    cmd = [PY, "-u", "scripts/finalize_roster.py", f"outputs/{book}/chapters", "--out", f"outputs/{book}/roster.json"]
    for log in _stream(cmd, _llm_env(base_url, model, profile)):
        yield log, gr.update()
    roster_path = out / "roster.json"
    if roster_path.is_file():
        to_cast(json.loads(roster_path.read_text(encoding="utf-8"))).save(out / "cast.json")
        yield "人物字典完成，并生成 cast.json\n", _roster_table(roster_path)
    else:
        yield "人物字典失败（见日志）\n", gr.update()


def step_script(book: str, chapter_label: str, base_url: str, model: str, profile: str) -> Iterator[str]:
    chapter_id = int(chapter_label.split("·")[0].strip())
    cmd = [
        PY,
        "-u",
        "scripts/build_script.py",
        f"outputs/{book}/chapters/ch{chapter_id:03d}.txt",
        "--out",
        f"outputs/{book}/ch{chapter_id:03d}",
    ]
    cast = APP_ROOT / f"outputs/{book}/cast.json"
    if cast.is_file():
        cmd += ["--cast", str(cast)]
    for log in _stream(cmd, _llm_env(base_url, model, profile)):
        yield log


def step_render(book: str, chapter_label: str) -> Iterator[str]:
    chapter_id = int(chapter_label.split("·")[0].strip())
    out = f"outputs/{book}/ch{chapter_id:03d}"
    voices = f"outputs/{book}/voices.json" if (APP_ROOT / f"outputs/{book}/voices.json").is_file() else None
    cmd = [
        PY,
        "-u",
        "scripts/render_book.py",
        "--script",
        f"{out}/script.csv",
        "--out",
        f"{out}/audio",
        "--cast",
        f"{out}/cast.json",
        "--target-lufs",
        "-16",
    ]
    if voices:
        cmd += ["--voices", voices]
    for log in _stream(cmd):
        yield log


def step_asr(book: str, chapter_label: str) -> Iterator[str]:
    chapter_id = int(chapter_label.split("·")[0].strip())
    out = f"outputs/{book}/ch{chapter_id:03d}"
    for log in _stream(
        [
            PY,
            "-u",
            "scripts/asr_check.py",
            "--script",
            f"{out}/script.json",
            "--rows",
            f"{out}/audio/rows",
            "--out",
            f"{out}/asr_check.json",
        ],
        env={"LD_LIBRARY_PATH": CACHE_LIB},
    ):
        yield log


def step_report(book: str, chapter_label: str) -> Iterator[tuple[str, str]]:
    chapter_id = int(chapter_label.split("·")[0].strip())
    out = APP_ROOT / "outputs" / book / f"ch{chapter_id:03d}"
    report = APP_ROOT / "outputs" / "report.html"
    cmd = [
        PY,
        "-u",
        "scripts/visualize_script.py",
        "--script",
        str(out / "script.json"),
        "--out",
        str(report),
        "--asr",
        str(out / "asr_check.json"),
    ]
    for log in _stream(cmd):
        yield log, gr.update()
    yield "报告已生成 outputs/report.html\n", report.read_text(encoding="utf-8") if report.is_file() else ""


def _roster_table(path: Path):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        [
            p["name"],
            p.get("gender", ""),
            ",".join(map(str, p.get("chapters", []))),
            ",".join(p.get("scan_names", [])),
            Path(p.get("voice_ref", "")).name,
        ]
        for p in data.get("characters", [])
    ]


def build() -> gr.Blocks:
    with gr.Blocks(title="AuK 有声书流水线") as demo:
        gr.Markdown("# AuK 有声书流水线\n每步可单独运行；渲染前请先停掉 LLM 服务以释放显存。")
        with gr.Row():
            txt = gr.Textbox(label="小说 txt", value="assets/txt/巴黎圣母院_维克多·雨果_TXT小说天堂.txt", scale=4)
            book = gr.Textbox(label="书名(输出目录)", value="nddp", scale=1)
        with gr.Row():
            base_url = gr.Textbox(label="LLM base_url", value="http://127.0.0.1:8080/v1", scale=3)
            model = gr.Textbox(label="LLM model", value="spark-4b", scale=2)
            profile = gr.Textbox(label="LLM profile", value="spark-4b", scale=2)
        chapter = gr.Dropdown(label="章节", choices=[], value=None)
        roster_df = gr.Dataframe(headers=["人物", "性别", "出场章节", "扫描称谓", "音色"], label="人物字典", wrap=True)
        log = gr.Textbox(label="日志", lines=22, max_lines=40, autoscroll=True)
        report = gr.HTML(label="四列报告")

        with gr.Row():
            b1 = gr.Button("1 导入/切章", variant="primary")
            b2 = gr.Button("2 人物字典")
            b3 = gr.Button("3 生成台本")
            b4 = gr.Button("4 渲染音频")
            b5 = gr.Button("5 ASR 体检")
            b6 = gr.Button("6 报告")
        with gr.Row():
            bl_on = gr.Button("▶ 启动 LLM")
            bl_off = gr.Button("■ 停止 LLM")

        b1.click(step_import, [txt, book], [log, chapter])
        b2.click(step_roster, [book, base_url, model, profile], [log, roster_df])
        b3.click(step_script, [book, chapter, base_url, model, profile], [log])
        b4.click(step_render, [book, chapter], [log])
        b5.click(step_asr, [book, chapter], [log])
        b6.click(step_report, [book, chapter], [log, report])
        bl_on.click(step_llm_start, None, log)
        bl_off.click(step_llm_stop, None, log)
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    build().queue().launch(server_name=args.host, server_port=args.port, show_error=True)
