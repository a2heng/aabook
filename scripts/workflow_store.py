#!/usr/bin/env python3
"""Per-book workflow overlay: prompts/params edited from the web, read by mark_script.

The web page (``/workflow`` on the file server) shows the marking flow and lets either side
edit prompts/parameters. Edits are stored in ``outputs/<book>/workflow.json`` and every save
is appended to ``outputs/<book>/workflow_changelog.jsonl`` so both sides can see what changed.

    python scripts/workflow_store.py show --book <book>
    python scripts/workflow_store.py set --book <book> --params '{"max_steps": 120}' --actor agent
    python scripts/workflow_store.py clear --book <book> --actor agent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

PROMPT_KEYS = ("local_system", "step_mark", "roster_system")
PARAM_RANGES = {"batch": (1, 200), "max_steps": (1, 2000)}

# The pipeline as a structure: EVERY stage reads and writes the same product -- the text.
# A chapter is not an object, only a tag in the text; speech/events are tags too.
PIPELINE = {
    "product": {
        "name": "文本（段落流）",
        "note": "没有章节对象：章节、台词、事件都只是文本里的标记，所有阶段读写同一段文本。",
    },
    "tags": [
        {"id": "chapter", "mark": "【第 N 章】", "label": "章节", "note": "边界标记，只用于切窗口"},
        {"id": "speech", "mark": "<角色名>…</角色名>", "label": "台词", "note": "标记外一律旁白"},
        {"id": "event", "mark": "[笑] [叹气] …", "label": "vocal event", "note": "写在台词内容开头"},
    ],
    "stages": [
        {
            "id": "prepare",
            "title": "prepare",
            "script": "scripts/build_book.py",
            "command": "build_book.py <txt> --out outputs/<book>",
            "input": "原始 TXT",
            "output": "段落流文本 + 章节标记",
            "detail": "cleaning + textnorm.keep_layout：段落/缩进/标点原样保留，只去 [ ] < >",
        },
        {
            "id": "mark",
            "title": "script（标注）",
            "script": "scripts/mark_script.py",
            "command": "mark_script.py 1 --count N --book <book>",
            "input": "文本（前情窗口/摘要 + 本章）",
            "output": "插入 <角色> 标记的文本 + roles.json",
            "detail": "maintain_roster 维护人物词典（含声线档案）；edit 工具 speak/delete/replace；无补漏",
            "editable": True,
        },
        {
            "id": "convert",
            "title": "convert",
            "script": "scripts/marks_to_script.py",
            "command": "marks_to_script.py --marked-dir outputs/<book>/script --out outputs/<book>",
            "input": "marked 文本",
            "output": "script.csv / script.json（逐行）",
            "detail": "机械解析 MARK_RE，无 LLM",
        },
        {
            "id": "voicebank",
            "title": "voicebank",
            "script": "scripts/build_voicebank_breeze.py",
            "command": "build_voicebank_breeze.py --book <book>",
            "input": "script.csv + roles/voice_profiles",
            "output": "references/*.wav + voicebank*.json",
            "detail": "Breeze voice design 造参考音；缺年龄/性别时先用 LLM 现定",
        },
        {
            "id": "render",
            "title": "render",
            "script": "scripts/render_book.py",
            "command": "render_book.py --script outputs/<book>/script.csv ...",
            "input": "script.csv + voicebank",
            "output": "render/{rows,chapters,book.wav,mp3}",
            "detail": "Breeze 逐行合成 + 拼接/响度；render 前停 LLM 腾 GPU",
        },
    ],
}


def _outputs(base: Path | None = None) -> Path:
    return Path(base) if base is not None else APP_ROOT / "outputs"


def overlay_path(book: str, base: Path | None = None) -> Path:
    return _outputs(base) / book / "workflow.json"


def changelog_path(book: str, base: Path | None = None) -> Path:
    return _outputs(base) / book / "workflow_changelog.jsonl"


def defaults() -> dict:
    """Built-in values, straight from the code (mark_script module constants)."""
    from scripts import mark_script

    return {
        "params": {
            "batch": 5,  # mark_script `--batch`: first N chapters fed in full, then the summary rolls
            "max_steps": 200,
            "think": mark_script.THINK,
        },
        "prompts": {
            "local_system": mark_script.LOCAL_SYSTEM,
            "step_mark": mark_script.STEP_MARK,
            "roster_system": mark_script.ROSTER_SYSTEM,
        },
    }


def load_overlay(book: str, base: Path | None = None) -> dict:
    path = overlay_path(book, base)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def changelog(book: str, limit: int = 30, base: Path | None = None) -> list[dict]:
    path = changelog_path(book, base)
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return []
    entries = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    entries.reverse()
    return entries


def merged(book: str, base: Path | None = None) -> dict:
    """Defaults + overlay, plus the raw overlay and recent changelog for the page."""
    builtin = defaults()
    overlay = load_overlay(book, base)
    params = {**builtin["params"], **(overlay.get("params") or {})}
    prompts = {**builtin["prompts"], **(overlay.get("prompts") or {})}
    return {
        "book": book,
        "params": params,
        "prompts": prompts,
        "pipeline": PIPELINE,
        "defaults": builtin,
        "overlay": overlay,
        "updated_at": overlay.get("updated_at"),
        "updated_by": overlay.get("updated_by"),
        "changed_params": sorted((overlay.get("params") or {}).keys()),
        "changed_prompts": sorted((overlay.get("prompts") or {}).keys()),
        "paths": {"overlay": str(overlay_path(book, base)), "changelog": str(changelog_path(book, base))},
        "changelog": changelog(book, base=base),
    }


def _clean_params(params: dict) -> dict:
    clean: dict = {}
    for key, value in (params or {}).items():
        if key == "think":
            clean[key] = bool(value)
            continue
        if key not in PARAM_RANGES:
            raise ValueError(f"unknown param: {key}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"param {key} 需要整数")
        low, high = PARAM_RANGES[key]
        clean[key] = max(low, min(high, value))
    return clean


def _clean_prompts(prompts: dict) -> dict:
    clean: dict = {}
    for key, value in (prompts or {}).items():
        if key not in PROMPT_KEYS:
            raise ValueError(f"unknown prompt: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt {key} 不能为空")
        clean[key] = value
    return clean


def _append(book: str, entry: dict, base: Path | None = None) -> None:
    path = changelog_path(book, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save(book: str, payload: dict, actor: str = "web", base: Path | None = None) -> dict:
    """Merge a partial update into the overlay and log what changed."""
    params = _clean_params(payload.get("params") or {})
    prompts = _clean_prompts(payload.get("prompts") or {})
    if not params and not prompts:
        raise ValueError("没有可保存的改动")
    current = load_overlay(book, base)
    overlay = {
        "params": {**(current.get("params") or {}), **params},
        "prompts": {**(current.get("prompts") or {}), **prompts},
        "updated_at": time.time(),
        "updated_by": actor,
    }
    path = overlay_path(book, base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    _append(book, {"ts": time.time(), "actor": actor, "params": params, "prompts": prompts}, base)
    return merged(book, base)


def clear(book: str, actor: str = "web", base: Path | None = None) -> dict:
    path = overlay_path(book, base)
    if path.is_file():
        path.unlink()
    _append(book, {"ts": time.time(), "actor": actor, "cleared": True}, base)
    return merged(book, base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workflow overlay store")
    parser.add_argument("action", choices=["show", "set", "clear"])
    parser.add_argument("--book", required=True)
    parser.add_argument("--params", default="{}", help="JSON partial params")
    parser.add_argument("--prompts", default="{}", help="JSON partial prompts")
    parser.add_argument("--actor", default="agent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "show":
        print(json.dumps(merged(args.book), ensure_ascii=False, indent=2))
        return
    if args.action == "clear":
        state = clear(args.book, actor=args.actor)
        print(f"[workflow] cleared -> {state['paths']['overlay']}")
        return
    payload = {"params": json.loads(args.params), "prompts": json.loads(args.prompts)}
    state = save(args.book, payload, actor=args.actor)
    print(json.dumps({"params": state["params"], "prompts": state["changed_prompts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
