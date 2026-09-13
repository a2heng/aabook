#!/usr/bin/env python3
"""Render a built script (script.json/.csv) into a colourful self-contained HTML report.

Shows, per segment: kind/role, break level, duration, flags, and the reading text
with agent-added punctuation highlighted (``ins``), next to the bit-exact source.
The full original chapter text is appended at the bottom for review.

Example:
    python scripts/visualize_script.py --script outputs/book/script.json --out outputs/report.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, OrderedDict
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(APP_ROOT))

from audiobook.schema import read_script  # noqa: E402

KIND_COLORS = {
    "narration": ("#7f8ea3", "#243040"),
    "dialogue": ("#38d39f", "#12332a"),
    "monologue": ("#c792ea", "#2c2140"),
}
BREAK_COLORS = {
    "sentence": "#4a90d9",
    "paragraph": "#e58f4e",
    "": "#5a6270",
}


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [asdict(row) for row in read_script(path)]


def resolve_source(script_path: Path, explicit: str | None) -> str:
    """Full original text for the bottom-of-page reference."""
    if explicit:
        return Path(explicit).read_text(encoding="utf-8")
    for candidate in (script_path.parent / "clean.txt", script_path.parent.parent / "clean.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return ""


_PAUSE_CHARS = set("，、；,;:")


def _emit_char(char: str, added: bool) -> str:
    esc = html.escape(char)
    if char in _PAUSE_CHARS:
        esc = f'<span class="pause" title="气口（段内短停顿）">{esc}</span>'
    if added:
        esc = f'<ins class="air" title="agent 新增">{esc}</ins>'
    return esc


def mark_added(raw: str, tts: str) -> str:
    """HTML for the TTS text: mark 气口 (commas) and agent-inserted characters."""
    if not raw or raw == tts:
        return "".join(_emit_char(char, False) for char in tts)
    matcher = SequenceMatcher(None, raw, tts, autojunk=False)
    pieces: list[str] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        added = tag in ("insert", "replace")
        pieces.extend(_emit_char(char, added) for char in tts[j1:j2])
    return "".join(pieces)


def count_added(raw: str, tts: str) -> int:
    if not raw or raw == tts:
        return 0
    matcher = SequenceMatcher(None, raw, tts, autojunk=False)
    return sum(j2 - j1 for tag, _i1, _i2, j1, j2 in matcher.get_opcodes() if tag in ("insert", "replace"))


def role_hue(role_id: str, role_name: str) -> int:
    key = role_id or role_name or ""
    return (sum(ord(ch) * 17 for ch in key) % 360) if key else 210


def render(rows: list[dict], title: str, source_text: str = "") -> str:
    kind_counts = Counter(row.get("kind", "") for row in rows)
    role_counts = Counter((row.get("role_name") or row.get("role_id") or "?") for row in rows)
    total_seconds = sum(float(row.get("target_duration_s") or 0) for row in rows)
    flagged = sum(1 for row in rows if row.get("flags"))
    air_total = sum(count_added(row.get("raw_text") or "", row.get("tts_text") or "") for row in rows)

    chapters: "OrderedDict[int, list[dict]]" = OrderedDict()
    for row in rows:
        chapters.setdefault(int(row.get("chapter_id") or 0), []).append(row)

    legend_roles = role_counts.most_common(24)
    max_role = max((count for _, count in legend_roles), default=1)
    role_bars = "".join(
        f'<div class="bar-row"><span class="bar-name">{html.escape(name)}</span>'
        f'<span class="bar" style="width:{max(4, int(count / max_role * 100))}%;'
        f'background:hsl({role_hue("", name)},65%,55%)"></span><span class="bar-count">{count}</span></div>'
        for name, count in legend_roles
    )

    cards: list[str] = []
    for chapter_id, chapter_rows in chapters.items():
        heading = chapter_rows[0].get("chapter_title") or f"chapter {chapter_id}"
        chapter_seconds = sum(float(r.get("target_duration_s") or 0) for r in chapter_rows)
        cards.append(
            f'<h2 class="chapter">第 {chapter_id} 章 · {html.escape(str(heading))}'
            f'<span class="chapter-meta">{len(chapter_rows)} 段 · {chapter_seconds:.1f}s</span></h2>'
        )
        for row in chapter_rows:
            kind = row.get("kind", "")
            accent, bg = KIND_COLORS.get(kind, ("#889", "#222"))
            hue = role_hue(row.get("role_id", ""), row.get("role_name", ""))
            break_level = row.get("break_level", "") or ""
            flags = [f for f in str(row.get("flags") or "").split(";") if f]
            flag_html = "".join(f'<span class="flag">{html.escape(f)}</span>' for f in flags)
            raw = row.get("raw_text", "") or ""
            tts = row.get("tts_text", "") or ""
            duration = float(row.get("target_duration_s") or 0)
            cards.append(
                f"""
<article class="card" style="--accent:{accent};background:{bg};border-left-color:hsl({hue},70%,55%)">
  <div class="meta">
    <span class="ord">#{row.get("order", "?")}</span>
    <span class="kind" style="background:{accent}">{html.escape(kind)}</span>
    <span class="role" style="color:hsl({hue},75%,72%)">{html.escape(row.get("role_name") or row.get("role_id") or "?")}</span>
    <span class="brk" style="border-color:{BREAK_COLORS.get(break_level, "#5a6270")}">{html.escape(break_level or "—")}</span>
    <span class="dur">{duration:.1f}s</span>
    <span class="seg">{html.escape(row.get("seg_id", ""))}</span>
    {flag_html}
  </div>
  <div class="panes">
    <div class="pane">
      <div class="pane-lbl">TTS 朗读（处理后）</div>
      <div class="tts">{mark_added(raw, tts)}</div>
    </div>
    <div class="pane">
      <div class="pane-lbl">raw 原文（完整）</div>
      <div class="rawtext">{html.escape(raw)}</div>
    </div>
  </div>
</article>"""
            )

    if source_text:
        source_section = (
            f'<h2 class="chapter">原文全文（对照用 · {len(source_text)} 字）'
            f'<span class="chapter-meta">未做任何改写</span></h2>'
            f'<pre class="fullsource">{html.escape(source_text)}</pre>'
        )
    else:
        source_section = '<h2 class="chapter">原文全文</h2><p class="none">未找到 clean.txt，用 --source 指定原文文件。</p>'

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 24px; background:#11151c; color:#dfe6f0;
    font: 15px/1.6 "Noto Sans CJK SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .sub {{ color:#8b97a8; margin-bottom: 20px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }}
  .stat {{ background:#1b2130; border:1px solid #2a3345; border-radius:10px; padding:10px 14px; min-width:110px; }}
  .stat b {{ display:block; font-size:20px; }}
  .stat span {{ color:#8b97a8; font-size:12px; }}
  .legend {{ background:#161b26; border:1px solid #2a3345; border-radius:10px; padding:14px 16px; margin-bottom:24px; }}
  .legend h3 {{ margin:0 0 10px; font-size:14px; color:#9fb0c8; }}
  .bar-row {{ display:flex; align-items:center; gap:8px; margin:3px 0; font-size:13px; }}
  .bar-name {{ width:120px; text-align:right; color:#bcc7d8; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .bar {{ height:12px; border-radius:6px; min-width:4px; }}
  .bar-count {{ color:#7f8ea3; font-size:12px; }}
  h2.chapter {{ margin:30px 0 12px; font-size:17px; color:#e8eef7; border-bottom:1px solid #2a3345; padding-bottom:6px; }}
  .chapter-meta {{ float:right; color:#7f8ea3; font-size:13px; font-weight:400; }}
  .card {{ border-left:4px solid; border-radius:8px; padding:10px 14px; margin:8px 0; }}
  .meta {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:12px; margin-bottom:6px; }}
  .kind {{ color:#0d1117; font-weight:700; border-radius:5px; padding:1px 8px; }}
  .role {{ font-weight:600; }}
  .brk {{ border:1px solid #5a6270; border-radius:5px; padding:0 6px; color:#aebad0; }}
  .dur {{ background:#0f141d; border-radius:5px; padding:1px 8px; color:#8fd3ff; }}
  .ttsdur {{ color:#ffe27a; }}
  .seg {{ color:#6b778a; font-family:ui-monospace,monospace; }}
  .flag {{ background:#3a2230; color:#ff9db0; border-radius:5px; padding:1px 6px; }}
  .tts {{ font-size:15.5px; word-break:break-word; }}
  .panes {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .pane {{ min-width:0; }}
  .pane-lbl {{ font-size:11px; color:#5a6270; margin-bottom:4px; letter-spacing:.06em; }}
  .rawtext {{ color:#9fb0c8; font-size:14px; line-height:1.7; word-break:break-word; }}
  .ord {{ color:#8fd3ff; font-family:ui-monospace,monospace; font-weight:700; }}
  @media (max-width:900px) {{ .panes {{ grid-template-columns:1fr; }} }}
  ins {{ background:#3d3410; color:#ffe27a; text-decoration:none; border-radius:3px; padding:0 1px; }}
  ins.air {{ background:#5a4a00; color:#ffe27a; border-bottom:2px solid #ffd34d; border-radius:3px; padding:0 2px; }}
  ins.air::before {{ content:"⏸"; font-size:9px; color:#ffd34d; margin-right:1px; vertical-align:super; }}
  .pause {{ color:#ffd34d; border-bottom:1px dotted #ffd34d; }}
  .none {{ color:#8b97a8; }}
  .fullsource {{ white-space:pre-wrap; background:#0f141d; border:1px solid #232c3d; border-radius:8px;
    padding:16px; color:#c4cedd; font:14px/1.9 "Noto Sans CJK SC","PingFang SC",system-ui,sans-serif; }}
</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="sub">每个卡片 = <b>一次 TTS</b> · 顶部 <span style="color:#8fd3ff">#序号</span> 为句子顺序 · 左栏 = TTS 朗读（<span style="color:#ffd34d">逗号=段内气口</span>，<ins class="air" style="background:#5a4a00;color:#ffe27a">⏸金色</ins>=agent 新增）· 右栏 = raw 原文（完整，审计用）</div>
  <div class="stats">
    <div class="stat"><b>{len(rows)}</b><span>segments</span></div>
    <div class="stat"><b>{total_seconds / 60:.1f}m</b><span>total</span></div>
    <div class="stat"><b>{kind_counts.get("dialogue", 0)}</b><span>dialogue</span></div>
    <div class="stat"><b>{kind_counts.get("monologue", 0)}</b><span>monologue</span></div>
    <div class="stat"><b>{kind_counts.get("narration", 0)}</b><span>narration</span></div>
    <div class="stat"><b>{air_total}</b><span>新增气口</span></div>
    <div class="stat"><b>{flagged}</b><span>flagged</span></div>
  </div>
  <div class="legend"><h3>角色分布（前 24）</h3>{role_bars}</div>
  {"".join(cards)}
  {source_section}
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a built script as HTML")
    parser.add_argument("--script", required=True, help="script.json or script.csv")
    parser.add_argument(
        "--out", default="outputs/report.html", help="output .html (fixed default so a browser refresh shows the latest)"
    )
    parser.add_argument("--source", default=None, help="original text file for the bottom reference")
    parser.add_argument("--title", default="AuK 预处理可视化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script)
    rows = load_rows(script_path)
    source_text = resolve_source(script_path, args.source)
    html_text = render(rows, args.title, source_text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows, source {len(source_text)} chars)")


if __name__ == "__main__":
    main()
