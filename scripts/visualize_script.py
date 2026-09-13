#!/usr/bin/env python3
"""Render a built script into a self-contained HTML report with four columns:

``原文(raw) -> 预处理 -> LLM -> ASR`` per row, plus the full original text at the
bottom. ``raw`` is mapped back from the cleaned span by alnum position, so the
original wording is always shown next to what the LLM produced.

Example:
    python scripts/visualize_script.py --script outputs/ep003/script.json \
        --asr outputs/ep003/asr_check.json --out outputs/report.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, OrderedDict
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.cleaning import split_chapters  # noqa: E402
from audiobook.schema import read_script  # noqa: E402

KIND_COLORS = {
    "narration": ("#7f8ea3", "#243040"),
    "dialogue": ("#38d39f", "#12332a"),
    "monologue": ("#c792ea", "#2c2140"),
}
BREAK_COLORS = {"sentence": "#4a90d9", "paragraph": "#e58f4e", "": "#5a6270"}


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return [asdict(row) for row in read_script(path)]


def _read_or(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else default


def role_hue(role_id: str, role_name: str) -> int:
    key = role_id or role_name or ""
    return (sum(ord(ch) * 17 for ch in key) % 360) if key else 210


def _alnum_positions(text: str) -> list[int]:
    return [index for index, char in enumerate(text) if char.isalnum()]


def _find(haystack: list[tuple[str, int]], needle: list[str], start: int) -> int | None:
    n = len(needle)
    if n == 0:
        return None
    for k in range(start, len(haystack) - n + 1):
        if all(haystack[k + t][0] == needle[t] for t in range(n)):
            return k
    return None


def map_source_spans(chapter_rows: list[dict], src_chapter: str, clean_chapter: str) -> None:
    """Fill ``row['_src']`` with the original span corresponding to ``row['raw_text']``.

    Cleaning only changes punctuation/whitespace, so the k-th alphanumeric character
    is stable between the original and the cleaned chapter.
    """
    src_skel = [(c, i) for i, c in enumerate(src_chapter) if c.isalnum()]
    clean_skel = [(c, i) for i, c in enumerate(clean_chapter) if c.isalnum()]
    cursor = 0
    for row in chapter_rows:
        raw = row.get("raw_text") or ""
        needle = [c for c, _ in [(c, i) for i, c in enumerate(raw) if c.isalnum()]]
        k = _find(clean_skel, needle, cursor)
        if k is None or not needle:
            row["_src"] = raw
            continue
        start = src_skel[k][1]
        end = src_skel[k + len(needle) - 1][1]
        row["_src"] = src_chapter[start : end + 1]
        cursor = k + len(needle)


def diff_html(base: str, text: str, kind: str = "chg") -> str:
    """Escape ``text``, wrapping the parts that differ from ``base`` in ``<ins class=kind>``."""
    if not base or base == text:
        return html.escape(text)
    pieces: list[str] = []
    for tag, _i1, _i2, j1, j2 in SequenceMatcher(None, base, text, autojunk=False).get_opcodes():
        segment = html.escape(text[j1:j2])
        pieces.append(segment if tag == "equal" else f'<ins class="{kind}">{segment}</ins>')
    return "".join(pieces)


def render(rows: list[dict], title: str, source_text: str = "", clean_text: str = "", asr_map: dict | None = None) -> str:
    asr_map = asr_map or {}
    kind_counts = Counter(row.get("kind", "") for row in rows)
    role_counts = Counter((row.get("role_name") or row.get("role_id") or "?") for row in rows)
    total_seconds = sum(float(row.get("target_duration_s") or 0) for row in rows)
    flagged = sum(1 for row in rows if row.get("flags"))
    asr_bad = 0

    chapters: "OrderedDict[int, list[dict]]" = OrderedDict()
    for row in rows:
        chapters.setdefault(int(row.get("chapter_id") or 0), []).append(row)

    src_chapters = split_chapters(source_text) if source_text else []
    clean_chapters = split_chapters(clean_text) if clean_text else []

    legend_roles = role_counts.most_common(24)
    max_role = max((count for _, count in legend_roles), default=1)
    role_bars = "".join(
        f'<div class="bar-row"><span class="bar-name">{html.escape(name)}</span>'
        f'<span class="bar" style="width:{max(4, int(count / max_role * 100))}%;'
        f'background:hsl({role_hue("", name)},65%,55%)"></span><span class="bar-count">{count}</span></div>'
        for name, count in legend_roles
    )

    cards: list[str] = []
    for position_index, (chapter_id, chapter_rows) in enumerate(chapters.items()):
        heading = chapter_rows[0].get("chapter_title") or f"chapter {chapter_id}"
        if position_index < len(src_chapters) and position_index < len(clean_chapters):
            map_source_spans(chapter_rows, src_chapters[position_index].text, clean_chapters[position_index].text)
        else:
            for row in chapter_rows:
                row["_src"] = row.get("raw_text") or ""
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
            src = row.get("_src") or ""
            clean = row.get("raw_text") or ""
            llm = row.get("tts_text") or ""
            asr = asr_map.get(row.get("seg_id", ""), "")
            if asr and asr != llm:
                asr_bad += 1
            cards.append(
                f"""
<article class="card" style="--accent:{accent};background:{bg};border-left-color:hsl({hue},70%,55%)">
  <div class="meta">
    <span class="ord">#{row.get("order", "?")}</span>
    <span class="kind" style="background:{accent}">{html.escape(kind)}</span>
    <span class="role" style="color:hsl({hue},75%,72%)">{html.escape(row.get("role_name") or row.get("role_id") or "?")}</span>
    <span class="brk" style="border-color:{BREAK_COLORS.get(break_level, "#5a6270")}">{html.escape(break_level or "—")}</span>
    <span class="dur">{float(row.get("target_duration_s") or 0):.1f}s</span>
    <span class="seg">{html.escape(row.get("seg_id", ""))}</span>
    {flag_html}
  </div>
  <div class="grid4">
    <div class="pane"><div class="pane-lbl">原文 raw</div><div class="rawtext">{html.escape(src)}</div></div>
    <div class="pane"><div class="pane-lbl">预处理</div><div class="rawtext">{diff_html(src, clean, "chg")}</div></div>
    <div class="pane"><div class="pane-lbl">LLM</div><div class="tts">{diff_html(clean, llm, "chg")}</div></div>
    <div class="pane"><div class="pane-lbl">ASR</div><div class="rawtext">{diff_html(llm, asr, "asr") if asr else "<span class='none'>（无 ASR）</span>"}</div></div>
  </div>
</article>"""
            )

    source_section = (
        f'<h2 class="chapter">原文全文（对照用 · {len(source_text)} 字）</h2>'
        f'<pre class="fullsource">{html.escape(source_text)}</pre>'
        if source_text
        else '<h2 class="chapter">原文全文</h2><p class="none">未找到 source.txt，用 --source 指定。</p>'
    )

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
  .meta {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:12px; margin-bottom:8px; }}
  .kind {{ color:#0d1117; font-weight:700; border-radius:5px; padding:1px 8px; }}
  .role {{ font-weight:600; }}
  .brk {{ border:1px solid #5a6270; border-radius:5px; padding:0 6px; color:#aebad0; }}
  .dur {{ background:#0f141d; border-radius:5px; padding:1px 8px; color:#8fd3ff; }}
  .ord {{ color:#8fd3ff; font-family:ui-monospace,monospace; font-weight:700; }}
  .seg {{ color:#6b778a; font-family:ui-monospace,monospace; }}
  .flag {{ background:#3a2230; color:#ff9db0; border-radius:5px; padding:1px 6px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .pane {{ min-width:0; }}
  .pane-lbl {{ font-size:11px; color:#5a6270; margin-bottom:4px; letter-spacing:.06em; }}
  .tts, .rawtext {{ font-size:14.5px; line-height:1.7; word-break:break-word; }}
  .rawtext {{ color:#9fb0c8; }}
  ins.chg {{ background:#3d3410; color:#ffe27a; text-decoration:none; border-radius:3px; padding:0 1px; }}
  ins.asr {{ background:#3a1f2a; color:#ff9db0; text-decoration:none; border-radius:3px; padding:0 1px; }}
  .none {{ color:#8b97a8; }}
  @media (max-width:1100px) {{ .grid4 {{ grid-template-columns:repeat(2,1fr); }} }}
  @media (max-width:640px) {{ .grid4 {{ grid-template-columns:1fr; }} }}
  .fullsource {{ white-space:pre-wrap; background:#0f141d; border:1px solid #232c3d; border-radius:8px;
    padding:16px; color:#c4cedd; font:14px/1.9 "Noto Sans CJK SC","PingFang SC",system-ui,sans-serif; }}
</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="sub">每行四列：<b>原文 raw</b> → <span style="color:#ffe27a">预处理</span> → <b>LLM</b> → <span style="color:#ff9db0">ASR</span>（高亮=有改动/差异）</div>
  <div class="stats">
    <div class="stat"><b>{len(rows)}</b><span>segments</span></div>
    <div class="stat"><b>{total_seconds / 60:.1f}m</b><span>total</span></div>
    <div class="stat"><b>{kind_counts.get("dialogue", 0)}</b><span>dialogue</span></div>
    <div class="stat"><b>{kind_counts.get("narration", 0)}</b><span>narration</span></div>
    <div class="stat"><b>{flagged}</b><span>flagged</span></div>
    <div class="stat" style="border-color:#ff9db0"><b style="color:#ff9db0">{asr_bad}</b><span>ASR ≠ LLM</span></div>
  </div>
  <div class="legend"><h3>角色分布（前 24）</h3>{role_bars}</div>
  {"".join(cards)}
  {source_section}
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a built script as HTML")
    parser.add_argument("--script", required=True, help="script.json or script.csv")
    parser.add_argument("--out", default="outputs/report.html", help="output .html (fixed default)")
    parser.add_argument("--source", default=None, help="original text (defaults to source.txt next to script)")
    parser.add_argument("--clean", default=None, help="cleaned text (defaults to clean.txt next to script)")
    parser.add_argument("--asr", default=None, help="asr_check.json to show an ASR column")
    parser.add_argument("--title", default="AuK 预处理可视化")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(args.script)
    rows = load_rows(script_path)
    source_text = _read_or(Path(args.source)) if args.source else _read_or(script_path.parent / "source.txt")
    clean_text = _read_or(Path(args.clean)) if args.clean else _read_or(script_path.parent / "clean.txt")
    asr_map: dict = {}
    asr_path = Path(args.asr) if args.asr else script_path.parent / "asr_check.json"
    if asr_path.is_file():
        asr_map = {item["seg_id"]: item.get("asr", "") for item in json.loads(asr_path.read_text(encoding="utf-8"))}
    html_text = render(rows, args.title, source_text, clean_text, asr_map)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows, source {len(source_text)}, asr {len(asr_map)})")


if __name__ == "__main__":
    main()
