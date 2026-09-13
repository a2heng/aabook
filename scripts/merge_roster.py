#!/usr/bin/env python3
"""Fold per-chapter ``new_roles.json`` annotations into the roster.

The per-chapter extraction treats the roster as a *reference* and marks unseen
speakers with ``*``. This stage collects those marks across chapters, verifies them
with context, merges aliases into existing people (or adds them as new people), and
rewrites ``roster.json`` / ``cast.json``.

    python scripts/merge_roster.py outputs/mybook --chapters outputs/mybook/chapters \\
        --roster outputs/mybook/roster.json --cast-out outputs/mybook/cast.json --voices outputs/refs_bwe
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(APP_ROOT / "scripts"))

from finalize_roster import CONTEXT_COUNT, context_windows, looks_non_person  # noqa: E402

from audiobook.cleaning import normalize_text, read_text  # noqa: E402
from audiobook.llm import LLMClient  # noqa: E402
from audiobook.namefinder import scan_chapters  # noqa: E402
from audiobook.roster import _chat, finalize, infer_gender, to_cast  # noqa: E402
from audiobook.textnorm import clean_for_llm  # noqa: E402

MERGE_SYSTEM = """你是中文小说人物字典整理器。
【已有角色字典】（规范名 + 别名）：
__EXISTING__

【各章新标注的说话人候选】（原文称呼/代号 + 上下文；可能包含泛称与噪声）：
__NEW__

请输出**合并后的完整人物列表**：
1. 新候选若其实是已有角色的另一种称呼（同一人）→ 并入该角色的 aliases；
2. 新候选若确实是新人物 → 新增一条，规范名用候选或按上下文补全更正式的称呼；
3. 明显不是具体人物的泛称/噪声（如“众人”“年轻女声”“巨龙”“神秘人”若无确定身份）→ 放进 dropped；
4. **原有角色一个都不能丢**；不要新增输入之外的人。
每人输出 name / aliases（全部称呼）/ scan_names（最小无歧义称呼，家族姓/头衔单独不可用）/ gender（男/女/未知）/ traits（2-3 条）。
只输出 JSON：{"characters":[{"name":..,"aliases":[..],"scan_names":[..],"gender":..,"traits":[..]}],"dropped":["..."]}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-chapter new roles into the roster")
    parser.add_argument("out", help="book output root, e.g. outputs/mybook")
    parser.add_argument("--chapters", default=None, help="chapter dir (default <out>/chapters)")
    parser.add_argument("--roster", default=None, help="roster.json (default <out>/roster.json)")
    parser.add_argument("--cast-out", default=None, help="cast.json (default <out>/cast.json)")
    parser.add_argument("--voices", default="outputs/refs_bwe")
    parser.add_argument("--min-count", type=int, default=2, help="keep new names seen at least N times")
    parser.add_argument("--top", type=int, default=150, help="cap new candidates fed to one merge call")
    return parser.parse_args()


def _labels(value) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    chapters_dir = Path(args.chapters) if args.chapters else out / "chapters"
    roster_path = Path(args.roster) if args.roster else out / "roster.json"
    cast_path = Path(args.cast_out) if args.cast_out else out / "cast.json"
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    existing = roster.get("characters", [])
    covered: set[str] = set()
    for person in existing:
        covered.update([person["name"], *person.get("aliases", []), *person.get("scan_names", [])])

    counts: Counter[str] = Counter()
    for path in sorted(out.glob("ch*/new_roles.json")):
        for name in json.loads(path.read_text(encoding="utf-8")):
            counts[str(name).strip()] += 1
    fresh = [
        (name, count)
        for name, count in counts.most_common()
        if count >= args.min_count and name not in covered and not looks_non_person(name)
    ][: args.top]
    print(f"[merge] chapters={len(list(out.glob('ch*/new_roles.json')))} raw_new={len(counts)} fresh={len(fresh)}")
    if not fresh:
        print("[merge] nothing new to fold in")
        return

    client = LLMClient.from_env()
    if client is None:
        raise SystemExit("no LLM configured")

    chapter_files = sorted(chapters_dir.glob("ch*.txt"))
    corpus = "\n".join(clean_for_llm(normalize_text(read_text(file))) for file in chapter_files)

    new_block = json.dumps(
        [{"name": name, "count": count, "contexts": context_windows(corpus, name, CONTEXT_COUNT)} for name, count in fresh],
        ensure_ascii=False,
    )
    system = MERGE_SYSTEM.replace(
        "__EXISTING__", json.dumps([{"name": p["name"], "aliases": p.get("aliases", [])} for p in existing], ensure_ascii=False)
    ).replace("__NEW__", new_block)
    payload = _chat(client, system, "只输出 JSON。")
    cards = payload.get("characters") if isinstance(payload, dict) else None
    if not isinstance(cards, list) or not cards:
        raise SystemExit("merge returned nothing")

    merged: list[dict] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        name = str(card.get("name") or "").strip()
        if not name:
            continue
        merged.append(
            {
                "name": name,
                "aliases": _labels(card.get("aliases")),
                "scan_names": _labels(card.get("scan_names")) or [name],
                "gender": str(card.get("gender") or "未知").strip() or "未知",
                "traits": _labels(card.get("traits")),
            }
        )
    print(f"[merge] people {len(existing) - 1} -> {len(merged)}")

    # Re-scan the whole book with every form so chapters stay accurate. Scan forms
    # are derived deterministically (model scan_names are unreliable on a 4B model).
    for person in merged:
        person["scan_names"] = sorted({person["name"], *person["aliases"]})
        if person.get("gender") in ("", "未知"):
            person["gender"] = infer_gender(corpus, person["scan_names"])
    chapter_texts = [clean_for_llm(normalize_text(read_text(file))) for file in chapter_files]
    all_names = sorted({form for person in merged for form in person["scan_names"] if form})
    hits = scan_chapters(chapter_texts, all_names)
    for person in merged:
        person["chapters"] = sorted({chapter for form in person["scan_names"] for chapter in hits.get(form, [])})

    result = finalize([person for person in merged if person["chapters"]], args.voices)
    roster_path.write_text(json.dumps({"characters": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    to_cast({"characters": result}).save(cast_path)
    print(f"[merge] wrote {roster_path} ({len(result) - 1} people) and {cast_path}")


if __name__ == "__main__":
    main()
