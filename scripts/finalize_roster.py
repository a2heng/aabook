#!/usr/bin/env python3
"""Build the character dictionary: statistical discovery -> context LLM filter -> full re-scan.

The candidate list is long and noisy, so the LLM never sees it all at once. Instead
each candidate is presented **with real context windows from the book**, so the model
can tell a person from a place/common word; a second pass verifies the picks and
recovers aliases; finally every chapter is re-scanned with the names + aliases so
silent appearances count.

    python scripts/finalize_roster.py outputs/mybook/chapters --out outputs/mybook/roster.json \\
        --cast-out outputs/mybook/cast.json --voices outputs/refs_bwe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

from audiobook.cleaning import normalize_text, read_text  # noqa: E402
from audiobook.llm import LLMClient  # noqa: E402
from audiobook.namefinder import FUNCTION, STOP, discover, scan_chapters, split_keywords  # noqa: E402
from audiobook.roster import _chat, finalize, infer_gender, to_cast  # noqa: E402
from audiobook.textnorm import clean_for_llm  # noqa: E402

CONTEXT_RADIUS = 16
CONTEXT_COUNT = 3
MERGE_BATCH = 80

CLASSIFY_SYSTEM = """下面每行是一个编号、一个从全书统计出的高频串、以及它在书中的若干上下文片段（…表示省略）。
请判断每个串是不是【人物姓名】（有血肉的个体，包括只出场几次的次要人物）。
**必须对每个编号输出一条**，漏掉的视为非人物。

非人物的典型：地名/国家/世界/星球/城市/组织/家族/种族/神殿神教/称号/物品/概念；普通词、动词短语、数量词（如“一些”“一个”“之前”“为什么”）。头衔/家族姓单独出现（如“塞西尔”“侯爵”）也不算人物。

对是人物者给出规范名 name（最正式完整称呼，如“高文·塞西尔”）与上下文能确认的别名 aliases（如“高文”“赫蒂姑妈”）。

只输出 JSON：
{"items":[{"i":1,"person":true,"name":"高文·塞西尔","aliases":["高文"]},{"i":2,"person":false}]}
"""

ENRICH_SYSTEM = """下面每行是编号、一个已确认的人物串、以及它在书中的上下文。请**二次验证并补全**，而且**每行都要输出一条、禁止漏行、禁止新增输入之外的人**：
- name：规范名（最正式完整称呼）
- aliases：指向同一人物的**全部**称呼（简称/全名/头衔/亲属称谓/代称都收）
- scan_names：用于全书复扫的最小无歧义称呼（家族姓/头衔单独不可用，如“高文”“赫蒂”）
- gender：结合上下文（他/她、称谓、身份）推断，尽量给 男 或 女，实在无法判断才写 未知
- traits：身份/特点 2-3 条
只输出 JSON：{"characters":[{"i":1,"name":..,"aliases":[..],"scan_names":[..],"gender":..,"traits":[..]}]}
"""

# Grammatical/geographic suffixes that are never a person on their own.
NON_PERSON_SUFFIX = (
    "王国",
    "帝国",
    "联邦",
    "公国",
    "共和国",
    "王都",
    "大陆",
    "世界",
    "星球",
    "位面",
    "家族",
    "氏族",
    "军团",
    "骑士团",
    "教会",
    "神殿",
    "神教",
    "教派",
    "商会",
    "协会",
    "学院",
    "部落",
    "种族",
    "联盟",
    "议会",
    "集团",
    "要塞",
    "之神",
    "之城",
    "之国",
    "之地",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize the character dictionary")
    parser.add_argument("input", help="chapter dir or txt")
    parser.add_argument("--out", default="outputs/roster.json")
    parser.add_argument("--cast-out", default=None, help="also write cast.json (extraction schema)")
    parser.add_argument("--voices", default="outputs/refs_bwe")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--top", type=int, default=800)
    parser.add_argument("--batch", type=int, default=35)
    parser.add_argument("--no-verify", action="store_true", help="skip the second context verification pass")
    return parser.parse_args()


def context_windows(corpus: str, gram: str, count: int = CONTEXT_COUNT, radius: int = CONTEXT_RADIUS) -> list[str]:
    """Up to ``count`` real book windows around ``gram`` (grounds the LLM verdict)."""
    windows: list[str] = []
    start = 0
    while len(windows) < count:
        found = corpus.find(gram, start)
        if found < 0:
            break
        low = max(0, found - radius)
        high = min(len(corpus), found + len(gram) + radius)
        windows.append(corpus[low:high].replace("\n", " ").strip())
        start = found + len(gram)
    return windows


def _labels(value) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _index(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def looks_non_person(gram: str) -> bool:
    """Deterministic block: place/org suffixes and pure function words."""
    if gram in STOP or set(gram) <= FUNCTION:
        return True
    return len(gram) > 2 and gram.endswith(NON_PERSON_SUFFIX)


def _is_person(entry: dict) -> bool:
    flag = entry.get("person")
    return not (flag is False or str(flag).strip().lower() in ("false", "0", "no"))


def classify(client: LLMClient, batch: list[dict], corpus: str) -> list[dict]:
    """Context-grounded person judgement; returns ``{gram,name,aliases}`` picks."""
    lines = []
    for index, item in enumerate(batch, start=1):
        windows = context_windows(corpus, item["gram"])
        evidence = " ／ ".join(windows) if windows else "（无上下文）"
        lines.append(f"{index}. {item['gram']}（出现{item['count']}次）例：{evidence}")
    payload = _chat(client, CLASSIFY_SYSTEM, "候选：\n" + "\n".join(lines) + "\n\n只输出 JSON。")
    items = payload.get("items") if isinstance(payload, dict) else None
    picks: list[dict] = []
    for entry in items if isinstance(items, list) else []:
        if not isinstance(entry, dict) or not _is_person(entry):
            continue
        index = _index(entry.get("i") or entry.get("index"))
        if not index or not 1 <= index <= len(batch):
            continue
        item = batch[index - 1]
        name = str(entry.get("name") or item["gram"]).strip() or item["gram"]
        picks.append({"gram": item["gram"], "name": name, "aliases": _labels(entry.get("aliases"))})
    return picks


def _dedup_people(people: list[dict]) -> list[dict]:
    """Fuse people across merge batches by alias overlap (deterministic)."""
    fused: list[dict] = []
    for person in people:
        tokens = {person["name"], *person.get("aliases", [])}
        hit = next((p for p in fused if tokens & {p["name"], *p.get("aliases", [])}), None)
        if hit is None:
            fused.append(person)
            continue
        hit["aliases"] = sorted({*hit.get("aliases", []), person["name"], *person.get("aliases", [])} - {hit["name"]})
        hit["scan_names"] = sorted({*hit.get("scan_names", []), *person.get("scan_names", [])})
        hit["traits"] = sorted({*hit.get("traits", []), *person.get("traits", [])})
        if hit.get("gender") in ("", "未知"):
            hit["gender"] = person.get("gender", "未知")
    return fused


def enrich(client: LLMClient, confirmed: list[dict], corpus: str) -> list[dict]:
    """Second pass: context verification + alias recovery.

    Non-destructive on purpose -- every confirmed candidate gets an output entry, so
    the model can only refine names/aliases/gender, never silently drop a person.
    """
    people: list[dict] = []
    for start in range(0, len(confirmed), MERGE_BATCH):
        batch = confirmed[start : start + MERGE_BATCH]
        items = [
            {
                "i": index,
                "gram": entry["gram"],
                "name": entry["name"],
                "aliases": entry["aliases"][:6],
                "contexts": context_windows(corpus, entry["gram"], 2),
            }
            for index, entry in enumerate(batch, start=1)
        ]
        payload = _chat(client, ENRICH_SYSTEM, "候选人物：\n" + json.dumps(items, ensure_ascii=False) + "\n\n只输出 JSON。")
        cards = payload.get("characters") if isinstance(payload, dict) else None
        got: dict[int, dict] = {}
        for card in cards if isinstance(cards, list) else []:
            if not isinstance(card, dict):
                continue
            index = _index(card.get("i") or card.get("index"))
            if index and 1 <= index <= len(batch):
                got[index] = card
        for index, entry in enumerate(batch, start=1):
            card = got.get(index, {})
            name = str(card.get("name") or entry["name"]).strip() or entry["name"]
            aliases = _labels(card.get("aliases")) or entry["aliases"]
            scan = [str(s).strip() for s in (card.get("scan_names") or []) if str(s).strip()] or [entry["gram"]]
            people.append(
                {
                    "name": name,
                    "aliases": aliases,
                    "scan_names": scan,
                    "gender": str(card.get("gender") or "未知").strip() or "未知",
                    "traits": _labels(card.get("traits")),
                }
            )
    return _dedup_people(people)


def main() -> None:
    args = parse_args()
    client = LLMClient.from_env()
    if client is None:
        raise SystemExit("no LLM configured")
    path = Path(args.input)
    files = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    chapter_texts = [clean_for_llm(normalize_text(read_text(file))) for file in files]
    # Discovery runs on a whole-book sample (the cast recurs across chapters); the
    # full-book re-scan below still sees every chapter. Sampling keeps the O(n^2)
    # n-gram overlap step fast on multi-million-character books.
    discovery_corpus = "\n".join(text for index, text in enumerate(chapter_texts) if index % 3 == 0)
    print(f"[discover] corpus={len(discovery_corpus)} chars (sampled 1/3 of {len(chapter_texts)} chapters)")

    candidates = discover(discovery_corpus, min_count=args.min_count, family_limit=args.top)[: args.top]
    person_keywords, blocked = split_keywords(candidates)
    by_gram = {item["gram"]: item for item in candidates}
    pool = [by_gram[gram] for gram in person_keywords if gram in by_gram and not looks_non_person(gram)]
    print(
        f"[discover] candidates={len(candidates)} person-keywords={len(pool)} "
        f"blocked={len(blocked) + len(person_keywords) - len(pool)}"
    )

    confirmed: list[dict] = []
    seen: set[str] = set()
    for start in range(0, len(pool), args.batch):
        batch = pool[start : start + args.batch]
        for pick in classify(client, batch, discovery_corpus):
            if pick["gram"] not in seen:
                seen.add(pick["gram"])
                confirmed.append(pick)
        print(f"[classify] batch {start // args.batch + 1}: people={len(confirmed)}", flush=True)
    if not confirmed:
        raise SystemExit("no people found")

    if args.no_verify:
        merged = [
            {"name": c["name"], "aliases": c["aliases"], "scan_names": [c["gram"]], "gender": "未知", "traits": []}
            for c in confirmed
        ]
    else:
        merged = enrich(client, confirmed, discovery_corpus)
    merged = [
        person for person in merged if not any(looks_non_person(form) for form in [person["name"], *person.get("aliases", [])])
    ]
    print(f"[verify] confirmed={len(confirmed)} -> people={len(merged)}")
    if not merged:
        raise SystemExit("verification dropped every candidate")

    # Re-scan the whole book with names + aliases so silent appearances and aliases count.
    # scan forms are derived deterministically (the model's scan_names are unreliable).
    for person in merged:
        person["scan_names"] = sorted({person["name"], *person["aliases"]})
        if person.get("gender") in ("", "未知"):
            person["gender"] = infer_gender(discovery_corpus, person["scan_names"])
    all_names = sorted({form for person in merged for form in person["scan_names"] if form})
    hits = scan_chapters(chapter_texts, all_names)

    result: list[dict] = []
    for person in merged:
        chapters = sorted({chapter for form in person["scan_names"] for chapter in hits.get(form, [])})
        if not chapters:
            continue
        person["chapters"] = chapters
        result.append(person)

    result = finalize(result, args.voices)
    out = {"characters": result}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[roster] wrote {args.out}: {len(result) - 1} people")
    if args.cast_out:
        to_cast(out).save(args.cast_out)
        print(f"[cast] wrote {args.cast_out}")
    for person in result[1:]:
        print(
            f"  {person['name']:16s} {person['gender']} aliases={person.get('aliases', [])[:4]} "
            f"scan={person.get('scan_names')} voice={Path(person['voice_ref']).name}"
        )


if __name__ == "__main__":
    main()
