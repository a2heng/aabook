"""Unsupervised name *candidates* via repeated substrings (frequency/cohesion/freedom).

No dictionaries and no length/keyword rules: an n-gram is a candidate when it
repeats a lot, its parts stick together far more than chance, and it appears in
many different left/right contexts. Callers then let the LLM pick the real people
and provide each one's minimal, distinctive *scan names*, which are used for a
deterministic full-book re-scan (so even silent appearances count).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

WORD = "\u4e00-\u9fff·"
RUN = re.compile(rf"[{WORD}]+")


def _entropy(counter: Counter[str]) -> float:
    size = sum(counter.values())
    return -sum((c / size) * math.log2(c / size) for c in counter.values())


def discover(
    text: str,
    *,
    min_count: int = 6,
    min_cohesion: float = 4.0,
    min_entropy: float = 1.0,
    max_n: int = 6,
    family_limit: int = 0,
) -> list[dict]:
    runs = RUN.findall(text)
    total = sum(len(run) for run in runs) or 1
    counts: dict[int, Counter[str]] = {n: Counter() for n in range(1, max_n + 1)}
    for run in runs:
        for n in range(1, max_n + 1):
            counter = counts[n]
            for index in range(len(run) - n + 1):
                counter[run[index : index + n]] += 1

    def probability(gram: str) -> float:
        return counts[len(gram)][gram] / total

    candidates = [gram for n in range(2, max_n + 1) for gram, count in counts[n].items() if count >= min_count]
    candidate_set = set(candidates)

    left: dict[str, Counter[str]] = defaultdict(Counter)
    right: dict[str, Counter[str]] = defaultdict(Counter)
    for run in runs:
        for n in range(2, max_n + 1):
            for index in range(len(run) - n + 1):
                gram = run[index : index + n]
                if gram not in candidate_set:
                    continue
                left[gram][run[index - 1] if index > 0 else "^"] += 1
                end = index + n
                right[gram][run[end] if end < len(run) else "$"] += 1

    scored: list[tuple[str, int, float, float]] = []
    for gram in candidates:
        best = math.inf
        for split in range(1, len(gram)):
            denom = probability(gram[:split]) * probability(gram[split:])
            if denom:
                best = min(best, probability(gram) / denom)
        freedom = min(_entropy(left[gram]), _entropy(right[gram]))
        if best >= min_cohesion and freedom >= min_entropy:
            scored.append((gram, counts[len(gram)][gram], best, freedom))

    kept: list[tuple[str, int, float, float]] = []
    for gram, count, cohesion, freedom in sorted(scored, key=lambda item: -len(item[0])):
        if any(gram in other and count <= other_count * 1.3 for other, other_count, _, _ in kept):
            continue
        kept.append((gram, count, cohesion, freedom))
    kept.sort(key=lambda item: -item[1])

    grams = [gram for gram, _, _, _ in kept]
    # The overlap family is O(n^2); only the first ``family_limit`` entries are ever
    # shown to the LLM, so the full matrix is computed only when it is small/needed.
    scope = grams[:family_limit] if family_limit > 0 else grams
    family_of = {gram: [other for other in scope if other != gram and (gram in other or other in gram)] for gram in scope}
    return [
        {
            "gram": gram,
            "count": count,
            "cohesion": round(cohesion, 1),
            "freedom": round(freedom, 2),
            "family": family_of.get(gram, []),
        }
        for gram, count, cohesion, freedom in kept
    ]


def scan_chapters(chapter_texts: list[str], names: list[str]) -> dict[str, list[int]]:
    """Deterministic re-scan: chapter ids where each name occurs (longest-first)."""
    ordered = sorted({name for name in names if name}, key=len, reverse=True)
    found: dict[str, list[int]] = {name: [] for name in ordered}
    for chapter_id, text in enumerate(chapter_texts, start=1):
        for name in ordered:
            if name in text:
                found[name].append(chapter_id)
    return found


# structural/"keyword" helpers: decide which discovered grams are *not* people
PLACE_SUFFIX = ("国", "王国", "帝国", "领", "城", "镇", "村", "山脉", "森林", "要塞", "王都", "联邦", "大陆", "军团")
FUNCTION = set("的了是和与对把被将向从在到这那每各某大小老于其之而且并则因所以为与及或者要能会可该此便又也还就")
STOP = {
    "自己",
    "他们",
    "我们",
    "什么",
    "怎么",
    "已经",
    "如果",
    "虽然",
    "竟然",
    "终于",
    "可是",
    "然后",
    "不是",
    "所有",
    "众人",
    "周围",
    "年轻",
    "其他",
    "有些",
    "对方",
    "身后",
    "面前",
    "身边",
    "声音",
    "时候",
}


def split_keywords(candidates: list[dict]) -> tuple[list[str], set[str]]:
    """Split discovered grams into person keywords vs family/place/function keywords.

    A gram is a family surname when >= 2 middle-dot names share it with different
    given names ("高文·塞西尔", "赫蒂·塞西尔"); it is a place when it forms a longer
    gram with a place suffix ("安苏王国").
    """
    grams = {item["gram"] for item in candidates}
    family: set[str] = set()
    for gram in grams:
        if "·" in gram:
            continue
        given_names = {other[: other.index(gram)] for other in grams if "·" in other and other.endswith(gram)}
        if len(given_names) >= 2:
            family.add(gram)
    place: set[str] = set()
    for gram in grams:
        for other in grams:
            if other != gram and other.startswith(gram) and other[len(gram) :].startswith(PLACE_SUFFIX):
                place.add(gram)
    person: list[str] = []
    for gram in sorted(grams, key=lambda g: -len(g)):
        if gram in family or gram in place or gram in FUNCTION or gram in STOP:
            continue
        person.append(gram)
    return person, family | place
