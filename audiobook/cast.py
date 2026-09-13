"""Global cast discovery (Pass 0): sampled candidate extraction + consolidation.

Scanning every chapter is too expensive for long books (a 1595-chapter novel
would be 1595 calls). Instead we scan a representative sample (head + evenly
spaced chapters), merge candidates deterministically, then run one LLM
consolidation pass to group aliases that do not overlap literally.
"""

from __future__ import annotations

import json
import re

from .cleaning import Chapter
from .llm import LLMClient, prompt_hash
from .schema import Cast, Role, slugify

_PAREN_RE = re.compile(r"[（(]([^（()）]+)[)）]")

PROMPT_ID = "cast.candidates.v1"
CONSOLIDATE_ID = "cast.consolidate.v1"
MAX_SCAN_CHARS = 6000
DEFAULT_SAMPLE = 48
HEAD_SAMPLE = 12

SYSTEM = """你是一个中文小说角色抽取器。给你一章小说正文，请找出其中出现的人物角色（含旁白）。
要求：
1. 只输出一个 JSON 对象，不要解释。
2. 合并同一角色的不同称呼为 aliases（如"小林/林老师/林某"）。
3. 旁白用 name="旁白", is_narrator=true。
4. example_utterances 给 1-3 句该角色最有代表性的原话（叙述角色可给其被描述的语气片段）。
JSON schema:
{"roles": [{"name": "角色名", "aliases": ["别名1"], "description": "身份/性格一句话", "example_utterances": ["原话1"], "is_narrator": false}]}
"""

USER_TEMPLATE = "章节标题：{title}\n\n正文：\n{text}\n"

CONSOLIDATE_SYSTEM = """你是中文小说角色合并器。下面是从多章抽取的角色候选，可能有重复、别名分散。
目标：把指向同一人物的候选合并成一个角色，并汇总其所有称呼。

只合并下面这类明确关系：
- 简称/全名：高文 与 高文·塞西尔；
- 头衔/敬称/代称：格鲁曼 与 格鲁曼·塞西尔 与 格鲁曼侯爵；
- 同一人物的别称：半精灵少女 与 琥珀。
不要合并：只是同姓、同场景出现、亲属关系但不同的两个人。

只输出 JSON：{"roles":[{"name":"规范名","aliases":["别名1","别名2"],"description":"..."}]}
要求：
1. name 用全书最常用的正式名（不要带括号）。
2. 保留该人物出现过的所有别名。
3. 输出必须覆盖候选里的每一个人（未合并的也要列出），不要新增候选里没有的角色。
4. 宁可不合并，也不要错合并。
"""


def sample_chapters(chapters: list[Chapter], max_chapters: int = DEFAULT_SAMPLE, head: int = HEAD_SAMPLE) -> list[Chapter]:
    if len(chapters) <= max_chapters:
        return list(chapters)
    head = min(head, max_chapters)
    picked = list(chapters[:head])
    rest = chapters[head:]
    step = max(1, len(rest) // max(1, max_chapters - head))
    for chapter in rest[::step]:
        if len(picked) >= max_chapters:
            break
        picked.append(chapter)
    return picked


def _as_role_dicts(payload: dict | list) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        roles = payload.get("roles")
        if isinstance(roles, list):
            return [item for item in roles if isinstance(item, dict)]
    return []


def _chat(client: LLMClient, system: str, user: str) -> dict | list | None:
    for thinking in (False, True):
        try:
            return client.chat_json(system, user, thinking=thinking)
        except Exception:  # noqa: BLE001 - try the other thinking mode
            continue
    return None


def _add_unique(cast: Cast, role: Role) -> None:
    role_id = role.role_id or slugify(role.name)
    if role_id in cast.roles:
        suffix = 2
        while f"{role_id}_{suffix}" in cast.roles:
            suffix += 1
        role_id = f"{role_id}_{suffix}"
    cast.roles[role_id] = Role(
        role_id=role_id,
        name=role.name,
        aliases=list(role.aliases),
        kind=role.kind,
        description=role.description,
        style_desc=role.style_desc,
        voice_ref=role.voice_ref,
        exemplars=list(role.exemplars),
    )


def _clean_display_name(name: str, aliases: list[str]) -> str:
    """Turn a composite name like "高文（高文·塞西尔）" into one canonical name."""
    match = _PAREN_RE.search(name)
    if not match:
        return name
    outside = _PAREN_RE.sub("", name).strip()
    candidates = [outside, match.group(1).strip(), *aliases]
    return max((c for c in candidates if c), key=len)


def _normalize_names(cast: Cast) -> None:
    rebuilt: dict[str, Role] = {}
    for role in cast.roles.values():
        cleaned = _clean_display_name(role.name, role.aliases)
        if cleaned != role.name:
            if role.name and role.name not in role.aliases:
                role.aliases.append(role.name)
            role.name = cleaned
        if role.kind != "narrator":
            role.role_id = slugify(role.name)
        rebuilt[role.role_id] = role
    cast.roles = rebuilt


def consolidate_cast(cast: Cast, client: LLMClient | None) -> Cast:
    """One LLM pass to merge semantically-equal roles, with a no-loss safety net."""
    if client is None or len(cast.roles) <= 1:
        return cast
    payload = _chat(client, CONSOLIDATE_SYSTEM, "角色候选：\n" + json.dumps(cast.to_dict(), ensure_ascii=False))
    roles = _as_role_dicts(payload) if payload is not None else []
    if not roles:
        return cast
    merged = Cast()
    merged.merge_candidates(roles)
    merged.consolidate()
    for role in cast.roles.values():
        for token in [role.name, *role.aliases]:
            if token and merged.resolve(token) is None:
                _add_unique(merged, role)
                break
    merged.consolidate()
    _normalize_names(merged)
    return merged


def discover_cast(
    chapters: list[Chapter],
    client: LLMClient | None,
    *,
    max_chapters: int = DEFAULT_SAMPLE,
    consolidate: bool = True,
) -> Cast:
    cast = Cast()
    if client is None:
        cast.narrator()
        return cast
    for chapter in sample_chapters(chapters, max_chapters):
        text = chapter.text[:MAX_SCAN_CHARS]
        payload = _chat(client, SYSTEM, USER_TEMPLATE.format(title=chapter.title, text=text))
        if payload is not None:
            cast.merge_candidates(_as_role_dicts(payload))
        cast.consolidate()
    if consolidate and chapters:
        cast = consolidate_cast(cast, client)
    cast.consolidate()
    cast.narrator()
    return cast


def cast_prompt_hash() -> str:
    return prompt_hash(PROMPT_ID, SYSTEM, USER_TEMPLATE)
