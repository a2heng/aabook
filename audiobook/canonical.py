"""Canonicalize role ids/names across script rows.

Cast consolidation is not perfect, so one character can end up split across
several ``role_id`` values (e.g. ``大英雄高文·塞西尔`` vs ``高文·塞西尔``). This
groups rows by cast resolution (exact/alias/fuzzy) and by core-name containment,
then rewrites them to a single canonical role so statistics and voices stay
consistent. Safe to run on any ``script.csv``, including older ones.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .schema import Cast, ScriptRow, core_name


def canonicalize_rows(rows: list[ScriptRow], cast: Cast | None = None) -> dict[str, str]:
    """Merge split roles in place; return ``{old_role_id: canonical_role_id}``."""
    if not rows:
        return {}

    names: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        role_id = row.role_id or "unknown"
        names.setdefault(role_id, row.role_name or role_id)
        counts[role_id] += 1

    parent = {role_id: role_id for role_id in names}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    if cast is not None:
        for role_id, name in list(names.items()):
            role = cast.resolve(name)
            if role is None:
                continue
            names.setdefault(role.role_id, role.name)
            parent.setdefault(role.role_id, role.role_id)
            union(role_id, role.role_id)

    cores = {role_id: core_name(name) for role_id, name in names.items()}
    ids = list(names)
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            core_left, core_right = cores[left], cores[right]
            if len(core_left) >= 2 and len(core_right) >= 2:
                if core_left in core_right or core_right in core_left:
                    union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for role_id in ids:
        groups[find(role_id)].append(role_id)

    mapping: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for members in groups.values():
        canonical = max(members, key=lambda role_id: (counts.get(role_id, 0), len(names.get(role_id, ""))))
        canonical_name = names.get(canonical, canonical)
        canonical_names[canonical] = canonical_name
        for role_id in members:
            mapping[role_id] = canonical

    for row in rows:
        role_id = row.role_id or "unknown"
        target = mapping.get(role_id, role_id)
        row.role_id = target
        row.role_name = canonical_names.get(target, row.role_name)
    return mapping
