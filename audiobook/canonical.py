"""Re-group script rows whose speaker was split across role ids.

Role ids are produced by exact cast resolution downstream, but older or hand-made
``script.csv`` files can still hold several ids for one character. This maps them
onto the cast role they exactly resolve to, so statistics and voices stay
consistent. No fuzzy/regex name guessing: canonical names come from the LLM
dictionary or the cast.
"""

from __future__ import annotations

from collections import Counter

from .schema import Cast, ScriptRow


def canonicalize_rows(rows: list[ScriptRow], cast: Cast | None = None) -> dict[str, str]:
    """Merge split roles in place; return ``{old_role_id: canonical_role_id}``."""
    if not rows or cast is None:
        return {}

    names: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        role_id = row.role_id or "unknown"
        names.setdefault(role_id, row.role_name or role_id)
        counts[role_id] += 1

    groups: dict[str, list[str]] = {}
    for role_id, name in names.items():
        role = cast.resolve(name) or cast.resolve(role_id)
        groups.setdefault(role.role_id if role else role_id, []).append(role_id)

    mapping: dict[str, str] = {}
    canonical_names: dict[str, str] = {}
    for target, members in groups.items():
        canonical = max(members, key=lambda role_id: (counts.get(role_id, 0), len(names.get(role_id, ""))))
        canonical_name = names[canonical]
        resolved = cast.resolve(canonical_name)
        if resolved is not None:
            canonical_name = resolved.name
        canonical_names[target] = canonical_name
        for role_id in members:
            mapping[role_id] = target

    for row in rows:
        role_id = row.role_id or "unknown"
        target = mapping.get(role_id, role_id)
        row.role_id = target
        row.role_name = canonical_names.get(target, row.role_name)
    return mapping
