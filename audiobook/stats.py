"""Role distribution statistics over extracted script rows.

Only *speaking* roles are counted: the narrator (who voices narration) and any
character that owns at least one dialogue/monologue row. Named characters that
never speak are reported separately and excluded from the distribution, even if
they are mentioned frequently in the narration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Cast, ScriptRow


@dataclass
class RoleStat:
    role_id: str
    name: str
    kind: str = "character"
    speaking: bool = False
    dialogue_rows: int = 0
    monologue_rows: int = 0
    narration_rows: int = 0
    tts_chars: int = 0
    est_seconds: float = 0.0
    mentions: int = 0
    chapters: set[int] = field(default_factory=set)
    first_order: int = 0

    @property
    def speaking_rows(self) -> int:
        return self.dialogue_rows + self.monologue_rows

    def to_dict(self) -> dict:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "kind": self.kind,
            "speaking": self.speaking,
            "dialogue_rows": self.dialogue_rows,
            "monologue_rows": self.monologue_rows,
            "narration_rows": self.narration_rows,
            "speaking_rows": self.speaking_rows,
            "tts_chars": self.tts_chars,
            "est_seconds": round(self.est_seconds, 1),
            "est_minutes": round(self.est_seconds / 60.0, 2),
            "mentions": self.mentions,
            "chapters": len(self.chapters),
        }


def _count_mentions(text: str, role) -> int:
    needles = {role.name, role.role_id, *role.aliases}
    return sum(text.count(needle) for needle in needles if needle)


def role_distribution(rows: list[ScriptRow], cast: Cast | None = None) -> dict[str, RoleStat]:
    stats: dict[str, RoleStat] = {}
    for row in rows:
        role_id = row.role_id or "unknown"
        stat = stats.get(role_id)
        if stat is None:
            stat = RoleStat(role_id=role_id, name=row.role_name or role_id)
            stats[role_id] = stat
        if row.kind == "dialogue":
            stat.dialogue_rows += 1
        elif row.kind == "monologue":
            stat.monologue_rows += 1
        else:
            stat.narration_rows += 1
        stat.tts_chars += len(row.tts_text or "")
        stat.est_seconds += row.target_duration_s or 0.0
        stat.chapters.add(row.chapter_id)
        if not stat.first_order or row.order < stat.first_order:
            stat.first_order = row.order

    if cast is not None:
        narration_text = "".join(row.tts_text for row in rows if row.kind == "narration")
        for role in cast.roles.values():
            stat = stats.get(role.role_id)
            if stat is None:
                stat = RoleStat(role_id=role.role_id, name=role.name, kind=role.kind)
                stats[role.role_id] = stat
            else:
                stat.name = role.name
                stat.kind = role.kind
            if role.kind != "narrator":
                stat.mentions = _count_mentions(narration_text, role)

    for stat in stats.values():
        if stat.kind == "narrator":
            stat.speaking = stat.dialogue_rows + stat.monologue_rows + stat.narration_rows > 0
        else:
            stat.speaking = stat.speaking_rows > 0
    return stats


def speaking_roles(stats: dict[str, RoleStat]) -> list[RoleStat]:
    roles = [stat for stat in stats.values() if stat.speaking]
    return sorted(roles, key=lambda stat: (stat.est_seconds, stat.speaking_rows), reverse=True)


def non_speaking_named(stats: dict[str, RoleStat]) -> list[RoleStat]:
    """Named characters that never speak (excluded), sorted by mentions."""
    result = [stat for stat in stats.values() if stat.kind != "narrator" and not stat.speaking]
    return sorted(result, key=lambda stat: stat.mentions, reverse=True)


def distribution_report(rows: list[ScriptRow], cast: Cast | None = None) -> dict:
    stats = role_distribution(rows, cast)
    speaking = speaking_roles(stats)
    non_speaking = non_speaking_named(stats)
    total_seconds = sum(stat.est_seconds for stat in speaking)
    return {
        "total_seconds": round(total_seconds, 1),
        "total_minutes": round(total_seconds / 60.0, 2),
        "speaking_roles": [stat.to_dict() for stat in speaking],
        "non_speaking_named": [stat.to_dict() for stat in non_speaking],
    }
