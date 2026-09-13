"""Data model for the audiobook script and global cast."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

KINDS = ("narration", "dialogue", "monologue")

EMOTIONS = ("happy", "angry", "sad", "fearful", "surprised", "disgusted", "calm", "excited")

EMOTION_MULTIPLIERS = {
    "sad": 1.22,
    "fearful": 1.16,
    "happy": 1.06,
    "angry": 1.06,
    "surprised": 1.06,
    "disgusted": 1.06,
    "calm": 1.06,
    "excited": 1.06,
}

NARRATOR_ID = "narrator"

_ALIAS_SPLIT_RE = re.compile(r"[\s,，、/|]+")
_TITLE_RE = re.compile(
    r"(骑士|侯爵|伯爵|子爵|男爵|将军|大人|先生|女士|小姐|夫人|陛下|国王|女王|法师|术士|"
    r"侍女|女仆|先祖|老祖宗|殿下|阁下|队长|团长|会长|长老|祭司|神父|修女|少爷|老爷|太太|"
    r"公子|姑娘|少女|少年|大汉|青年|女孩|男孩|大公|公爵|亲王|王子|公主)"
)
_CORE_SEP_RE = re.compile(r"[·・.\-_—\s]+")


def core_name(name: str) -> str:
    """Strip titles and separators to compare Chinese name variants.

    e.g. ``拜伦·柯克`` -> ``拜伦柯克``, ``拜伦骑士`` -> ``拜伦``.
    """
    core = _TITLE_RE.sub("", name)
    core = _CORE_SEP_RE.sub("", core)
    return core.strip()


def slugify(name: str) -> str:
    """Stable ASCII-ish id for a role."""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name.strip().lower()).strip("_")
    return cleaned or "role"


@dataclass
class Role:
    role_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    kind: str = "character"  # narrator | character
    description: str = ""
    style_desc: str = ""
    voice_ref: str = ""
    exemplars: list[str] = field(default_factory=list)

    def matches(self, label: str) -> bool:
        needle = label.strip()
        return needle == self.name or needle == self.role_id or needle in self.aliases


@dataclass
class Cast:
    roles: dict[str, Role] = field(default_factory=dict)

    def add(self, role: Role) -> Role:
        existing = self.roles.get(role.role_id)
        if existing is None:
            self.roles[role.role_id] = role
            return role
        for alias in role.aliases:
            if alias and alias not in existing.aliases:
                existing.aliases.append(alias)
        if not existing.description and role.description:
            existing.description = role.description
        for sample in role.exemplars:
            if sample not in existing.exemplars:
                existing.exemplars.append(sample)
        return existing

    def resolve(self, label: str) -> Role | None:
        needle = label.strip()
        if not needle:
            return None
        for role in self.roles.values():
            if role.matches(needle):
                return role
        core = core_name(needle)
        if len(core) >= 2:
            for role in self.roles.values():
                role_core = core_name(role.name)
                if len(role_core) >= 2 and (core in role_core or role_core in core):
                    return role
        return None

    def narrator(self) -> Role:
        role = self.roles.get(NARRATOR_ID)
        if role is None:
            role = Role(role_id=NARRATOR_ID, name="旁白", kind="narrator")
            self.roles[NARRATOR_ID] = role
        return role

    def to_dict(self) -> dict:
        return {"roles": [asdict(role) for role in self.roles.values()]}

    @classmethod
    def from_dict(cls, data: dict) -> Cast:
        cast = cls()
        for item in data.get("roles", []):
            role = Role(**{key: item.get(key) for key in Role.__dataclass_fields__ if key in item})
            cast.roles[role.role_id] = role
        return cast

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Cast:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def consolidate(self) -> None:
        """Merge roles whose names/aliases overlap exactly (case-insensitive).

        Handles duplicates that ``merge_candidates`` misses when two candidate
        entries only share an alias indirectly (e.g. "高文" vs "高文·塞西尔").
        Semantic merges still need the LLM pass in :mod:`audiobook.cast`.
        """
        parent = {role_id: role_id for role_id in self.roles}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        owner: dict[str, str] = {}
        for role_id, role in self.roles.items():
            for token in {role_id, role.name, *role.aliases}:
                key = token.strip().lower()
                if not key:
                    continue
                if key in owner:
                    union(owner[key], role_id)
                else:
                    owner[key] = role_id

        ids = list(self.roles)
        cores = {role_id: core_name(self.roles[role_id].name) for role_id in ids}
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                core_left, core_right = cores[left], cores[right]
                if len(core_left) >= 2 and len(core_right) >= 2:
                    if core_left in core_right or core_right in core_left:
                        union(left, right)

        grouped: dict[str, list[Role]] = {}
        for role_id, role in self.roles.items():
            grouped.setdefault(find(role_id), []).append(role)

        merged: dict[str, Role] = {}
        for members in grouped.values():
            canonical = max(members, key=lambda role: (len(role.name), len(role.aliases), len(role.exemplars)))
            for role in members:
                if role is canonical:
                    continue
                for alias in [role.name, *role.aliases]:
                    if alias and alias != canonical.name and alias not in canonical.aliases:
                        canonical.aliases.append(alias)
                for sample in role.exemplars:
                    if sample not in canonical.exemplars:
                        canonical.exemplars.append(sample)
                if not canonical.description and role.description:
                    canonical.description = role.description
            canonical.aliases = [alias for alias in sorted(canonical.aliases) if alias and alias != canonical.name]
            merged[canonical.role_id] = canonical
        self.roles = merged

    def merge_candidates(self, candidates: list[dict]) -> None:
        """Merge LLM candidate roles, reconciling aliases into one role each."""
        for candidate in candidates:
            name = (candidate.get("name") or "").strip()
            if not name:
                continue
            aliases = [alias for alias in _as_list(candidate.get("aliases")) if alias and alias != name]
            role = None
            for existing in self.roles.values():
                if existing.matches(name) or any(existing.matches(alias) for alias in aliases):
                    role = existing
                    break
            if role is None:
                is_narrator = bool(candidate.get("is_narrator")) or name in ("旁白", "叙述", "旁白君")
                role_id = NARRATOR_ID if is_narrator else slugify(name)
                role = Role(role_id=role_id, name=name, kind="narrator" if is_narrator else "character")
            role.aliases = sorted({alias for alias in [*role.aliases, *aliases] if alias != role.name})
            if not role.description:
                role.description = (candidate.get("description") or "").strip()
            for sample in _as_list(candidate.get("example_utterances")):
                sample = sample.strip()
                if sample and sample not in role.exemplars:
                    role.exemplars.append(sample)
            self.add(role)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in _ALIAS_SPLIT_RE.split(value) if part]
    return [str(item) for item in value]


@dataclass
class ScriptRow:
    order: int = 0
    chapter_id: int = 0
    chapter_title: str = ""
    seg_id: str = ""
    kind: str = "narration"
    role_id: str = ""
    role_name: str = ""
    raw_text: str = ""
    tts_text: str = ""
    punct_edited: bool = False
    break_level: str = ""
    auk_task: str = "zero_shot_tts"
    voice_ref: str = ""
    style_desc: str = ""
    emotion: str = ""
    emotion_multiplier: float = 1.0
    base_seg_id: str = ""
    speed: float = 1.0
    volume_gain_db: float = 0.0
    pitch_semitones: float = 0.0
    target_duration_s: float = 0.0
    duration_source: str = "est"
    pe_instruction: str = ""
    seed: int = -1
    nfe: int = 0
    cfg: float = 0.0
    extract_conf: float = 1.0
    needs_pass2: bool = False
    prompt_id: str = ""
    prompt_hash: str = ""
    model_id: str = ""
    flags: str = ""
    notes: str = ""

    def add_flag(self, flag: str) -> None:
        existing = [part for part in self.flags.split(";") if part]
        if flag not in existing:
            existing.append(flag)
            self.flags = ";".join(existing)


_BOOL_FIELDS = {"punct_edited", "needs_pass2"}
_INT_FIELDS = {"order", "chapter_id", "seed", "nfe"}
_FLOAT_FIELDS = {"emotion_multiplier", "speed", "volume_gain_db", "pitch_semitones", "target_duration_s", "cfg", "extract_conf"}


def write_script(rows: list[ScriptRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f.name for f in fields(ScriptRow)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_script_json(rows: list[ScriptRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(row) for row in rows], ensure_ascii=False, indent=2), encoding="utf-8")


_SQLITE_TYPES = {
    "order": "INTEGER",
    "chapter_id": "INTEGER",
    "seed": "INTEGER",
    "nfe": "INTEGER",
    "emotion_multiplier": "REAL",
    "speed": "REAL",
    "volume_gain_db": "REAL",
    "pitch_semitones": "REAL",
    "target_duration_s": "REAL",
    "cfg": "REAL",
    "extract_conf": "REAL",
    "punct_edited": "INTEGER",
    "needs_pass2": "INTEGER",
}


def write_script_sqlite(rows: list[ScriptRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [(f.name, _SQLITE_TYPES.get(f.name, "TEXT")) for f in fields(ScriptRow)]
    ddl = ", ".join(f'"{name}" {sql_type}' for name, sql_type in columns)
    placeholders = ", ".join("?" for _ in columns)
    names = ", ".join(f'"{name}"' for name, _ in columns)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE IF EXISTS script")
        connection.execute(f"CREATE TABLE script ({ddl})")
        for row in rows:
            values = [int(v) if isinstance(v, bool) else v for v in asdict(row).values()]
            connection.execute(f"INSERT INTO script ({names}) VALUES ({placeholders})", values)


def read_script(path: str | Path) -> list[ScriptRow]:
    rows: list[ScriptRow] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            row = ScriptRow()
            for key, value in record.items():
                if key in _BOOL_FIELDS:
                    setattr(row, key, str(value).strip().lower() in ("1", "true", "yes"))
                elif key in _INT_FIELDS:
                    setattr(row, key, int(float(value or 0)))
                elif key in _FLOAT_FIELDS:
                    setattr(row, key, float(value or 0))
                elif hasattr(row, key):
                    setattr(row, key, value)
            rows.append(row)
    return rows
