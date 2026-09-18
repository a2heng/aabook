"""Data model for the audiobook script and global cast."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

NARRATOR_ID = "narrator"


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
        """Exact name / role_id / alias match only; canonicalization is the LLM's job."""
        needle = label.strip()
        if not needle:
            return None
        for role in self.roles.values():
            if role.matches(needle):
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
