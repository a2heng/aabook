"""Character dictionary -> extraction ``Cast``, plus gender-correct voice assignment.

The dictionary itself is produced by ``scripts/finalize_roster.py`` (frequency
discovery -> batched LLM filter -> full-book re-scan). This module bridges that
``roster.json`` into the extraction ``Cast`` and gives each person a
gender-correct reference wav from the optimized voice bank.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .llm import LLMClient
from .namefinder import FUNCTION, STOP
from .schema import Cast, Role, slugify

MALE_VOICES = [
    "青年才俊",
    "耿直君子",
    "温润君子",
    "高尚郎君",
    "高傲少爷",
    "高冷男神",
    "少年统帅",
    "逍遥浪客",
    "文弱书生",
    "霸道哥哥",
    "温吞少爷",
    "玉面公子",
]
FEMALE_VOICES = [
    "青春女神",
    "知性御姐",
    "俏皮姐姐",
    "温暖御姐",
    "甜美萌妹",
    "傲娇女王",
    "甜心少女",
    "娇媚女神",
    "稚嫩少女",
    "冰山女王",
    "纯情闺蜜",
    "开朗妹妹",
]
NARRATOR_VOICE = "温润君子"

# Fixed generic speakers for everyone not in the (small) main cast: the model never
# invents names; unknown dialogue is bucketed by gender into one of these four.
PASSERBY_ROLES = [
    {"id": "passerby_m1", "name": "路人男1", "gender": "男"},
    {"id": "passerby_m2", "name": "路人男2", "gender": "男"},
    {"id": "passerby_f1", "name": "路人女1", "gender": "女"},
    {"id": "passerby_f2", "name": "路人女2", "gender": "女"},
]
PASSERBY_NAMES = {spec["name"] for spec in PASSERBY_ROLES}


def _chat(client: LLMClient, system: str, user: str) -> dict | list | None:
    for thinking in (False, True):
        try:
            return client.chat_json(system, user, thinking=thinking)
        except Exception:  # noqa: BLE001 - try the other thinking mode
            continue
    return None


def infer_gender(corpus: str, forms: list[str]) -> str:
    """Cheap fallback when the LLM is unsure: gendered pronouns right after a name."""
    male = female = 0
    for form in forms:
        if not form:
            continue
        start = 0
        while True:
            found = corpus.find(form, start)
            if found < 0:
                break
            window = corpus[found + len(form) : found + len(form) + 8]
            if "她" in window:
                female += 1
            elif "他" in window:
                male += 1
            start = found + len(form)
    if male == 0 and female == 0:
        return "未知"
    return "男" if male >= female else "女"


def voice_pool(gender: str) -> list[str]:
    return FEMALE_VOICES if gender == "女" else MALE_VOICES


def assign_voice(name: str, gender: str, voice_dir: str | Path, used: set[str]) -> str:
    """Pick a gender-correct reference wav, avoiding collisions when possible."""
    directory = Path(voice_dir)
    pool = voice_pool(gender)
    digest = int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16)
    for offset in range(len(pool)):
        candidate = pool[(digest + offset) % len(pool)]
        path = directory / f"{candidate}.wav"
        if path.is_file() and (candidate not in used or offset == len(pool) - 1):
            used.add(candidate)
            return str(path)
    return str(directory / f"{pool[digest % len(pool)]}.wav")


_PRONOUNS = frozenset(
    {"他们", "她们", "我们", "你们", "咱们", "大家", "众人", "这位", "那位", "这些", "那些", "其中", "自己", "对方"}
)


def valid_form(form: str) -> bool:
    """A form usable as an alias/scan target: not a pronoun, stop word or function run."""
    return bool(form) and form not in _PRONOUNS and form not in STOP and not set(form) <= FUNCTION


def finalize(people: list[dict], voice_dir: str | Path) -> list[dict]:
    used: set[str] = {NARRATOR_VOICE}
    people = [person for person in people if person.get("name") not in ("旁白", "叙述") and person.get("id") != "narrator"]
    # A bare family surname ("塞西尔" from "高文·塞西尔") is not a person of its own.
    all_forms = [form for person in people for form in [person["name"], *person.get("aliases", [])] if form]
    surnames = {form.rsplit("·", 1)[-1] for form in all_forms if "·" in form}
    people = [person for person in people if person["name"] not in surnames]
    people = sorted(people, key=lambda p: (not p.get("chapters"), p["name"]))
    for person in people:
        person["aliases"] = [form for form in person.get("aliases", []) if valid_form(form)]
        person["scan_names"] = sorted({form for form in [person["name"], *person["aliases"]] if valid_form(form)}) or [
            person["name"]
        ]
        person["id"] = slugify(person["name"])
        person["voice_ref"] = assign_voice(person["name"], person.get("gender", "未知"), voice_dir, used)
    narrator = {
        "id": "narrator",
        "name": "旁白",
        "aliases": [],
        "gender": "男",
        "traits": ["叙述者"],
        "chapters": sorted({c for p in people for c in p.get("chapters", [])}),
        "role": "旁白",
        "voice_ref": str(Path(voice_dir) / f"{NARRATOR_VOICE}.wav"),
    }
    return [narrator, *people]


def load_main_names(cast_path: str | Path) -> list[str] | None:
    """The fixed main-cast allowlist next to a cast file (``main_cast.json``), if any."""
    path = Path(cast_path).parent / "main_cast.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return [str(item) for item in data]
    return None


def to_cast(roster: dict, main_names: list[str] | None = None) -> Cast:
    """Bridge the roster dictionary into the extraction ``Cast`` schema.

    ``main_names`` (if given) keeps only those people as named roles -- the fixed
    main cast with fixed voices; everyone else is served by the generic passerby
    roles appended below.
    """
    main = set(main_names) if main_names else None
    cast = Cast()
    narrator = cast.narrator()
    for person in roster.get("characters", []):
        if person.get("id") == "narrator" or person.get("name") == "旁白":
            narrator.voice_ref = person.get("voice_ref", narrator.voice_ref)
            continue
        if main is not None and person["name"] not in main:
            continue
        cast.add(
            Role(
                role_id=person.get("id") or slugify(person["name"]),
                name=person["name"],
                aliases=list(person.get("aliases", [])),
                kind="character",
                description="；".join(person.get("traits", [])),
                voice_ref=person.get("voice_ref", ""),
            )
        )
    voice_dir = Path(narrator.voice_ref).parent if narrator.voice_ref else Path("outputs/refs_bwe")
    used: set[str] = {Path(role.voice_ref).stem for role in cast.roles.values() if role.voice_ref}
    for spec in PASSERBY_ROLES:
        cast.add(
            Role(
                role_id=spec["id"],
                name=spec["name"],
                aliases=[],
                kind="character",
                description="路人/配角",
                voice_ref=assign_voice(spec["name"], spec["gender"], voice_dir, used),
            )
        )
    return cast
