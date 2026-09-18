#!/usr/bin/env python3
"""Manufacture per-role reference voices with Breeze TTS 2 voice design.

For every speaking role in a ``script.csv`` (plus the narrator):
1. voice-design a reference clip from a natural-language description;
2. use the exact design sample text as the reference transcript (cloning requires it);
3. write ``voicebank.json`` (role -> wav) and ``voicebank_meta.json`` (role -> meta).

The result feeds ``scripts/render_book.py --voices ... --voice-meta ...``.

Examples:
    python scripts/build_voicebank_breeze.py --book mybook --styles styles.json
    python scripts/build_voicebank_breeze.py --book mybook --script outputs/mybook/script.csv --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_ROOT))

for _key in list(os.environ):
    if "proxy" in _key.lower():
        del os.environ[_key]
os.chdir(APP_ROOT)

from audiobook.llm import LLMClient, config_from_env  # noqa: E402
from audiobook.tts import BreezeConfig, BreezeRenderer  # noqa: E402
from audiobook.schema import read_script  # noqa: E402

DESIGN_CFG = 4.0  # official voice-design guidance
MAX_SAMPLE_CHARS = 40  # keep the reference clip short (~2-9 s)
NARRATOR_SAMPLE = "故事，要从很久以前说起。"
DEFAULT_STYLE = "日常说话，语气自然。"
# The narrator is frozen in voices/narrator.json (git-tracked); characters only vary by age+gender.
NARRATOR_JSON = APP_ROOT / "voices" / "narrator.json"

# Voice profiles (age/gender) are assigned HERE, when the final cast is known -- not during
# marking. `voice_profiles.json` is just a cache; delete it (or --force-profiles) to redo.
PROFILE_SYSTEM = """你在给一部小说的有声书定声线档案。根据人物名、别名和部分台词，只判断两件事：
- age：少年|青年|中年|老年
- gender：男|女
依据：台词自称、他人称呼（爷爷/少女/老仆 等）、身份与上下文；拿不准取最接近的一档。
只输出 JSON：{"规范名": {"age": "青年", "gender": "男"}}"""

AGE_CHOICES = ("少年", "青年", "中年", "老年")
GENDER_CHOICES = ("男", "女")
PROFILE_BATCH = 30  # roles per LLM call (keeps the request small)


def valid_profile(profile) -> bool:
    return isinstance(profile, dict) and profile.get("age") in AGE_CHOICES and profile.get("gender") in GENDER_CHOICES


def load_narrator() -> dict:
    if NARRATOR_JSON.is_file():
        return json.loads(NARRATOR_JSON.read_text(encoding="utf-8"))
    return {}


def load_profiles(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def voice_instruction(profile: dict | None) -> str:
    """Only age + gender are effective; everyone speaks everyday Mandarin."""
    if not profile:
        return ""
    age = str(profile.get("age", "")).strip()
    gender = str(profile.get("gender", "")).strip()
    if not age or not gender:
        return ""
    return f"一位{age}{gender}性，日常说话，语气自然。"


def ensure_profiles(script_dir: Path, lines: dict[str, list[str]], kinds: dict[str, str], force: bool = False) -> dict[str, dict]:
    """Assign age/gender to every character with the LLM (voicebank stage, not marking).

    The whole cast is known here, so the only input needed is the character info: canonical
    name + aliases + a few lines. Existing profiles are kept unless ``force``; without an LLM
    the voicebank still works with the default style.
    """
    path = script_dir / "voice_profiles.json"
    profiles = load_profiles(path)
    characters = [name for name, kind in kinds.items() if kind == "character"]
    missing = [name for name in characters if force or not valid_profile(profiles.get(name))]
    if not missing:
        return profiles
    config = config_from_env()
    if config is None:
        print(f"[profiles] {len(missing)} 个角色缺年龄/性别，但未配置 LLM；用默认描述造声", flush=True)
        return profiles
    roles: dict[str, list[str]] = {}
    roles_path = script_dir / "roles.json"
    if roles_path.is_file():
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
    alias_to_canonical: dict[str, str] = {}
    for canonical, labels in roles.items():
        alias_to_canonical[canonical] = canonical
        for label in labels:
            alias_to_canonical.setdefault(str(label), canonical)
    llm = LLMClient(config)
    added = 0
    for start in range(0, len(missing), PROFILE_BATCH):
        batch = missing[start : start + PROFILE_BATCH]
        block: list[str] = []
        for name in batch:
            aliases = "、".join(str(label) for label in roles.get(name, []) if label and label != name)
            block.append(f"{name}" + (f"（别名：{aliases}）" if aliases else ""))
            for line in lines.get(name, [])[:6]:
                text = line.strip().replace("\n", " ")
                if text:
                    block.append(f"    · {text[:60]}")
        prompt = "【人物】\n" + "\n".join(block) + "\n\n只给这些人物输出 JSON（键=规范名）。"
        try:
            result = llm.chat_json(PROFILE_SYSTEM, prompt, thinking=False)
        except Exception as error:  # noqa: BLE001 - profiles are best-effort
            print(f"[profiles] LLM 失败：{str(error)[:120]}", flush=True)
            break
        if not isinstance(result, dict):
            continue
        for key, entry in result.items():
            name = alias_to_canonical.get(str(key).strip())
            if name is None or name not in batch or not isinstance(entry, dict):
                continue
            age = str(entry.get("age", "")).strip()
            gender = str(entry.get("gender", "")).strip()
            if age in AGE_CHOICES and gender in GENDER_CHOICES:
                profiles[name] = {**(profiles.get(name) or {}), "age": age, "gender": gender}
                added += 1
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[profiles] 新增/更新 {added} 个 -> {path}", flush=True)
    return profiles


def role_seed(role_name: str, base_seed: int) -> int:
    """Deterministic per-role seed: same role -> same voice, different roles differ."""
    digest = hashlib.sha1(role_name.encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manufacture role voices with Breeze voice design")
    parser.add_argument("--book", required=True, help="book name (outputs/<book>)")
    parser.add_argument("--script", default=None, help="script.csv (default outputs/<book>/script.csv)")
    parser.add_argument("--out", default=None, help="output dir (default outputs/<book>)")
    parser.add_argument("--styles", default=None, help="JSON {role: voice description} overrides")
    parser.add_argument("--samples", default=None, help="JSON {role: reference text} overrides (prefer neutral declaratives)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--narrator-gender", default=None, help="frozen narrator to use (男/女)")
    parser.add_argument("--no-llm-profiles", action="store_true", help="keep existing profiles; do not ask the LLM")
    parser.add_argument("--force-profiles", action="store_true", help="re-assign age/gender for every character")
    parser.add_argument("--force", action="store_true", help="redo roles that already have a reference")
    parser.add_argument("--breeze-url", default=None)
    parser.add_argument("--breeze-no-start", action="store_true")
    return parser.parse_args()


def _clip(text: str) -> str:
    if len(text) <= MAX_SAMPLE_CHARS:
        return text
    head = text[:MAX_SAMPLE_CHARS]
    end = -1
    for mark in "。！？，、":
        end = max(end, head.rfind(mark))
    return head[: end + 1] if end >= 8 else head


def sample_for(role_name: str, lines: list[str], kind: str) -> str:
    """Prefer a neutral declarative line: questions/interjections leak their prosody
    into every clone, so design references should be calm statements."""
    if kind == "narrator" or role_name in ("旁白", "叙述", "narrator"):
        return NARRATOR_SAMPLE
    clean = [line.strip() for line in lines if line and line.strip()]
    declaratives = [line for line in clean if line.endswith(("。", "！")) and len(line) >= 10]
    for line in declaratives:
        return _clip(line)
    for line in clean:
        if len(line) >= 10:
            return _clip(line)
    return f"我是{role_name}，很高兴见到你。"


def main() -> None:
    args = parse_args()
    book = Path("outputs") / args.book
    script = Path(args.script) if args.script else book / "script.csv"
    out_dir = Path(args.out) if args.out else book
    refs_dir = out_dir / "references"
    refs_dir.mkdir(parents=True, exist_ok=True)

    rows = read_script(script)
    lines: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for row in rows:
        name = row.role_name or row.role_id
        lines.setdefault(name, []).append(row.tts_text)
        kinds.setdefault(name, "narrator" if row.role_id == "narrator" or name in ("旁白", "叙述") else "character")

    styles = json.loads(Path(args.styles).read_text(encoding="utf-8")) if args.styles else {}
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8")) if args.samples else {}
    if args.no_llm_profiles:
        profiles = load_profiles(book / "script" / "voice_profiles.json")
    else:  # age/gender belong to the voicebank stage; the marking stage only keeps names
        profiles = ensure_profiles(book / "script", lines, kinds, force=args.force_profiles)

    config = BreezeConfig(cfg_scale=DESIGN_CFG, seed=args.seed)
    if args.breeze_url:
        config.base_url = args.breeze_url.rstrip("/")
    renderer = BreezeRenderer(config)
    if not args.breeze_no_start:
        renderer.start()

    voices: dict[str, str] = {}
    meta: dict[str, dict] = {}
    try:
        narrator = load_narrator()
        for name, role_lines in lines.items():
            destination = refs_dir / f"{name}.wav"
            profile = profiles.get(name) or {}
            if kinds[name] == "narrator":
                gender = args.narrator_gender or narrator.get("default", "男")
                entry = (narrator.get("voices") or {}).get(gender)
                source = APP_ROOT / "voices" / entry["file"] if entry else None
                if entry and source is not None and source.is_file():
                    if args.force or not destination.is_file():
                        shutil.copy(source, destination)
                    print(f"[narrator] {name} <- {source.name} ({gender})", flush=True)
                    voices[name] = str(destination)
                    meta[name] = {
                        "wav_path": str(destination),
                        "ref_text": narrator.get("ref_text", ""),
                        "style_desc": entry.get("instruction", ""),
                        "seed": narrator.get("seed"),
                    }
                    continue
            style = styles.get(name) or voice_instruction(profile) or DEFAULT_STYLE
            sample = samples.get(name) or profile.get("sample") or sample_for(name, role_lines, kinds[name])
            seed = role_seed(name, args.seed)
            if destination.is_file() and not args.force:
                print(f"[skip] {name} 已有参考音", flush=True)
            else:
                print(f"[design] {name} ({kinds[name]}) seed={seed} <- {style!r}", flush=True)
                renderer.config.seed = seed
                audio, rate = renderer.design_voice(sample, style)
                if not np.isfinite(audio).all():
                    raise RuntimeError(f"non-finite design audio for {name}")
                sf.write(str(destination), np.clip(audio, -1.0, 1.0), rate, subtype="PCM_16")
            # The reference transcript is the text we asked it to say, not ASR (ASR
            # typos would silently degrade cloning; the text is known exactly).
            print(f"    -> {destination} | ref_text {sample[:40]!r}", flush=True)
            voices[name] = str(destination)
            meta[name] = {
                "wav_path": str(destination),
                "ref_text": sample,
                "style_desc": style,
                "seed": seed,
            }
    finally:
        renderer.close()

    (out_dir / "voicebank.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "voicebank_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[voicebank] {len(voices)} voices -> {out_dir / 'voicebank.json'}")


if __name__ == "__main__":
    main()
