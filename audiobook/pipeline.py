"""Orchestrate the front-end: clean -> cast -> extract -> script.csv."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .agent import RoleAgent
from .canonical import canonicalize_rows
from .cleaning import Chapter, normalize_text, read_text, split_chapters
from .duration import MAX_SEGMENT_SECONDS, estimate_text_duration
from .extract import Unit, extract_chapter
from .instructions import emotion_multiplier, render_instruction
from .llm import LLMClient
from .postprocess import merge_adjacent_narration
from .schema import Cast, ScriptRow, write_script, write_script_json, write_script_sqlite
from .segment import segment_units
from .stats import distribution_report
from .textnorm import clean_for_llm

CONFIDENCE_THRESHOLD = 0.6


@dataclass
class BuildResult:
    script_path: Path
    cast_path: Path
    qa_path: Path
    stats_path: Path
    chapters: int
    rows: int
    issues: int


def _write_chapters(chapters: list[Chapter], out_dir: Path) -> None:
    chapter_dir = out_dir / "chapters"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for chapter in chapters:
        (chapter_dir / f"ch{chapter.chapter_id:03d}.txt").write_text(chapter.text + "\n", encoding="utf-8")


def _assign_voices(cast: Cast, voices: dict[str, str] | None) -> None:
    for label, path in (voices or {}).items():
        role = cast.resolve(label)
        if role is not None:
            role.voice_ref = path


def _rows_for_units(chapter: Chapter, units: list[Unit], cast: Cast, model_id: str, start_order: int) -> list[ScriptRow]:
    """One ``Unit`` -> one ``ScriptRow`` (1:1).

    Segmentation already happened at the source-aligned unit level, so raw_text and
    tts_text belong to the same span; re-splitting the string here (and re-attaching
    the whole unit raw to every part) is what used to corrupt the audit trail.
    """
    rows: list[ScriptRow] = []
    for unit_index, unit in enumerate(units, start=1):
        tts = unit.tts_text
        if not any(char.isalnum() for char in tts):
            continue  # never let a punctuation-only fragment become a TTS row
        role = cast.roles.get(unit.role_id)
        row = ScriptRow(
            order=start_order + unit_index,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            seg_id=f"ch{chapter.chapter_id:03d}_s{unit_index:04d}",
            kind=unit.kind,
            role_id=unit.role_id,
            role_name=unit.role_name,
            raw_text=unit.raw_text,
            tts_text=tts,
            punct_edited=False,
            break_level=unit.break_level,
            auk_task="zero_shot_tts",
            voice_ref=role.voice_ref if role else "",
            style_desc=role.style_desc if role else "",
            emotion=unit.emotion,
            emotion_multiplier=emotion_multiplier(unit.emotion),
            target_duration_s=round(estimate_text_duration(tts), 3),
            duration_source="est",
            pe_instruction=render_instruction("zero_shot_tts", tts),
            extract_conf=unit.confidence,
            needs_pass2=unit.confidence < CONFIDENCE_THRESHOLD,
            prompt_id=unit.prompt_id,
            prompt_hash=unit.prompt_hash,
            model_id=model_id,
        )
        for flag in unit.flags:
            row.add_flag(flag)
        _qa_row(row)
        rows.append(row)
    return rows


def build_rows(
    chapter: Chapter,
    cast: Cast,
    client: LLMClient | None,
    model_id: str,
    start_order: int = 0,
) -> list[ScriptRow]:
    """Extract/agent/segment one chapter into source-aligned rows (1:1 with units)."""
    units = extract_chapter(chapter, cast, client)
    if client is not None:
        RoleAgent(cast, client).run(units)
        units = segment_units(units, cast, client)
    return _rows_for_units(chapter, units, cast, model_id, start_order)


def _qa_row(row: ScriptRow) -> None:
    if not row.role_id:
        row.add_flag("unresolved_role")
    if not row.voice_ref:
        row.add_flag("no_voice_ref")
    if row.target_duration_s > MAX_SEGMENT_SECONDS:
        row.add_flag("over_length")
    if row.needs_pass2:
        row.add_flag("low_confidence")


def _qa_report(rows: list[ScriptRow]) -> dict:
    flags: dict[str, list[str]] = {}
    for row in rows:
        for flag in filter(None, row.flags.split(";")):
            flags.setdefault(flag, []).append(row.seg_id)
    issues = sum(len(seg_ids) for seg_ids in flags.values())
    return {"rows": len(rows), "issues": issues, "flags": flags}


def _load_cast(
    out_dir: Path,
    cast_source: str | Path | None,
    refresh_cast: bool,
) -> tuple[Cast, Path]:
    """Return the cast, persisting it to ``out_dir/cast.json``.

    Precedence: explicit ``cast_source`` > existing ``out_dir/cast.json`` (unless
    ``refresh_cast``) > narrator-only fallback. The real dictionary is built by
    ``scripts/finalize_roster.py``; reusing the file keeps the cast stable across
    runs and lets a human hand-edit roles/aliases/voice_ref.
    """
    cast_file = out_dir / "cast.json"
    source = Path(cast_source) if cast_source else None
    if source is not None:
        cast = Cast.load(source)
    elif cast_file.exists() and not refresh_cast:
        cast = Cast.load(cast_file)
    else:
        cast = Cast()
        cast.narrator()
        print("[cast] no cast provided -> narrator-only fallback", file=sys.stderr)
    cast.save(cast_file)
    return cast, cast_file


def build_script(
    input_path: str | Path,
    out_dir: str | Path,
    *,
    client: LLMClient | None = None,
    voices: dict[str, str] | None = None,
    model_id: str = "",
    cast_source: str | Path | None = None,
    refresh_cast: bool = False,
) -> BuildResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_text(input_path)
    source = normalize_text(raw)
    clean = clean_for_llm(source)
    (out_dir / "source.txt").write_text(source, encoding="utf-8")
    (out_dir / "clean.txt").write_text(clean, encoding="utf-8")

    chapters = split_chapters(clean)
    _write_chapters(chapters, out_dir)

    cast, cast_path = _load_cast(out_dir, cast_source, refresh_cast)
    _assign_voices(cast, voices)
    cast.save(cast_path)

    resolved_model = model_id or (client.model_id if client else "none")
    rows: list[ScriptRow] = []
    for index, chapter in enumerate(chapters, start=1):
        chapter_rows = build_rows(chapter, cast, client, resolved_model, len(rows))
        rows.extend(chapter_rows)
        print(
            f"[extract] {index}/{len(chapters)} {chapter.title[:24]} rows={len(chapter_rows)} total={len(rows)}",
            file=sys.stderr,
            flush=True,
        )

    canonicalize_rows(rows, cast)

    if os.environ.get("AUDIOBOOK_MERGE_NARRATION", "1").lower() not in ("0", "off", "false", "no"):
        merged_dicts, merged_pairs = merge_adjacent_narration([asdict(row) for row in rows])
        if merged_pairs:
            rows = [ScriptRow(**item) for item in merged_dicts]
            for index, row in enumerate(rows, start=1):
                row.order = index  # keep #order contiguous after absorbing rows
            print(
                f"[merge] narration merged {len(merged_pairs)} pair(s) -> {len(rows)} rows",
                file=sys.stderr,
                flush=True,
            )

    script_path = out_dir / "script.csv"
    write_script(rows, script_path)
    write_script_json(rows, out_dir / "script.json")
    write_script_sqlite(rows, out_dir / "script.sqlite")

    report = _qa_report(rows)
    qa_path = out_dir / "qa_report.json"
    qa_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    stats_path = out_dir / "role_stats.json"
    stats_path.write_text(json.dumps(distribution_report(rows, cast), ensure_ascii=False, indent=2), encoding="utf-8")

    return BuildResult(
        script_path=script_path,
        cast_path=cast_path,
        qa_path=qa_path,
        stats_path=stats_path,
        chapters=len(chapters),
        rows=len(rows),
        issues=report["issues"],
    )
