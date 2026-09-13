"""Orchestrate the front-end: clean -> cast -> extract -> script.csv."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .agent import RoleAgent
from .canonical import canonicalize_rows
from .cast import discover_cast
from .cleaning import Chapter, normalize_text, read_text, split_chapters
from .duration import MAX_SEGMENT_SECONDS, estimate_text_duration, segment_tts_text
from .extract import Unit, extract_chapter
from .instructions import emotion_multiplier, render_instruction
from .llm import LLMClient
from .schema import Cast, ScriptRow, write_script, write_script_json, write_script_sqlite
from .stats import distribution_report

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
    rows: list[ScriptRow] = []
    order = start_order
    for unit_index, unit in enumerate(units, start=1):
        role = cast.roles.get(unit.role_id)
        segments = segment_tts_text(unit.tts_text)
        for part_index, segment in enumerate(segments, start=1):
            order += 1
            suffix = "" if len(segments) == 1 else chr(ord("a") + part_index - 1)
            row = ScriptRow(
                order=order,
                chapter_id=chapter.chapter_id,
                chapter_title=chapter.title,
                seg_id=f"ch{chapter.chapter_id:03d}_s{unit_index:04d}{suffix}",
                kind=unit.kind,
                role_id=unit.role_id,
                role_name=unit.role_name,
                raw_text=unit.raw_text,
                tts_text=segment.text,
                punct_edited=segment.punct_edited,
                auk_task="zero_shot_tts",
                voice_ref=role.voice_ref if role else "",
                style_desc=role.style_desc if role else "",
                emotion=unit.emotion,
                emotion_multiplier=emotion_multiplier(unit.emotion),
                target_duration_s=round(estimate_text_duration(segment.text), 3),
                duration_source="est",
                pe_instruction=render_instruction("zero_shot_tts", segment.text),
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


def _load_or_discover_cast(
    chapters: list[Chapter],
    out_dir: Path,
    client: LLMClient | None,
    cast_source: str | Path | None,
    refresh_cast: bool,
) -> tuple[Cast, Path]:
    """Return the cast, persisting it to ``out_dir/cast.json``.

    Precedence: explicit ``cast_source`` > existing ``out_dir/cast.json`` (unless
    ``refresh_cast``) > fresh discovery. Reusing the file keeps the cast stable
    across runs and lets a human hand-edit roles/aliases/voice_ref.
    """
    cast_file = out_dir / "cast.json"
    source = Path(cast_source) if cast_source else None
    if source is not None:
        cast = Cast.load(source)
    elif cast_file.exists() and not refresh_cast:
        cast = Cast.load(cast_file)
    else:
        cast = discover_cast(chapters, client)
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
    clean = normalize_text(raw)
    (out_dir / "clean.txt").write_text(clean, encoding="utf-8")

    chapters = split_chapters(clean)
    _write_chapters(chapters, out_dir)

    cast, cast_path = _load_or_discover_cast(chapters, out_dir, client, cast_source, refresh_cast)
    _assign_voices(cast, voices)
    cast.save(cast_path)

    resolved_model = model_id or (client.model_id if client else "none")
    rows: list[ScriptRow] = []
    for index, chapter in enumerate(chapters, start=1):
        units = extract_chapter(chapter, cast, client)
        if client is not None:
            RoleAgent(cast, client).run(units)
        rows.extend(_rows_for_units(chapter, units, cast, resolved_model, len(rows)))
        print(
            f"[extract] {index}/{len(chapters)} {chapter.title[:24]} units={len(units)} rows={len(rows)}",
            file=sys.stderr,
            flush=True,
        )

    canonicalize_rows(rows, cast)

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
