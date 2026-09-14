"""Assemble per-row wavs into chapter tracks, then (optionally) the whole book.

Loudness is normalized to ``TARGET_LUFS`` (EBU-ish) with a hard peak ceiling, so
chapters can be concatenated without level jumps between speakers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_LUFS = -16.0
PEAK_CEILING = 0.95
SAME_SPEAKER_GAP = 0.25
SPEAKER_CHANGE_GAP = 0.40
KIND_CHANGE_GAP = 0.50


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sample_rate)


def normalize_loudness(
    audio: np.ndarray, sample_rate: int, target: float = TARGET_LUFS, peak: float = PEAK_CEILING
) -> np.ndarray:
    output = audio.astype(np.float64)
    try:
        import pyloudnorm as pyln

        measured = float(pyln.Meter(sample_rate).integrated_loudness(output))
        if np.isfinite(measured) and measured > -70.0:
            output *= 10.0 ** ((target - measured) / 20.0)
    except Exception:  # noqa: BLE001 - loudness is best-effort
        pass
    ceiling = float(np.max(np.abs(output))) if output.size else 0.0
    if ceiling > peak:
        output *= peak / ceiling
    return output.astype(np.float32)


def _gap_seconds(previous, current) -> float:
    if previous is None:
        return 0.0
    if previous.kind != current.kind:
        return KIND_CHANGE_GAP
    if previous.role_id != current.role_id:
        return SPEAKER_CHANGE_GAP
    return SAME_SPEAKER_GAP


def assemble_rows(
    items: list[tuple[Path, object]],
    out_path: str | Path,
    *,
    sample_rate: int = 24000,
    normalize: bool = True,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    """Concatenate ``(wav_path, row)`` in order with speaker-aware gaps."""
    pieces: list[np.ndarray] = []
    previous = None
    for path, row in items:
        audio, _ = load_mono(path)
        gap = _gap_seconds(previous, row)
        if gap > 0:
            pieces.append(np.zeros(int(sample_rate * gap), dtype=np.float32))
        pieces.append(audio)
        previous = row
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if normalize and merged.size:
        merged = normalize_loudness(merged, sample_rate, target_lufs)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), merged, sample_rate, subtype="PCM_16")
    return merged


def assemble_chapters(
    chapter_paths: list[tuple[int, Path]],
    out_path: str | Path,
    *,
    sample_rate: int = 24000,
    gap: float = 0.8,
    normalize: bool = True,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for index, (_, path) in enumerate(chapter_paths):
        audio, _ = load_mono(path)
        if index:
            pieces.append(np.zeros(int(sample_rate * gap), dtype=np.float32))
        pieces.append(audio)
    merged = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    if normalize and merged.size:
        merged = normalize_loudness(merged, sample_rate, target_lufs)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), merged, sample_rate, subtype="PCM_16")
    return merged
