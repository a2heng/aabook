"""Per-role voice bank.

No real recordings exist for each character, so we bootstrap one:

1. ``instruct_tts`` synthesizes a sample from the role's personality
   (``style_desc``/``description``) plus one of its lines (``exemplars``);
2. ASR (faster-whisper) transcribes that sample to get the reference text;
3. every later line is rendered with ``zero_shot_tts`` cloning that reference,
   so the voice stays stable and matches the character description.

Outputs ``voicebank.json`` (role -> wav) and ``voicebank_meta.json`` (role -> ref
text / style), both reusable by ``scripts/render_book.py --voices``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .duration import estimate_text_duration
from .instructions import INSTRUCT_TEMPLATE
from .renderer import Renderer
from .schema import Cast, Role

NARRATOR_STYLE = "沉稳、清晰、娓娓道来的旁白叙述，语速适中，情绪克制"
NARRATOR_SAMPLE = "故事，要从很久以前说起。"
DEFAULT_STYLE = "自然、清晰、有感情的说话方式"


class Asr:
    """Thin faster-whisper wrapper (optional; failures leave ref_text empty)."""

    def __init__(self, model: str = "small", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, path: str | Path) -> str:
        segments, _info = self._model.transcribe(str(path), language="zh", beam_size=1, vad_filter=False)
        return "".join(segment.text for segment in segments).strip()


@dataclass
class Reference:
    role_id: str
    name: str
    wav_path: str
    ref_text: str
    style_desc: str
    sample_text: str


def _style_for(role: Role) -> str:
    if role.kind == "narrator":
        return role.style_desc or NARRATOR_STYLE
    return role.style_desc or role.description or DEFAULT_STYLE


def _sample_for(role: Role) -> str:
    if role.kind == "narrator":
        return NARRATOR_SAMPLE
    for exemplar in role.exemplars:
        text = (exemplar or "").strip()
        if len(text) >= 4:
            return text
    return f"我是{role.name}，很高兴见到你。"


def build_reference(renderer: Renderer, role: Role, out_dir: str | Path, *, asr: Asr | None = None) -> Reference:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    style = _style_for(role)
    sample = _sample_for(role)
    instruction = INSTRUCT_TEMPLATE.format(style_desc=style, text=sample)
    seconds = min(20.0, max(1.0, estimate_text_duration(sample) * 1.2))

    messages = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    audio, sample_rate = renderer.engine.generate(
        messages,
        audio=None,
        gen_seconds=seconds,
        nfe=renderer.config.nfe,
        cfg_strength=renderer.config.cfg,
        seed=renderer.config.seed,
    )
    array = audio.detach().to("cpu").float().squeeze(0).numpy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"non-finite reference audio for {role.name}")
    wav_path = out_dir / f"{role.role_id}.wav"
    sf.write(str(wav_path), np.clip(array, -1.0, 1.0), sample_rate, subtype="FLOAT")

    ref_text = ""
    if asr is not None:
        try:
            ref_text = asr.transcribe(wav_path)
        except Exception:  # noqa: BLE001 - ASR is best-effort
            ref_text = ""
    return Reference(
        role_id=role.role_id,
        name=role.name,
        wav_path=str(wav_path),
        ref_text=ref_text,
        style_desc=style,
        sample_text=sample,
    )


def build_voice_bank(
    renderer: Renderer,
    cast: Cast,
    out_dir: str | Path,
    *,
    asr: Asr | None = None,
    log=None,
) -> dict[str, Reference]:
    bank: dict[str, Reference] = {}
    for role in cast.roles.values():
        if log:
            log(f"[voicebank] {role.name} ({role.kind}) ...")
        reference = build_reference(renderer, role, out_dir, asr=asr)
        bank[role.role_id] = reference
        if log:
            log(f"    text={reference.ref_text[:40]!r}")
    write_voice_bank(bank, Path(out_dir).parent)
    return bank


def write_voice_bank(bank: dict[str, Reference], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    voices = {ref.name: ref.wav_path for ref in bank.values()}
    meta = {
        ref.name: {
            "role_id": ref.role_id,
            "wav_path": ref.wav_path,
            "ref_text": ref.ref_text,
            "style_desc": ref.style_desc,
            "sample_text": ref.sample_text,
        }
        for ref in bank.values()
    }
    (out_dir / "voicebank.json").write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "voicebank_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
