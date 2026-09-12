"""``funasr`` shim backed by faster-whisper.

The pristine AuK PE code lazily does ``from funasr import AutoModel`` and expects
SenseVoice-style output (``{"text": "<|zh|>..."}``). This shim keeps that
interface but runs faster-whisper under the hood, so no funasr/modelscope.
"""

from __future__ import annotations

import os


def _resolve_model(name: str | None) -> str:
    override = os.environ.get("AUK_ASR_MODEL")
    if override:
        return override
    if not name or "SenseVoice" in name or "/" in name:
        return "small"
    return name


def _tag(language: str | None) -> str:
    language = (language or "").lower()
    if language == "en":
        return "en"
    if language in {"zh", "yue"}:
        return "zh"
    return "zh"


class AutoModel:
    def __init__(self, model: str | None = None, device: str = "cpu", ncpu: int = 4, **kwargs):
        from faster_whisper import WhisperModel

        self.model_name = _resolve_model(model)
        self._device = device or "cpu"
        self._compute_type = os.environ.get("AUK_ASR_COMPUTE_TYPE", "int8")
        self._model = WhisperModel(self.model_name, device=self._device, compute_type=self._compute_type)

    def generate(self, input=None, **kwargs):
        if isinstance(input, (list, tuple)):
            paths = list(input)
        else:
            paths = [input]
        results = []
        for path in paths:
            segments, info = self._model.transcribe(path, language=None, beam_size=1, vad_filter=False)
            text = "".join(segment.text for segment in segments).strip()
            results.append({"text": f"<|{_tag(getattr(info, 'language', None))}|>{text}"})
        return results
