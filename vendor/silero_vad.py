"""``silero_vad`` shim backed by faster-whisper's bundled Silero VAD (onnxruntime).

Lets the pristine AuK PE code (``from silero_vad import ...``) work without the
``silero-vad`` / ``torchaudio`` packages.
"""

from __future__ import annotations

import numpy as np
import torch


def load_silero_vad(*args, **kwargs):
    return object()


def get_speech_timestamps(
    audio,
    model=None,
    sampling_rate: int = 16000,
    return_seconds: bool = False,
    **kwargs,
):
    from faster_whisper.vad import VadOptions, get_speech_timestamps as _get_speech_timestamps

    if isinstance(audio, torch.Tensor):
        audio = audio.detach().to(torch.float32).cpu().numpy()
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    stamps = _get_speech_timestamps(signal, VadOptions(), sampling_rate=int(sampling_rate))
    if return_seconds:
        return [{"start": item["start"] / sampling_rate, "end": item["end"] / sampling_rate} for item in stamps]
    return stamps
