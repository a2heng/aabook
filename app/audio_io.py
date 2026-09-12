"""Audio I/O and resampling helpers (soundfile + scipy), replacing torchaudio."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioInfo:
    sample_rate: int
    num_frames: int


def load_audio(path: str) -> tuple[torch.Tensor, int]:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sample_rate)


def audio_info(path: str) -> AudioInfo:
    info = sf.info(path)
    return AudioInfo(sample_rate=int(info.samplerate), num_frames=int(info.frames))


def resample(waveform: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if waveform.numel() == 0 or int(orig_sr) == int(new_sr):
        return waveform
    from scipy.signal import resample_poly

    divisor = math.gcd(int(orig_sr), int(new_sr))
    up = int(new_sr) // divisor
    down = int(orig_sr) // divisor
    array = waveform.detach().to(dtype=torch.float32).cpu().numpy()
    resampled = resample_poly(array, up, down, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(resampled)).to(dtype=waveform.dtype, device=waveform.device)


def save_audio(
    path: str,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    encoding: str | None = None,
    bits_per_sample: int | None = None,
) -> None:
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Audio tensor must have shape [channels, samples], got {tuple(audio.shape)}.")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    frames = audio.detach().to(torch.float32).cpu().numpy().T
    use_pcm16 = bits_per_sample == 16 or encoding in ("PCM_S", "PCM_16")
    if use_pcm16:
        frames = np.clip(frames, -1.0, 1.0)
        subtype = "PCM_16"
    else:
        subtype = "FLOAT"
    sf.write(path, frames, int(sample_rate), subtype=subtype)
