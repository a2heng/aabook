"""Minimal ``torchaudio`` shim backed by :mod:`app.audio_io`.

The AuK submodule is kept pristine and still does ``import torchaudio``; this
module is placed first on ``sys.path`` so those calls resolve to soundfile+scipy
instead. Only the subset AuK uses is implemented.
"""

from __future__ import annotations

from app import audio_io


class _Info:
    __slots__ = ("sample_rate", "num_frames")

    def __init__(self, sample_rate: int, num_frames: int):
        self.sample_rate = sample_rate
        self.num_frames = num_frames


def load(path, *args, **kwargs):
    return audio_io.load_audio(path)


def save(path, tensor, sample_rate, *args, **kwargs):
    return audio_io.save_audio(path, tensor, sample_rate, **kwargs)


def info(path, *args, **kwargs):
    resolved = audio_io.audio_info(path)
    return _Info(resolved.sample_rate, resolved.num_frames)


class _Resample:
    def __init__(self, orig_freq, new_freq, *args, **kwargs):
        self.orig_freq = orig_freq
        self.new_freq = new_freq

    def __call__(self, waveform):
        return audio_io.resample(waveform, self.orig_freq, self.new_freq)


class transforms:  # noqa: N801 - mimic torchaudio.transforms namespace
    Resample = _Resample


class functional:  # noqa: N801 - mimic torchaudio.functional namespace
    @staticmethod
    def resample(waveform, orig_freq, new_freq, *args, **kwargs):
        return audio_io.resample(waveform, orig_freq, new_freq)
