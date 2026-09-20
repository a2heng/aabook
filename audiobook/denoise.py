"""PureVox denoise for chapter audio (file in -> file out, GPU preferred).

Wraps the trained model ``purevox_denoise_202609_ep0167`` (48 kHz, 960/480 STFT, 0.62 M).
Two backends:

- ``torch`` (default): loads the original checkpoint with CUDA and runs the OFFLINE forward
  over the whole file in ~60 s chunks with a small overlap/crossfade (RTF ≈ 0.001 on a
  4070 Ti Super, ~0.7 GB per 60 s chunk) -- the streaming hop loop is far slower in Python.
- ``onnx`` (fallback, no torch): the exported streaming graph, hop by hop on CPU.

    from audiobook.denoise import denoise_file
    denoise_file("in.wav", "out.wav")
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

APP_ROOT = Path(__file__).resolve().parent.parent

FS = 48000
HOP = 480
DEFAULT_ONNX = APP_ROOT / "ckpts" / "denoise" / "purevox_denoise_202609_ep0167.onnx"
DEFAULT_SCRIPT = Path(
    os.environ.get("AUDIOBOOK_DENOISE_SCRIPT", "/home/a2heng/d/PureVoxModel/2_denoise/purevox_denoise_202609.py")
)
DEFAULT_CKPT = Path(
    os.environ.get(
        "AUDIOBOOK_DENOISE_CKPT",
        "/home/a2heng/d/PureVoxModel/7_output/output_purevox_denoise_202609/checkpoints/checkpoint_epoch_167.tar",
    )
)
# 分块：每块处理 35s（前后各 2.5s 上下文），只保留中间 30s，段间 0.25s 交叉淡化
CHUNK_SECONDS = float(os.environ.get("AUDIOBOOK_DENOISE_CHUNK", "35"))
STEP_SECONDS = float(os.environ.get("AUDIOBOOK_DENOISE_STEP", "30"))
CONTEXT_SECONDS = float(os.environ.get("AUDIOBOOK_DENOISE_CONTEXT", "2.5"))
X_FADE_SECONDS = float(os.environ.get("AUDIOBOOK_DENOISE_XFADE", "0.25"))


def _resample(mono: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or mono.size == 0:
        return mono.astype(np.float32)
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(sr, target)
    return resample_poly(mono.astype(np.float32), target // divisor, sr // divisor).astype(np.float32)


def _load_purevox_module(script: Path):
    """Import the training script without requiring tensorboard (only SummaryWriter is used)."""
    if "torch.utils.tensorboard" not in sys.modules:
        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = object  # type: ignore[attr-defined]
        sys.modules["torch.utils.tensorboard"] = stub
    spec = importlib.util.spec_from_file_location("purevox_denoise_202609", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Denoiser:
    """Whole-file denoiser: torch/CUDA (chunked offline) with an ONNX/CPU fallback."""

    def __init__(self, backend: str = "auto", device: str = "auto") -> None:
        self.backend = backend
        self.torch_model: Any = None
        self.onnx_session: Any = None
        self.onnx_cache = 0
        if backend in ("auto", "torch") and DEFAULT_SCRIPT.is_file() and DEFAULT_CKPT.is_file():
            try:
                self._init_torch(device)
                self.backend = "torch"
            except Exception as error:  # noqa: BLE001 - fall back to onnx
                if backend == "torch":
                    raise
                print(f"[denoise] torch backend unavailable ({str(error)[:120]}); falling back to ONNX/CPU", flush=True)
        if self.torch_model is None:
            self._init_onnx()
            self.backend = "onnx"

    def _init_torch(self, device: str) -> None:
        import torch

        module = _load_purevox_module(DEFAULT_SCRIPT)
        checkpoint = torch.load(str(DEFAULT_CKPT), map_location="cpu", weights_only=False)
        model = module.PureVoxDenoise202609()
        model.load_state_dict(checkpoint["model"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        self.device = device
        self.torch_model = model
        self.chunk = max(1, int(CHUNK_SECONDS * FS))  # 35s processed per forward
        self.step = max(1, int(STEP_SECONDS * FS))  # 30s kept
        self.context = max(0, int(CONTEXT_SECONDS * FS))  # 2.5s context each side
        self.xfade = max(0, int(X_FADE_SECONDS * FS))

    def _init_onnx(self) -> None:
        import onnxruntime as ort

        if not DEFAULT_ONNX.is_file():
            raise FileNotFoundError(f"denoise model not found: {DEFAULT_ONNX}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 2
        self.onnx_session = ort.InferenceSession(str(DEFAULT_ONNX), options, providers=["CPUExecutionProvider"])
        self.onnx_cache = int(self.onnx_session.get_inputs()[1].shape[1])

    # -- backends ----------------------------------------------------------
    def _torch_forward(self, mono: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(mono).float().unsqueeze(0).to(self.device)
            output = self.torch_model(tensor)
        return output[0].detach().cpu().numpy().astype(np.float32)

    def _process_torch(self, mono: np.ndarray) -> np.ndarray:
        """35 s forward per block (2.5 s context each side), keep ONLY the middle 30 s.

        The context is discarded; the kept middles tile the file exactly (30 s steps), so
        there is no crossfade -- every sample comes from a block that saw its full
        neighbourhood."""
        total = len(mono)
        if total <= self.chunk:
            return self._torch_forward(mono)[:total]
        out = np.zeros(total, dtype=np.float32)
        start = 0
        while start < total:
            low = max(0, start - self.context)
            high = min(total, start + self.step + self.context)
            enhanced = self._torch_forward(mono[low:high])
            keep_low = start - low
            keep_len = min(self.step, total - start)
            out[start : start + keep_len] = enhanced[keep_low : keep_low + keep_len]
            start += self.step
        return out

    def _process_onnx(self, mono: np.ndarray) -> np.ndarray:
        total = len(mono)
        hops = total // HOP
        if hops == 0:
            return mono
        body = mono[: hops * HOP]
        cache = np.zeros((1, self.onnx_cache), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for index in range(hops + 1):
            hop = body[index * HOP : (index + 1) * HOP].reshape(1, HOP) if index < hops else np.zeros((1, HOP), np.float32)
            enhanced, cache = self.onnx_session.run(None, {"mix_hop": hop, "cache_in": cache})
            pieces.append(enhanced)
        output = np.concatenate(pieces, axis=-1)[0, HOP : HOP + hops * HOP]
        rest = total - hops * HOP
        if rest > 0:
            output = np.concatenate([output, np.zeros(rest, dtype=np.float32)])
        return output

    # -- public ------------------------------------------------------------
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise one waveform; returns the same sample rate as the input."""
        mono = audio.mean(axis=1) if audio.ndim > 1 else audio
        work = _resample(mono, sample_rate, FS)
        output = self._process_torch(work) if self.torch_model is not None else self._process_onnx(work)
        return _resample(output, FS, sample_rate)


def denoise_file(src: str | Path, dst: str | Path | None = None, backend: str = "auto") -> Path:
    """Denoise a wav file; default output is ``<stem>_denoised.wav`` next to it."""
    src = Path(src)
    data, sample_rate = sf.read(str(src), dtype="float32", always_2d=True)
    output = Denoiser(backend=backend).process(data, sample_rate)
    target = Path(dst) if dst else src.with_name(f"{src.stem}_denoised.wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.wav")
    sf.write(str(temp), output, sample_rate, subtype="PCM_16")
    os.replace(temp, target)
    return target
