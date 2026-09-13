"""Runtime patches so the pristine AuK submodule fits on a 16 GB GPU.

Kept entirely outside the submodule:

* Qwen2.5-Omni text encoder is loaded with 8-bit bitsandbytes quantization.
* The DiT model is kept in bfloat16 instead of being upcast to float32.
"""

from __future__ import annotations

import logging
import os

import torch


logger = logging.getLogger(__name__)
_APPLIED = False


def _apply_qwen_8bit() -> None:
    if os.environ.get("AUK_QWEN_8BIT", "1") not in ("1", "true", "True"):
        return
    try:
        from transformers import BitsAndBytesConfig, Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError:  # transformers not installed yet
        return

    original = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("quantization_config", BitsAndBytesConfig(load_in_8bit=True))
        return original(cls, *args, **kwargs)

    Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained = from_pretrained


def _apply_dit_bf16() -> None:
    try:
        from auk.model import CFMEdit
    except ImportError:
        return

    original = CFMEdit.to

    def to(self, *args, **kwargs):
        if args and args[0] == torch.float32:
            args = (torch.bfloat16, *args[1:])
        elif kwargs.get("dtype") == torch.float32:
            kwargs["dtype"] = torch.bfloat16
        return original(self, *args, **kwargs)

    CFMEdit.to = to


class _DropNoise(logging.Filter):
    """Drop the per-layer bitsandbytes quantization spam from the root log."""

    _NEEDLES = ("MatMul8bitLt", "inputs will be cast")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(needle in message for needle in self._NEEDLES)


def _quiet_logs() -> None:
    # bitsandbytes logs one warning per quantized layer (thousands of lines).
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("bitsandbytes"):
            logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

    root = logging.getLogger()
    if not any(isinstance(existing, _DropNoise) for existing in root.filters):
        root.addFilter(_DropNoise())
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - optional
        pass


def apply_patches() -> None:
    """Idempotently install all runtime patches."""
    global _APPLIED
    if _APPLIED:
        return
    _quiet_logs()
    _apply_qwen_8bit()
    _apply_dit_bf16()
    _APPLIED = True
    logger.info("AuK runtime patches applied (Qwen 8-bit, DiT bf16).")
