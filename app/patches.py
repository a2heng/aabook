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


def apply_patches() -> None:
    """Idempotently install all runtime patches."""
    global _APPLIED
    if _APPLIED:
        return
    _apply_qwen_8bit()
    _apply_dit_bf16()
    _APPLIED = True
    logger.info("AuK runtime patches applied (Qwen 8-bit, DiT bf16).")
