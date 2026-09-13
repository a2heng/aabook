"""Mechanical AuK instruction rendering (no LLM, deterministic)."""

from __future__ import annotations

from .schema import EMOTION_MULTIPLIERS, EMOTIONS

ZERO_SHOT_TEMPLATE = 'Say the following with the same voice: "{text}"'
INSTRUCT_TEMPLATE = '请基于下面的描述: "{style_desc}",生成语音内容"{text}".'
EMOTION_EDIT_TEMPLATE = "将情感转变为{emotion}。"


def render_instruction(auk_task: str, text: str, *, style_desc: str = "", emotion: str = "") -> str:
    if auk_task == "zero_shot_tts":
        return ZERO_SHOT_TEMPLATE.format(text=text)
    if auk_task == "instruct_tts":
        return INSTRUCT_TEMPLATE.format(style_desc=style_desc, text=text)
    if auk_task == "emotion_edit":
        return EMOTION_EDIT_TEMPLATE.format(emotion=emotion)
    raise ValueError(f"Unsupported auk_task: {auk_task!r}")


def emotion_multiplier(emotion: str) -> float:
    if not emotion:
        return 1.0
    if emotion not in EMOTIONS:
        return 1.0
    return EMOTION_MULTIPLIERS.get(emotion, 1.06)
