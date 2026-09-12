from __future__ import annotations

import argparse
import math
import os
import re

import gradio as gr
import torch

from app import audio_io
from app.template_helper import tpl_instances
from auk.infer.infer_auk import AukInfer, get_gen_duration
from auk.infer.pe import PromptEnhancer, PromptEnhancerError


# Static examples derived from the public AuK demo sample list.
# Each row is: (audio path relative to the repository root, instruction, target duration seconds).
DEMO_EXAMPLE_GROUPS: dict[str, dict[str, list[tuple[str | None, str, float]]]] = {
    "Speech Generation": {
        "Zero-shot TTS": [
            (
                "assets/demo-input-audio/zero-shot-tts/ref.wav",
                "Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'",
                6.0,
            ),
        ],
        "Instruct TTS": [
            (
                None,
                "Say the following in the voice described here: “一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断，带着审视人心的敏锐感。讲话时节奏变化明显，可以先压低声音缓缓铺垫，再突然加重关键字。既有豪迈，也隐约带着危险与猜疑”, and say: “宁可我负天下人，休教天下人负我。”",
                3.36,
            ),
            (
                None,
                "Say the following in the voice described here: “一位中年女性，在面对面质问多年前抛弃自己的人时，用沙哑、干涩的嗓音倾诉三十年压抑的恨意与委屈，仿佛眼泪已经流尽。语速缓慢而沉稳，字字清晰冰冷，整体音量中等偏低，但说到愤恨处声音微微发颤、压抑着哽咽，音色苍凉而有力，句尾带着渗骨的决绝，最后一句陡然拔高、几乎用尽全身力气砸出来”, and say: “哼，我的眼泪早哭干了，我没有委屈，我有的是恨，是悔，是三十年一天一天我自己受的苦。你大概已经忘了你做的事了！”",
                15.0,
            ),
            (
                None,
                "Say the following in the voice described here: “一位莎士比亚戏剧风格的经典反派，正在揭露真相前细细品味这一刻。声音是浑厚且富有戏剧张力的男中音，胸腔共鸣饱满，语速从容舒缓，带着刻意的停顿与强调；辅音被夸张地滚动、咬字清晰而富有舞台感。语调先佯装甜蜜温柔，几乎带着宠溺，随后在词句间骤然转冷，锋利如刀刃，每个字都缠绕着危险而愉悦的得意之情”, and say: “You see, my dear, the trap was never meant for you. It was always meant for him.”",
                7.0,
            ),
            (
                None,
                "Say the following in the voice described here: “仿佛在葬礼上宣布噩耗一般，声音轻柔而哽咽，强忍着泪水，话语断断续续、每句之间都有停顿，字字愈发沉重，最后一句的尾音被悲伤撕裂、颤抖着透出压抑不住的哭腔。语调缓慢低沉，气息不稳，清晰度因哽咽而略受影响，整体氛围凝重而哀恸”, and say: “He always said he'd come back. He always kept his word. Until now.”",
                7.0,
            ),
        ],
    },
    "Content Editing": {
        "Speech Content Editing": [
            (
                "assets/demo-input-audio/content-edit/content.wav",
                "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.",
                7.0,
            ),
        ],
        "Lyric Editing": [
            (
                "assets/demo-input-audio/vocal-edit/vocaledit-en-1-input.wav",
                "Replace “rear view” with “like you” in the lyrics",
                5.44,
            ),
        ],
    },
    "Acoustic Editing": {
        "Pitch Editing": [
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低3个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低2个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调降低1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高1个半音。", 5.0),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高2个半音。", 5.12),
            ("assets/demo-input-audio/pitch/pitch-1-input.wav", "将音调升高3个半音。", 5.12),
        ],
        "Speed Editing": [
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.5倍。", 20.6),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为0.75倍。", 13.74),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.25倍。", 8.24),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为1.5倍。", 6.86),
            ("assets/demo-input-audio/speed/speed-edit-1-input.wav", "将语速调整为2.0倍。", 5.16),
        ],
        "Volume Editing": [
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低15分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量降低5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高5分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高10分贝。", 3.78),
            ("assets/demo-input-audio/energy/energy-edit-1-input.wav", "将音量升高15分贝。", 3.78),
        ],
    },
    "Paralinguistic Editing": {
        "Emotion Editing": [
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a happy tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a sad tone", 7.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a angry tone", 6.6),
            ("assets/demo-input-audio/emotion-edit/en-1-input.wav", "Say this in a afraid tone", 7.22),
        ],
        "Timbre Editing": [
            (
                "assets/demo-input-audio/vc/vc-1-input.wav",
                "Keep the words and change the timbre to: “这位说话人的声音低沉而浑厚，语速平稳，吐字清晰。他的说话风格沉稳而富有思考，带有平静的反思特质。”",
                6.56,
            ),
        ],
        "De-accent": [
            ("assets/demo-input-audio/accent/accent-anhui-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.0),
            (
                "assets/demo-input-audio/accent/accent-dongbei-input.wav",
                "请去掉这段语音里的方言口音，保持说话人音色一致。",
                17.52,
            ),
            ("assets/demo-input-audio/accent/accent-fujian-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 4.06),
            ("assets/demo-input-audio/accent/accent-hubei-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 2.2),
            ("assets/demo-input-audio/accent/accent-hunan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.52),
            ("assets/demo-input-audio/accent/accent-sichuan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 6.74),
            ("assets/demo-input-audio/accent/accent-tibetan-input.wav", "请去掉这段语音里的方言口音，保持说话人音色一致。", 3.22),
        ],
        "Nonverbal Editing": [
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a breath before “We tested”", 10.44),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a sneeze before “only one of them”", 12.0),
            ("assets/demo-input-audio/nv/en-c-input.wav", "Add a pause after “only one of them”", 11.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the humming", 22.0),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the hiss", 22.82),
            ("assets/demo-input-audio/nv/en-d-input.wav", "Remove the sobbing", 21.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a sigh before “这个月”", 13.0),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a filler before “主要的缺口”", 12.38),
            ("assets/demo-input-audio/nv/zh-a-input.wav", "Add a cough before “再决定”", 13.0),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the laughter", 12.58),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the gasp of surprise", 12.72),
            ("assets/demo-input-audio/nv/zh-b-input.wav", "Remove the sharp inhale", 12.62),
        ],
        "Whisper Conversion": [
            ("assets/demo-input-audio/whisper/wh-w2n-zh-input.wav", "Turn this into a whisper", 8.36),
        ],
    },
    "Enhancement & Separation": {
        "Speech Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Remove the background noise and make the voice cleaner", 5.0),
        ],
        "Speech Separation": [
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the speaker who says “get what”", 28.0),
            ("assets/demo-input-audio/ss/en-1-input.wav", "Keep only the first speaker", 28.0),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the speaker who says “警队规矩”", 19.28),
            ("assets/demo-input-audio/ss/zh-1-input.wav", "Keep only the second speaker", 18.24),
        ],
        "Vocal Extraction": [
            (
                "assets/demo-input-audio/vocal-extraction/vocal-1-input.wav",
                "Extract the vocals and remove the accompaniment",
                10.88,
            ),
        ],
        "Audio Quality Enhancement": [
            ("assets/demo-input-audio/se/se-zh-1-input.wav", "Improve the audio quality and make it clearer", 5.0),
        ],
    },
}

# Display labels (Chinese) for the English example-group keys above.
DEMO_EXAMPLE_LABELS: dict[str, str] = {
    "Speech Generation": "语音合成",
    "Zero-shot TTS": "零样本合成",
    "Instruct TTS": "指令合成",
    "Content Editing": "内容编辑",
    "Speech Content Editing": "语音内容编辑",
    "Lyric Editing": "歌词编辑",
    "Acoustic Editing": "声学编辑",
    "Pitch Editing": "音调编辑",
    "Speed Editing": "语速编辑",
    "Volume Editing": "音量编辑",
    "Paralinguistic Editing": "副语言编辑",
    "Emotion Editing": "情感编辑",
    "Timbre Editing": "音色编辑",
    "De-accent": "去口音",
    "Nonverbal Editing": "非语言编辑",
    "Whisper Conversion": "耳语转换",
    "Enhancement & Separation": "增强与分离",
    "Speech Enhancement": "语音增强",
    "Speech Separation": "说话人分离",
    "Vocal Extraction": "人声提取",
    "Audio Quality Enhancement": "音质提升",
}


# variant label -> checkpoint path (filled in main() from CLI args)
CKPT_PATHS: dict[str, str] = {}
CONFIG_PATHS: dict[str, str | None] = {}
# variant label -> loaded engine, populated lazily on first use and reused after
ENGINES: dict[str, AukInfer] = {}
ENGINE_KWARGS: dict = {}
DEVICE_PATHS: dict[str, str | None] = {}

BASE_LABEL = "AuK (Base)"
FLASH_LABEL = "AuK-Flash ⚡"

SAMPLING_PRESETS = {
    BASE_LABEL: {"nfe": 32, "cfg": 2.0, "interactive": True},
    FLASH_LABEL: {"nfe": 4, "cfg": 0.0, "interactive": False},
}

LOUDNESS_LIMIT_TASK_TYPES = frozenset({"extract_vocals", "vocal_edit"})
VOCAL_OUTPUT_TARGET_LUFS = -14.0
VOCAL_OUTPUT_PEAK_CEILING = 0.95

# Local duration estimation used when Duration is 0 and Prompt Enhancer is off.
# Rough sentence-level estimate: speaking time + punctuation pauses.
SECONDS_PER_CJK_CHAR = 0.22
SECONDS_PER_LATIN_WORD = 0.40
SECONDS_PER_STRONG_PAUSE = 0.30  # 。！？!?…
SECONDS_PER_WEAK_PAUSE = 0.12  # ，、；：,;:
MIN_AUTO_DURATION = 0.5
MAX_AUTO_DURATION = 300.0
MAX_MANUAL_DURATION = 120.0
# Long synthesis is chunked by sentences to keep each forward pass short enough to fit VRAM.
MAX_SEGMENT_SECONDS = 20.0
SEGMENT_GAP_SECONDS = 0.15
# Manual multiplier applied on top of the auto estimate (buttons, step 0.1).
DURATION_RATES = [round(1.0 + index * 0.1, 1) for index in range(11)]
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")
_STRONG_PAUSE_RE = re.compile(r"[。！？!?…]+")
_WEAK_PAUSE_RE = re.compile(r"[，、；：,;:]+")
_DOUBLE_QUOTED_RE = re.compile(r"[\"“”«»「」『』]([^\"“”«»「」『』]+)[\"“”«»「」『』]")
# Single quotes, allowing apostrophes inside contractions (e.g. it's) but not as delimiters.
_SINGLE_QUOTED_RE = re.compile(r"(?<![A-Za-z])['‘’]([^'‘’]*(?:['‘’][A-Za-z][^'‘’]*)*)['‘’](?![A-Za-z])")
# Synthesis tasks (instruct/zero-shot TTS): duration follows the target text, not the reference audio.
_TTS_HINT_RE = re.compile(
    r"same voice|voice clon|clone (?:this|the) voice|"
    r"音色克隆|克隆.{0,6}(?:音色|声音)|生成语音内容|"
    r"用.{0,8}(?:声音|音色).{0,4}(?:说|念|讲)",
    re.IGNORECASE,
)

DEMO_CSS = """
#auk-main-row {
    align-items: flex-start;
}
#auk-input-audio {
    min-height: 210px !important;
}
#auk-output-audio {
    min-height: 200px !important;
}
#auk-pe-output {
    margin-top: 12px;
}
.auk-pe-card textarea {
    min-height: 92px !important;
    height: 92px !important;
}
.auk-sampling-note {
    margin-top: 2px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.auk-duration-note {
    margin-top: 2px;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.auk-duration-note p {
    margin: 0;
    line-height: 1.6;
}
#auk-est-duration {
    margin-top: 0;
    color: var(--body-text-color-subdued);
    font-size: 12px;
    font-weight: 600;
}
.auk-rate-row {
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    margin: 2px 0 !important;
}
.auk-rate-row .auk-rate-label {
    flex: 0 0 auto !important;
    min-width: 0 !important;
}
.auk-rate-row .auk-rate-label p {
    margin: 0 !important;
    font-size: 12px;
    color: var(--body-text-color-subdued);
    white-space: nowrap;
}
.auk-rate-row button {
    width: auto !important;
    min-width: 0 !important;
    min-height: 0 !important;
    padding: 2px 8px !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
}
#auk-title,
#auk-title *,
#auk-supported,
#auk-supported * {
    overflow: visible !important;
    white-space: normal !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
}
.auk-tpl-row {
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
    margin: 2px 0 !important;
    overflow: visible !important;
}
.auk-tpl-row > *,
.auk-tpl-row > .form,
.auk-tpl-row .auk-tpl-cat {
    flex: 0 0 auto !important;
    min-width: 0 !important;
}
.auk-tpl-row .auk-tpl-cat {
    flex-basis: 60px !important;
    width: 60px !important;
    margin: 0 !important;
}
.auk-tpl-row .auk-tpl-cat p {
    margin: 0 !important;
    font-size: 12px;
    color: var(--body-text-color-subdued);
    white-space: nowrap;
}
.auk-tpl-row button {
    width: auto !important;
    min-width: 0 !important;
    min-height: 0 !important;
    padding: 2px 8px !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    white-space: nowrap !important;
}
.auk-examples-table {
    width: 100% !important;
    overflow-x: hidden !important;
}
.auk-examples-table table {
    width: 100% !important;
    table-layout: fixed !important;
}
.auk-examples-table table tbody tr > td:last-child,
.auk-examples-table table tbody tr > td:last-child * {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}
"""


APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_path(*parts: str) -> str:
    return os.path.join(APP_ROOT, *parts)


def submodule_path(*parts: str) -> str:
    return os.path.join(APP_ROOT, "third_party", "AuK", *parts)


def _speech_text(instruction: str | None) -> str:
    text = instruction or ""
    quotes = [quote for quote in _DOUBLE_QUOTED_RE.findall(text) if quote.strip()]
    if not quotes:
        quotes = [quote for quote in _SINGLE_QUOTED_RE.findall(text) if quote.strip()]
    return quotes[-1] if quotes else text


def estimate_text_duration(text: str | None) -> float:
    text = text or ""
    cjk = len(_CJK_CHAR_RE.findall(text))
    words = len(_LATIN_WORD_RE.findall(text))
    strong_pauses = len(_STRONG_PAUSE_RE.findall(text))
    weak_pauses = len(_WEAK_PAUSE_RE.findall(text))
    speech = cjk * SECONDS_PER_CJK_CHAR + words * SECONDS_PER_LATIN_WORD
    pauses = strong_pauses * SECONDS_PER_STRONG_PAUSE + weak_pauses * SECONDS_PER_WEAK_PAUSE
    return speech + pauses


def _replace_speech_text(instruction: str, new_text: str) -> str:
    text = instruction or ""
    matches = list(_DOUBLE_QUOTED_RE.finditer(text))
    if not matches:
        matches = list(_SINGLE_QUOTED_RE.finditer(text))
    if not matches:
        return new_text
    match = matches[-1]
    return text[: match.start(1)] + new_text + text[match.end(1) :]


def _split_oversized(text: str, max_seconds: float) -> list[str]:
    if estimate_text_duration(text) <= max_seconds:
        return [text]
    parts = [part.strip() for part in re.split(r"(?<=[，、；：,;:])\s*", text) if part.strip()]
    if len(parts) > 1:
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{part}" if current else part
            if current and estimate_text_duration(candidate) > max_seconds:
                chunks.append(current)
                current = part
            else:
                current = candidate
        if current:
            chunks.append(current)
        if all(estimate_text_duration(chunk) <= max_seconds for chunk in chunks):
            return chunks
    approx_chars = max(8, int(max_seconds / SECONDS_PER_CJK_CHAR))
    return [text[index : index + approx_chars] for index in range(0, len(text), approx_chars)]


def _split_text_for_duration(text: str | None, max_seconds: float) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    pieces = [piece.strip() for piece in re.findall(r"[^。！？!?…\n]+[。！？!?…]*", text)]
    pieces = [piece for piece in pieces if piece] or [text]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{piece}" if current else piece
        if current and estimate_text_duration(candidate) > max_seconds:
            chunks.extend(_split_oversized(current, max_seconds))
            current = piece
        else:
            current = candidate
    if current:
        chunks.extend(_split_oversized(current, max_seconds))
    return [chunk for chunk in chunks if chunk.strip()]


def estimate_duration(instruction: str | None, audio: str | None, rate: float = 1.0) -> float:
    try:
        rate = min(2.0, max(0.1, float(rate or 1.0)))
    except (TypeError, ValueError):
        rate = 1.0
    # Synthesis tasks follow the target text even when a reference audio is loaded;
    # only audio-editing tasks fall back to the source audio length.
    is_synthesis = bool(_TTS_HINT_RE.search(instruction or ""))
    if audio and not is_synthesis:
        try:
            info = audio_io.audio_info(audio)
            if info.sample_rate > 0 and info.num_frames > 0:
                return min(MAX_AUTO_DURATION, max(MIN_AUTO_DURATION, info.num_frames / info.sample_rate * rate))
        except Exception:
            pass
    seconds = estimate_text_duration(_speech_text(instruction)) * rate
    if seconds <= 0:
        seconds = 3.0 * rate
    return min(MAX_AUTO_DURATION, max(MIN_AUTO_DURATION, seconds))


def update_duration_estimate(instruction: str | None, audio: str | None, gen_seconds: float | None, rate: float = 1.0):
    if gen_seconds and float(gen_seconds) > 0:
        return f"预计时长: {float(gen_seconds):.2f} s（手动指定）"
    return f"预计时长: {estimate_duration(instruction, audio, rate):.2f} s（自动估算 ×{float(rate or 1.0):.1f}）"


def update_sampling_controls(variant: str):
    preset = SAMPLING_PRESETS.get(variant, SAMPLING_PRESETS[BASE_LABEL])
    interactive = preset["interactive"]
    return (
        gr.update(value=preset["nfe"], interactive=interactive),
        gr.update(value=preset["cfg"], interactive=interactive),
    )


def get_engine(variant: str) -> AukInfer:
    if variant not in ENGINES:
        ckpt_path = CKPT_PATHS.get(variant)
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise gr.Error(f"未找到 {variant} 权重文件：{ckpt_path}")
        # config.yaml ships next to the checkpoint (release dirs bundle their own)
        ckpt_dir_config = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "config.yaml")
        config_path = CONFIG_PATHS.get(variant) or ckpt_dir_config
        if not os.path.isfile(config_path):
            raise gr.Error(f"未在 {variant} 权重旁找到 config.yaml：{config_path}")
        gr.Info(f"正在加载 {variant} …（首次约 15 秒）")
        ENGINES[variant] = AukInfer(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=DEVICE_PATHS.get(variant) or ENGINE_KWARGS.get("device"),
            **{key: value for key, value in ENGINE_KWARGS.items() if key != "device"},
        )
    return ENGINES[variant]


def _limit_vocal_output_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    output = waveform.to(torch.float64)
    try:
        import pyloudnorm as pyln

        measured_lufs = float(pyln.Meter(sample_rate).integrated_loudness(output.numpy()))
        if math.isfinite(measured_lufs) and measured_lufs > VOCAL_OUTPUT_TARGET_LUFS:
            gain = 10.0 ** ((VOCAL_OUTPUT_TARGET_LUFS - measured_lufs) / 20.0)
            output = output * min(1.0, gain)
    except Exception:
        pass

    if output.numel():
        peak = float(output.abs().max())
        if math.isfinite(peak) and peak > VOCAL_OUTPUT_PEAK_CEILING:
            output = output * (VOCAL_OUTPUT_PEAK_CEILING / peak)
    return output


def _format_output_audio(out_audio: torch.Tensor, sample_rate: int, task_type: str | None = None):
    waveform = out_audio.detach().to(torch.float64).cpu()
    if task_type in LOUDNESS_LIMIT_TASK_TYPES and waveform.ndim == 2:
        if waveform.shape[0] <= 8:
            waveform = waveform.mean(dim=0)
        elif waveform.shape[1] <= 8:
            waveform = waveform.mean(dim=1)
    elif waveform.ndim == 2 and 1 in waveform.shape:
        waveform = waveform.reshape(-1)
    if waveform.ndim != 1:
        raise gr.Error(f"期望单声道输出音频，实际形状为 {tuple(waveform.shape)}。")
    if not torch.isfinite(waveform).all():
        raise gr.Error("生成的音频包含 NaN 或 Inf。")
    if task_type in LOUDNESS_LIMIT_TASK_TYPES:
        waveform = _limit_vocal_output_loudness(waveform, sample_rate)
    pcm16 = torch.round(waveform.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy()
    return int(sample_rate), pcm16


def _generate_once(engine, instruction, audio, gen_seconds, nfe, cfg, seed):
    content = [{"type": "text", "text": instruction}]
    if audio:
        content.append({"type": "audio", "audio": audio})
    messages = [{"role": "user", "content": content}]
    return engine.generate(
        messages,
        audio=(audio or None),
        gen_seconds=gen_seconds,
        nfe=int(nfe),
        cfg_strength=float(cfg),
        seed=(int(seed) if seed is not None else None),
    )


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    output = waveform.detach().to(torch.float64).cpu()
    if output.ndim == 2:
        if output.shape[0] <= 8:
            output = output.mean(dim=0)
        else:
            output = output.mean(dim=1)
    return output.reshape(-1)


def _generate_chunked(engine, instruction, audio, chunks, total_seconds, nfe, cfg, seed, task_type):
    total_text = sum(estimate_text_duration(chunk) for chunk in chunks) or 1.0
    pieces: list[torch.Tensor] = []
    sample_rate = None
    for index, chunk in enumerate(chunks):
        share = estimate_text_duration(chunk) / total_text
        seconds = max(MIN_AUTO_DURATION, total_seconds * share)
        chunk_instruction = _replace_speech_text(instruction, chunk)
        out_audio, sample_rate = _generate_once(
            engine,
            chunk_instruction,
            audio,
            seconds,
            nfe,
            cfg,
            seed,
        )
        pieces.append(_to_mono(out_audio))
        if index != len(chunks) - 1 and SEGMENT_GAP_SECONDS > 0:
            pieces.append(torch.zeros(max(1, round(sample_rate * SEGMENT_GAP_SECONDS)), dtype=torch.float64))
    return _format_output_audio(torch.cat(pieces), sample_rate or 24000, task_type=task_type)


def run_generate(variant, audio, instruction, gen_seconds, ref_text, gen_text, nfe, cfg, seed, task_type=None):
    if not instruction:
        raise gr.Error("请填写指令。")
    if not audio and not gen_seconds and not gen_text:
        raise gr.Error("无参考音频的指令合成需要提供目标时长或目标文本来确定长度。")

    instruction = instruction.strip()
    engine = get_engine(variant)
    # nfe/cfg are used as-is for base AuK; AuK-Flash ignores them (locked 4-step / CFG-off recipe).
    gen_seconds = get_gen_duration(
        audio=(audio or None),
        ref_text=(ref_text or None),
        gen_text=(gen_text or None),
        gen_seconds=(gen_seconds or None),
    )

    synthesis = _TTS_HINT_RE.search(instruction) and task_type in (None, "instruct_tts", "zero_shot_tts")
    if synthesis:
        chunks = _split_text_for_duration(_speech_text(instruction), MAX_SEGMENT_SECONDS)
        if len(chunks) > 1:
            total_seconds = gen_seconds or sum(estimate_text_duration(chunk) for chunk in chunks)
            if total_seconds > MAX_SEGMENT_SECONDS:
                return _generate_chunked(engine, instruction, audio, chunks, total_seconds, nfe, cfg, seed, task_type)

    out_audio, sr = _generate_once(engine, instruction, audio, gen_seconds, nfe, cfg, seed)
    return _format_output_audio(out_audio, sr, task_type=task_type)


def run_generate_with_pe(use_pe, variant, audio, instruction, gen_seconds, duration_rate, ref_text, gen_text, nfe, cfg, seed):
    if not use_pe:
        effective_duration = float(gen_seconds) if gen_seconds else estimate_duration(instruction, audio, duration_rate)
        return run_generate(variant, audio, instruction, effective_duration, ref_text, gen_text, nfe, cfg, seed), None

    prepared = None
    try:
        requested_duration = float(gen_seconds or 0)
        prepared = PromptEnhancer().prepare(
            instruction,
            audio,
            target_duration=requested_duration if requested_duration > 0 else None,
        )
        effective_duration = requested_duration if requested_duration > 0 else prepared.gen_seconds
        generated = run_generate(
            variant,
            prepared.audio,
            prepared.instruction,
            effective_duration,
            prepared.ref_text,
            prepared.gen_text,
            nfe,
            cfg,
            seed,
            prepared.task_type,
        )
        asr = getattr(prepared, "asr", None)
        asr_text = asr.text if asr and asr.text else None
        task = str(getattr(prepared, "task_type", ""))
        operation_subtype = getattr(prepared, "operation_subtype", None)
        if operation_subtype:
            task = f"{task} / {operation_subtype}"
        metadata = {
            "task": task,
            "duration": f"{effective_duration:.2f} s",
            "asr_content": asr_text or "",
            "instruction": prepared.instruction,
        }
        return generated, metadata
    except (PromptEnhancerError, ValueError, FileNotFoundError) as exc:
        raise gr.Error(f"Prompt Enhancer 失败：{type(exc).__name__}: {exc}") from None
    finally:
        if prepared is not None:
            prepared.cleanup()


def update_pe_outputs(metadata: dict | None):
    if not metadata:
        return "", "", ""
    return (
        f"任务：{metadata.get('task') or ''}\n目标时长：{metadata.get('duration') or ''}",
        str(metadata.get("asr_content") or ""),
        str(metadata.get("instruction") or ""),
    )


def load_demo_example(index: int | None, examples: list[tuple[str | None, str, float]]):
    if index is None or not 0 <= int(index) < len(examples):
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    audio, instruction, duration = examples[int(index)]
    return audio, instruction, duration, True, 42


def _resolve_demo_examples(examples: list[tuple[str | None, str, float]]) -> list[tuple[str | None, str, float]]:
    resolved = []
    for audio, instruction, duration in examples:
        audio_path = submodule_path(*audio.split("/")) if audio else None
        if audio_path is None or os.path.isfile(audio_path):
            resolved.append((audio_path, instruction, duration))
    return resolved


def _demo_dataset_samples(task: str, examples: list[tuple[str | None, str, float]]) -> list[list[str | None]]:
    if task == "Instruct TTS":
        return [[instruction] for _, instruction, _ in examples]
    return [[audio, instruction] for audio, instruction, _ in examples]


def build_demo() -> gr.Blocks:
    available_variants = [label for label, path in CKPT_PATHS.items() if path and os.path.isfile(path)]
    if not available_variants:
        raise RuntimeError("未配置有效的模型权重。")
    initial_variant = available_variants[0]
    initial_sampling = SAMPLING_PRESETS.get(initial_variant, SAMPLING_PRESETS[BASE_LABEL])
    with gr.Blocks(title="AuK") as demo:
        gr.Markdown(
            '<div align="center">\n\n# AuK：面向语音生成与编辑的开源基础模型\n\n</div>',
            elem_id="auk-title",
        )
        gr.Markdown(
            "**支持的任务**\n"
            "- **语音合成**：零样本合成 · 指令合成\n"
            "- **内容编辑**：歌词编辑 · 语音内容编辑\n"
            "- **声学编辑**：音调编辑 · 语速编辑 · 音量编辑\n"
            "- **副语言编辑**：情感 · 音色 · 去口音 · 非语言编辑 · 耳语转换\n"
            "- **增强与分离**：语音增强 · 说话人分离 · 人声／伴奏分离\n\n"
            "时长优先级：手动时长 > 参考文本+目标文本估算 > 匹配源音频长度。",
            elem_id="auk-supported",
        )
        # Instruction box is created here so the top instance buttons can fill it.
        in_instr = gr.Textbox(label="指令", lines=4, render=False)

        gr.Markdown("### 指令实例（点击填入下方「指令」输入框，可自行修改）")
        for category, items in tpl_instances():
            with gr.Row(elem_classes="auk-tpl-row"):
                gr.Markdown(f"**{category}**", elem_classes="auk-tpl-cat")
                for label, instruction in items:
                    gr.Button(label, size="sm", variant="secondary", min_width=0).click(
                        fn=lambda text=instruction: text,
                        inputs=[],
                        outputs=[in_instr],
                        queue=False,
                        api_visibility="private",
                    )

        with gr.Row(equal_height=True, elem_id="auk-main-row"):
            with gr.Column(elem_id="auk-input-column"):  # left: inputs (audio + text)
                in_audio = gr.Audio(
                    label="输入音频（可选；指令合成可不填）",
                    type="filepath",
                    elem_id="auk-input-audio",
                )
                in_instr.render()
                with gr.Column(elem_id="auk-input-bottom"):
                    use_pe = gr.Checkbox(
                        value=False,
                        label="使用 Prompt Enhancer",
                        info="默认关闭。关闭后，时长为 0 时本地自动估算（优先音频时长，否则按文本长度）。",
                    )
                    # Keep empty internal states for the existing generation callback
                    # after removing the two manual text boxes from the UI.
                    in_ref_text = gr.State(value="")
                    in_gen_text = gr.State(value="")
            with gr.Column(elem_id="auk-output-column"):  # right: inference settings + output
                in_variant = gr.Radio(
                    choices=available_variants,
                    value=initial_variant,
                    label="模型",
                )
                in_secs = gr.Slider(
                    0,
                    MAX_MANUAL_DURATION,
                    value=0,
                    step=0.5,
                    label="时长（秒）",
                )
                gr.Markdown(
                    "0 = 自动估算（有音频按音频时长，否则按文本长度）；大于 0 时覆盖估算。",
                    elem_classes="auk-duration-note",
                )
                est_duration = gr.Markdown(
                    "预计时长: --",
                    elem_id="auk-est-duration",
                    elem_classes="auk-duration-note",
                )
                in_duration_rate = gr.State(value=1.0)
                with gr.Row(elem_classes="auk-rate-row"):
                    gr.Markdown("时长倍率", elem_classes="auk-rate-label")
                    for rate in DURATION_RATES:
                        gr.Button(f"{rate:.1f}", size="sm", variant="secondary", min_width=0).click(
                            fn=lambda value=rate: value,
                            inputs=[],
                            outputs=[in_duration_rate],
                            queue=False,
                            api_visibility="private",
                        ).then(
                            update_duration_estimate,
                            inputs=[in_instr, in_audio, in_secs, in_duration_rate],
                            outputs=est_duration,
                            queue=False,
                            api_visibility="private",
                            show_progress="hidden",
                        )
                with gr.Row():
                    in_nfe = gr.Slider(
                        4,
                        64,
                        value=initial_sampling["nfe"],
                        step=1,
                        label="NFE 步数",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_cfg = gr.Slider(
                        0.0,
                        5.0,
                        value=initial_sampling["cfg"],
                        step=0.1,
                        label="CFG 强度",
                        interactive=initial_sampling["interactive"],
                        scale=2,
                    )
                    in_seed = gr.Number(label="随机种子（可选）", value=42, precision=0, scale=1)
                gr.Markdown(
                    "Base：NFE 32 / CFG 2.0 · Flash：NFE 4 / CFG 0（锁定）",
                    elem_classes="auk-sampling-note",
                )
                gen_btn = gr.Button("生成", variant="primary")
                with gr.Column(elem_id="auk-output-bottom"):
                    out_audio = gr.Audio(label="输出音频", elem_id="auk-output-audio")
                    pe_metadata_state = gr.State(value=None)

        with gr.Column(elem_id="auk-pe-output"):
            gr.Markdown("### Prompt Enhancer 输出")
            with gr.Row(equal_height=True):
                pe_summary = gr.Textbox(
                    label="任务概览",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=1,
                )
                pe_asr_content = gr.Textbox(
                    label="ASR 转写",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )
                pe_instruction = gr.Textbox(
                    label="指令",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    elem_classes="auk-pe-card",
                    scale=2,
                )

        in_variant.change(
            update_sampling_controls,
            inputs=in_variant,
            outputs=[in_nfe, in_cfg],
            queue=False,
            api_visibility="private",
        )
        generate_event = gen_btn.click(
            run_generate_with_pe,
            inputs=[
                use_pe,
                in_variant,
                in_audio,
                in_instr,
                in_secs,
                in_duration_rate,
                in_ref_text,
                in_gen_text,
                in_nfe,
                in_cfg,
                in_seed,
            ],
            outputs=[out_audio, pe_metadata_state],
            api_name="run_generate_with_pe",
        )
        generate_event.then(
            update_pe_outputs,
            inputs=pe_metadata_state,
            outputs=[pe_summary, pe_asr_content, pe_instruction],
            queue=False,
            api_visibility="private",
        )
        for duration_input in (in_instr, in_audio, in_secs):
            duration_input.change(
                update_duration_estimate,
                inputs=[in_instr, in_audio, in_secs, in_duration_rate],
                outputs=est_duration,
                queue=False,
                api_visibility="private",
                show_progress="hidden",
            )

        gr.Markdown("## 示例\n点击示例加载到上方面板，然后点 **生成**。也可以更换随机种子多试几次。")
        with gr.Tabs():
            for category, tasks in DEMO_EXAMPLE_GROUPS.items():
                with gr.Tab(DEMO_EXAMPLE_LABELS.get(category, category)):
                    with gr.Tabs():
                        for task, task_examples in tasks.items():
                            examples = _resolve_demo_examples(task_examples)
                            if not examples:
                                continue
                            with gr.Tab(DEMO_EXAMPLE_LABELS.get(task, task)):
                                is_instruct_tts = task == "Instruct TTS"
                                components = [gr.Textbox(label="指令", render=False)]
                                headers = ["指令"]
                                if not is_instruct_tts:
                                    components.insert(0, gr.Audio(label="音频", render=False))
                                    headers.insert(0, "音频")
                                dataset = gr.Dataset(
                                    components=components,
                                    samples=_demo_dataset_samples(task, examples),
                                    headers=headers,
                                    type="index",
                                    layout="table",
                                    samples_per_page=8,
                                    show_label=False,
                                    elem_classes="auk-examples-table",
                                )
                                dataset.select(
                                    lambda index, rows=examples: load_demo_example(index, rows),
                                    inputs=dataset,
                                    outputs=[in_audio, in_instr, in_secs, use_pe, in_seed],
                                    queue=False,
                                    api_visibility="private",
                                    show_progress="hidden",
                                )

    return demo


def main():
    p = argparse.ArgumentParser(description="AuK Gradio demo")
    p.add_argument(
        "--base_ckpt",
        default=None,
        help="AuK checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument(
        "--flash_ckpt",
        default=None,
        help="AuK-Flash checkpoint. If any checkpoint flag is supplied, only explicitly supplied variants are shown.",
    )
    p.add_argument("--base_config", default=None)
    p.add_argument("--flash_config", default=None)
    p.add_argument("--base_device", default=None, help="Device for AuK, for example cuda:0.")
    p.add_argument("--flash_device", default=None, help="Device for AuK-Flash, for example cuda:1.")
    p.add_argument("--preload", action="store_true")
    p.add_argument("--qwen_path", default=None)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--device", default=None)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    explicitly_configured = args.base_ckpt is not None or args.flash_ckpt is not None
    candidates = {
        BASE_LABEL: (
            args.base_ckpt if explicitly_configured else default_path("ckpts", "AuK", "auk_base.safetensors"),
            args.base_config,
        ),
        FLASH_LABEL: (
            args.flash_ckpt if explicitly_configured else default_path("ckpts", "AuK-Flash", "auk_flash.safetensors"),
            args.flash_config,
        ),
    }
    for label, (checkpoint, config) in candidates.items():
        if checkpoint is None:
            continue
        if not os.path.isfile(checkpoint):
            if explicitly_configured:
                p.error(f"{label} checkpoint not found: {checkpoint}")
            continue
        if config is not None and not os.path.isfile(config):
            p.error(f"{label} config not found: {config}")
        CKPT_PATHS[label] = checkpoint
        CONFIG_PATHS[label] = config

    ENGINE_KWARGS.update(device=args.device, dtype=args.dtype, qwen_path=args.qwen_path)
    DEVICE_PATHS[BASE_LABEL] = args.base_device
    DEVICE_PATHS[FLASH_LABEL] = args.flash_device

    if not CKPT_PATHS:
        p.error("No valid --base_ckpt or --flash_ckpt was found.")
    if args.preload:
        for variant in CKPT_PATHS:
            get_engine(variant)

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=DEMO_CSS,
    )


if __name__ == "__main__":
    main()
