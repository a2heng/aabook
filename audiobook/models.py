"""Per-model profiles: follow each author's README/template/sampling guidance.

The preprocessing LLM is served locally over an OpenAI-compatible endpoint; the
profile tells the client which sampling params and thinking controls to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelProfile:
    key: str
    repo: str
    gguf: str
    arch: str
    context: int
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 4096
    # How to turn thinking off for one request (passed as extra_body.chat_template_kwargs).
    thinking_off_kwargs: dict = field(default_factory=dict)
    # Gemma-style: thinking is enabled by this token at the start of the system prompt.
    thinking_system_token: str = ""
    notes: str = ""


PROFILES: dict[str, ModelProfile] = {
    "ornith-9b": ModelProfile(
        key="ornith-9b",
        repo="ornith-ai/Ornith-1.5-9B",
        gguf="ckpts/llm/Ornith-1.5-9B-Q8_0.gguf",
        arch="qwen35 (hybrid, MTP)",
        context=262144,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        thinking_off_kwargs={"enable_thinking": False},
        notes="Reasoning model: <think> block; reasoning in reasoning_content. General sampling per README.",
    ),
    "spark-4b": ModelProfile(
        key="spark-4b",
        repo="XHToken/Spark-X2.5-4B",
        gguf="ckpts/llm/Spark-X2.5-4B-Q8_0.gguf",
        arch="spark2_5",
        context=1048576,
        temperature=1.0,
        top_p=0.95,
        top_k=-1,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_off_kwargs={"enable_thinking": False},
        notes="Custom template (<｜start▁of▁sentence｜><|System|>). Thinking on by default.",
    ),
    "gemma-4-12b": ModelProfile(
        key="gemma-4-12b",
        repo="unsloth/gemma-4-12B-it-qat-GGUF",
        gguf="ckpts/llm/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
        arch="gemma4",
        context=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_system_token="<|think|>",
        notes="12B unified, QAT UD-Q4_K_XL; MTP draft at ckpts/llm/MTP/mtp-gemma-4-12B-it-Q8_0.gguf.",
    ),
    "gemma-4-26b-a4b": ModelProfile(
        key="gemma-4-26b-a4b",
        repo="unsloth/gemma-4-26B-A4B-it-qat-GGUF",
        gguf="ckpts/llm/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        arch="gemma4",
        context=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_system_token="<|think|>",
        notes="26B-A4B MoE (~4B active), QAT UD-Q4_K_XL; MTP draft at ckpts/llm/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf.",
    ),
    "gemma-4-e2b": ModelProfile(
        key="gemma-4-e2b",
        repo="google/gemma-4-E2B-it",
        gguf="ckpts/llm/gemma-4-E2B-it-Q8_0.gguf",
        arch="gemma4",
        context=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_system_token="<|think|>",
        notes="2.3B effective; thinking only via <|think|> at system start (off by default).",
    ),
    "gemma-4-e4b": ModelProfile(
        key="gemma-4-e4b",
        repo="google/gemma-4-E4B-it",
        gguf="ckpts/llm/gemma-4-E4B-it-Q8_0.gguf",
        arch="gemma4",
        context=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        thinking_system_token="<|think|>",
        notes="Enable thinking by putting <|think|> at the start of the system prompt; output <|channel>thought...<channel|>.",
    ),
}

DEFAULT_PROFILE = "ornith-9b"


def get_profile(name: str | None = None) -> ModelProfile:
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
