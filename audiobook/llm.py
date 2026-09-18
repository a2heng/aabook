"""OpenAI-compatible LLM client for offline preprocessing.

Works with any local server (llama.cpp / Ollama / vLLM) exposing an OpenAI API.
Config precedence: ``AUDIOBOOK_LLM_*`` then ``LLM_*`` (the Prompt Enhancer vars).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Qwen3.8-27B README "Best Practices" - thinking mode sampling.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.0


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
    thinking_off_kwargs: dict = field(default_factory=dict)
    thinking_system_token: str = ""
    notes: str = ""


# Qwen3.5 official sampling (model card "Best Practices") + froggeric template for Qwen3.8.
PROFILES: dict[str, ModelProfile] = {
    "qwen3.5-9b": ModelProfile(
        key="qwen3.5-9b",
        repo="unsloth/Qwen3.5-9B-MTP-GGUF",
        gguf="ckpts/llm/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        arch="qwen35 (dense, built-in MTP head)",
        context=262144,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        thinking_off_kwargs={"enable_thinking": False},
        thinking_system_token="",
        notes="Official Qwen3.5 thinking-mode sampling; tool calls via the model's own chat template "
        "(do NOT pass qwen_chat_template.jinja); MTP speculation: --spec-type draft-mtp --spec-draft-n-max 6.",
    ),
    "gemma-4-e4b": ModelProfile(
        key="gemma-4-e4b",
        repo="unsloth/gemma-4-E4B-it-qat-GGUF",
        gguf="ckpts/llm/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        arch="gemma4 (E4B, QAT Q4, MTP)",
        context=131072,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_off_kwargs={"enable_thinking": False},
        thinking_system_token="<|think|>",
        notes="Official sampling; thinking via <|think|> at system start; native MTP draft.",
    ),
    "ornith-1.5-9b": ModelProfile(
        key="ornith-1.5-9b",
        repo="ornith-ai/Ornith-1.5-9B-GGUF",
        gguf="ckpts/llm/Ornith-1.5-9B-Q8_0.gguf",
        arch="qwen3.5 (Ornith-1.5 9B dense, reasoning)",
        context=262144,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        notes="Official sampling; built-in Qwen template (do not pass qwen_chat_template.jinja); "
        "thinking by default, tool calls via <tool_call> parsed by llama.cpp --jinja.",
    ),
    "qwen3.8-27b": ModelProfile(
        key="qwen3.8-27b",
        repo="unsloth/Qwen3.8-27B-GGUF",
        gguf="ckpts/llm/Qwen3.8-27B-UD-IQ4_XS.gguf",
        arch="qwen3.8 (dense)",
        context=262144,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        thinking_off_kwargs={"enable_thinking": False},
        notes="assets/llm/qwen_chat_template.jinja (froggeric v22.5); reasoning in reasoning_content.",
    ),
}

DEFAULT_PROFILE = "qwen3.5-9b"


def get_profile(name: str | None = None) -> ModelProfile:
    return PROFILES.get(name or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = DEFAULT_MIN_P
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_tokens: int = 4096
    thinking_system_token: str = ""


def config_from_env() -> LLMConfig | None:
    base_url = os.environ.get("AUDIOBOOK_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("AUDIOBOOK_LLM_API_KEY") or os.environ.get("LLM_API_KEY") or "sk-local"
    model = os.environ.get("AUDIOBOOK_LLM_MODEL") or os.environ.get("LLM_MODEL_NAME")
    if not base_url or not model:
        return None
    profile = get_profile(os.environ.get("AUDIOBOOK_LLM_PROFILE"))
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=profile.temperature,
        top_p=profile.top_p,
        top_k=profile.top_k,
        min_p=profile.min_p,
        presence_penalty=profile.presence_penalty,
        repetition_penalty=profile.repetition_penalty,
        max_tokens=int(os.environ.get("AUDIOBOOK_LLM_MAX_TOKENS", profile.max_tokens)),
        thinking_system_token=profile.thinking_system_token,
    )


def prompt_hash(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def reasoning_of(message) -> str | None:
    """``reasoning_content`` for llama.cpp ``--reasoning-format deepseek`` replies."""
    value = getattr(message, "reasoning_content", None)
    if value:
        return str(value)
    extra = getattr(message, "model_extra", None) or {}
    return extra.get("reasoning_content")


def raw_log(record: dict) -> None:
    """Append one raw LLM I/O record to ``AUDIOBOOK_LLM_RAW_LOG`` (JSONL) when set."""
    path = os.environ.get("AUDIOBOOK_LLM_RAW_LOG")
    if not path:
        return
    record["ts"] = round(time.time(), 3)
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def extract_json(text: str) -> dict | list:
    """Best-effort JSON extraction from an LLM reply (handles code fences / prose)."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "{[":
            try:
                payload, _ = decoder.raw_decode(text[index:])
                return payload
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no valid JSON in model reply: {text[:200]!r}")


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None

    @classmethod
    def from_env(cls) -> LLMClient | None:
        config = config_from_env()
        return cls(config) if config else None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=float(os.environ.get("AUDIOBOOK_LLM_TIMEOUT", 300)),
            )
        return self._client

    @property
    def model_id(self) -> str:
        return self.config.model

    def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        thinking: bool | None = None,
        json_mode: bool = False,
    ) -> str:
        config = self.config
        key = prompt_hash(
            config.model,
            str(config.temperature),
            str(config.top_k),
            str(config.presence_penalty),
            str(thinking),
            str(json_mode),
            json.dumps(messages, ensure_ascii=False),
        )
        cache_path = Path(os.environ.get("AUDIOBOOK_LLM_CACHE", ".cache/llm")) / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]
        extra_body = {
            "top_k": config.top_k,
            "min_p": config.min_p,
            "repetition_penalty": config.repetition_penalty,
        }
        if thinking is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": thinking}
        extra: dict = {}
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        raw_log(
            {
                "kind": "request",
                "source": "llm",
                "model": config.model,
                "messages": messages,
                "thinking": thinking,
                "json_mode": json_mode,
                "max_tokens": max_tokens or config.max_tokens,
            }
        )
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            top_p=config.top_p,
            presence_penalty=config.presence_penalty,
            max_tokens=max_tokens or config.max_tokens,
            extra_body=extra_body,
            **extra,
        )
        message = response.choices[0].message
        text = message.content or ""
        raw_log(
            {
                "kind": "response",
                "source": "llm",
                "model": config.model,
                "duration_s": round(time.perf_counter() - started, 2),
                "content": message.content,
                "reasoning": reasoning_of(message),
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            }
        )
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
        return text

    def chat_json(
        self,
        system: str,
        user: str,
        max_tokens: int | None = None,
        thinking: bool | None = None,
        json_mode: bool = True,
    ) -> dict | list:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return extract_json(self.chat(messages, max_tokens, thinking=thinking, json_mode=json_mode))

    def chat_json_messages(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        thinking: bool | None = None,
        json_mode: bool = True,
    ) -> dict | list:
        return extract_json(self.chat(messages, max_tokens, thinking=thinking, json_mode=json_mode))
