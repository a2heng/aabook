"""OpenAI-compatible LLM client for offline preprocessing.

Works with any local server (llama.cpp / Ollama / vLLM) exposing an OpenAI API.
Config precedence: ``AUDIOBOOK_LLM_*`` then ``LLM_*`` (the Prompt Enhancer vars).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

# Qwen3.8-27B README "Best Practices" - thinking mode sampling.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_REPETITION_PENALTY = 1.0


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


def config_from_env() -> LLMConfig | None:
    from .models import get_profile

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
    )


def prompt_hash(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


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
        text = response.choices[0].message.content or ""
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
