"""Chat-completions LLM adapter, provider-agnostic by injection.

Takes an already-built client (openai.OpenAI, openai.AzureOpenAI, or a stand-in) and
never names a provider; all wiring lives in app/config.py. Behaviour flags cover the
backend differences: the thinking toggle (self-hosted only, never sent to Azure),
json_mode (Azure/OpenAI), and the reasoning dialect (gpt-5/o-series use
max_completion_tokens and reject temperature/seed). The caller validates the output.
"""
from __future__ import annotations

import json
import time
from typing import Any

from app.domain.ports import LLMResponse


class OpenAIChatLLM:
    def __init__(
        self,
        client: Any,                     # anything exposing .chat.completions.create
        *,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        template_thinking: bool = False,
        json_mode: bool = False,
        reasoning: bool = False,
    ) -> None:
        self._client = client
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._template_thinking = template_thinking
        self._json_mode = json_mode
        self._reasoning = reasoning

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None,
        model: str,
        prompt_version: str,
        temperature: float = 0.0,
        seed: int | None = 0,
    ) -> LLMResponse:
        system = system_prompt
        request: dict = {"model": model}
        if self._reasoning:
            # Reasoning models (o-series, gpt-5): the cap is max_completion_tokens
            # (shared with hidden reasoning tokens, so it must be generous), and
            # temperature/seed are rejected; only the model default is allowed.
            request["max_completion_tokens"] = self._max_tokens
        else:
            request["temperature"] = temperature
            request["max_tokens"] = self._max_tokens
            if seed is not None:
                request["seed"] = seed  # best-effort determinism; ignored where unsupported

        if response_schema is not None:
            system = (
                f"{system_prompt}\n\n"
                "Reply with STRICT JSON only --- no prose, no code fences --- "
                "matching exactly this JSON Schema:\n"
                f"{json.dumps(response_schema, ensure_ascii=False)}"
            )
            if self._json_mode:
                request["response_format"] = {"type": "json_object"}

        if self._template_thinking and not self._enable_thinking:
            # Open-model reasoning switch. Verified on mlx_lm.server: 546 -> 8 output
            # tokens (18.7s -> 0.5s). Only set for backends that understand it.
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        request["messages"] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

        start = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        usage = response.usage  # None on servers that omit token accounting
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model or model,
            prompt_version=prompt_version,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
