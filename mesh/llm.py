from __future__ import annotations

import time
from typing import Protocol

from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class LLMClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> LLMResponse: ...


class GroqClient:
    def __init__(self, api_key: str):
        from groq import Groq

        self.client = Groq(api_key=api_key)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
        )


class GeminiClient:
    def __init__(self, api_key: str):
        from google import genai

        self.client = genai.Client(api_key=api_key)

    def complete(
        self,
        *,
        model: str,
        messages: list[dict],
        json_mode: bool = True,
        temperature: float = 0.0,
    ) -> LLMResponse:
        started = time.perf_counter()
        contents = "\n\n".join(
            f"{message.get('role', 'user').upper()}: {message.get('content', '')}"
            for message in messages
        )
        config = {"temperature": temperature}
        if json_mode:
            config["response_mime_type"] = "application/json"
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=getattr(response, "text", "") or "",
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_ms=latency_ms,
        )
