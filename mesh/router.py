from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from mesh.config import Settings
from mesh.llm import LLMClient, LLMResponse
from mesh.pricing import estimate_cost


class TaskPolicy(BaseModel):
    provider: str
    model: str
    fallback_provider: str | None = None
    fallback_model: str | None = None
    reason: str = ""


class RouterPolicy(BaseModel):
    tasks: dict[str, TaskPolicy]

    @classmethod
    def load(
        cls,
        path: Path | str = Path("eval/router_policy.json"),
        settings: Settings | None = None,
    ) -> "RouterPolicy":
        path = Path(path)
        if path.exists():
            return cls.model_validate(json.loads(path.read_text()))
        if settings is None:
            raise ValueError(
                "Router policy file not found and no settings were provided"
            )
        return cls(tasks={
            "extract": TaskPolicy(
                provider="groq",
                model=settings.groq_model,
                fallback_provider="gemini",
                fallback_model=settings.gemini_model,
                reason="default policy",
            ),
            "propose": TaskPolicy(
                provider="groq",
                model=settings.groq_model,
                fallback_provider="gemini",
                fallback_model=settings.gemini_model,
                reason="default policy",
            ),
        })


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


class ModelRouter:
    def __init__(
        self,
        clients: dict[str, LLMClient],
        policy: RouterPolicy,
        settings: Settings | None = None,
    ):
        self.clients = clients
        self.policy = policy
        self.settings = settings

    def choose(self, task: str) -> TaskPolicy:
        try:
            return self.policy.tasks[task]
        except KeyError as exc:
            raise KeyError(f"No router policy for task {task}") from exc

    def _attempt(
        self,
        provider: str,
        model: str,
        messages: list[dict],
        schema: type[BaseModel],
        retry_invalid: bool = True,
    ) -> tuple[BaseModel, LLMResponse, float]:
        client = self.clients.get(provider)
        if client is None:
            raise ValueError(f"No client configured for provider {provider}")

        response = client.complete(model=model, messages=messages)
        cost = estimate_cost(model, response.input_tokens, response.output_tokens)
        try:
            return schema.model_validate_json(_clean_json(response.text)), response, cost
        except ValidationError as first_error:
            if not retry_invalid:
                raise ValueError(
                    f"Invalid JSON from {provider}/{model}: {first_error}"
                ) from first_error

            retry_messages = list(messages) + [{
                "role": "user",
                "content": (
                    "Your previous response was invalid JSON or did not match "
                    f"the schema: {first_error}. Return only valid JSON."
                ),
            }]
            retry_response = client.complete(model=model, messages=retry_messages)
            cost += estimate_cost(
                model,
                retry_response.input_tokens,
                retry_response.output_tokens,
            )
            try:
                parsed = schema.model_validate_json(_clean_json(retry_response.text))
                return parsed, retry_response, cost
            except ValidationError as second_error:
                raise ValueError(
                    f"Invalid JSON from {provider}/{model}: {second_error}"
                ) from second_error

    def complete_json(
        self,
        *,
        task: str,
        messages: list[dict],
        schema: type[BaseModel],
        force_provider: str | None = None,
        force_model: str | None = None,
    ) -> tuple[BaseModel, LLMResponse, float]:
        if force_provider and force_model:
            return self._attempt(force_provider, force_model, messages, schema)

        policy = self.choose(task)
        errors: list[Exception] = []
        try:
            return self._attempt(policy.provider, policy.model, messages, schema)
        except Exception as exc:
            errors.append(exc)

        if policy.fallback_provider and policy.fallback_model:
            try:
                return self._attempt(
                    policy.fallback_provider,
                    policy.fallback_model,
                    messages,
                    schema,
                )
            except Exception as exc:
                errors.append(exc)

        raise ValueError(
            f"All model attempts failed for task {task}: {errors}"
        ) from errors[-1]

    def complete_fixed(
        self,
        *,
        provider: str,
        model: str,
        messages: list[dict],
        schema: type[BaseModel],
    ) -> tuple[BaseModel, LLMResponse, float]:
        return self._attempt(provider, model, messages, schema)
