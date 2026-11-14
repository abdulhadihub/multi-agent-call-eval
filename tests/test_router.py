import pytest
from pydantic import BaseModel
from mesh.llm import LLMResponse
from mesh.router import ModelRouter, RouterPolicy, TaskPolicy


class Out(BaseModel):
    answer: str


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, model, messages, json_mode=True, temperature=0.0):
        self.calls.append(model)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            model=model,
            input_tokens=10,
            output_tokens=5,
            latency_ms=20,
        )


POLICY = RouterPolicy(tasks={
    "extract": TaskPolicy(
        provider="groq",
        model="strong",
        fallback_provider="gemini",
        fallback_model="flash",
    ),
})


def test_primary_model_success():
    groq = FakeClient(['{"answer": "ok"}'])
    router = ModelRouter({"groq": groq, "gemini": FakeClient([])}, POLICY, None)
    result, response, cost = router.complete_json(task="extract", messages=[], schema=Out)
    assert result.answer == "ok"
    assert groq.calls == ["strong"]
    assert cost >= 0


def test_fallback_on_primary_failure():
    router = ModelRouter(
        {
            "groq": FakeClient([RuntimeError("rate limit")]),
            "gemini": FakeClient(['{"answer": "ok"}']),
        },
        POLICY,
        None,
    )
    result, response, cost = router.complete_json(task="extract", messages=[], schema=Out)
    assert result.answer == "ok"
    assert response.model == "flash"


def test_invalid_json_retries_then_fails():
    router = ModelRouter(
        {
            "groq": FakeClient(["not json", "not json"]),
            "gemini": FakeClient([]),
        },
        POLICY,
        None,
    )
    with pytest.raises(ValueError):
        router.complete_json(task="extract", messages=[], schema=Out)


class RateLimited(Exception):
    status_code = 429


def test_retries_on_rate_limit():
    router = ModelRouter(
        {"groq": FakeClient([RateLimited("slow down"), '{"answer": "ok"}'])},
        POLICY,
        None,
    )
    result, response, cost = router.complete_json(
        task="extract",
        messages=[],
        schema=Out,
    )
    assert result.answer == "ok"
