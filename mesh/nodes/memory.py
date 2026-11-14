from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel

from mesh.state import TraceEvent

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "memory_note.md"
MIN_SIMILARITY = 0.25


class MemoryNoteOutput(BaseModel):
    note: str


def _cited_ids(note: str) -> set[str]:
    return set(re.findall(r"int_[a-z0-9_]+", note))


def _fallback_note(hits: list[dict]) -> str:
    top = hits[0]
    if top.get("resolution"):
        return (
            f"Matches {top['interaction_id']}. "
            f"Prior resolution: {top['resolution']}."
        )
    return f"Matches {top['interaction_id']}. No resolution recorded."


def memory_node(state: dict, deps) -> dict:
    trace = list(state.get("trace", []))
    customer_id = state["extraction"]["customer_id"]
    embedding = deps.embedder.embed(state["transcript"])
    hits = deps.memory.search(
        customer_id=customer_id,
        query_embedding=embedding,
        top_k=3,
        before=state.get("transcript_occurred_at"),
        session_id=state.get("session_id"),
        exclude_run_id=state.get("run_id"),
    )
    hits = [hit for hit in hits if hit.get("similarity", 0.0) >= MIN_SIMILARITY]

    if not hits:
        trace.append(TraceEvent(
            node="memory",
            model=None,
            detail={"hits": 0},
        ).model_dump())
        return {
            "memories": [],
            "memory_note": "No prior interactions.",
            "trace": trace,
        }

    messages = [
        {"role": "system", "content": PROMPT_PATH.read_text()},
        {
            "role": "user",
            "content": (
                "Current extraction:\n"
                f"{json.dumps(state['extraction'], indent=2)}\n\n"
                "Prior interactions:\n"
                f"{json.dumps(hits, indent=2)}"
            ),
        },
    ]

    note: str | None = None
    model: str | None = None
    tokens_in = 0
    tokens_out = 0
    cost = 0.0
    latency_ms = 0
    try:
        parsed, response, cost = deps.router.complete_fixed(
            provider="groq",
            model=deps.settings.groq_fast_model,
            messages=messages,
            schema=MemoryNoteOutput,
        )
        model = response.model
        tokens_in = response.input_tokens
        tokens_out = response.output_tokens
        latency_ms = response.latency_ms
        retrieved = {hit["interaction_id"] for hit in hits}
        cited = _cited_ids(parsed.note)
        if cited and cited <= retrieved:
            note = parsed.note
    except Exception:
        note = None

    if note is None:
        note = _fallback_note(hits)

    trace.append(TraceEvent(
        node="memory",
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        latency_ms=latency_ms,
        detail={"hits": len(hits)},
    ).model_dump())

    return {
        "memories": hits,
        "memory_note": note,
        "trace": trace,
    }
