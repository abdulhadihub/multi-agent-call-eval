from __future__ import annotations

from mesh.state import TraceEvent


def escalate_node(state: dict, deps) -> dict:
    trace = list(state.get("trace", []))
    reason = state.get("escalation_reason") or "escalated"
    trace.append(TraceEvent(
        node="escalate",
        event="escalated",
        detail={"reason": reason},
    ).model_dump())
    return {
        "escalated": True,
        "escalation_reason": reason,
        "trace": trace,
        "memories": state.get("memories", []),
    }
