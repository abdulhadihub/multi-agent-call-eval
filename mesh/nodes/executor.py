from __future__ import annotations

from dataclasses import asdict, is_dataclass

from mesh.state import TraceEvent
from tools.gateway import ToolDenied


def _result_dict(result):
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


def executor_node(state: dict, deps) -> dict:
    trace = list(state.get("trace", []))
    proposal = state.get("proposal") or {}
    calls = proposal.get("tool_calls", [])

    if not calls:
        return {"tool_results": []}

    approval = state.get("approval") or {}
    if not approval.get("approved"):
        trace.append(TraceEvent(
            node="executor",
            event="skipped",
            detail={"reason": "rejected"},
        ).model_dump())
        return {"tool_results": [], "trace": trace}

    customer = state.get("customer")
    denied: list[tuple[dict, str]] = []
    for call in calls:
        try:
            deps.gateway.validate(
                agent="executor",
                tool=call["tool"],
                args=call["arguments"],
                customer=customer,
            )
        except ToolDenied as exc:
            denied.append((call, exc.error_code))

    if denied:
        for call, error_code in denied:
            deps.store.add_audit_event({
                "tool_call_id": call.get("tool_call_id", "unknown"),
                "run_id": state["run_id"],
                "session_id": state["session_id"],
                "agent": "executor",
                "tool": call.get("tool", "unknown"),
                "args_hash": "preflight",
                "event_type": "denied",
                "detail": {"error": error_code},
            })
        trace.append(TraceEvent(
            node="executor",
            event="escalated",
            detail={"denied": [error_code for _, error_code in denied]},
        ).model_dump())
        return {
            "tool_results": [],
            "trace": trace,
            "escalated": True,
            "escalation_reason": "tool_denied",
        }

    results = []
    for call in calls:
        result = deps.gateway.execute(
            agent="executor",
            tool=call["tool"],
            args=call["arguments"],
            run_id=state["run_id"],
            session_id=state["session_id"],
            approved=True,
            customer=customer,
        )
        results.append(_result_dict(result))

    trace.append(TraceEvent(
        node="executor",
        event="completed",
        detail={"tool_results": len(results)},
    ).model_dump())
    return {"tool_results": results, "trace": trace}
