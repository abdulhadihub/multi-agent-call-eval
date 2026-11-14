from __future__ import annotations

from langgraph.types import interrupt

from mesh.state import ApprovalDecision, TraceEvent


def approval_node(state: dict, deps) -> dict:
    proposal = state["proposal"]
    decision = interrupt({
        "proposal": proposal,
        "customer": state.get("customer"),
        "run_id": state["run_id"],
    })
    parsed = ApprovalDecision.model_validate(decision)
    tool_calls = proposal.get("tool_calls", [])

    deps.gateway.record_proposed(
        run_id=state["run_id"],
        session_id=state["session_id"],
        agent="action",
        tool_calls=tool_calls,
    )
    deps.gateway.record_decision(
        run_id=state["run_id"],
        session_id=state["session_id"],
        approved=parsed.approved,
        approved_by=parsed.approved_by,
        tool_calls=tool_calls,
    )

    trace = list(state.get("trace", []))
    trace.append(TraceEvent(
        node="approval",
        event="approved" if parsed.approved else "rejected",
        detail={"approved_by": parsed.approved_by},
    ).model_dump())

    return {"approval": parsed.model_dump(), "trace": trace}
