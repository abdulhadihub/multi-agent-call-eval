from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from mesh.state import TraceEvent
from tools.registry import TOOLS, catalog_for

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "action.md"


class ProposedToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    requires_approval: bool = True


class ActionOutput(BaseModel):
    tool_calls: list[ProposedToolCall] = Field(default_factory=list)
    rationale: str = ""
    risk_level: str = "low"
    requires_approval: bool = False


def _fail(trace: list, errors: list, reason: str) -> dict:
    return {
        "trace": trace,
        "errors": errors,
        "escalated": True,
        "escalation_reason": reason,
    }


def action_node(state: dict, deps) -> dict:
    trace = list(state.get("trace", []))
    errors = list(state.get("errors", []))

    customer = state.get("customer")
    if customer is None:
        customer = deps.store.get_customer(state["extraction"]["customer_id"])
    if customer is None:
        return _fail(trace, errors + ["customer_not_found"], "customer_not_found")

    catalog = catalog_for(
        {"crm:propose"},
        set(customer.get("consent_scopes", [])),
    )
    messages = [
        {"role": "system", "content": PROMPT_PATH.read_text()},
        {
            "role": "user",
            "content": (
                "Extraction:\n"
                f"{json.dumps(state.get('extraction', {}), indent=2, default=str)}\n\n"
                "Memory note:\n"
                f"{state.get('memory_note', '')}\n\n"
                "Open tickets:\n"
                f"{json.dumps(state.get('tickets', []), indent=2, default=str)}\n\n"
                "Tool catalog:\n"
                f"{json.dumps(catalog, indent=2, default=str)}"
            ),
        },
    ]

    try:
        parsed, response, cost = deps.router.complete_json(
            task="propose",
            messages=messages,
            schema=ActionOutput,
        )
    except Exception as exc:
        return _fail(trace, errors + [str(exc)], "action_failed")

    catalog_names = {item["name"] for item in catalog}
    tool_calls: list[dict] = []
    for call in parsed.tool_calls:
        spec = TOOLS.get(call.tool)
        if spec is None or not spec.is_write or call.tool not in catalog_names:
            trace.append(TraceEvent(
                node="action",
                event="failed",
                detail={"error": "invalid_tool", "tool": call.tool},
            ).model_dump())
            return _fail(trace, errors + ["invalid_tool"], "invalid_tool")

        try:
            spec.input_model(**call.arguments)
        except ValidationError as exc:
            trace.append(TraceEvent(
                node="action",
                event="failed",
                detail={"error": "invalid_arguments", "tool": call.tool},
            ).model_dump())
            return _fail(trace, errors + [str(exc)], "invalid_arguments")

        tool_call_id = deps.gateway.make_tool_call_id(
            state["run_id"],
            call.tool,
            call.arguments,
        )
        tool_calls.append({
            "tool": call.tool,
            "arguments": call.arguments,
            "rationale": call.rationale,
            "requires_approval": True,
            "tool_call_id": tool_call_id,
        })

    proposal = {
        "tool_calls": tool_calls,
        "rationale": parsed.rationale,
        "risk_level": parsed.risk_level,
        "requires_approval": bool(tool_calls),
    }
    trace.append(TraceEvent(
        node="action",
        model=response.model,
        tokens_in=response.input_tokens,
        tokens_out=response.output_tokens,
        cost_usd=cost,
        latency_ms=response.latency_ms,
        detail={
            "tool_calls": len(tool_calls),
            "risk_level": parsed.risk_level,
        },
    ).model_dump())

    return {
        "proposal": proposal,
        "trace": trace,
        "escalated": False,
        "escalation_reason": None,
    }
