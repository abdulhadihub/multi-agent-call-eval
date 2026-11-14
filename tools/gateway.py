from __future__ import annotations

import json
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from pydantic import ValidationError

from stores.sqlite import SqliteStore
from tools.registry import TOOLS
from tools.simulated_apis import SimulatedCompanyAPI

AGENT_SCOPES: dict[str, set[str]] = {
    "extractor": {"crm:read"},
    "memory": {"history:read"},
    "action": {"crm:propose"},
    "executor": {"crm:execute"},
}


class ToolDenied(Exception):
    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


@dataclass
class ToolResult:
    tool_call_id: str
    tool: str
    status: str
    result: Any = None
    error: str | None = None
    latency_ms: int = 0
    duplicate: bool = False


def _args_hash(args: dict) -> str:
    payload = json.dumps(args, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


class ToolGateway:
    def __init__(
        self,
        store: SqliteStore,
        api: SimulatedCompanyAPI,
        settings: Any = None,
    ):
        self.store = store
        self.api = api
        self.settings = settings

    def make_tool_call_id(self, run_id: str, tool: str, args: dict) -> str:
        payload = f"{run_id}|{tool}|{json.dumps(args, sort_keys=True, default=str)}"
        return sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_customer(
        self,
        tool: str,
        args: dict,
        customer: dict | None,
    ) -> dict | None:
        if customer is not None:
            return customer
        if "customer_id" in args:
            return self.store.get_customer(args["customer_id"])
        if "ticket_id" in args:
            ticket = self.store.get_ticket(args["ticket_id"])
            if ticket:
                return self.store.get_customer(ticket["customer_id"])
        if tool == "lookup_customer" and "name" in args:
            return self.api.lookup_customer(args["name"])
        return None

    def validate(
        self,
        agent: str,
        tool: str,
        args: dict,
        customer: dict | None,
    ) -> None:
        spec = TOOLS.get(tool)
        if spec is None:
            raise ToolDenied("unknown_tool")

        scopes = AGENT_SCOPES.get(agent, set())
        if spec.is_write:
            if not ({"crm:propose", "crm:execute"} & scopes):
                raise ToolDenied("scope_denied")
        elif "crm:read" not in scopes:
            raise ToolDenied("scope_denied")

        if tool != "lookup_customer" and customer is None:
            raise ToolDenied("customer_not_found")

        if (
            spec.consent_scope
            and customer is not None
            and spec.consent_scope not in customer.get("consent_scopes", [])
        ):
            raise ToolDenied("consent_denied")

        try:
            spec.input_model(**args)
        except ValidationError as exc:
            raise ToolDenied("invalid_arguments") from exc

        if spec.amount_cap is not None:
            amount = float(args.get("amount", 0))
            if amount > spec.amount_cap:
                raise ToolDenied("amount_cap_exceeded")

    def execute(
        self,
        *,
        agent: str,
        tool: str,
        args: dict,
        run_id: str,
        session_id: str,
        approved: bool,
        customer: dict | None = None,
    ) -> ToolResult:
        tool_call_id = self.make_tool_call_id(run_id, tool, args)
        args_hash = _args_hash(args)

        self.store.add_audit_event({
            "tool_call_id": tool_call_id,
            "run_id": run_id,
            "session_id": session_id,
            "agent": agent,
            "tool": tool,
            "args_hash": args_hash,
            "event_type": "attempted",
            "detail": {},
        })

        spec = TOOLS.get(tool)
        try:
            if spec is not None and spec.is_write:
                if "crm:execute" not in AGENT_SCOPES.get(agent, set()):
                    raise ToolDenied("scope_denied")
            customer = self._resolve_customer(tool, args, customer)
            self.validate(agent, tool, args, customer)
            if spec is not None and spec.is_write and not approved:
                raise ToolDenied("approval_required")
        except ToolDenied as denied:
            self.store.add_audit_event({
                "tool_call_id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": tool,
                "args_hash": args_hash,
                "event_type": "denied",
                "detail": {"error": denied.error_code},
            })
            return ToolResult(
                tool_call_id=tool_call_id,
                tool=tool,
                status="denied",
                error=denied.error_code,
            )

        existing = self.store.get_tool_call(tool_call_id)
        if existing is not None:
            stored = existing.get("result")
            return ToolResult(
                tool_call_id=tool_call_id,
                tool=tool,
                status=existing["status"],
                result=json.loads(stored) if stored else None,
                duplicate=True,
            )

        started = time.perf_counter()
        try:
            result = self._call(tool, args, session_id)
            latency_ms = int((time.perf_counter() - started) * 1000)
            self.store.record_tool_call({
                "id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": tool,
                "args_hash": args_hash,
                "status": "executed",
                "result": json.dumps(result, sort_keys=True, default=str),
                "created_at": None,
            })
            self.store.add_audit_event({
                "tool_call_id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": tool,
                "args_hash": args_hash,
                "event_type": "executed",
                "latency_ms": latency_ms,
                "detail": {},
            })
            return ToolResult(
                tool_call_id=tool_call_id,
                tool=tool,
                status="executed",
                result=result,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            detail = {"error": str(exc)}
            self.store.record_tool_call({
                "id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": tool,
                "args_hash": args_hash,
                "status": "failed",
                "result": json.dumps(detail, sort_keys=True, default=str),
                "created_at": None,
            })
            self.store.add_audit_event({
                "tool_call_id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": tool,
                "args_hash": args_hash,
                "event_type": "failed",
                "latency_ms": latency_ms,
                "detail": detail,
            })
            return ToolResult(
                tool_call_id=tool_call_id,
                tool=tool,
                status="failed",
                error=str(exc),
                latency_ms=latency_ms,
            )

    def _call(self, tool: str, args: dict, session_id: str) -> Any:
        if tool == "lookup_customer":
            return self.api.lookup_customer(args["name"])
        if tool == "get_open_tickets":
            return self.api.get_open_tickets(args["customer_id"], session_id)
        if tool == "create_ticket":
            return self.api.create_ticket(
                args["customer_id"],
                args["subject"],
                args["priority"],
                args.get("description", ""),
                session_id,
            )
        if tool == "update_ticket":
            return self.api.update_ticket(
                args["ticket_id"],
                args["fields"],
                session_id,
            )
        if tool == "issue_refund":
            return self.api.issue_refund(
                args["customer_id"],
                args["amount"],
                args["reason"],
                session_id,
            )
        if tool == "schedule_followup":
            return self.api.schedule_followup(
                args["customer_id"],
                args["date"],
                args["channel"],
                session_id,
            )
        raise ToolDenied("unknown_tool")

    def _tool_call_parts(self, run_id: str, call: dict) -> tuple[str, str]:
        args = call.get("arguments", {})
        tool_call_id = call.get("tool_call_id") or self.make_tool_call_id(
            run_id, call["tool"], args
        )
        return tool_call_id, _args_hash(args)

    def record_proposed(
        self,
        *,
        run_id: str,
        session_id: str,
        agent: str,
        tool_calls: list[dict],
    ) -> None:
        for call in tool_calls:
            tool_call_id, args_hash = self._tool_call_parts(run_id, call)
            self.store.add_audit_event({
                "tool_call_id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": call["tool"],
                "args_hash": args_hash,
                "event_type": "proposed",
                "detail": {"rationale": call.get("rationale")},
            })

    def record_decision(
        self,
        *,
        run_id: str,
        session_id: str,
        approved: bool,
        approved_by: str,
        tool_calls: list[dict],
    ) -> None:
        event_type = "approved" if approved else "rejected"
        for call in tool_calls:
            tool_call_id, args_hash = self._tool_call_parts(run_id, call)
            self.store.add_audit_event({
                "tool_call_id": tool_call_id,
                "run_id": run_id,
                "session_id": session_id,
                "agent": "approval",
                "tool": call["tool"],
                "args_hash": args_hash,
                "event_type": event_type,
                "detail": {"approved_by": approved_by},
            })
