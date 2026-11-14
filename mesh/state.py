from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class ExtractionOutput(BaseModel):
    customer_name: str
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    action_items: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class TraceEvent(BaseModel):
    node: str
    event: str = "completed"
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    detail: dict[str, Any] = Field(default_factory=dict)


class ApprovalDecision(BaseModel):
    approved: bool
    approved_by: str
    note: str | None = None


class MeshState(TypedDict, total=False):
    run_id: str
    session_id: str
    transcript_id: str
    transcript: str
    transcript_occurred_at: str
    customer: dict[str, Any] | None
    extraction: dict[str, Any] | None
    memories: list[dict[str, Any]]
    memory_note: str | None
    proposal: dict[str, Any] | None
    approval: dict[str, Any] | None
    tool_results: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    audit: list[dict[str, Any]]
    confidence: float
    escalated: bool
    escalation_reason: str | None
    errors: list[str]
