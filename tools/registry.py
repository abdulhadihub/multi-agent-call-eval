from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field


class LookupCustomerInput(BaseModel):
    name: str


class GetOpenTicketsInput(BaseModel):
    customer_id: str


class CreateTicketInput(BaseModel):
    customer_id: str
    subject: str
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    description: str = ""


class UpdateTicketInput(BaseModel):
    ticket_id: str
    fields: dict[str, Any]


class IssueRefundInput(BaseModel):
    customer_id: str
    amount: float = Field(gt=0)
    reason: str


class ScheduleFollowupInput(BaseModel):
    customer_id: str
    date: str
    channel: Literal["email", "phone", "sms"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    is_write: bool
    consent_scope: str
    amount_cap: float | None = None


TOOLS: dict[str, ToolSpec] = {
    "lookup_customer": ToolSpec(
        name="lookup_customer",
        description="Look up a customer by name or alias.",
        input_model=LookupCustomerInput,
        is_write=False,
        consent_scope="crm:read",
    ),
    "get_open_tickets": ToolSpec(
        name="get_open_tickets",
        description="List open support tickets for a customer.",
        input_model=GetOpenTicketsInput,
        is_write=False,
        consent_scope="crm:read",
    ),
    "create_ticket": ToolSpec(
        name="create_ticket",
        description="Create a support ticket for a customer.",
        input_model=CreateTicketInput,
        is_write=True,
        consent_scope="crm:write",
    ),
    "update_ticket": ToolSpec(
        name="update_ticket",
        description="Update fields on an existing support ticket.",
        input_model=UpdateTicketInput,
        is_write=True,
        consent_scope="crm:write",
    ),
    "issue_refund": ToolSpec(
        name="issue_refund",
        description="Issue a refund to a customer, up to the policy cap.",
        input_model=IssueRefundInput,
        is_write=True,
        consent_scope="refund",
        amount_cap=500.0,
    ),
    "schedule_followup": ToolSpec(
        name="schedule_followup",
        description="Schedule a follow-up with a customer.",
        input_model=ScheduleFollowupInput,
        is_write=True,
        consent_scope="crm:write",
    ),
}


def catalog_for(
    agent_scopes: set[str],
    consent_scopes: set[str],
) -> list[dict]:
    catalog = []
    for spec in TOOLS.values():
        if spec.is_write and "crm:propose" not in agent_scopes:
            continue
        if not spec.is_write and "crm:read" not in agent_scopes:
            continue
        if spec.consent_scope not in consent_scopes:
            continue
        catalog.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_model.model_json_schema(),
        })
    return catalog
