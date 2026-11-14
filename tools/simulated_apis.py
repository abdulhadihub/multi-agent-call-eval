from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from stores.sqlite import SqliteStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _deterministic_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    digest = sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{digest}"


class SimulatedCompanyAPI:
    def __init__(self, store: SqliteStore):
        self.store = store

    def lookup_customer(self, name: str) -> dict | None:
        needle = name.strip().lower()
        for customer in self.store.list_customers():
            if customer["name"].lower() == needle:
                return customer
            if any(alias.lower() == needle for alias in customer["aliases"]):
                return customer
        return None

    def get_open_tickets(self, customer_id: str, session_id: str) -> list[dict]:
        return self.store.get_open_tickets(customer_id, session_id)

    def create_ticket(
        self,
        customer_id: str,
        subject: str,
        priority: str,
        description: str,
        session_id: str,
    ) -> dict:
        ticket = {
            "id": _deterministic_id("T", session_id, customer_id, subject, description),
            "customer_id": customer_id,
            "subject": subject,
            "status": "open",
            "priority": priority,
            "description": description,
            "scope": "session",
            "session_id": session_id,
            "updated_at": _now(),
        }
        self.store.add_ticket(ticket)
        return ticket

    def update_ticket(
        self,
        ticket_id: str,
        fields: dict,
        session_id: str,
    ) -> dict:
        self.store.update_ticket(ticket_id, fields)
        ticket = self.store.get_ticket(ticket_id)
        if ticket is None:
            raise ValueError("ticket_not_found")
        return ticket

    def issue_refund(
        self,
        customer_id: str,
        amount: float,
        reason: str,
        session_id: str,
    ) -> dict:
        refund = {
            "id": _deterministic_id("R", session_id, customer_id, amount, reason),
            "customer_id": customer_id,
            "amount": amount,
            "reason": reason,
            "scope": "session",
            "session_id": session_id,
            "created_at": _now(),
        }
        self.store.add_refund(refund)
        return refund

    def schedule_followup(
        self,
        customer_id: str,
        date: str,
        channel: str,
        session_id: str,
    ) -> dict:
        followup = {
            "id": _deterministic_id("F", session_id, customer_id, date, channel),
            "customer_id": customer_id,
            "due_date": date,
            "channel": channel,
            "scope": "session",
            "session_id": session_id,
            "created_at": _now(),
        }
        self.store.add_followup(followup)
        return followup
