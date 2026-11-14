from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _dumps(value: Any, default: str = "[]") -> str:
    if value is None:
        return default
    return json.dumps(value, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class SqliteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init_schema(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text()
        with self._connect() as conn:
            conn.executescript(schema)

    @staticmethod
    def _customer(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "aliases": _loads(row["aliases"], []),
            "consent_scopes": _loads(row["consent_scopes"], []),
            "metadata": _loads(row["metadata"], {}),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _interaction(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "customer_id": row["customer_id"],
            "transcript_id": row["transcript_id"],
            "summary": row["summary"],
            "resolution": row["resolution"],
            "intent": row["intent"],
            "action_items": _loads(row["action_items"], []),
            "occurred_at": row["occurred_at"],
            "scope": row["scope"],
            "session_id": row["session_id"],
            "run_id": row["run_id"],
        }

    @staticmethod
    def _ticket(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "customer_id": row["customer_id"],
            "subject": row["subject"],
            "status": row["status"],
            "priority": row["priority"],
            "description": row["description"],
            "scope": row["scope"],
            "session_id": row["session_id"],
            "updated_at": row["updated_at"],
        }

    def upsert_customer(self, customer: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO customers
                    (id, name, aliases, consent_scopes, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    aliases = excluded.aliases,
                    consent_scopes = excluded.consent_scopes,
                    metadata = excluded.metadata
                """,
                (
                    customer["id"],
                    customer["name"],
                    _dumps(customer.get("aliases"), "[]"),
                    _dumps(customer.get("consent_scopes"), "[]"),
                    _dumps(customer.get("metadata"), "{}"),
                    customer.get("created_at") or _now(),
                ),
            )

    def get_customer(self, customer_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM customers WHERE id = ?", (customer_id,)
            ).fetchone()
        return self._customer(row) if row else None

    def list_customers(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM customers ORDER BY id").fetchall()
        return [self._customer(row) for row in rows]

    def add_interaction(self, interaction: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO interactions
                    (id, customer_id, transcript_id, summary, resolution, intent,
                     action_items, occurred_at, scope, session_id, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    transcript_id = excluded.transcript_id,
                    summary = excluded.summary,
                    resolution = excluded.resolution,
                    intent = excluded.intent,
                    action_items = excluded.action_items,
                    occurred_at = excluded.occurred_at,
                    scope = excluded.scope,
                    session_id = excluded.session_id,
                    run_id = excluded.run_id
                """,
                (
                    interaction["id"],
                    interaction["customer_id"],
                    interaction["transcript_id"],
                    interaction["summary"],
                    interaction.get("resolution"),
                    interaction.get("intent"),
                    _dumps(interaction.get("action_items"), "[]"),
                    interaction["occurred_at"],
                    interaction.get("scope", "seed"),
                    interaction.get("session_id"),
                    interaction.get("run_id"),
                ),
            )

    def list_interactions(
        self,
        customer_id: str,
        before: str | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM interactions WHERE customer_id = ?"
        params: list[Any] = [customer_id]
        if session_id is None:
            query += " AND scope = 'seed'"
        else:
            query += " AND (scope = 'seed' OR session_id = ?)"
            params.append(session_id)
        if before is not None:
            query += " AND occurred_at < ?"
            params.append(before)
        query += " ORDER BY occurred_at, rowid"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._interaction(row) for row in rows]

    def add_ticket(self, ticket: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tickets
                    (id, customer_id, subject, status, priority, description,
                     scope, session_id, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    subject = excluded.subject,
                    status = excluded.status,
                    priority = excluded.priority,
                    description = excluded.description,
                    scope = excluded.scope,
                    session_id = excluded.session_id,
                    updated_at = excluded.updated_at
                """,
                (
                    ticket["id"],
                    ticket["customer_id"],
                    ticket["subject"],
                    ticket["status"],
                    ticket["priority"],
                    ticket.get("description", ""),
                    ticket.get("scope", "seed"),
                    ticket.get("session_id"),
                    ticket.get("updated_at") or _now(),
                ),
            )

    def get_ticket(self, ticket_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
            ).fetchone()
        return self._ticket(row) if row else None

    def get_open_tickets(self, customer_id: str, session_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM tickets
                WHERE customer_id = ?
                  AND status = 'open'
                  AND (scope = 'seed' OR session_id = ?)
                ORDER BY updated_at, rowid
                """,
                (customer_id, session_id),
            ).fetchall()
        return [self._ticket(row) for row in rows]

    def update_ticket(self, ticket_id: str, fields: dict) -> None:
        allowed = {"subject", "status", "priority", "description"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = fields.get("updated_at") or _now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tickets SET {assignments} WHERE id = ?",
                (*updates.values(), ticket_id),
            )

    def add_refund(self, refund: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO refunds
                    (id, customer_id, amount, reason, scope, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    refund["id"],
                    refund["customer_id"],
                    refund["amount"],
                    refund["reason"],
                    refund.get("scope", "session"),
                    refund.get("session_id"),
                    refund.get("created_at") or _now(),
                ),
            )

    def add_followup(self, followup: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO followups
                    (id, customer_id, due_date, channel, scope, session_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    followup["id"],
                    followup["customer_id"],
                    followup["due_date"],
                    followup["channel"],
                    followup.get("scope", "session"),
                    followup.get("session_id"),
                    followup.get("created_at") or _now(),
                ),
            )

    def record_tool_call(self, tool_call: dict) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO tool_calls
                    (id, run_id, session_id, agent, tool, args_hash, status,
                     result, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call["id"],
                    tool_call["run_id"],
                    tool_call["session_id"],
                    tool_call["agent"],
                    tool_call["tool"],
                    tool_call["args_hash"],
                    tool_call["status"],
                    tool_call.get("result"),
                    tool_call.get("created_at") or _now(),
                ),
            )
            return cursor.rowcount == 1

    def get_tool_call(self, tool_call_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tool_calls WHERE id = ?", (tool_call_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "agent": row["agent"],
            "tool": row["tool"],
            "args_hash": row["args_hash"],
            "status": row["status"],
            "result": row["result"],
            "created_at": row["created_at"],
        }

    def count_tool_calls(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]

    def add_audit_event(self, event: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events
                    (tool_call_id, run_id, session_id, agent, tool, args_hash,
                     event_type, latency_ms, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["tool_call_id"],
                    event["run_id"],
                    event["session_id"],
                    event["agent"],
                    event["tool"],
                    event["args_hash"],
                    event["event_type"],
                    event.get("latency_ms"),
                    _dumps(event.get("detail"), "{}"),
                    event.get("created_at") or _now(),
                ),
            )

    def list_audit_events(self, run_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "tool_call_id": row["tool_call_id"],
                "run_id": row["run_id"],
                "session_id": row["session_id"],
                "agent": row["agent"],
                "tool": row["tool"],
                "args_hash": row["args_hash"],
                "event_type": row["event_type"],
                "latency_ms": row["latency_ms"],
                "detail": _loads(row["detail"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def add_run(self, run: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs
                    (id, session_id, transcript_id, status, started_at,
                     finished_at, confidence, escalated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    finished_at = excluded.finished_at,
                    confidence = excluded.confidence,
                    escalated = excluded.escalated
                """,
                (
                    run["id"],
                    run["session_id"],
                    run["transcript_id"],
                    run.get("status", "running"),
                    run.get("started_at") or _now(),
                    run.get("finished_at"),
                    run.get("confidence"),
                    int(run.get("escalated", False)),
                ),
            )

    def finish_run(self, run_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, _now(), run_id),
            )

    def count_runs(self, session_id: str, since: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM runs WHERE session_id = ? AND started_at >= ?",
                (session_id, since),
            ).fetchone()[0]
