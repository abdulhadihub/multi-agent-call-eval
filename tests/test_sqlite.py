from stores.sqlite import SqliteStore


def _add_acme(store):
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })


def test_store_round_trip(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })
    assert store.get_customer("acme")["name"] == "Acme Logistics"

    store.add_interaction({
        "id": "int_acme_001", "customer_id": "acme", "transcript_id": "acme_001",
        "summary": "Double charge", "resolution": "Refund issued",
        "intent": "billing_dispute", "action_items": [],
        "occurred_at": "2026-08-01T10:00:00Z", "scope": "seed",
        "session_id": None, "run_id": None,
    })
    rows = store.list_interactions("acme")
    assert len(rows) == 1
    assert rows[0]["resolution"] == "Refund issued"


def test_tool_call_is_idempotent(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    row = {
        "id": "call-1", "run_id": "run-1", "session_id": "s1",
        "agent": "executor", "tool": "issue_refund", "args_hash": "abc",
        "status": "executed", "result": "{}", "created_at": "2026-08-01T00:00:00Z",
    }
    assert store.record_tool_call(row) is True
    assert store.record_tool_call(row) is False
    assert store.count_tool_calls() == 1


def test_session_scoped_tickets(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    _add_acme(store)
    store.add_ticket({
        "id": "T-1", "customer_id": "acme", "subject": "Seed",
        "status": "open", "priority": "normal", "description": "",
        "scope": "seed", "session_id": None, "updated_at": "2026-08-01T00:00:00Z",
    })
    store.add_ticket({
        "id": "T-2", "customer_id": "acme", "subject": "Session",
        "status": "open", "priority": "high", "description": "",
        "scope": "session", "session_id": "s1", "updated_at": "2026-08-01T00:00:00Z",
    })
    assert [t["id"] for t in store.get_open_tickets("acme", "s1")] == ["T-1", "T-2"]
    assert [t["id"] for t in store.get_open_tickets("acme", "s2")] == ["T-1"]


def test_get_ticket_resolves_customer(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    _add_acme(store)
    store.add_ticket({
        "id": "T-1", "customer_id": "acme", "subject": "Seed",
        "status": "open", "priority": "normal", "description": "",
        "scope": "seed", "session_id": None, "updated_at": "2026-08-01T00:00:00Z",
    })
    assert store.get_ticket("T-1")["customer_id"] == "acme"
    assert store.get_ticket("missing") is None
