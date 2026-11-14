from stores.sqlite import SqliteStore
from tools.gateway import ToolGateway
from tools.simulated_apis import SimulatedCompanyAPI


def make_gateway(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": [],
        "consent_scopes": ["crm:read", "crm:write", "refund"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    store.upsert_customer({
        "id": "blueharbor", "name": "Blue Harbor Retail", "aliases": [],
        "consent_scopes": ["crm:read", "crm:write"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    return ToolGateway(store, SimulatedCompanyAPI(store), None)


def test_memory_agent_cannot_write(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="memory", tool="issue_refund",
        args={"customer_id": "acme", "amount": 10, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "scope_denied"


def test_revoked_consent_denies_refund(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="issue_refund",
        args={"customer_id": "blueharbor", "amount": 10, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "consent_denied"


def test_write_requires_approval(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="create_ticket",
        args={"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"},
        run_id="r1", session_id="s1", approved=False,
    )
    assert result.status == "denied"
    assert result.error == "approval_required"


def test_refund_cap_is_enforced(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="issue_refund",
        args={"customer_id": "acme", "amount": 5000, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "amount_cap_exceeded"


def test_duplicate_execution_is_prevented(tmp_path):
    gw = make_gateway(tmp_path)
    args = {"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"}
    first = gw.execute(agent="executor", tool="create_ticket", args=args, run_id="r1", session_id="s1", approved=True)
    second = gw.execute(agent="executor", tool="create_ticket", args=args, run_id="r1", session_id="s1", approved=True)
    assert first.status == "executed"
    assert second.status == "executed"
    assert second.duplicate is True
    assert len(gw.store.get_open_tickets("acme", "s1")) == 1


def test_audit_events_are_append_only(tmp_path):
    gw = make_gateway(tmp_path)
    gw.execute(
        agent="executor", tool="create_ticket",
        args={"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    events = gw.store.list_audit_events("r1")
    assert [e["event_type"] for e in events] == ["attempted", "executed"]


def test_mismatched_customer_id_is_denied(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="issue_refund",
        args={"customer_id": "blueharbor", "amount": 10, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
        customer=gw.store.get_customer("acme"),
    )
    assert result.status == "denied"
    assert result.error == "customer_mismatch"


def test_create_ticket_mismatched_customer_is_denied(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="create_ticket",
        args={
            "customer_id": "blueharbor",
            "subject": "x",
            "priority": "normal",
            "description": "x",
        },
        run_id="r1", session_id="s1", approved=True,
        customer=gw.store.get_customer("acme"),
    )
    assert result.status == "denied"
    assert result.error == "customer_mismatch"


def test_get_open_tickets_mismatched_customer_is_denied(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="extractor", tool="get_open_tickets",
        args={"customer_id": "blueharbor"},
        run_id="r1", session_id="s1", approved=False,
        customer=gw.store.get_customer("acme"),
    )
    assert result.status == "denied"
    assert result.error == "customer_mismatch"


def test_update_ticket_ownership_is_denied(tmp_path):
    gw = make_gateway(tmp_path)
    gw.store.add_ticket({
        "id": "T-3001", "customer_id": "blueharbor", "subject": "Other",
        "status": "open", "priority": "normal", "description": "",
        "scope": "seed", "session_id": None,
        "updated_at": "2026-08-01T00:00:00Z",
    })
    result = gw.execute(
        agent="executor", tool="update_ticket",
        args={"ticket_id": "T-3001", "fields": {"status": "resolved"}},
        run_id="r1", session_id="s1", approved=True,
        customer=gw.store.get_customer("acme"),
    )
    assert result.status == "denied"
    assert result.error == "ticket_ownership_denied"
