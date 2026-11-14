from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mesh.graph import build_graph
from stores.sqlite import SqliteStore
from tests.conftest import fake_router, make_deps
from tools.gateway import ToolGateway
from tools.simulated_apis import SimulatedCompanyAPI


def run_to_interrupt(graph, state, config):
    for event in graph.stream(state, config, stream_mode="updates"):
        if "__interrupt__" in event:
            return event["__interrupt__"][0].value
    return None


def test_happy_path_approves_and_executes():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        '{"note":"No prior interactions."}',
        '{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Charged twice"},"rationale":"Track it","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    state = {
        "run_id": "r1", "session_id": "s1", "transcript_id": "acme_003",
        "transcript_occurred_at": "2026-09-02T10:00:00Z",
        "transcript": "Hi, this is Priya from Acme Logistics. Charged twice.",
        "trace": [], "errors": [],
    }
    payload = run_to_interrupt(graph, state, config)
    assert payload["proposal"]["tool_calls"][0]["tool"] == "create_ticket"
    result = graph.invoke(
        Command(resume={"approved": True, "approved_by": "demo_user"}),
        config,
    )
    assert result["tool_results"][0]["status"] == "executed"
    assert result["approval"]["approved"] is True


def test_low_confidence_stops_before_memory_and_action():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Atlas Freight","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.1}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    state = {
        "run_id": "r2", "session_id": "s2", "transcript_id": "atlas_002",
        "transcript_occurred_at": "2026-09-16T10:00:00Z",
        "transcript": "Hi, this is Atlas Freight. It is not working.",
        "trace": [], "errors": [],
    }
    result = graph.invoke(state, config)
    assert result["escalated"] is True
    assert result["escalation_reason"] == "low_confidence"
    assert result["memories"] == []


def _real_deps(tmp_path, router):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })
    store.upsert_customer({
        "id": "blueharbor", "name": "Blue Harbor Retail",
        "aliases": ["Blue Harbor"],
        "consent_scopes": ["crm:read", "crm:write"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })
    gateway = ToolGateway(store, SimulatedCompanyAPI(store), None)
    deps = make_deps(store=store, gateway=gateway, router=router)
    return store, deps


def _acme_state(run_id):
    return {
        "run_id": run_id, "session_id": "s1", "transcript_id": "acme_003",
        "transcript_occurred_at": "2026-09-02T10:00:00Z",
        "transcript": "Hi, this is Priya from Acme Logistics. Charged twice.",
        "trace": [], "errors": [],
    }


def _refund_router(customer_id, amount):
    return fake_router(
        '{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        '{"note":"No prior interactions."}',
        '{"tool_calls":[{"tool":"issue_refund","arguments":{"customer_id":"'
        + customer_id
        + '","amount":'
        + str(amount)
        + ',"reason":"demo"},"rationale":"Refund","requires_approval":true}],'
        '"rationale":"Refund","risk_level":"high","requires_approval":true}',
    )


def test_real_gateway_blocks_over_cap_refund(tmp_path):
    store, deps = _real_deps(
        tmp_path, _refund_router("acme", 5000)
    )
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "real-cap"}}
    payload = run_to_interrupt(graph, _acme_state("r1"), config)
    assert payload["proposal"]["tool_calls"][0]["tool"] == "issue_refund"

    result = graph.invoke(
        Command(resume={"approved": True, "approved_by": "demo_user"}),
        config,
    )
    assert result["escalated"] is True
    assert result["escalation_reason"] == "tool_denied"
    events = store.list_audit_events("r1")
    assert any(
        event["event_type"] == "denied"
        and event["detail"].get("error") == "amount_cap_exceeded"
        for event in events
    )


def test_real_gateway_blocks_mismatched_customer(tmp_path):
    store, deps = _real_deps(
        tmp_path, _refund_router("blueharbor", 10)
    )
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "real-mismatch"}}
    run_to_interrupt(graph, _acme_state("r2"), config)

    result = graph.invoke(
        Command(resume={"approved": True, "approved_by": "demo_user"}),
        config,
    )
    assert result["escalated"] is True
    assert result["escalation_reason"] == "tool_denied"
    events = store.list_audit_events("r2")
    assert any(
        event["event_type"] == "denied"
        and event["detail"].get("error") == "customer_mismatch"
        for event in events
    )


def test_rejection_writes_audit_and_skips_execution():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        '{"note":"No prior interactions."}',
        '{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Charged twice"},"rationale":"Track it","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t3"}}
    state = {
        "run_id": "r3", "session_id": "s3", "transcript_id": "acme_003",
        "transcript_occurred_at": "2026-09-02T10:00:00Z",
        "transcript": "Hi, this is Priya from Acme Logistics. Charged twice.",
        "trace": [], "errors": [],
    }
    run_to_interrupt(graph, state, config)
    result = graph.invoke(
        Command(resume={"approved": False, "approved_by": "demo_user"}),
        config,
    )
    assert result["approval"]["approved"] is False
    assert result["tool_results"] == []
    assert [e["event_type"] for e in deps.store.list_audit_events("r3")] == ["proposed", "rejected"]
