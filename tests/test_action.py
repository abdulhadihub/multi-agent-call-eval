from mesh.nodes.action import action_node
from tests.conftest import fake_router, make_deps


def test_action_builds_validated_proposal():
    deps = make_deps(router=fake_router('{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Customer charged twice"},"rationale":"Track the issue","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}'))
    state = {
        "run_id": "r1", "session_id": "s1",
        "extraction": {"customer_id": "acme", "intent": "billing_dispute", "action_items": []},
        "memories": [], "memory_note": "No prior interactions.",
        "trace": [], "errors": [],
    }
    update = action_node(state, deps)
    assert update["proposal"]["requires_approval"] is True
    assert update["proposal"]["tool_calls"][0]["tool"] == "create_ticket"
    assert update["proposal"]["tool_calls"][0]["tool_call_id"].startswith("r1:")


def test_unknown_tool_escalates():
    deps = make_deps(router=fake_router('{"tool_calls":[{"tool":"drop_database","arguments":{},"rationale":"bad","requires_approval":true}],"rationale":"bad","risk_level":"high","requires_approval":true}'))
    state = {
        "run_id": "r1", "session_id": "s1",
        "extraction": {"customer_id": "acme"},
        "memories": [], "memory_note": "",
        "trace": [], "errors": [],
    }
    update = action_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "invalid_tool"


def test_no_action_skips_approval():
    deps = make_deps(router=fake_router('{"tool_calls":[],"rationale":"No action needed","risk_level":"low","requires_approval":false}'))
    state = {
        "run_id": "r1", "session_id": "s1",
        "extraction": {"customer_id": "acme"},
        "memories": [], "memory_note": "",
        "trace": [], "errors": [],
    }
    update = action_node(state, deps)
    assert update["proposal"]["tool_calls"] == []
    assert update["proposal"]["requires_approval"] is False
