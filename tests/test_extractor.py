from mesh.nodes.extractor import extractor_node
from tests.conftest import fake_router, make_deps


def test_extractor_resolves_known_customer():
    deps = make_deps(router=fake_router('{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "acme_003", "transcript": "We were charged twice.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["extraction"]["customer_id"] == "acme"
    assert update["escalated"] is False
    assert update["trace"][0]["node"] == "extractor"


def test_unknown_customer_escalates():
    deps = make_deps(router=fake_router('{"customer_name":"Unknown Corp","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.9}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "x", "transcript": "Hello.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "customer_not_found"


def test_low_confidence_escalates():
    deps = make_deps(router=fake_router('{"customer_name":"Atlas Freight","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.2}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "atlas_002", "transcript": "It is not working.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "low_confidence"


def test_invalid_model_output_escalates():
    deps = make_deps(router=fake_router(error=ValueError("invalid json")))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "x", "transcript": "Hello.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "extraction_failed"
