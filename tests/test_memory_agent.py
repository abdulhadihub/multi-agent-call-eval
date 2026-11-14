from mesh.nodes.memory import memory_node
from tests.conftest import fake_memory, fake_router, make_deps


def test_memory_hit_produces_cited_note():
    hits = [{
        "interaction_id": "int_acme_002",
        "summary": "Refund confirmed",
        "resolution": "Refund issued",
        "similarity": 0.91,
        "occurred_at": "2026-08-05T00:00:00Z",
    }]
    deps = make_deps(
        memory=fake_memory(hits),
        router=fake_router('{"note":"This matches int_acme_002, resolved by refund."}'),
    )
    state = {
        "run_id": "r1", "session_id": "s1", "transcript_id": "acme_003",
        "transcript": "Charged twice again.",
        "extraction": {"customer_id": "acme"},
        "trace": [], "errors": [],
    }
    update = memory_node(state, deps)
    assert update["memories"] == hits
    assert "int_acme_002" in update["memory_note"]
    assert update["trace"][0]["node"] == "memory"


def test_no_history_returns_empty_note():
    deps = make_deps(
        memory=fake_memory([]),
        router=fake_router(error=AssertionError("should not call model")),
    )
    state = {
        "run_id": "r1", "session_id": "s1", "transcript_id": "cedar_001",
        "transcript": "Pricing question.",
        "extraction": {"customer_id": "cedar"},
        "trace": [], "errors": [],
    }
    update = memory_node(state, deps)
    assert update["memories"] == []
    assert update["memory_note"] == "No prior interactions."
    assert update["trace"][0]["model"] is None


def test_search_gets_customer_and_time_scope():
    captured = {}

    class Memory:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []

    deps = make_deps(
        memory=Memory(),
        router=fake_router(error=AssertionError("not called")),
    )
    state = {
        "run_id": "r1", "session_id": "s1", "transcript_id": "x",
        "transcript": "x", "extraction": {"customer_id": "acme"},
        "transcript_occurred_at": "2026-09-02T10:00:00Z",
        "trace": [], "errors": [],
    }
    memory_node(state, deps)
    assert captured["customer_id"] == "acme"
    assert captured["before"] == "2026-09-02T10:00:00Z"
    assert captured["session_id"] == "s1"
    assert captured["exclude_run_id"] == "r1"
