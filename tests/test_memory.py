from uuid import uuid4

import chromadb
from stores.chroma import ChromaMemory


def _memory():
    return ChromaMemory(
        chromadb.EphemeralClient(),
        collection_name=f"test-{uuid4().hex}",
    )


def test_search_filters_by_customer_and_time():
    memory = _memory()
    memory.add(
        interaction_id="i1", customer_id="acme", text="double charge",
        embedding=[1.0, 0.0], summary="Double charge", resolution="Refund",
        occurred_at="2026-08-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    memory.add(
        interaction_id="i2", customer_id="acme", text="future charge",
        embedding=[0.9, 0.1], summary="Future", resolution=None,
        occurred_at="2026-09-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    memory.add(
        interaction_id="i3", customer_id="northwind", text="double charge",
        embedding=[1.0, 0.0], summary="Other customer", resolution=None,
        occurred_at="2026-08-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    hits = memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0],
        top_k=3, before="2026-08-15T00:00:00Z",
    )
    assert [h["interaction_id"] for h in hits] == ["i1"]
    assert hits[0]["similarity"] > 0.99


def test_session_memory_never_crosses_sessions():
    memory = _memory()
    memory.add(
        interaction_id="s1-run", customer_id="acme", text="session one",
        embedding=[1.0, 0.0], summary="Session one", resolution=None,
        occurred_at="2026-08-02T00:00:00Z", scope="session",
        session_id="s1", run_id="run-1",
    )
    assert memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3, session_id="s2",
    ) == []
    assert len(memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3, session_id="s1",
    )) == 1


def test_session_hits_ignore_time_cutoff():
    memory = _memory()
    memory.add(
        interaction_id="session-1", customer_id="acme", text="session",
        embedding=[1.0, 0.0], summary="Session", resolution=None,
        occurred_at="2026-11-14T00:00:00Z", scope="session",
        session_id="s1", run_id="run-1",
    )
    hits = memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3,
        before="2026-09-02T00:00:00Z", session_id="s1",
        exclude_run_id="run-2",
    )
    assert [hit["interaction_id"] for hit in hits] == ["session-1"]


def test_current_run_is_excluded():
    memory = _memory()
    memory.add(
        interaction_id="current", customer_id="acme", text="current",
        embedding=[1.0, 0.0], summary="Current", resolution=None,
        occurred_at="2026-08-02T00:00:00Z", scope="session",
        session_id="s1", run_id="run-1",
    )
    assert memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3,
        session_id="s1", exclude_run_id="run-1",
    ) == []
