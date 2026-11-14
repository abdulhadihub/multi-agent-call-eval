from pathlib import Path
from uuid import uuid4

import chromadb
from stores.chroma import ChromaMemory
from stores.seed import seed_demo_data
from stores.sqlite import SqliteStore
from tests.conftest import FakeEmbedder


def test_seed_is_idempotent_and_searchable(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    memory = ChromaMemory(chromadb.EphemeralClient())
    embedder = FakeEmbedder()

    first = seed_demo_data(store, memory, embedder, Path("data"))
    second = seed_demo_data(store, memory, embedder, Path("data"))

    assert first.customers == 6
    assert first.interactions == 15
    assert second.interactions == 15
    assert len(store.list_interactions("acme")) == 3
    assert len(store.get_open_tickets("acme", "s1")) == 1

    hits = memory.search(
        customer_id="acme",
        query_embedding=embedder.embed("duplicate charge refund invoice"),
        top_k=3,
        before="2026-09-02T10:00:00Z",
    )
    assert hits
    assert hits[0]["interaction_id"] in {"int_acme_001", "int_acme_002"}


def test_reset_clears_session_data(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    memory = ChromaMemory(
        chromadb.EphemeralClient(),
        collection_name=f"reset-{uuid4().hex}",
    )
    embedder = FakeEmbedder()
    seed_demo_data(store, memory, embedder, Path("data"))

    store.add_interaction({
        "id": "session-1", "customer_id": "acme", "transcript_id": "session",
        "summary": "Session row", "resolution": None, "intent": "other",
        "action_items": [], "occurred_at": "2026-09-03T00:00:00Z",
        "scope": "session", "session_id": "s1", "run_id": "run-1",
    })
    store.add_ticket({
        "id": "T-SESSION", "customer_id": "acme", "subject": "Session ticket",
        "status": "open", "priority": "normal", "description": "",
        "scope": "session", "session_id": "s1", "updated_at": "2026-09-03T00:00:00Z",
    })
    memory.add(
        interaction_id="session-1", customer_id="acme", text="session",
        embedding=embedder.embed("session"), summary="Session row",
        resolution=None, occurred_at="2026-09-03T00:00:00Z",
        scope="session", session_id="s1", run_id="run-1",
    )

    seed_demo_data(store, memory, embedder, Path("data"), reset=True)

    assert len(store.list_interactions("acme", session_id="s1")) == 3
    assert len(store.get_open_tickets("acme", "s1")) == 1
    assert memory.collection.get(where={"scope": "session"})["ids"] == []
