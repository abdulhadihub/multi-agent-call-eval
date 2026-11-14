from uuid import uuid4

import chromadb
from stores.chroma import ChromaMemory
from stores.session import persist_session_run
from stores.sqlite import SqliteStore
from tests.conftest import FakeEmbedder


def _acme(store):
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })


def test_persisted_session_run_is_retrievable(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    _acme(store)
    memory = ChromaMemory(
        chromadb.EphemeralClient(),
        collection_name=f"session-{uuid4().hex}",
    )
    embedder = FakeEmbedder()

    interaction_id = persist_session_run(
        store,
        memory,
        embedder,
        session_id="s1",
        run_id="run-1",
        transcript_id="acme_003",
        transcript="Charged twice again.",
        extraction={
            "customer_id": "acme",
            "intent": "billing_dispute",
            "action_items": [],
        },
        proposal={"rationale": "Refund issued"},
        occurred_at="2026-11-14T00:00:00Z",
    )

    rows = store.list_interactions("acme", session_id="s1")
    assert any(row["id"] == interaction_id for row in rows)

    hits = memory.search(
        customer_id="acme",
        query_embedding=embedder.embed("Charged twice again."),
        top_k=3,
        before="2026-09-02T00:00:00Z",
        session_id="s1",
        exclude_run_id="run-2",
    )
    assert any(hit["interaction_id"] == interaction_id for hit in hits)

    same_run = memory.search(
        customer_id="acme",
        query_embedding=embedder.embed("Charged twice again."),
        top_k=3,
        session_id="s1",
        exclude_run_id="run-1",
    )
    assert all(hit["interaction_id"] != interaction_id for hit in same_run)
