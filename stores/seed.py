from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stores.chroma import ChromaMemory
from stores.embeddings import Embedder
from stores.sqlite import SqliteStore

SEED_CREATED_AT = "2026-01-01T00:00:00Z"


@dataclass
class SeedReport:
    customers: int
    interactions: int
    tickets: int
    embeddings: int


def seed_demo_data(
    store: SqliteStore,
    memory: ChromaMemory,
    embedder: Embedder,
    data_dir: Path = Path("data"),
    reset: bool = False,
) -> SeedReport:
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text())
    transcripts = {
        path.stem: json.loads(path.read_text())
        for path in (data_dir / "transcripts").glob("*.json")
    }

    if reset:
        store.clear_session_data()
        memory.delete_scope("session")

    tickets = 0
    for customer in ground_truth["customers"]:
        store.upsert_customer({
            "id": customer["id"],
            "name": customer["name"],
            "aliases": customer.get("aliases", []),
            "consent_scopes": customer.get("consent_scopes", []),
            "metadata": {},
            "created_at": SEED_CREATED_AT,
        })
        for ticket in customer.get("seed_tickets", []):
            store.add_ticket({
                "id": ticket["id"],
                "customer_id": customer["id"],
                "subject": ticket["subject"],
                "status": ticket.get("status", "open"),
                "priority": ticket.get("priority", "normal"),
                "description": ticket.get("description", ""),
                "scope": "seed",
                "session_id": None,
                "updated_at": SEED_CREATED_AT,
            })
            tickets += 1

    interactions = 0
    embeddings = 0
    for entry in ground_truth["transcripts"]:
        transcript = transcripts[entry["id"]]
        interaction_id = f"int_{entry['id']}"
        store.add_interaction({
            "id": interaction_id,
            "customer_id": entry["customer_id"],
            "transcript_id": entry["id"],
            "summary": entry["summary"],
            "resolution": entry.get("resolution"),
            "intent": entry["intent"],
            "action_items": entry.get("action_items", []),
            "occurred_at": entry["occurred_at"],
            "scope": "seed",
            "session_id": None,
            "run_id": None,
        })

        text = f"{transcript['title']}\n{transcript['transcript']}"
        memory.add(
            interaction_id=interaction_id,
            customer_id=entry["customer_id"],
            text=text,
            embedding=embedder.embed(text),
            summary=entry["summary"],
            resolution=entry.get("resolution"),
            occurred_at=entry["occurred_at"],
            scope="seed",
            session_id=None,
            run_id=None,
        )
        interactions += 1
        embeddings += 1

    return SeedReport(
        customers=len(ground_truth["customers"]),
        interactions=interactions,
        tickets=tickets,
        embeddings=embeddings,
    )
