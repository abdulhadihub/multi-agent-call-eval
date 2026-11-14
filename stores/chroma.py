from __future__ import annotations

from typing import Any

import chromadb


class ChromaMemory:
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str = "interactions",
    ):
        self.collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        *,
        interaction_id: str,
        customer_id: str,
        text: str,
        embedding: list[float],
        summary: str,
        resolution: str | None,
        occurred_at: str,
        scope: str,
        session_id: str | None,
        run_id: str | None,
    ) -> None:
        metadata: dict[str, Any] = {
            "interaction_id": interaction_id,
            "customer_id": customer_id,
            "summary": summary,
            "resolution": resolution or "",
            "occurred_at": occurred_at,
            "scope": scope,
            "session_id": session_id or "",
            "run_id": run_id or "",
        }
        self.collection.upsert(
            ids=[interaction_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )

    def delete_scope(self, scope: str) -> None:
        if self.collection.count() == 0:
            return
        self.collection.delete(where={"scope": scope})

    def search(
        self,
        *,
        customer_id: str,
        query_embedding: list[float],
        top_k: int = 3,
        before: str | None = None,
        session_id: str | None = None,
        exclude_run_id: str | None = None,
        seed_only: bool = False,
    ) -> list[dict]:
        total = self.collection.count()
        if total == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(top_k * 4, 10), total),
            where={"customer_id": customer_id},
            include=["metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        hits: list[dict] = []
        for interaction_id, metadata, distance in zip(ids, metadatas, distances):
            metadata = metadata or {}
            scope = metadata.get("scope", "seed")
            if seed_only and scope != "seed":
                continue
            if scope == "session":
                if session_id is None or metadata.get("session_id") != session_id:
                    continue
                if exclude_run_id and metadata.get("run_id") == exclude_run_id:
                    continue
            if before is not None and metadata.get("occurred_at", "") >= before:
                continue
            hits.append({
                "interaction_id": interaction_id,
                "summary": metadata.get("summary", ""),
                "resolution": metadata.get("resolution") or None,
                "occurred_at": metadata.get("occurred_at", ""),
                "similarity": 1.0 - float(distance),
            })

        hits.sort(key=lambda hit: hit["similarity"], reverse=True)
        return hits[:top_k]
