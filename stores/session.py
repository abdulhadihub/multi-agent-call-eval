from __future__ import annotations

from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def persist_session_run(
    store,
    memory,
    embedder,
    *,
    session_id: str,
    run_id: str,
    transcript_id: str,
    transcript: str,
    extraction: dict,
    proposal: dict,
    occurred_at: str | None = None,
) -> str | None:
    extraction = extraction or {}
    customer_id = extraction.get("customer_id")
    if not customer_id:
        return None

    interaction_id = f"int_session_{run_id}"
    summary = f"Session run for {transcript_id}"
    resolution = (proposal or {}).get("rationale") or None
    timestamp = occurred_at or _now()

    store.add_interaction({
        "id": interaction_id,
        "customer_id": customer_id,
        "transcript_id": transcript_id,
        "summary": summary,
        "resolution": resolution,
        "intent": extraction.get("intent"),
        "action_items": extraction.get("action_items", []),
        "occurred_at": timestamp,
        "scope": "session",
        "session_id": session_id,
        "run_id": run_id,
    })

    memory.add(
        interaction_id=interaction_id,
        customer_id=customer_id,
        text=transcript,
        embedding=embedder.embed(transcript),
        summary=summary,
        resolution=resolution,
        occurred_at=timestamp,
        scope="session",
        session_id=session_id,
        run_id=run_id,
    )
    return interaction_id
