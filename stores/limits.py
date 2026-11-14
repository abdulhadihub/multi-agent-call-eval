from __future__ import annotations

from datetime import datetime, timezone

SESSION_EPOCH = "1970-01-01T00:00:00Z"


def run_cap_error(
    store,
    session_id: str,
    session_limit: int,
    daily_limit: int,
    now: datetime | None = None,
) -> str | None:
    if store.count_runs(session_id, SESSION_EPOCH) >= session_limit:
        return f"Session run limit reached ({session_limit} runs)."

    current = now or datetime.now(timezone.utc)
    since = current.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ).isoformat().replace("+00:00", "Z")
    if store.count_all_runs(since) >= daily_limit:
        return f"Daily run limit reached ({daily_limit} runs)."
    return None
