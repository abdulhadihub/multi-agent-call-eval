from datetime import datetime, timezone

from stores.limits import run_cap_error
from stores.sqlite import SqliteStore


def _store(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    return store


def _add_run(store, run_id, session_id):
    store.add_run({
        "id": run_id,
        "session_id": session_id,
        "transcript_id": "x",
        "status": "completed",
        "started_at": "2026-11-14T10:00:00Z",
    })


def test_session_cap(tmp_path):
    store = _store(tmp_path)
    _add_run(store, "r1", "s1")
    _add_run(store, "r2", "s1")
    now = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    assert run_cap_error(store, "s1", 2, 100, now=now) == (
        "Session run limit reached (2 runs)."
    )
    assert run_cap_error(store, "s2", 2, 100, now=now) is None


def test_daily_cap_is_global(tmp_path):
    store = _store(tmp_path)
    _add_run(store, "r1", "s1")
    _add_run(store, "r2", "s2")
    now = datetime(2026, 11, 14, 12, tzinfo=timezone.utc)
    assert run_cap_error(store, "s9", 20, 2, now=now) == (
        "Daily run limit reached (2 runs)."
    )
