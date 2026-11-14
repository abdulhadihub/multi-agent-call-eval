from __future__ import annotations

DECISIVE = {"executed", "denied", "failed", "rejected"}


def group_audit_events(events: list[dict]) -> list[dict]:
    groups: dict[str, dict] = {}
    for event in events:
        key = event.get("tool_call_id") or "unknown"
        group = groups.setdefault(key, {
            "tool_call_id": key,
            "tool": event.get("tool", "unknown"),
            "events": [],
            "status": "pending",
        })
        group["events"].append(event)
        if event.get("event_type") in DECISIVE:
            group["status"] = event["event_type"]
    return list(groups.values())


def render_audit(events: list[dict]) -> None:
    import pandas as pd
    import streamlit as st

    groups = group_audit_events(events)
    if not groups:
        st.caption("No audit events for this run.")
        return

    rows = []
    for group in groups:
        rows.append({
            "tool call": group["tool_call_id"][:12],
            "tool": group["tool"],
            "status": group["status"],
            "timeline": " -> ".join(
                event.get("event_type", "") for event in group["events"]
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
