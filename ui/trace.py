from __future__ import annotations

import pandas as pd
import streamlit as st


def render_trace(trace: list[dict]) -> None:
    if not trace:
        st.caption("No trace events yet.")
        return

    rows = []
    for event in trace:
        rows.append({
            "node": event.get("node", ""),
            "event": event.get("event", ""),
            "model": event.get("model") or "-",
            "tokens in": event.get("tokens_in", 0),
            "tokens out": event.get("tokens_out", 0),
            "cost USD": round(float(event.get("cost_usd", 0.0)), 6),
            "latency ms": event.get("latency_ms", 0),
        })
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
    )
