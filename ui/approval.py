from __future__ import annotations


def render_approval(pending: dict) -> str | None:
    import streamlit as st

    proposal = pending.get("proposal") or {}
    customer = pending.get("customer") or {}
    st.markdown("### Approval required")
    if customer:
        st.caption(
            f"Customer: {customer.get('name', customer.get('id', '-'))} "
            f"({customer.get('id', '-')})"
        )
    if proposal.get("rationale"):
        st.markdown(proposal["rationale"])
    st.caption(f"Risk level: {proposal.get('risk_level', '-')}")

    for call in proposal.get("tool_calls", []):
        title = (
            f"{call.get('tool', 'tool')} · "
            f"{call.get('rationale', '')}"
        )
        with st.expander(title):
            st.json(call.get("arguments", {}))
            st.caption(
                f"Idempotency key: {call.get('tool_call_id', '-')}"
            )

    left, right = st.columns(2)
    with left:
        if st.button("Approve and execute", type="primary"):
            return "approve"
    with right:
        if st.button("Reject"):
            return "reject"
    return None
