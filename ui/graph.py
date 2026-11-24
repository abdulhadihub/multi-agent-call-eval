from __future__ import annotations

import streamlit as st

BOX_W = 130
BOX_H = 44

NODES = {
    "extractor": (30, 50, "Extractor"),
    "memory": (190, 50, "Memory"),
    "action": (350, 50, "Action"),
    "approval": (510, 50, "Approval"),
    "executor": (670, 50, "Executor"),
    "escalate": (350, 170, "Escalate"),
}

COLORS = {
    "pending": ("#e5e7eb", "#374151"),
    "running": ("#f59e0b", "#ffffff"),
    "done": ("#0f766e", "#ffffff"),
    "waiting": ("#2563eb", "#ffffff"),
    "error": ("#b91c1c", "#ffffff"),
}


def _edge(x1: float, y1: float, x2: float, y2: float, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#6b7280" stroke-width="2"{dash} marker-end="url(#arrow)"/>'
    )


def _svg(node_states: dict[str, str]) -> str:
    parts = [
        '<svg viewBox="0 0 830 250" width="100%" height="240" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Agent mesh graph">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/></marker></defs>',
    ]

    for start, end in (
        ("extractor", "memory"),
        ("memory", "action"),
        ("action", "approval"),
        ("approval", "executor"),
    ):
        x1, y1, _ = NODES[start]
        x2, y2, _ = NODES[end]
        parts.append(_edge(
            x1 + BOX_W,
            y1 + BOX_H / 2,
            x2,
            y2 + BOX_H / 2,
        ))

    ax, ay, _ = NODES["action"]
    ex, ey, _ = NODES["escalate"]
    parts.append(_edge(
        ax + BOX_W / 2,
        ay + BOX_H,
        ex + BOX_W / 2,
        ey,
    ))

    x1, y1, _ = NODES["extractor"]
    path = (
        f"M {x1 + BOX_W / 2} {y1 + BOX_H} "
        f"L {x1 + BOX_W / 2} 135 "
        f"L {ex + BOX_W / 2} 135 "
        f"L {ex + BOX_W / 2} {ey}"
    )
    parts.append(
        f'<path d="{path}" fill="none" stroke="#6b7280" stroke-width="2" '
        'stroke-dasharray="6 5" marker-end="url(#arrow)"/>'
    )

    for node, (x, y, label) in NODES.items():
        state = node_states.get(node, "pending")
        fill, text_color = COLORS.get(state, COLORS["pending"])
        parts.append(
            f'<g><rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
            f'rx="10" fill="{fill}" stroke="#374151" stroke-width="1.5"/>'
            f'<text x="{x + BOX_W / 2}" y="{y + 28}" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="13" '
            f'font-weight="600" fill="{text_color}">{label}</text></g>'
        )

    parts.append("</svg>")
    return (
        '<div style="font-family: Inter, Arial, sans-serif;">'
        + "".join(parts)
        + "</div>"
    )


def render_graph(node_states: dict[str, str]) -> None:
    st.markdown(_svg(node_states), unsafe_allow_html=True)
    st.caption(
        "Pending · Running · Finished · Waiting for approval · Error"
    )
