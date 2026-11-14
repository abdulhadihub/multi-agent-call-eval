from __future__ import annotations

import json
from pathlib import Path

METRICS = [
    ("n", "Transcripts"),
    ("customer_accuracy", "Customer accuracy"),
    ("intent_accuracy", "Intent accuracy"),
    ("action_f1", "Action item F1"),
    ("memory_recall_at_1", "Memory recall@1"),
    ("memory_recall_at_3", "Memory recall@3"),
    ("memory_negative_correct", "Memory negative correct"),
    ("guardrail_correct", "Guardrail correct"),
    ("citation_accuracy", "Memory citation accuracy"),
    ("tool_accuracy", "Tool selection accuracy"),
    ("args_accuracy", "Tool argument accuracy"),
    ("escalation_rate", "Escalation rate"),
    ("latency_p50", "Latency p50 (ms)"),
    ("latency_p95", "Latency p95 (ms)"),
    ("total_tokens", "Tokens"),
    ("mean_cost_usd", "Mean cost (USD)"),
]


def render_evaluation(
    results_path: Path,
    router_results_path: Path,
    charts_dir: Path,
) -> None:
    import pandas as pd
    import streamlit as st

    results_path = Path(results_path)
    if not results_path.exists():
        st.info("No evaluation results yet. Run `python -m eval.harness`.")
        return
    try:
        results = json.loads(results_path.read_text())
    except json.JSONDecodeError:
        st.warning("Evaluation results are still being written.")
        return

    summary = results.get("summary", {})
    rows = []
    for key, label in METRICS:
        rows.append({
            "metric": label,
            "system": summary.get("system", {}).get(key, 0),
            "baseline": summary.get("baseline", {}).get(key, 0),
            "routed": summary.get("routed", {}).get(key, 0),
        })
    st.subheader("System vs baseline")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    groups = summary.get("groups", {})
    if groups:
        group_rows = []
        for group, data in groups.items():
            system_group = data.get("system", {})
            baseline_group = data.get("baseline", {})
            group_rows.append({
                "group": group,
                "system n": system_group.get("n", 0),
                "system guardrail": round(
                    float(system_group.get("guardrail_correct", 0.0)),
                    3,
                ),
                "baseline n": baseline_group.get("n", 0),
                "baseline guardrail": round(
                    float(baseline_group.get("guardrail_correct", 0.0)),
                    3,
                ),
            })
        st.subheader("By group")
        st.dataframe(
            pd.DataFrame(group_rows),
            hide_index=True,
            width="stretch",
        )

    charts_dir = Path(charts_dir)
    for filename, caption in (
        ("accuracy.png", "Accuracy"),
        ("latency.png", "Latency"),
        ("cost.png", "Estimated cost"),
    ):
        chart_path = charts_dir / filename
        if chart_path.exists():
            st.image(str(chart_path), caption=caption)

    failures = results.get("failures", [])
    st.subheader("Failure cases")
    if failures:
        failure_rows = [
            {
                "run": failure.get("run"),
                "transcript": failure.get("transcript_id"),
                "diagnosis": failure.get("diagnosis"),
            }
            for failure in failures
        ]
        st.dataframe(
            pd.DataFrame(failure_rows),
            hide_index=True,
            width="stretch",
        )
        with st.expander("Expected and actual values"):
            for failure in failures[:10]:
                st.markdown(
                    f"**{failure.get('run')} / "
                    f"{failure.get('transcript_id')}**: "
                    f"{failure.get('diagnosis')}"
                )
                st.json({
                    "expected": failure.get("expected"),
                    "actual": failure.get("actual"),
                })
            if len(failures) > 10:
                st.caption(
                    f"Showing 10 of {len(failures)} failures."
                )
    else:
        st.caption("No failures recorded.")

    st.subheader("Router benchmark")
    router_path = Path(router_results_path)
    if not router_path.exists():
        st.caption(
            "No router benchmark yet. Run "
            "`python -m eval.router_benchmark`."
        )
        return
    try:
        router_results = json.loads(router_path.read_text())
    except json.JSONDecodeError:
        st.caption("Router benchmark is still being written.")
        return

    router_rows = []
    for task in ("extract", "propose"):
        for row in router_results.get(task, []):
            router_rows.append({
                "task": task,
                "provider": row.get("provider"),
                "model": row.get("model"),
                "F1": round(float(row.get("f1", 0.0)), 3),
                "cost USD": round(float(row.get("cost_usd", 0.0)), 6),
                "latency ms": round(float(row.get("latency_ms", 0.0))),
                "errors": row.get("errors", 0),
            })
    st.dataframe(
        pd.DataFrame(router_rows),
        hide_index=True,
        width="stretch",
    )
