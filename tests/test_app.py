from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_loads_without_api_calls():
    at = AppTest.from_file(str(APP)).run(timeout=15)
    assert not at.exception
    assert at.title[0].value == "Agent Mesh"
    assert at.selectbox[0].label == "Transcript"
    assert at.button[0].label == "Run"


def test_run_view_renders_memory_and_tool_results():
    at = AppTest.from_file(str(APP))
    at.session_state["last_result"] = {
        "escalated": False,
        "extraction": {
            "customer_id": "acme",
            "intent": "billing_dispute",
            "confidence": 0.9,
        },
        "memory_note": "Matches int_acme_002. Prior resolution: Refund issued.",
        "memories": [{
            "interaction_id": "int_acme_002",
            "summary": "Refund confirmed",
            "similarity": 0.91,
        }],
        "tool_results": [{
            "tool": "issue_refund",
            "status": "executed",
            "result": {"id": "R-1"},
        }],
    }
    at.run(timeout=15)
    assert not at.exception
    assert any(
        "Matches int_acme_002" in markdown.value
        for markdown in at.markdown
    )
    assert any(
        "issue_refund" in str(dataframe.value)
        for dataframe in at.dataframe
    )
