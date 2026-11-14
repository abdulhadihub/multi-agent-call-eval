from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


def test_app_loads_without_api_calls():
    at = AppTest.from_file(str(APP)).run(timeout=15)
    assert not at.exception
    assert at.title[0].value == "Agent Mesh"
    assert at.selectbox[0].label == "Transcript"
    assert at.button[0].label == "Run"
