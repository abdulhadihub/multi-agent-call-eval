from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ui.graph import render_graph

NODE_STATES = ("extractor", "memory", "action", "approval", "executor", "escalate")

METHOD = """
## How the mesh works

- **Extractor** reads the transcript and the customer directory, then returns a structured record and a confidence score.
- **Memory** embeds the transcript and retrieves prior interactions for the same customer, before the current call.
- **Action** proposes write tools from a catalog filtered by customer consent.
- **Approval** pauses the graph. A human approves or rejects the whole proposal.
- **Executor** pre-validates every call, then executes only approved writes.
- **Escalate** stops the run when confidence is low or a policy check fails.

Every tool attempt writes an append-only audit event. The router selects a model
per task from the committed benchmark policy.
"""


def _load_transcripts() -> list[dict]:
    paths = sorted(Path("data/transcripts").glob("*.json"))
    return [json.loads(path.read_text()) for path in paths]


@st.cache_resource
def _build_deps():
    import chromadb

    from mesh.config import Settings
    from mesh.deps import MeshDeps
    from mesh.llm import GeminiClient, GroqClient
    from mesh.router import ModelRouter, RouterPolicy
    from stores.chroma import ChromaMemory
    from stores.embeddings import FastEmbedder
    from stores.seed import seed_demo_data
    from stores.sqlite import SqliteStore
    from tools.gateway import ToolGateway
    from tools.simulated_apis import SimulatedCompanyAPI

    settings = Settings.load()
    store = SqliteStore("data/demo.db")
    store.init_schema()
    memory = ChromaMemory(
        chromadb.PersistentClient(path="data/chroma")
    )
    embedder = FastEmbedder(settings.embedding_model)
    api = SimulatedCompanyAPI(store)
    gateway = ToolGateway(store, api, settings)
    seed_demo_data(store, memory, embedder, Path("data"))

    clients = {"groq": GroqClient(settings.groq_api_key)}
    if settings.gemini_api_key:
        clients["gemini"] = GeminiClient(settings.gemini_api_key)
    policy = RouterPolicy.load(settings=settings)
    router = ModelRouter(clients, policy, settings)

    return MeshDeps(
        router=router,
        gateway=gateway,
        store=store,
        memory=memory,
        embedder=embedder,
        api=api,
        settings=settings,
    )


@st.cache_resource
def _build_graph():
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from mesh.graph import build_graph

    deps = _build_deps()
    conn = sqlite3.connect("data/demo.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return build_graph(deps, checkpointer)


def _run_transcript(transcript: dict, graph) -> None:
    session_id = st.session_state["session_id"]
    run_id = uuid4().hex
    config = {"configurable": {"thread_id": f"{session_id}:{run_id}"}}
    node_states = {node: "pending" for node in NODE_STATES}
    node_states["extractor"] = "running"

    st.session_state["config"] = config
    st.session_state["run_id"] = run_id
    st.session_state["pending"] = None
    st.session_state["last_result"] = None
    st.session_state["trace"] = []
    st.session_state["node_states"] = node_states

    state = {
        "run_id": run_id,
        "session_id": session_id,
        "transcript_id": transcript["id"],
        "transcript": transcript["transcript"],
        "transcript_occurred_at": transcript["occurred_at"],
        "trace": [],
        "errors": [],
        "memories": [],
    }

    try:
        for event in graph.stream(state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                pending = event["__interrupt__"][0].value
                st.session_state["pending"] = pending
                node_states["approval"] = "waiting"
                st.session_state["node_states"] = dict(node_states)
                break

            for node, update in event.items():
                if str(node).startswith("__"):
                    continue
                node_states[node] = "done"
                if isinstance(update, dict):
                    if update.get("trace") is not None:
                        st.session_state["trace"] = update["trace"]
                    if update.get("escalated"):
                        st.session_state["last_result"] = update
            st.session_state["node_states"] = dict(node_states)

        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            st.session_state["last_result"] = dict(snapshot.values)
            if snapshot.values.get("trace") is not None:
                st.session_state["trace"] = snapshot.values["trace"]
    except Exception as exc:
        node_states["extractor"] = "error"
        st.session_state["node_states"] = dict(node_states)
        st.session_state["last_result"] = {
            "escalated": True,
            "escalation_reason": "app_error",
            "errors": [str(exc)],
        }


def _render_run_state() -> None:
    node_states = st.session_state.get(
        "node_states",
        {node: "pending" for node in NODE_STATES},
    )
    left, right = st.columns([1.15, 1.0])

    with left:
        render_graph(node_states)

    with right:
        pending = st.session_state.get("pending")
        last_result = st.session_state.get("last_result")
        if pending:
            st.warning("Approval required")
            st.json(pending.get("proposal", pending))
        elif last_result:
            if last_result.get("escalated"):
                st.error(
                    "Escalated: "
                    f"{last_result.get('escalation_reason', 'unknown')}"
                )
            else:
                st.success("Run complete")
            extraction = last_result.get("extraction") or {}
            if extraction:
                st.markdown(
                    f"**Customer** {extraction.get('customer_id', '-')}  \n"
                    f"**Intent** {extraction.get('intent', '-')}  \n"
                    f"**Confidence** {extraction.get('confidence', '-')}"
                )
            memories = last_result.get("memories") or []
            st.caption(f"Memory hits: {len(memories)}")
        else:
            st.caption("Choose a transcript and run the mesh.")

    trace = st.session_state.get("trace", [])
    if trace:
        from ui.trace import render_trace

        st.subheader("Trace")
        render_trace(trace)


def main() -> None:
    st.set_page_config(page_title="Agent Mesh", layout="wide")
    st.title("Agent Mesh")
    st.session_state.setdefault("session_id", uuid4().hex)

    transcripts = _load_transcripts()
    if not transcripts:
        st.error("No transcripts found in data/transcripts.")
        return

    run_tab, evaluation_tab, method_tab = st.tabs(
        ["Run", "Evaluation", "Method"]
    )

    with run_tab:
        by_title = {
            transcript.get("title", transcript["id"]): transcript
            for transcript in transcripts
        }
        selected = st.selectbox("Transcript", list(by_title))
        if st.button("Run", type="primary"):
            _run_transcript(by_title[selected], _build_graph())
        _render_run_state()

    with evaluation_tab:
        st.markdown(
            "Evaluation artifacts are generated by "
            "`python -m eval.harness`. The evaluation view renders the "
            "committed baseline comparison, router benchmark, and failure "
            "cases."
        )

    with method_tab:
        st.markdown(METHOD)


main()
