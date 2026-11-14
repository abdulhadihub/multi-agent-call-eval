from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import streamlit as st

from ui.graph import render_graph

NODE_STATES = ("extractor", "memory", "action", "approval", "executor", "escalate")

METHOD = """
## How the mesh works

- **Extractor** reads the transcript and customer directory, then returns a structured record and a confidence score.
- **Memory** embeds the transcript and retrieves prior interactions for the same customer, before the current call.
- **Action** proposes write tools from a catalog filtered by customer consent.
- **Approval** pauses the graph. A human approves or rejects the whole proposal.
- **Executor** pre-validates every call, then runs only approved writes.
- **Escalate** stops the run when confidence is low or a policy check fails.

## Tool scopes

| Agent | Scope | Can do |
| --- | --- | --- |
| Extractor | crm:read | Look up customers and open tickets |
| Memory | history:read | Query the customer vector memory |
| Action | crm:propose | Propose write calls, never execute |
| Executor | crm:execute | Execute approved writes |

Customer consent is checked separately from agent scope. Every tool attempt
writes an append-only audit event.

## Data and limits

- 15 synthetic transcripts across 6 customers. No CallCenterEN text is redistributed.
- Scoring is deterministic. No LLM judge.
- n=15 is directional, not statistically powered.
- Free-tier model catalogs and quotas can change.

## Cost

Each demo run uses two model calls, about 2,000 to 3,000 tokens. The
paid-equivalent cost is under one cent per run. Evaluation and benchmark
artifacts are generated offline and committed.
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


def _resume_run(approved: bool) -> None:
    from langgraph.types import Command

    config = st.session_state.get("config")
    if not config:
        return

    graph = _build_graph()
    try:
        result = graph.invoke(
            Command(resume={
                "approved": approved,
                "approved_by": "demo_user",
            }),
            config,
        )
    except Exception as exc:
        st.session_state["pending"] = None
        st.session_state["last_result"] = {
            "escalated": True,
            "escalation_reason": "resume_error",
            "errors": [str(exc)],
        }
        st.rerun()
        return

    st.session_state["pending"] = None
    st.session_state["last_result"] = result
    st.session_state["trace"] = result.get("trace", [])
    node_states = st.session_state.get("node_states", {})
    node_states["approval"] = "done"
    node_states["executor"] = "done"
    if result.get("escalated"):
        node_states["escalate"] = "done"
    st.session_state["node_states"] = node_states
    st.rerun()


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
            from ui.approval import render_approval

            decision = render_approval(pending)
            if decision:
                _resume_run(decision == "approve")
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

    run_id = st.session_state.get("run_id")
    if run_id and not st.session_state.get("pending"):
        try:
            deps = _build_deps()
            events = deps.store.list_audit_events(run_id)
        except Exception:
            events = []
        if events:
            from ui.audit import render_audit

            st.subheader("Audit")
            render_audit(events)


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
        from ui.evaluation import render_evaluation

        render_evaluation(
            Path("eval/results.json"),
            Path("eval/router_results.json"),
            Path("eval/charts"),
        )

    with method_tab:
        st.markdown(METHOD)


main()
