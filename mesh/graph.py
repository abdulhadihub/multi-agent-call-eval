from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from mesh.nodes.action import action_node
from mesh.nodes.approval import approval_node
from mesh.nodes.escalate import escalate_node
from mesh.nodes.executor import executor_node
from mesh.nodes.extractor import extractor_node
from mesh.nodes.memory import memory_node
from mesh.state import MeshState


def _after_extractor(state: dict) -> str:
    return "escalate" if state.get("escalated") else "memory"


def _after_action(state: dict) -> str:
    if state.get("escalated"):
        return "escalate"
    proposal = state.get("proposal") or {}
    return "approval" if proposal.get("requires_approval") else "executor"


def build_graph(deps, checkpointer=None):
    graph = StateGraph(MeshState)
    graph.add_node("extractor", lambda state: extractor_node(state, deps))
    graph.add_node("memory", lambda state: memory_node(state, deps))
    graph.add_node("action", lambda state: action_node(state, deps))
    graph.add_node("approval", lambda state: approval_node(state, deps))
    graph.add_node("executor", lambda state: executor_node(state, deps))
    graph.add_node("escalate", lambda state: escalate_node(state, deps))

    graph.add_edge(START, "extractor")
    graph.add_conditional_edges(
        "extractor",
        _after_extractor,
        {"escalate": "escalate", "memory": "memory"},
    )
    graph.add_edge("memory", "action")
    graph.add_conditional_edges(
        "action",
        _after_action,
        {"escalate": "escalate", "approval": "approval", "executor": "executor"},
    )
    graph.add_edge("approval", "executor")
    graph.add_edge("executor", END)
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)
