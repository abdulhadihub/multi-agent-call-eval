from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.baseline import run_baseline
from eval.scoring import score_action, score_extraction, score_memory
from mesh.config import Settings
from mesh.deps import MeshDeps
from mesh.llm import GeminiClient, GroqClient
from mesh.nodes.action import action_node
from mesh.nodes.extractor import extractor_node
from mesh.nodes.memory import memory_node
from mesh.router import ModelRouter, RouterPolicy
from stores.chroma import ChromaMemory
from stores.embeddings import FastEmbedder
from stores.seed import seed_demo_data
from stores.sqlite import SqliteStore
from tools.gateway import ToolGateway
from tools.registry import catalog_for
from tools.simulated_apis import SimulatedCompanyAPI

GROUPS = ("memory_positive", "memory_negative", "guardrail", "standard")


class _ForcedRouter:
    def __init__(self, router, provider: str, model: str):
        self.router = router
        self.provider = provider
        self.model = model

    def complete_json(self, *, task, messages, schema, **kwargs):
        return self.router.complete_json(
            task=task,
            messages=messages,
            schema=schema,
            force_provider=self.provider,
            force_model=self.model,
        )

    def complete_fixed(self, *, provider, model, messages, schema):
        return self.router.complete_fixed(
            provider=self.provider,
            model=self.model,
            messages=messages,
            schema=schema,
        )


def _group(entry: dict) -> str:
    tags = set(entry.get("tags", []))
    if tags & {"prompt_injection", "low_confidence", "revoked_consent"}:
        return "guardrail"
    if "memory_positive" in tags:
        return "memory_positive"
    if "memory_negative" in tags:
        return "memory_negative"
    return "standard"


def _expected_prior(prior: str | None) -> str | None:
    if prior is None:
        return None
    return prior if prior.startswith("int_") else f"int_{prior}"


def _expected_payload(entry: dict) -> dict:
    return {
        "intent": entry.get("intent"),
        "action_items": entry.get("action_items", []),
        "expected_tool_calls": entry.get("expected_tool_calls", []),
        "prior_interaction_id": entry.get("prior_interaction_id"),
        "tags": entry.get("tags", []),
    }


def _row_from_state(entry: dict, state: dict, latency_ms: int) -> dict:
    extraction = state.get("extraction") or {}
    proposal = state.get("proposal") or {}
    memories = state.get("memories", [])

    scores = {"extraction": score_extraction(entry, extraction)}
    if entry.get("memory_probe"):
        expected_prior = _expected_prior(entry.get("prior_interaction_id"))
        scores["memory"] = score_memory(
            expected_prior,
            memories,
            negative=expected_prior is None,
        )
    scores["action"] = score_action(
        entry.get("expected_tool_calls", []),
        proposal.get("tool_calls", []),
    )

    trace = state.get("trace", [])
    return {
        "transcript_id": entry["id"],
        "customer_id": entry["customer_id"],
        "group": _group(entry),
        "escalated": bool(state.get("escalated")),
        "escalation_reason": state.get("escalation_reason"),
        "extraction": extraction,
        "memories": memories,
        "memory_note": state.get("memory_note"),
        "proposal": proposal,
        "trace": trace,
        "scores": scores,
        "latency_ms": latency_ms,
        "tokens_in": sum(event.get("tokens_in", 0) for event in trace),
        "tokens_out": sum(event.get("tokens_out", 0) for event in trace),
        "cost_usd": sum(event.get("cost_usd", 0.0) for event in trace),
        "error": None,
        "errors": state.get("errors", []),
        "expected": _expected_payload(entry),
    }


def _run_nodes(deps: MeshDeps, entry: dict) -> tuple[dict, int]:
    state = {
        "run_id": f"eval-{entry['id']}",
        "session_id": "eval",
        "transcript_id": entry["id"],
        "transcript": entry["transcript"],
        "transcript_occurred_at": entry.get("occurred_at"),
        "trace": [],
        "errors": [],
        "memories": [],
    }
    started = time.perf_counter()
    try:
        state.update(extractor_node(state, deps))
        if not state.get("escalated"):
            state.update(memory_node(state, deps))
            if not state.get("escalated"):
                state.update(action_node(state, deps))
    except Exception as exc:
        errors = list(state.get("errors", []))
        errors.append(str(exc))
        state["errors"] = errors
        state["escalated"] = True
        state["escalation_reason"] = "harness_error"
    latency_ms = int((time.perf_counter() - started) * 1000)
    return state, latency_ms


def run_system(
    deps: MeshDeps,
    transcripts: list[dict],
    provider: str | None = None,
    model: str | None = None,
    delay: float = 0.0,
) -> list[dict]:
    if provider and model:
        deps = replace(deps, router=_ForcedRouter(deps.router, provider, model))
    rows = []
    for entry in transcripts:
        state, latency_ms = _run_nodes(deps, entry)
        rows.append(_row_from_state(entry, state, latency_ms))
        if delay:
            time.sleep(delay)
    return rows


def run_baseline_all(
    deps: MeshDeps,
    transcripts: list[dict],
    provider: str,
    model: str,
    delay: float = 0.0,
) -> list[dict]:
    customers = {customer["id"]: customer for customer in deps.store.list_customers()}
    directory = [
        {
            "id": customer["id"],
            "name": customer["name"],
            "aliases": customer.get("aliases", []),
        }
        for customer in customers.values()
    ]
    rows = []
    for entry in transcripts:
        customer = customers.get(entry["customer_id"], {})
        catalog = catalog_for(
            {"crm:propose"},
            set(customer.get("consent_scopes", [])),
        )
        started = time.perf_counter()
        try:
            result = run_baseline(
                transcript=entry["transcript"],
                customer_directory=directory,
                tool_catalog=catalog,
                router=deps.router,
                provider=provider,
                model=model,
            )
            extraction = result.get("extraction", {})
            proposal = result.get("proposal", {})
            meta = result.get("meta", {})
            escalated = False
            escalation_reason = None
            error = None
        except Exception as exc:
            extraction = {}
            proposal = {}
            meta = {}
            escalated = True
            escalation_reason = "harness_error"
            error = str(exc)
        latency_ms = int((time.perf_counter() - started) * 1000)
        scores = {
            "extraction": score_extraction(entry, extraction),
            "action": score_action(
                entry.get("expected_tool_calls", []),
                proposal.get("tool_calls", []),
            ),
        }
        rows.append({
            "transcript_id": entry["id"],
            "customer_id": entry["customer_id"],
            "group": _group(entry),
            "escalated": escalated,
            "escalation_reason": escalation_reason,
            "extraction": extraction,
            "memories": [],
            "memory_note": None,
            "proposal": proposal,
            "trace": [],
            "scores": scores,
            "latency_ms": latency_ms,
            "tokens_in": meta.get("input_tokens", 0),
            "tokens_out": meta.get("output_tokens", 0),
            "cost_usd": meta.get("cost_usd", 0.0),
            "error": error,
            "expected": _expected_payload(entry),
        })
        if delay:
            time.sleep(delay)
    return rows


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def values(node: str, key: str) -> list[float]:
        collected = []
        for row in rows:
            value = row["scores"].get(node, {}).get(key)
            if value is None:
                continue
            if value is True:
                collected.append(1.0)
            elif value is False:
                collected.append(0.0)
            else:
                collected.append(float(value))
        return collected

    latencies = [row["latency_ms"] for row in rows]
    ordered = sorted(latencies)
    p95_index = min(
        len(ordered) - 1,
        max(0, math.ceil(0.95 * len(ordered)) - 1),
    )
    memory_rows = [row for row in rows if "memory" in row["scores"]]
    positive = [
        row for row in memory_rows
        if row["expected"].get("prior_interaction_id")
    ]
    negative = [
        row for row in memory_rows
        if not row["expected"].get("prior_interaction_id")
    ]

    return {
        "n": len(rows),
        "customer_accuracy": _mean(values("extraction", "customer_correct")),
        "intent_accuracy": _mean(values("extraction", "intent_correct")),
        "action_f1": _mean(values("extraction", "action_f1")),
        "memory_recall_at_1": _mean([
            1.0 if row["scores"]["memory"]["recall_at_1"] else 0.0
            for row in positive
        ]),
        "memory_recall_at_3": _mean([
            1.0 if row["scores"]["memory"]["recall_at_3"] else 0.0
            for row in positive
        ]),
        "memory_negative_correct": _mean([
            1.0 if row["scores"]["memory"]["correct_empty"] else 0.0
            for row in negative
        ]),
        "tool_accuracy": _mean(values("action", "tool_correct")),
        "args_accuracy": _mean(values("action", "args_correct")),
        "escalation_rate": _mean([
            1.0 if row["escalated"] else 0.0 for row in rows
        ]),
        "latency_p50": statistics.median(latencies),
        "latency_p95": ordered[p95_index],
        "total_tokens": sum(
            row["tokens_in"] + row["tokens_out"] for row in rows
        ),
        "total_cost_usd": sum(row["cost_usd"] for row in rows),
        "mean_cost_usd": _mean([row["cost_usd"] for row in rows]),
    }


def summarize(
    system: list[dict],
    baseline: list[dict],
    routed: list[dict],
) -> dict:
    return {
        "system": _aggregate(system),
        "baseline": _aggregate(baseline),
        "routed": _aggregate(routed),
    }


def _failure_rows(system: list[dict], baseline: list[dict]) -> list[dict]:
    failures = []
    for label, rows in (("system", system), ("baseline", baseline)):
        for row in rows:
            extraction = row["scores"].get("extraction", {})
            action = row["scores"].get("action", {})
            memory = row["scores"].get("memory")
            reasons = []
            if row["escalated"]:
                reasons.append("escalated")
            if extraction.get("customer_correct") is False:
                reasons.append("wrong_customer")
            if extraction.get("intent_correct") is False:
                reasons.append("wrong_intent")
            if extraction.get("action_f1", 1.0) < 1.0:
                reasons.append("action_item_error")
            if action.get("tool_correct") is False:
                reasons.append("wrong_tool")
            if action.get("args_correct") is False:
                reasons.append("wrong_argument")
            if memory:
                if (
                    row["expected"].get("prior_interaction_id")
                    and not memory.get("recall_at_3")
                ):
                    reasons.append("retrieval_miss")
                if (
                    not row["expected"].get("prior_interaction_id")
                    and not memory.get("correct_empty")
                ):
                    reasons.append("false_retrieval")
            if reasons:
                failures.append({
                    "run": label,
                    "transcript_id": row["transcript_id"],
                    "diagnosis": reasons[0],
                    "reasons": reasons,
                    "expected": row["expected"],
                    "actual": {
                        "extraction": row["extraction"],
                        "proposal": row["proposal"],
                    },
                })
    return failures


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_report(results: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = results["summary"]

    lines = [
        "# Agent Mesh Evaluation",
        "",
        f"Generated: {results.get('generated_at', '')}",
        "",
        "## Summary",
        "",
        "| Metric | System | Baseline | Routed |",
        "| --- | --- | --- | --- |",
    ]
    metrics = [
        ("n", "Transcripts"),
        ("customer_accuracy", "Customer accuracy"),
        ("intent_accuracy", "Intent accuracy"),
        ("action_f1", "Action item F1"),
        ("memory_recall_at_1", "Memory recall@1"),
        ("memory_recall_at_3", "Memory recall@3"),
        ("memory_negative_correct", "Memory negative correct"),
        ("tool_accuracy", "Tool selection accuracy"),
        ("args_accuracy", "Tool argument accuracy"),
        ("escalation_rate", "Escalation rate"),
        ("latency_p50", "Latency p50 (ms)"),
        ("latency_p95", "Latency p95 (ms)"),
        ("total_tokens", "Tokens"),
        ("mean_cost_usd", "Mean cost (USD)"),
    ]
    for key, label in metrics:
        lines.append(
            f"| {label} | "
            f"{_fmt(summary.get('system', {}).get(key, 0))} | "
            f"{_fmt(summary.get('baseline', {}).get(key, 0))} | "
            f"{_fmt(summary.get('routed', {}).get(key, 0))} |"
        )

    lines += [
        "",
        "## Per transcript",
        "",
        "| ID | Group | Escalated | Customer | Intent | Action F1 | Memory R@3 | Tool | Args | Latency ms |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results["system"]:
        scores = row["scores"]
        memory = scores.get("memory", {})
        lines.append(
            f"| {row['transcript_id']} | {row['group']} | "
            f"{row['escalated']} | "
            f"{scores['extraction']['customer_correct']} | "
            f"{scores['extraction']['intent_correct']} | "
            f"{_fmt(scores['extraction']['action_f1'])} | "
            f"{memory.get('recall_at_3', '-')} | "
            f"{scores['action']['tool_correct']} | "
            f"{scores['action']['args_correct']} | "
            f"{row['latency_ms']} |"
        )

    lines += ["", "## Failures", ""]
    if results["failures"]:
        for failure in results["failures"]:
            lines.append(
                f"- **{failure['run']} / {failure['transcript_id']}**: "
                f"{failure['diagnosis']}. "
                f"Expected: `{json.dumps(failure['expected'], default=str)}`. "
                f"Actual: `{json.dumps(failure['actual'], default=str)}`."
            )
    else:
        lines.append("No failures recorded.")

    lines += [
        "",
        "## Limitations",
        "",
        "- n=15 synthetic transcripts. Results are directional, not statistically powered.",
        "- Scoring uses exact and fuzzy rules. No LLM judge.",
        "- Free-tier model catalogs and limits can change.",
        "- The baseline is one stateless prompt, not a tuned single-agent system.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_charts(results: dict, charts_dir: Path) -> None:
    charts_dir = Path(charts_dir)
    charts_dir.mkdir(parents=True, exist_ok=True)
    summary = results["summary"]
    labels = ["system", "baseline", "routed"]
    colors = ["#2f6f6f", "#b45309", "#4b5563"]

    metrics = [
        ("customer_accuracy", "Customer"),
        ("intent_accuracy", "Intent"),
        ("action_f1", "Action F1"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    palette = ["#2f6f6f", "#7aa6a6", "#c9d6d6"]
    for index, (key, label) in enumerate(metrics):
        values = [
            summary.get(name, {}).get(key, 0) for name in labels
        ]
        positions = [x + index * width for x in range(len(labels))]
        ax.bar(positions, values, width=width, label=label, color=palette[index])
    ax.set_xticks([x + width for x in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Extraction accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "accuracy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for index, key in enumerate(("latency_p50", "latency_p95")):
        values = [
            summary.get(name, {}).get(key, 0) for name in labels
        ]
        positions = [x + index * width for x in range(len(labels))]
        ax.bar(positions, values, width=width, label=key, color=palette[index])
    ax.set_xticks([x + width for x in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms")
    ax.set_title("Latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "latency.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [summary.get(name, {}).get("mean_cost_usd", 0) for name in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("USD per transcript")
    ax.set_title("Estimated cost")
    fig.tight_layout()
    fig.savefig(charts_dir / "cost.png", dpi=150)
    plt.close(fig)


def _load_transcripts(data_dir: Path, limit: int | None = None) -> list[dict]:
    ground_truth = json.loads((data_dir / "ground_truth.json").read_text())
    entries = ground_truth["transcripts"]
    if limit is not None:
        entries = entries[:limit]
    transcripts = []
    for entry in entries:
        source = json.loads(
            (data_dir / "transcripts" / f"{entry['id']}.json").read_text()
        )
        transcripts.append({
            **entry,
            "transcript": source["transcript"],
            "title": source["title"],
        })
    return transcripts


def _build_deps(settings: Settings, data_dir: Path) -> MeshDeps:
    store = SqliteStore(data_dir / "demo.db")
    store.init_schema()
    memory = ChromaMemory(
        chromadb.PersistentClient(path=str(data_dir / "chroma"))
    )
    embedder = FastEmbedder(settings.embedding_model)
    api = SimulatedCompanyAPI(store)
    gateway = ToolGateway(store, api, settings)
    seed_demo_data(store, memory, embedder, data_dir)

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="eval")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args(argv)

    settings = Settings.load()
    provider = args.provider or "groq"
    model = args.model or settings.groq_model
    data_dir = Path(args.data_dir)

    deps = _build_deps(settings, data_dir)
    transcripts = _load_transcripts(data_dir, args.limit)

    system = run_system(deps, transcripts, provider, model, delay=args.delay)
    baseline = run_baseline_all(
        deps, transcripts, provider, model, delay=args.delay
    )
    routed = run_system(deps, transcripts, delay=args.delay)

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "system": system,
        "baseline": baseline,
        "routed": routed,
        "summary": summarize(system, baseline, routed),
        "failures": _failure_rows(system, baseline),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    report_path = output_dir / "report.md"
    write_report(results, report_path)
    write_charts(results, output_dir / "charts")
    print(report_path)


if __name__ == "__main__":
    main()
