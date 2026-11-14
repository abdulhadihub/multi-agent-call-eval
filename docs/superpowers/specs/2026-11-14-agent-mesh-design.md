# Agent Mesh for Customer Conversation Intelligence

Status: Draft for approval
Date: 2026-11-14

## Intent

A public, self-serve demo that shows three specialized agents working as a mesh on customer call transcripts. The artifact is aimed at hiring managers at early-stage agent and AI companies. A recruiter should be able to open a link, run one transcript, and see extraction, persistent memory, scoped tool calls, an approval gate, and an audit trail. The repository should show measured evaluation against a stateless baseline, a benchmark-driven model router, and documented failure cases.

## Why this shape

The original plan described three agents but built three functions in a fixed pipeline. The demo must make the mesh real in the code, not only in the README. LangGraph gives a typed shared state, per-node context, checkpointing per run, an interrupt for approval, and a graph that can be rendered live. The evaluation and tool layer carry the strongest hiring signal, so they are first-class parts of the product, not appendices.

## Goals

- Three agents with distinct responsibilities, prompts, tool scopes, and state slices.
- Persistent per-customer memory that retrieves prior interactions and surfaces prior resolutions.
- Tool calls through a gateway with agent scopes, customer consent, approval before writes, idempotency, and an audit trail for every attempt.
- A router that chooses a model per task from measured benchmark results and records the reason.
- An evaluation harness comparing the mesh with a stateless single-agent baseline on accuracy, cost, and latency, with failure cases.
- A public link that survives cold starts with no external database or paid service.
- A three-minute video script and README that explain the system to a non-specialist hiring manager.

## Non-goals

- Real CRM, ticketing, or refund integrations. All company APIs are simulated.
- Audio transcription. Inputs are text transcripts.
- Multi-tenant auth, SSO, or production security.
- Fine-tuning or self-hosting models.
- A statistically rigorous model benchmark. The corpus is small and results are directional.
- A hosted vector database or managed Postgres.
- Real-time streaming of partial model output.

## Anti-goals

- Agents that are only function names in a README.
- A router that is a dictionary of environment variables.
- An evaluation that reports one aggregate score and hides failures.
- A public demo that breaks because a free database project paused.
- Infrastructure that costs money or needs a credit card.

## Constraints

- Free tiers only. Groq and Gemini keys exist and are verified. Hugging Face Spaces free tier hosts the demo.
- No redistribution of CallCenterEN transcripts. The corpus is synthetic and authored for this demo.
- The public demo accepts only curated transcripts. No free-text input from visitors.
- Python 3.12 locally, Python 3.11 in the container. LangGraph, Streamlit, SQLite, Chroma, fastembed (ONNX).
- Target corpus of 15 transcripts, 6 customers, about 1,500 tokens each.
- Demo cold start target under 60 seconds.
- Build effort 8 to 12 focused days.

## Approaches considered

1. LangGraph plus Streamlit with embedded stores. Chosen. It makes the mesh real, gives a live visual, and keeps deployment to one service.
2. LangGraph plus FastAPI and a custom frontend. Better visuals, roughly double the effort and more deployment risk.
3. The original function pipeline with a static diagram. Fastest, but the mesh claim does not hold under code review.

## Architecture

```
Streamlit UI
  Run | Evaluation | Method
        |
        v
  Orchestrator (deterministic)
        |
        v
  LangGraph StateGraph
    extractor -> memory -> action -> [approval interrupt] -> executor
        |           |         |
        |           |         +--> Tool gateway -> simulated company APIs
        |           +------------> Chroma (customer-scoped vectors)
        +------------------------> Model router (benchmark policy)
        |
        v
  SQLite (customers, interactions, tool calls, audit, runs)
```

### State

One typed `MeshState` holds run metadata, transcript, customer record, extraction, retrieved memories, memory note, action proposal, approval decision, tool results, trace events, audit events, confidence, and an escalation flag. Nodes return partial updates. The checkpointer uses one thread per run, keyed by `session_id:run_id`, so a paused approval can resume after a page interaction without exposing another visitor's thread. Customer continuity lives in the memory layer, not in the checkpointer.

### Agents

**Extractor.** Reads the transcript and a read-only customer directory. Returns customer attribution, intent, entities, action items, evidence spans, and a confidence score. It cannot write. The orchestrator resolves the directory through the read tool `lookup_customer` before the model call, so the extractor gets scoped context without a tool loop.

**Memory.** Embeds the transcript, retrieves the top prior interactions for the attributed customer only, and writes a short note explaining whether a prior issue matches and what resolved it. It cannot write to company systems.

**Action.** Reads the extraction, memories, a current ticket snapshot, and a filtered tool catalog. Produces a structured proposal with tool calls, arguments, rationale, risk level, and whether approval is required. It proposes write tools only and never executes. The orchestrator runs `get_open_tickets` through the gateway before this node.

**Executor.** Validates the proposal against policy, consent, and approval. Executes simulated tools, records idempotency keys, and writes audit events. It is not an LLM agent.

Approval covers the whole proposal. A proposal with several write calls is approved or rejected as one unit. Each call still gets its own audit events and idempotency key. Partial approval is out of scope.

The orchestrator routes by state and confidence. Routing logic is deterministic Python, so the happy path costs no extra model call.

### Tool and permission layer

Six simulated tools.

- `lookup_customer(name)` read.
- `get_open_tickets(customer_id)` read.
- `create_ticket(customer_id, subject, priority, description)` write.
- `update_ticket(ticket_id, fields)` write.
- `issue_refund(customer_id, amount, reason)` write, amount cap.
- `schedule_followup(customer_id, date, channel)` write.

The gateway checks three things before any call. The agent must hold the tool scope. The customer must have granted the corresponding consent scope. Write tools require an approval decision. Every transition writes an append-only audit event with agent, tool call ID, arguments hash, event type, latency, and timestamp. Event types are `attempted`, `denied`, `proposed`, `approved`, `rejected`, `executed`, and `failed`. The UI groups events by tool call ID. A unique idempotency key on run, tool, and arguments prevents duplicate execution.

Scope matrix.

- Extractor holds `crm:read` and calls `lookup_customer` and `get_open_tickets`.
- Memory holds `history:read` and queries Chroma. It holds no company tool scopes.
- Action holds `crm:propose` and proposes write tools. It cannot execute.
- Executor holds `crm:execute` and runs approved writes.

Customer consent scopes are checked separately from agent scopes. A customer can revoke `refund` while the agent still holds the scope, and the gateway denies the call.

### Model router

An offline benchmark runs three current free models across the 15 transcripts on the two routed tasks, extraction and action proposal. It records field accuracy, JSON validity, latency, token use, and paid-equivalent cost. The result is a policy file mapping task to primary model and fallback. The live router reads that policy, logs the reason, and falls back on rate limits or errors. The UI shows the chosen model and the evidence behind it. The memory note uses the fast model and carries no routing claim.

### Memory and storage

SQLite holds customers, interactions, tool calls, audit events, and run metadata. Chroma holds embeddings with customer ID and scope metadata. Seed data is built from the committed corpus at boot. Shared seed memory is read-only. Writes and audit events are scoped to the session, so one visitor cannot pollute another visitor's demo. A completed visitor run can be added to a session-scoped collection, so a second run in the same session can retrieve the first. Retrieval excludes the current run and never crosses sessions. Evaluation uses seed memory only, which keeps the benchmark deterministic. Per-agent working memory is represented by the state slice each node reads and writes, not by a separate memory framework.

### Evaluation harness

The corpus has 15 synthetic transcripts across 6 customers. Three repeat customers have three calls each, which creates 9 transcripts and genuine memory positives. Three single-call customers have one call each, which creates 3 transcripts and memory negatives. Three guardrail transcripts complete the set and belong to existing customers. The cases are a vague low-confidence call, a prompt-injection attempt, and a refund with revoked consent.

Three runs are reported on the same corpus.

- System run. Mesh nodes with memory and tools, pinned to one primary model. Routing is disabled so the delta measures structure, not model choice.
- Baseline run. One stateless prompt with the same transcript, customer directory, tool catalog, and output schema. It has no memory, no routing, and no tool execution.
- Routed run. The production configuration with the policy router enabled. It is reported as an operational result and is not used to claim the architecture delta.

Metrics.

- Extraction. Customer accuracy, intent accuracy, action-item precision, recall, and F1. JSON validity.
- Memory. Recall at 1 and 3 for positive cases. Correct empty result rate for negative cases. Citation accuracy for the prior resolution note.
- Action. Tool selection accuracy and argument field accuracy.
- System. Median and p95 latency, tokens per transcript, paid-equivalent cost, escalation rate.
- Failure report. Every mismatch with transcript ID, expected, actual, and a rule-based diagnosis.

Scoring rules. No LLM judge is used.

- Customer attribution. Exact match on the canonical customer ID after normalizing case and whitespace. Ground truth includes aliases.
- Intent. Exact match against a fixed taxonomy of eight intents.
- Action items. Structured ground truth with type, target, amount, and due date. Enum, ID, amount, and date fields must match exactly. Text fields match when normalized token overlap is at least 0.8.
- Memory hit. A positive case is correct when the ground-truth prior interaction ID appears in the top k. A negative case is correct when the top similarity is below 0.35 and no interaction is returned above the floor.
- Citation. The note must cite at least one interaction ID. Correct when every cited ID is in the retrieved set and the primary citation is the ground-truth prior.
- Action proposal. Tool selection needs an exact tool name match. Argument fields use the same exact and fuzzy rules as action items.
- Failure diagnosis. Categories are authored by the harness. Wrong customer, missed action item, extra action item, retrieval miss, false retrieval, wrong tool, wrong argument, invalid JSON, and escalated. No generated diagnoses.

Scope and approval guardrails have a deterministic test suite separate from the corpus. It injects known bad proposals. These are a memory agent write attempt, an over-cap refund, a revoked customer consent scope, an unapproved write, and a duplicate execution. Scope violation block rate and approval compliance come from those tests.

Results are written to `eval/results.json`, `eval/report.md`, and committed charts. The Evaluation tab renders the committed artifacts. The public demo does not run the eval on demand.

### Guardrails

- Least privilege for every agent.
- Customer consent scopes checked separately from agent scopes.
- Approval interrupt before any write.
- Refund amount limit and a policy check independent of the model.
- Idempotency keys for writes.
- Extractor confidence below `CONFIDENCE_THRESHOLD` routes the run to an escalation terminal node. Memory and action do not run, and the UI hides approval. Policy failures at the executor, such as an over-cap refund or missing consent, take a separate escalation path.
- Transcript text is treated as data, never as instructions. Tool calls are validated against schemas and policy.
- Append-only audit log.
- Per-session run cap and daily cap to protect free-tier quotas.

### UI

Three tabs.

**Run.** Transcript selector, live graph, per-node trace, model and token counts, memory hits, proposal card, approval buttons, tool results, audit timeline. The graph highlights the active node as events stream from LangGraph.

**Evaluation.** Baseline, system, and routed comparison table, charts for accuracy, latency, and cost, router benchmark table, failure case explorer.

**Method.** Architecture diagram, agent scopes, permission matrix, data notes, limitations, and cost model.

The UI uses Streamlit with custom HTML and SVG components for the graph and trace. No separate frontend framework.

### Deployment

A single Docker image on Hugging Face Spaces. Python 3.11 slim base. fastembed downloads the ONNX embedding model at build time, so there is no torch dependency. The app seeds SQLite and Chroma on boot from committed data. Runtime secrets are `GROQ_API_KEY`, `GEMINI_API_KEY`, `DEMO_MODE`, `CONFIDENCE_THRESHOLD`, and `SESSION_RUN_LIMIT`. There is no external database.

### Environment and setup

App runtime reads these variables.

- `GROQ_API_KEY`
- `GROQ_MODEL` and `GROQ_FAST_MODEL`, pinned to models available at build time.
- `GEMINI_API_KEY` and `GEMINI_MODEL`, used for the router benchmark and fallback.
- `EMBEDDING_MODEL` and `EMBEDDING_DIM`, default `all-MiniLM-L6-v2` at 384 dimensions.
- `DEMO_MODE`, `CONFIDENCE_THRESHOLD`, `SESSION_RUN_LIMIT`, `LOG_LEVEL`.

`HF_TOKEN` is deployment-only and never read by the app. Supabase and Jina variables are removed.

### Repository layout

```
app.py
mesh/
  graph.py
  state.py
  nodes/
  router.py
  prompts/
tools/
  registry.py
  gateway.py
  simulated_apis.py
stores/
  sqlite.py
  chroma.py
  seed.py
data/
  transcripts/*.json
  ground_truth.json
eval/
  harness.py
  baseline.py
  router_benchmark.py
  results.json
  report.md
  charts/
tests/
Dockerfile
requirements.txt
README.md
docs/
```

### Cost model

Each demo run uses two model calls, extraction and action proposal, about 2,000 to 3,000 tokens in total. The memory note can use the fast model or be deterministic. At current list prices the paid-equivalent cost is well under one cent per run. Free-tier daily token limits allow dozens of runs. The evaluation and router benchmark run offline and their artifacts are committed, so they cost nothing at view time.

## Review decisions

These were settled in the builder review and are recorded here so they can be checked.

- Checkpoint threads are per run, keyed by `session_id:run_id`. Customer continuity comes from the memory layer.
- The architecture comparison pins one model. The router is benchmarked and reported separately, then shown as an operational run.
- The corpus is 15 transcripts. The third guardrail case is a revoked consent refund attached to an existing customer.
- Evaluation uses exact and fuzzy string rules. No LLM judge.
- Approval covers the whole proposal. Each tool call is audited separately.
- The audit log is append-only events grouped by tool call ID.
- Read tools run through the gateway before the action node. Write tools are proposed and only run after approval.
- The router benchmark covers extraction and action proposal. The memory note uses the fast model and is not routed.
- Escalation triggers on extractor confidence. Memory and action do not run after it.
- Session memory is retrievable within a session and never crosses sessions. Evaluation uses seed memory only.

## What is left to the builder

- Exact prompt wording and JSON schemas.
- Transcript content and ground truth labels.
- The three benchmark models, chosen after checking availability at build time.
- Chart styling and copy.
- Cold-start tuning and Docker layer order.
- Video pacing and narration.

## Risks

- Free-tier model catalogs change. Pin model IDs and record them in the eval report.
- LangGraph and Streamlit state can drift on reruns. Use the checkpointer and session state deliberately, with an integration test.
- Chroma's dependency tree can conflict. Pin versions and verify the Docker build early.
- Synthetic transcripts may read as artificial. Base them on real call patterns and label them honestly.
- A 15-item benchmark is directional. Say so in the report.
