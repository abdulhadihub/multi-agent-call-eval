# Agent Mesh Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a public three-agent customer conversation intelligence demo with a LangGraph mesh, scoped tool calls, benchmark-driven model routing, and a reproducible evaluation against a stateless baseline.

**Architecture:** A Streamlit app drives a LangGraph state machine with extractor, memory, action, approval, executor, and escalation nodes. Agents share typed state but receive scoped context. Every model call goes through a policy router and every company action goes through a permission gateway with append-only audit events. SQLite holds structured records and Chroma holds customer-scoped embeddings. The app runs as one Docker image on Hugging Face Spaces.

**Tech Stack:** Python 3.12 local, Python 3.11 container, LangGraph, Streamlit, Groq, Google Gen AI, Chroma, fastembed, SQLite, Pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-agent-mesh-design.md`

## Global Constraints

- Free tiers only. No paid service, managed database, credit card, or external runtime database.
- Public demo accepts curated transcripts only. No free-text input from visitors.
- Corpus is 15 synthetic transcripts across 6 customers. Do not commit CallCenterEN text.
- Ground truth in `data/ground_truth.json` is the source of truth for scoring. No LLM judge.
- Embeddings are local and 384-dimensional. Use fastembed with ONNX, not torch.
- Every model call goes through `mesh/router.py`. Every tool call goes through `tools/gateway.py`.
- Every write requires an approval decision and writes append-only audit events.
- Approval covers the whole proposal. Each tool call gets its own idempotency key and audit events.
- Append-only audit event types: `attempted`, `denied`, `proposed`, `approved`, `rejected`, `executed`, `failed`.
- Commit as the repository owner. No attribution trailers, co-author lines, or assistant mentions.
- Make small commits at the end of each task.
- Demo cold start target under 60 seconds.

## Review Focus

These are the failures the spec implies but no single feature test owns. Each gets a test in the task named.

1. Ambiguous or unknown customer attribution. The extractor may invent a customer or pick the wrong one. Expected behavior is low confidence or escalation, never a silent write to an unrelated customer. Test in Task 9.
2. Empty or cross-customer memory. The memory agent may cite a prior interaction that does not exist or belongs to another customer. Expected behavior is an empty result, no citation, and no cross-customer hit. Test in Task 10.
3. Provider failure. Rate limit, timeout, or invalid JSON from the primary model. Expected behavior is a fallback model or escalation, with no partial write. Test in Task 8 and Task 12.
4. Adversarial transcript. Transcript text instructs the agent to issue an unauthorized refund. Expected behavior is to treat it as data, deny the call, and write an audit event. Test in Task 7 and Task 12.
5. Duplicate or concurrent actions. Double-click approval, resume twice, or two sessions acting at once. Expected behavior is one execution, one idempotent result, and no cross-session state. Test in Task 7 and Task 10.

---

### Task 1: Project scaffold, dependencies, and settings

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `mesh/__init__.py`, `mesh/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Modify: `.env.example`, `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `Settings` with fields `groq_api_key`, `groq_model`, `groq_fast_model`, `gemini_api_key`, `gemini_model`, `embedding_model`, `embedding_dim`, `demo_mode`, `confidence_threshold`, `session_run_limit`, `daily_run_limit`, `log_level`. Methods `Settings.from_env(env: Mapping[str, str] | None = None) -> Settings` and `Settings.load() -> Settings`.

- [ ] **Step 1: Write the failing settings test**

```python
# tests/test_config.py
import pytest
from pydantic import ValidationError
from mesh.config import Settings

BASE = {
    "GROQ_API_KEY": "gsk_test",
    "GROQ_MODEL": "openai/gpt-oss-120b",
    "GROQ_FAST_MODEL": "openai/gpt-oss-20b",
    "GEMINI_MODEL": "gemini-2.5-flash",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "EMBEDDING_DIM": "384",
    "DEMO_MODE": "true",
    "CONFIDENCE_THRESHOLD": "0.6",
    "SESSION_RUN_LIMIT": "20",
    "DAILY_RUN_LIMIT": "100",
    "LOG_LEVEL": "INFO",
}

def test_settings_load_from_mapping():
    settings = Settings.from_env(BASE)
    assert settings.groq_model == "openai/gpt-oss-120b"
    assert settings.gemini_api_key is None
    assert settings.embedding_dim == 384
    assert settings.demo_mode is True
    assert settings.confidence_threshold == 0.6

def test_empty_optional_key_becomes_none():
    settings = Settings.from_env({**BASE, "GEMINI_API_KEY": ""})
    assert settings.gemini_api_key is None

def test_missing_required_key_raises():
    with pytest.raises(ValidationError):
        Settings.from_env({})
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL because `mesh.config` does not exist.

- [ ] **Step 3: Create the environment and install dependencies**

```bash
uv venv
uv pip install langgraph langgraph-checkpoint-sqlite streamlit groq google-genai chromadb fastembed pydantic python-dotenv pandas matplotlib pytest
```

After the suite passes at the end of this task, freeze exact versions into `requirements.txt` with `uv pip freeze > requirements.txt` and commit the pinned file.

- [ ] **Step 4: Implement `Settings` in `mesh/config.py`**

Use a Pydantic `BaseModel`. Read keys with `os.getenv`, convert empty strings to `None`, parse booleans from `true`/`false`, and parse integers and floats. `Settings.load()` calls `load_dotenv()` once and then `from_env()`.

Create `pyproject.toml` with only this content.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

Create `.env.example` with these exact values and no inline comments.

```bash
# LLM
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FAST_MODEL=openai/gpt-oss-20b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIM=384

# App
DEMO_MODE=true
CONFIDENCE_THRESHOLD=0.6
SESSION_RUN_LIMIT=20
DAILY_RUN_LIMIT=100
LOG_LEVEL=INFO

# Deployment only
HF_TOKEN=
```

Set `.gitignore` to include `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.streamlit/`, `data/*.db`, `data/chroma/`, and `*.log`.

- [ ] **Step 5: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: 3 passed.

```bash
git add requirements.txt pyproject.toml mesh/__init__.py mesh/config.py tests/test_config.py .env.example .gitignore
git commit -m "feat: set up project configuration"
```

---

### Task 2: SQLite store and schema

**Files:**
- Create: `stores/__init__.py`, `stores/schema.sql`, `stores/sqlite.py`, `tests/test_sqlite.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `SqliteStore(path: str | Path)`. Methods `init_schema()`, `upsert_customer(dict)`, `get_customer(customer_id)`, `list_customers()`, `add_interaction(dict)`, `list_interactions(customer_id, before=None, session_id=None)`, `add_ticket(dict)`, `get_open_tickets(customer_id, session_id)`, `update_ticket(ticket_id, fields)`, `add_refund(dict)`, `add_followup(dict)`, `record_tool_call(dict) -> bool`, `get_tool_call(tool_call_id)`, `add_audit_event(dict)`, `list_audit_events(run_id)`, `add_run(dict)`, `finish_run(run_id, status)`, `count_runs(session_id, since)`, `count_tool_calls()`.

- [ ] **Step 1: Write the failing store test**

```python
# tests/test_sqlite.py
from stores.sqlite import SqliteStore

def test_store_round_trip(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
        "metadata": {}, "created_at": "2026-08-01T00:00:00Z",
    })
    assert store.get_customer("acme")["name"] == "Acme Logistics"

    store.add_interaction({
        "id": "int_acme_001", "customer_id": "acme", "transcript_id": "acme_001",
        "summary": "Double charge", "resolution": "Refund issued",
        "intent": "billing_dispute", "action_items": [],
        "occurred_at": "2026-08-01T10:00:00Z", "scope": "seed",
        "session_id": None, "run_id": None,
    })
    rows = store.list_interactions("acme")
    assert len(rows) == 1
    assert rows[0]["resolution"] == "Refund issued"

def test_tool_call_is_idempotent(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    row = {
        "id": "call-1", "run_id": "run-1", "session_id": "s1",
        "agent": "executor", "tool": "issue_refund", "args_hash": "abc",
        "status": "executed", "result": "{}", "created_at": "2026-08-01T00:00:00Z",
    }
    assert store.record_tool_call(row) is True
    assert store.record_tool_call(row) is False
    assert store.count_tool_calls() == 1

def test_session_scoped_tickets(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.add_ticket({
        "id": "T-1", "customer_id": "acme", "subject": "Seed",
        "status": "open", "priority": "normal", "description": "",
        "scope": "seed", "session_id": None, "updated_at": "2026-08-01T00:00:00Z",
    })
    store.add_ticket({
        "id": "T-2", "customer_id": "acme", "subject": "Session",
        "status": "open", "priority": "high", "description": "",
        "scope": "session", "session_id": "s1", "updated_at": "2026-08-01T00:00:00Z",
    })
    assert [t["id"] for t in store.get_open_tickets("acme", "s1")] == ["T-1", "T-2"]
    assert [t["id"] for t in store.get_open_tickets("acme", "s2")] == ["T-1"]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_sqlite.py -v`
Expected: FAIL because `stores.sqlite` does not exist.

- [ ] **Step 3: Write `stores/schema.sql` and `SqliteStore`**

Create tables `customers`, `interactions`, `tickets`, `refunds`, `followups`, `tool_calls`, `audit_events`, and `runs`. Use text IDs for deterministic seed data and ISO timestamps. Store JSON fields as text and parse them on read. Add indexes on `interactions(customer_id)`, `audit_events(run_id)`, and `runs(session_id, started_at)`.

Use `sqlite3.connect(self.path, timeout=5.0)` per operation and `PRAGMA journal_mode=WAL` in `init_schema()`. `record_tool_call` uses `INSERT OR IGNORE` and returns whether a row was inserted. `get_open_tickets` returns rows where `scope='seed'` or `session_id` matches. `list_interactions` applies `occurred_at < before` when `before` is provided.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_sqlite.py -v`
Expected: 3 passed.

```bash
git add stores/__init__.py stores/schema.sql stores/sqlite.py tests/test_sqlite.py
git commit -m "feat: add SQLite store and schema"
```

---

### Task 3: Embeddings and Chroma memory

**Files:**
- Create: `stores/embeddings.py`, `stores/chroma.py`, `tests/test_memory.py`
- Modify: `tests/conftest.py` (new file if absent)

**Interfaces:**
- Consumes: nothing.
- Produces: `Embedder` protocol with `dim` and `embed(text) -> list[float]`, `embed_many(texts) -> list[list[float]]`. `FastEmbedder(model_name: str)`. `ChromaMemory(client: chromadb.ClientAPI, collection_name="interactions")` with `add(...)` and `search(*, customer_id, query_embedding, top_k=3, before=None, session_id=None, exclude_run_id=None, seed_only=False)`.

- [ ] **Step 1: Write the failing memory test**

```python
# tests/test_memory.py
import chromadb
from stores.chroma import ChromaMemory

def test_search_filters_by_customer_and_time():
    memory = ChromaMemory(chromadb.EphemeralClient())
    memory.add(
        interaction_id="i1", customer_id="acme", text="double charge",
        embedding=[1.0, 0.0], summary="Double charge", resolution="Refund",
        occurred_at="2026-08-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    memory.add(
        interaction_id="i2", customer_id="acme", text="future charge",
        embedding=[0.9, 0.1], summary="Future", resolution=None,
        occurred_at="2026-09-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    memory.add(
        interaction_id="i3", customer_id="northwind", text="double charge",
        embedding=[1.0, 0.0], summary="Other customer", resolution=None,
        occurred_at="2026-08-01T00:00:00Z", scope="seed",
        session_id=None, run_id=None,
    )
    hits = memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0],
        top_k=3, before="2026-08-15T00:00:00Z",
    )
    assert [h["interaction_id"] for h in hits] == ["i1"]
    assert hits[0]["similarity"] > 0.99

def test_session_memory_never_crosses_sessions():
    memory = ChromaMemory(chromadb.EphemeralClient())
    memory.add(
        interaction_id="s1-run", customer_id="acme", text="session one",
        embedding=[1.0, 0.0], summary="Session one", resolution=None,
        occurred_at="2026-08-02T00:00:00Z", scope="session",
        session_id="s1", run_id="run-1",
    )
    assert memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3, session_id="s2",
    ) == []
    assert len(memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3, session_id="s1",
    )) == 1

def test_current_run_is_excluded():
    memory = ChromaMemory(chromadb.EphemeralClient())
    memory.add(
        interaction_id="current", customer_id="acme", text="current",
        embedding=[1.0, 0.0], summary="Current", resolution=None,
        occurred_at="2026-08-02T00:00:00Z", scope="session",
        session_id="s1", run_id="run-1",
    )
    assert memory.search(
        customer_id="acme", query_embedding=[1.0, 0.0], top_k=3,
        session_id="s1", exclude_run_id="run-1",
    ) == []
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_memory.py -v`
Expected: FAIL because `stores.chroma` does not exist.

- [ ] **Step 3: Implement the embedder and Chroma store**

`FastEmbedder` uses `fastembed.TextEmbedding(model_name=...)`. Verify that `sentence-transformers/all-MiniLM-L6-v2` is in the local fastembed model list. If it is not, use `BAAI/bge-small-en-v1.5`, which also produces 384 dimensions, and update `EMBEDDING_MODEL` in `.env.example`. Raise a clear error at startup if the dimension is not 384.

`ChromaMemory` uses `get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`. `add` uses `collection.upsert` with `ids`, `embeddings`, `documents`, and `metadatas`. `search` requests `max(top_k * 4, 10)` results with `where={"customer_id": customer_id}`, then filters in Python by `before`, scope, session, and run. Return `{"interaction_id", "summary", "resolution", "similarity", "occurred_at"}`, sorted by similarity descending, limited to `top_k`. Return `[]` when the collection is empty.

In `tests/conftest.py`, add a `FakeEmbedder` with `dim = 384` that returns a deterministic vector from `hashlib.sha256(text)`. Do not use it in production.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_memory.py -v`
Expected: 3 passed.

```bash
git add stores/embeddings.py stores/chroma.py tests/test_memory.py tests/conftest.py
git commit -m "feat: add embedded vector memory"
```

---

### Task 4: Synthetic corpus and ground truth

**Files:**
- Create: `data/transcripts/acme_001.json`, `acme_002.json`, `acme_003.json`, `northwind_001.json`, `northwind_002.json`, `northwind_003.json`, `northwind_004.json`, `blueharbor_001.json`, `blueharbor_002.json`, `blueharbor_003.json`, `blueharbor_004.json`, `cedar_001.json`, `lumen_001.json`, `atlas_001.json`, `atlas_002.json`, `data/ground_truth.json`, `tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: transcript JSON files with `id`, `title`, `occurred_at`, and `transcript`. `ground_truth.json` with `customers` and `transcripts`.

- [ ] **Step 1: Write the failing corpus test**

```python
# tests/test_corpus.py
import json
from pathlib import Path

DATA = Path("data")
GT = json.loads((DATA / "ground_truth.json").read_text())
TRANSCRIPTS = {p.stem: json.loads(p.read_text()) for p in (DATA / "transcripts").glob("*.json")}

def test_corpus_shape():
    assert len(TRANSCRIPTS) == 15
    assert len({t["customer_id"] for t in GT["transcripts"]}) == 6
    assert set(TRANSCRIPTS) == {t["id"] for t in GT["transcripts"]}

def test_prior_interactions_exist_and_precede():
    by_id = {t["id"]: t for t in GT["transcripts"]}
    for entry in GT["transcripts"]:
        prior = entry.get("prior_interaction_id")
        if prior:
            assert prior in by_id
            assert by_id[prior]["occurred_at"] < entry["occurred_at"]

def test_guardrail_tags_and_consent():
    tags = {tag for t in GT["transcripts"] for tag in t["tags"]}
    assert {"memory_positive", "memory_negative", "prompt_injection", "low_confidence", "revoked_consent"} <= tags
    blueharbor = next(c for c in GT["customers"] if c["id"] == "blueharbor")
    assert "refund" not in blueharbor["consent_scopes"]
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_corpus.py -v`
Expected: FAIL because the corpus is missing.

- [ ] **Step 3: Write the 15 transcripts and ground truth**

Each transcript file has this shape.

```json
{
  "id": "acme_003",
  "title": "Acme Logistics: duplicate charge again",
  "occurred_at": "2026-09-02T10:00:00Z",
  "transcript": "Agent: Thank you for calling. How can I help?\nCustomer: We were charged twice again..."
}
```

Use realistic, messy dialogue of 300 to 700 words. Include the customer name, a concrete issue, and at least one implied action. Write these cases.

| ID | Customer | Date | Purpose | Prior |
| --- | --- | --- | --- | --- |
| acme_001 | acme | 2026-08-01 | Double charge, refund resolved | null |
| acme_002 | acme | 2026-08-05 | Refund confirmation follow-up | acme_001 |
| acme_003 | acme | 2026-09-02 | Duplicate charge recurs | acme_002 |
| northwind_001 | northwind | 2026-07-10 | Integration sync failure | null |
| northwind_002 | northwind | 2026-07-12 | Workaround confirmed | northwind_001 |
| northwind_003 | northwind | 2026-08-20 | Sync failure recurs | northwind_002 |
| northwind_004 | northwind | 2026-08-25 | Prompt injection inside transcript | northwind_002 |
| blueharbor_001 | blueharbor | 2026-06-15 | Refund request and approval | null |
| blueharbor_002 | blueharbor | 2026-06-18 | Refund status follow-up | blueharbor_001 |
| blueharbor_003 | blueharbor | 2026-07-01 | Refund follow-up, memory positive | blueharbor_002 |
| blueharbor_004 | blueharbor | 2026-07-05 | Refund with revoked consent | blueharbor_002 |
| cedar_001 | cedar | 2026-09-10 | Upgrade pricing question | null |
| lumen_001 | lumen | 2026-09-12 | Seat transfer request | null |
| atlas_001 | atlas | 2026-09-15 | Delivery complaint | null |
| atlas_002 | atlas | 2026-09-16 | Vague low-confidence call | null |

Set `prior_interaction_id` only for rows with a value in the Prior column. Set `memory_probe: true` on `acme_003`, `northwind_003`, `blueharbor_003`, `cedar_001`, `lumen_001`, and `atlas_001`. The first three are positive probes and the last three are negative probes.

Add these customer records.

| ID | Name | Consent scopes | Seed history |
| --- | --- | --- | --- |
| acme | Acme Logistics | crm:read, crm:write, refund | true |
| northwind | Northwind Health | crm:read, crm:write, refund | true |
| blueharbor | Blue Harbor Retail | crm:read, crm:write | true |
| cedar | Cedar Analytics | crm:read, crm:write, refund | true |
| lumen | Lumen Studio | crm:read, crm:write, refund | true |
| atlas | Atlas Freight | crm:read, crm:write, refund | true |

Each customer also has `aliases` and `seed_tickets`. Give each repeat customer one open ticket. Example for Acme:

```json
{
  "id": "acme",
  "name": "Acme Logistics",
  "aliases": ["Acme"],
  "consent_scopes": ["crm:read", "crm:write", "refund"],
  "seed_history": true,
  "seed_tickets": [
    {"id": "T-1001", "subject": "Duplicate invoice charge", "status": "open", "priority": "high"}
  ]
}
```

Each transcript entry has `id`, `customer_id`, `occurred_at`, `intent`, `action_items`, `expected_tool_calls`, `prior_interaction_id`, `memory_probe`, `summary`, `resolution`, and `tags`. Use the intent taxonomy `billing_dispute`, `refund_status`, `technical_issue`, `cancellation`, `upgrade`, `account_access`, `complaint`, and `other`. Use action item types `refund`, `callback`, `ticket_update`, `followup`, and `escalation`.

Add guardrail metadata. `northwind_004` has `tags: ["prompt_injection"]` and `expected_guardrail: "amount_cap"`. `blueharbor_004` has `tags: ["revoked_consent"]` and `expected_guardrail: "consent_denied"`. `atlas_002` has `tags: ["low_confidence"]` and `expected_escalation: true`. The injection transcript contains a line that asks the agent to ignore policy and refund 5000. The revoked consent transcript is a normal refund request for Blue Harbor, whose consent scopes exclude `refund`.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_corpus.py -v`
Expected: 3 passed.

```bash
git add data/transcripts data/ground_truth.json tests/test_corpus.py
git commit -m "data: add synthetic transcript corpus and ground truth"
```

---

### Task 5: Demo seed

**Files:**
- Create: `stores/seed.py`, `tests/test_seed.py`

**Interfaces:**
- Consumes: `SqliteStore`, `ChromaMemory`, `Embedder`, and the corpus.
- Produces: `SeedReport` dataclass with `customers`, `interactions`, `tickets`, `embeddings`. `seed_demo_data(store, memory, embedder, data_dir=Path("data"), reset=False) -> SeedReport`.

- [ ] **Step 1: Write the failing seed test**

```python
# tests/test_seed.py
import chromadb
from pathlib import Path
from stores.chroma import ChromaMemory
from stores.seed import seed_demo_data
from stores.sqlite import SqliteStore
from tests.conftest import FakeEmbedder

def test_seed_is_idempotent_and_searchable(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    memory = ChromaMemory(chromadb.EphemeralClient())
    embedder = FakeEmbedder()

    first = seed_demo_data(store, memory, embedder, Path("data"))
    second = seed_demo_data(store, memory, embedder, Path("data"))

    assert first.customers == 6
    assert first.interactions == 15
    assert second.interactions == 15
    assert len(store.list_interactions("acme")) == 3
    assert len(store.get_open_tickets("acme", "s1")) == 1

    hits = memory.search(
        customer_id="acme",
        query_embedding=embedder.embed("duplicate charge refund invoice"),
        top_k=3,
        before="2026-09-02T10:00:00Z",
    )
    assert hits
    assert hits[0]["interaction_id"] in {"int_acme_001", "int_acme_002"}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_seed.py -v`
Expected: FAIL because `stores.seed` does not exist.

- [ ] **Step 3: Implement the seed**

For each customer, upsert the record and insert `seed_tickets` as tickets with `scope="seed"`. For each transcript, insert an interaction with `id=f"int_{transcript_id}"`, the transcript summary and resolution from ground truth, `scope="seed"`, and the transcript timestamp. Embed `title + transcript text` and upsert into Chroma with the interaction metadata. Use `reset=True` to delete session rows and session vectors before reseeding. `seed_demo_data` must be safe to run on every app boot.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_seed.py -v`
Expected: 1 passed.

```bash
git add stores/seed.py tests/test_seed.py
git commit -m "feat: add demo seed data"
```

---

### Task 6: Tool registry and simulated company APIs

**Files:**
- Create: `tools/__init__.py`, `tools/registry.py`, `tools/simulated_apis.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: `SqliteStore`.
- Produces: Pydantic input models `LookupCustomerInput`, `GetOpenTicketsInput`, `CreateTicketInput`, `UpdateTicketInput`, `IssueRefundInput`, `ScheduleFollowupInput`. `ToolSpec` with `name`, `description`, `input_model`, `is_write`, `consent_scope`, `amount_cap`. `TOOLS: dict[str, ToolSpec]`. `catalog_for(agent_scopes: set[str], consent_scopes: set[str]) -> list[dict]`. `SimulatedCompanyAPI(store)` with the six tool methods, each taking `session_id` for writes and reads.

- [ ] **Step 1: Write the failing tool test**

```python
# tests/test_tools.py
import pytest
from pydantic import ValidationError
from stores.sqlite import SqliteStore
from tools.registry import catalog_for, TOOLS
from tools.simulated_apis import SimulatedCompanyAPI

def test_catalog_filters_by_scope_and_consent():
    names = {t["name"] for t in catalog_for({"crm:propose"}, {"crm:write", "refund"})}
    assert names == {"create_ticket", "update_ticket", "issue_refund", "schedule_followup"}
    names = {t["name"] for t in catalog_for({"crm:propose"}, {"crm:write"})}
    assert "issue_refund" not in names

def test_refund_amount_must_be_positive():
    with pytest.raises(ValidationError):
        TOOLS["issue_refund"].input_model(customer_id="acme", amount=0, reason="x")

def test_simulated_api_scopes_writes(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": [],
        "consent_scopes": ["crm:read", "crm:write"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    api = SimulatedCompanyAPI(store)
    ticket = api.create_ticket("acme", "Double charge", "high", "Customer charged twice", "s1")
    assert ticket["session_id"] == "s1"
    assert [t["id"] for t in api.get_open_tickets("acme", "s2")] == []
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL because `tools.registry` does not exist.

- [ ] **Step 3: Implement the registry and simulated APIs**

Define the six tool specs with exact input models. The consent scopes are `crm:read` for reads, `crm:write` for create, update, and follow-up, and `refund` for refunds. `issue_refund` is a write with `amount_cap=500.0`.

`SimulatedCompanyAPI` reads and writes only through `SqliteStore`. It never checks permissions and never calls an external service. `lookup_customer` matches the name or an alias, case-insensitively. Writes use the current session ID and a generated deterministic ID for tickets, refunds, and follow-ups.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: 3 passed.

```bash
git add tools/__init__.py tools/registry.py tools/simulated_apis.py tests/test_tools.py
git commit -m "feat: add tool registry and simulated company APIs"
```

---

### Task 7: Tool gateway with scopes, consent, approval, audit, and idempotency

**Files:**
- Create: `tools/gateway.py`, `tests/test_gateway.py`

**Interfaces:**
- Consumes: `SqliteStore`, `SimulatedCompanyAPI`, `ToolSpec`, `Settings`.
- Produces: `ToolResult` with `tool_call_id`, `tool`, `status`, `result`, `error`, `latency_ms`, `duplicate`. `ToolGateway(store, api, settings)` with `make_tool_call_id(run_id, tool, args) -> str`, `validate(agent, tool, args, customer) -> None`, `record_proposed(...)`, `record_decision(...)`, `execute(*, agent, tool, args, run_id, session_id, approved, customer=None) -> ToolResult`. The gateway resolves the customer from `customer`, `args["customer_id"]`, or the ticket owner, in that order. `lookup_customer` checks consent against the resolved customer when one is found, and is allowed to return not found without a consent check.

- [ ] **Step 1: Write the failing gateway tests**

```python
# tests/test_gateway.py
import pytest
from stores.sqlite import SqliteStore
from tools.gateway import ToolGateway
from tools.simulated_apis import SimulatedCompanyAPI

def make_gateway(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": [],
        "consent_scopes": ["crm:read", "crm:write", "refund"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    store.upsert_customer({
        "id": "blueharbor", "name": "Blue Harbor Retail", "aliases": [],
        "consent_scopes": ["crm:read", "crm:write"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    return ToolGateway(store, SimulatedCompanyAPI(store), None)

def test_memory_agent_cannot_write(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="memory", tool="issue_refund",
        args={"customer_id": "acme", "amount": 10, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "scope_denied"

def test_revoked_consent_denies_refund(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="issue_refund",
        args={"customer_id": "blueharbor", "amount": 10, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "consent_denied"

def test_write_requires_approval(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="create_ticket",
        args={"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"},
        run_id="r1", session_id="s1", approved=False,
    )
    assert result.status == "denied"
    assert result.error == "approval_required"

def test_refund_cap_is_enforced(tmp_path):
    gw = make_gateway(tmp_path)
    result = gw.execute(
        agent="executor", tool="issue_refund",
        args={"customer_id": "acme", "amount": 5000, "reason": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    assert result.status == "denied"
    assert result.error == "amount_cap_exceeded"

def test_duplicate_execution_is_prevented(tmp_path):
    gw = make_gateway(tmp_path)
    args = {"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"}
    first = gw.execute(agent="executor", tool="create_ticket", args=args, run_id="r1", session_id="s1", approved=True)
    second = gw.execute(agent="executor", tool="create_ticket", args=args, run_id="r1", session_id="s1", approved=True)
    assert first.status == "executed"
    assert second.status == "executed"
    assert second.duplicate is True
    assert len(gw.store.get_open_tickets("acme", "s1")) == 1

def test_audit_events_are_append_only(tmp_path):
    gw = make_gateway(tmp_path)
    gw.execute(
        agent="executor", tool="create_ticket",
        args={"customer_id": "acme", "subject": "x", "priority": "normal", "description": "x"},
        run_id="r1", session_id="s1", approved=True,
    )
    events = gw.store.list_audit_events("r1")
    assert [e["event_type"] for e in events] == ["attempted", "executed"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_gateway.py -v`
Expected: FAIL because `tools.gateway` does not exist.

- [ ] **Step 3: Implement the gateway**

Scope rules are fixed. Reads require `crm:read`. Proposing a write requires `crm:propose`. Executing a write requires `crm:execute`. Only the executor agent may execute. The action agent may only validate and record proposals.

`validate` checks tool existence, agent scope, customer consent, amount cap, and Pydantic arguments. `execute` calls `validate`, checks approval for writes, computes a deterministic `tool_call_id` with `sha256(f"{run_id}|{tool}|{json.dumps(args, sort_keys=True)}")`, records an `attempted` audit event, returns the existing result if the tool call ID is already recorded, otherwise executes and records `executed` or `failed`. Denials record a `denied` event and return a `ToolResult` with the error code. Never raise for a denied action.

`record_proposed` writes one `proposed` event per tool call. `record_decision` writes `approved` or `rejected` events for every tool call ID in the proposal. The `approved` field passed to `execute` comes from the approval decision.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_gateway.py -v`
Expected: 6 passed.

```bash
git add tools/gateway.py tests/test_gateway.py
git commit -m "feat: add permission gateway and audit trail"
```

---

### Task 8: LLM clients, pricing, and benchmark-driven router

**Files:**
- Create: `mesh/llm.py`, `mesh/pricing.py`, `mesh/router.py`, `tests/test_router.py`

**Interfaces:**
- Consumes: `Settings`.
- Produces: `LLMResponse` with `text`, `model`, `input_tokens`, `output_tokens`, `latency_ms`. `LLMClient` protocol with `complete(*, model, messages, json_mode=True, temperature=0.0) -> LLMResponse`. `GroqClient`, `GeminiClient`. `estimate_cost(model, input_tokens, output_tokens) -> float`. `TaskPolicy` and `RouterPolicy`. `ModelRouter(clients, policy, settings)` with `choose(task)`, `complete_json(*, task, messages, schema, force_provider=None, force_model=None) -> tuple[BaseModel, LLMResponse, float]`, and `complete_fixed(*, provider, model, messages, schema) -> tuple[BaseModel, LLMResponse, float]`.

- [ ] **Step 1: Write the failing router tests**

```python
# tests/test_router.py
import pytest
from pydantic import BaseModel
from mesh.llm import LLMResponse
from mesh.router import ModelRouter, RouterPolicy, TaskPolicy

class Out(BaseModel):
    answer: str

class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
    def complete(self, *, model, messages, json_mode=True, temperature=0.0):
        self.calls.append(model)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model=model, input_tokens=10, output_tokens=5, latency_ms=20)

POLICY = RouterPolicy(tasks={
    "extract": TaskPolicy(provider="groq", model="strong", fallback_provider="gemini", fallback_model="flash"),
})

def test_primary_model_success():
    groq = FakeClient(['{"answer": "ok"}'])
    router = ModelRouter({"groq": groq, "gemini": FakeClient([])}, POLICY, None)
    result, response, cost = router.complete_json(task="extract", messages=[], schema=Out)
    assert result.answer == "ok"
    assert groq.calls == ["strong"]
    assert cost >= 0

def test_fallback_on_primary_failure():
    router = ModelRouter(
        {"groq": FakeClient([RuntimeError("rate limit")]), "gemini": FakeClient(['{"answer": "ok"}'])},
        POLICY, None,
    )
    result, response, cost = router.complete_json(task="extract", messages=[], schema=Out)
    assert result.answer == "ok"
    assert response.model == "flash"

def test_invalid_json_retries_then_fails():
    router = ModelRouter({"groq": FakeClient(["not json", "not json"]), "gemini": FakeClient([])}, POLICY, None)
    with pytest.raises(ValueError):
        router.complete_json(task="extract", messages=[], schema=Out)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: FAIL because `mesh.router` does not exist.

- [ ] **Step 3: Implement the provider clients, pricing, and router**

`GroqClient` uses `groq.Groq` and `chat.completions.create` with `response_format={"type": "json_object"}` and `temperature=0`. `GeminiClient` uses `google.genai.Client` and `models.generate_content` with `response_mime_type="application/json"`. Normalize token usage from each provider.

`mesh/pricing.py` contains a `MODEL_PRICES` dict mapping model IDs to `(input_per_million, output_per_million)` and a comment with the lookup date and source. `estimate_cost` returns `0.0` for unknown models and logs a warning.

`RouterPolicy` loads `eval/router_policy.json` when present. Otherwise it defaults to `GROQ_MODEL` for `extract` and `propose`, with `GEMINI_MODEL` as fallback. `ModelRouter.complete_json` tries the primary once, then the fallback once. On JSON validation failure it retries the same model once with an appended error message, then tries the fallback. It returns the parsed model, the response, and the estimated cost. It records the chosen model and reason in a `choice` object that nodes add to the trace.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_router.py -v`
Expected: 3 passed.

```bash
git add mesh/llm.py mesh/pricing.py mesh/router.py tests/test_router.py
git commit -m "feat: add provider clients and benchmark-driven router"
```

---

### Task 9: State, dependencies, and extractor agent

**Files:**
- Create: `mesh/state.py`, `mesh/deps.py`, `mesh/nodes/__init__.py`, `mesh/nodes/extractor.py`, `mesh/prompts/extractor.md`, `tests/test_extractor.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `ModelRouter`, `ToolGateway`, `SqliteStore`, `ChromaMemory`, `Embedder`, `SimulatedCompanyAPI`, `Settings`.
- Produces: `MeshState`, `MeshDeps`, `ExtractionOutput`, `TraceEvent`, `ApprovalDecision`. `extractor_node(state, deps) -> dict`.

- [ ] **Step 1: Write the failing extractor tests**

```python
# tests/test_extractor.py
from dataclasses import replace
from mesh.nodes.extractor import extractor_node
from tests.conftest import make_deps, fake_router

def test_extractor_resolves_known_customer():
    deps = make_deps(router=fake_router('{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "acme_003", "transcript": "We were charged twice.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["extraction"]["customer_id"] == "acme"
    assert update["escalated"] is False
    assert update["trace"][0]["node"] == "extractor"

def test_unknown_customer_escalates():
    deps = make_deps(router=fake_router('{"customer_name":"Unknown Corp","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.9}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "x", "transcript": "Hello.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "customer_not_found"

def test_low_confidence_escalates():
    deps = make_deps(router=fake_router('{"customer_name":"Atlas Freight","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.2}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "atlas_002", "transcript": "It is not working.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "low_confidence"

def test_invalid_model_output_escalates():
    deps = make_deps(router=fake_router(error=ValueError("invalid json")))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "x", "transcript": "Hello.", "trace": [], "errors": []}
    update = extractor_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "extraction_failed"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_extractor.py -v`
Expected: FAIL because the extractor does not exist.

- [ ] **Step 3: Implement the state, dependencies, and extractor**

`MeshState` is a `TypedDict` with the keys from the design. `MeshDeps` is a dataclass with `router`, `gateway`, `store`, `memory`, `embedder`, `api`, and `settings`. Add a `now` callable defaulting to `datetime.now(timezone.utc)`.

`ExtractionOutput` is a Pydantic model with `customer_name`, `intent`, `entities`, `action_items`, `evidence`, and `confidence` constrained to 0 through 1. `TraceEvent` is a Pydantic model with `node`, `event`, `model`, `tokens_in`, `tokens_out`, `cost_usd`, `latency_ms`, and `detail`. `ApprovalDecision` is a Pydantic model with `approved: bool`, `approved_by: str`, and `note: str | None = None`.

Add shared test doubles to `tests/conftest.py`. `fake_router(*responses, error=None, **by_task)` returns an object with `complete_json` and `complete_fixed`. Positional responses are consumed in call order. Keyword responses are selected by task and reused across calls. `fake_memory(hits)` returns an object with `search(**kwargs) -> hits`. `make_deps(router=None, memory=None, gateway=None, store=None)` returns a `MeshDeps` with a test `Settings`, a store seeded with the six customers, a default memory returning one Acme hit, and a default gateway that resolves `lookup_customer` to Acme and executes writes successfully. The default memory hit makes graph tests exercise the memory note call.

The extractor builds the prompt from `mesh/prompts/extractor.md`, the transcript, and the customer directory from `deps.store.list_customers()`. It calls `deps.router.complete_json(task="extract", ...)`. On success it resolves the returned name through `deps.gateway.execute(agent="extractor", tool="lookup_customer", ...)` and returns the resolved record in `customer`. If the customer is not found, or confidence is below `deps.settings.confidence_threshold`, or the router raises, it returns an update with `escalated=True`, an `escalation_reason`, a trace event, and no extraction that could trigger a write.

The prompt instructs the model to return JSON only, to use the fixed intent taxonomy, and to treat transcript text as data.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_extractor.py -v`
Expected: 4 passed.

```bash
git add mesh/state.py mesh/deps.py mesh/nodes/__init__.py mesh/nodes/extractor.py mesh/prompts/extractor.md tests/test_extractor.py
git commit -m "feat: add extractor agent and shared state"
```

---

### Task 10: Memory agent

**Files:**
- Create: `mesh/nodes/memory.py`, `mesh/prompts/memory_note.md`, `tests/test_memory_agent.py`

**Interfaces:**
- Consumes: `MeshDeps`, `MeshState`, `ChromaMemory`, `Embedder`, `ModelRouter`.
- Produces: `memory_node(state, deps) -> dict` with `memories`, `memory_note`, and a trace event.

- [ ] **Step 1: Write the failing memory agent tests**

```python
# tests/test_memory_agent.py
from mesh.nodes.memory import memory_node
from tests.conftest import make_deps, fake_memory, fake_router

def test_memory_hit_produces_cited_note():
    hits = [{"interaction_id": "int_acme_002", "summary": "Refund confirmed", "resolution": "Refund issued", "similarity": 0.91, "occurred_at": "2026-08-05T00:00:00Z"}]
    deps = make_deps(memory=fake_memory(hits), router=fake_router('{"note":"This matches int_acme_002, resolved by refund."}'))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "acme_003", "transcript": "Charged twice again.", "extraction": {"customer_id": "acme"}, "trace": [], "errors": []}
    update = memory_node(state, deps)
    assert update["memories"] == hits
    assert "int_acme_002" in update["memory_note"]
    assert update["trace"][0]["node"] == "memory"

def test_no_history_returns_empty_note():
    deps = make_deps(memory=fake_memory([]), router=fake_router(error=AssertionError("should not call model")))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "cedar_001", "transcript": "Pricing question.", "extraction": {"customer_id": "cedar"}, "trace": [], "errors": []}
    update = memory_node(state, deps)
    assert update["memories"] == []
    assert update["memory_note"] == "No prior interactions."
    assert update["trace"][0]["model"] is None

def test_search_gets_customer_and_time_scope():
    captured = {}
    class Memory:
        def search(self, **kwargs):
            captured.update(kwargs)
            return []
    deps = make_deps(memory=Memory(), router=fake_router(error=AssertionError("not called")))
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "x", "transcript": "x", "extraction": {"customer_id": "acme"}, "transcript_occurred_at": "2026-09-02T10:00:00Z", "trace": [], "errors": []}
    memory_node(state, deps)
    assert captured["customer_id"] == "acme"
    assert captured["before"] == "2026-09-02T10:00:00Z"
    assert captured["session_id"] == "s1"
    assert captured["exclude_run_id"] == "r1"
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_memory_agent.py -v`
Expected: FAIL because `mesh.nodes.memory` does not exist.

- [ ] **Step 3: Implement the memory agent**

Embed the transcript with `deps.embedder.embed`. Call `deps.memory.search(customer_id=..., query_embedding=..., top_k=3, before=state["transcript_occurred_at"], session_id=state["session_id"], exclude_run_id=state["run_id"])`. Filter hits below `0.25` similarity. With no hits, return `"No prior interactions."` and no model call. With hits, call `deps.router.complete_fixed(provider="groq", model=deps.settings.groq_fast_model, ...)` with the memory note prompt and the hits. The prompt requires the note to cite interaction IDs exactly. Validate that every cited ID is in the retrieved set. If validation fails, fall back to a deterministic note such as `"Matches int_acme_002. Prior resolution: Refund issued."`.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_memory_agent.py -v`
Expected: 3 passed.

```bash
git add mesh/nodes/memory.py mesh/prompts/memory_note.md tests/test_memory_agent.py
git commit -m "feat: add memory agent with customer-scoped retrieval"
```

---

### Task 11: Action agent

**Files:**
- Create: `mesh/nodes/action.py`, `mesh/prompts/action.md`, `tests/test_action.py`

**Interfaces:**
- Consumes: `MeshDeps`, `MeshState`, `catalog_for`, `ToolGateway`.
- Produces: `ProposedToolCall`, `ActionOutput`, `action_node(state, deps) -> dict` with `proposal`.

- [ ] **Step 1: Write the failing action tests**

```python
# tests/test_action.py
from mesh.nodes.action import action_node
from tests.conftest import make_deps, fake_router

def test_action_builds_validated_proposal():
    deps = make_deps(router=fake_router('{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Customer charged twice"},"rationale":"Track the issue","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}'))
    state = {"run_id": "r1", "session_id": "s1", "extraction": {"customer_id": "acme", "intent": "billing_dispute", "action_items": []}, "memories": [], "memory_note": "No prior interactions.", "trace": [], "errors": []}
    update = action_node(state, deps)
    assert update["proposal"]["requires_approval"] is True
    assert update["proposal"]["tool_calls"][0]["tool"] == "create_ticket"
    assert update["proposal"]["tool_calls"][0]["tool_call_id"].startswith("r1:")

def test_unknown_tool_escalates():
    deps = make_deps(router=fake_router('{"tool_calls":[{"tool":"drop_database","arguments":{},"rationale":"bad","requires_approval":true}],"rationale":"bad","risk_level":"high","requires_approval":true}'))
    state = {"run_id": "r1", "session_id": "s1", "extraction": {"customer_id": "acme"}, "memories": [], "memory_note": "", "trace": [], "errors": []}
    update = action_node(state, deps)
    assert update["escalated"] is True
    assert update["escalation_reason"] == "invalid_tool"

def test_no_action_skips_approval():
    deps = make_deps(router=fake_router('{"tool_calls":[],"rationale":"No action needed","risk_level":"low","requires_approval":false}'))
    state = {"run_id": "r1", "session_id": "s1", "extraction": {"customer_id": "acme"}, "memories": [], "memory_note": "", "trace": [], "errors": []}
    update = action_node(state, deps)
    assert update["proposal"]["tool_calls"] == []
    assert update["proposal"]["requires_approval"] is False
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_action.py -v`
Expected: FAIL because `mesh.nodes.action` does not exist.

- [ ] **Step 3: Implement the action agent**

Read the customer from `state["customer"]`. If it is absent, load it with `deps.store.get_customer(state["extraction"]["customer_id"])`. Build the tool catalog with `catalog_for({"crm:propose"}, customer["consent_scopes"])`. Call `deps.router.complete_json(task="propose", ...)` with the extraction, memory note, ticket snapshot, and catalog. Validate every proposed tool exists in `TOOLS`, is a write tool, validates against its input model, and is in the filtered catalog. Compute `tool_call_id = deps.gateway.make_tool_call_id(run_id, tool, args)` for each call. If any validation fails, set `escalated=True`, `escalation_reason="invalid_tool"` or `"invalid_arguments"`, and do not return a proposal. Set `requires_approval` to true when `tool_calls` is non-empty. Record `proposed` audit events only after the proposal is accepted by the graph, not inside the action node.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_action.py -v`
Expected: 3 passed.

```bash
git add mesh/nodes/action.py mesh/prompts/action.md tests/test_action.py
git commit -m "feat: add action agent with validated tool proposals"
```

---

### Task 12: Approval, executor, escalation, and graph

**Files:**
- Create: `mesh/nodes/approval.py`, `mesh/nodes/executor.py`, `mesh/nodes/escalate.py`, `mesh/graph.py`, `tests/test_graph.py`

**Interfaces:**
- Consumes: all nodes and `MeshDeps`.
- Produces: `approval_node(state, deps) -> dict`, `executor_node(state, deps) -> dict`, `escalate_node(state, deps) -> dict`, `build_graph(deps, checkpointer) -> CompiledStateGraph`.

- [ ] **Step 1: Write the failing graph tests**

```python
# tests/test_graph.py
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from mesh.graph import build_graph
from tests.conftest import make_deps, fake_router

def run_to_interrupt(graph, state, config):
    for event in graph.stream(state, config, stream_mode="updates"):
        if "__interrupt__" in event:
            return event["__interrupt__"][0].value
    return None

def test_happy_path_approves_and_executes():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        '{"note":"No prior interactions."}',
        '{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Charged twice"},"rationale":"Track it","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t1"}}
    state = {"run_id": "r1", "session_id": "s1", "transcript_id": "acme_003", "transcript_occurred_at": "2026-09-02T10:00:00Z", "transcript": "Charged twice.", "trace": [], "errors": []}
    payload = run_to_interrupt(graph, state, config)
    assert payload["proposal"]["tool_calls"][0]["tool"] == "create_ticket"
    result = graph.invoke(Command(resume={"approved": True, "approved_by": "demo_user"}), config)
    assert result["tool_results"][0]["status"] == "executed"
    assert result["approval"]["approved"] is True

def test_low_confidence_stops_before_memory_and_action():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Atlas Freight","intent":"other","entities":{},"action_items":[],"evidence":[],"confidence":0.1}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t2"}}
    state = {"run_id": "r2", "session_id": "s2", "transcript_id": "atlas_002", "transcript_occurred_at": "2026-09-16T10:00:00Z", "transcript": "It is not working.", "trace": [], "errors": []}
    result = graph.invoke(state, config)
    assert result["escalated"] is True
    assert result["escalation_reason"] == "low_confidence"
    assert result["memories"] == []

def test_rejection_writes_audit_and_skips_execution():
    deps = make_deps(router=fake_router(
        '{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        '{"note":"No prior interactions."}',
        '{"tool_calls":[{"tool":"create_ticket","arguments":{"customer_id":"acme","subject":"Duplicate charge","priority":"high","description":"Charged twice"},"rationale":"Track it","requires_approval":true}],"rationale":"Create a ticket","risk_level":"low","requires_approval":true}',
    ))
    graph = build_graph(deps, MemorySaver())
    config = {"configurable": {"thread_id": "t3"}}
    state = {"run_id": "r3", "session_id": "s3", "transcript_id": "acme_003", "transcript_occurred_at": "2026-09-02T10:00:00Z", "transcript": "Charged twice.", "trace": [], "errors": []}
    run_to_interrupt(graph, state, config)
    result = graph.invoke(Command(resume={"approved": False, "approved_by": "demo_user"}), config)
    assert result["approval"]["approved"] is False
    assert result["tool_results"] == []
    assert [e["event_type"] for e in deps.store.list_audit_events("r3")] == ["proposed", "rejected"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_graph.py -v`
Expected: FAIL because `mesh.graph` does not exist.

- [ ] **Step 3: Implement the nodes and graph**

`approval_node` calls `interrupt({"proposal": state["proposal"], "customer": state["customer"], "run_id": state["run_id"]})`. It validates the resume value with an `ApprovalDecision` model and records `proposed` events for every tool call, then `approved` or `rejected` events through `deps.gateway.record_decision`.

`executor_node` does nothing when there are no tool calls. For a rejected proposal it returns `{"tool_results": []}`. For an approved proposal it pre-validates every call with `deps.gateway.validate` before executing any. If any validation fails, it records denied events for all calls and returns an escalation. If all pass, it executes each call with `approved=True` and returns `tool_results`.

`escalate_node` sets `escalated=True`, preserves the reason, and records a trace event. It never calls a model or a tool.

Graph edges are `START -> extractor`. After the extractor, go to `escalate` when `escalated` is true, otherwise `memory`. Then `memory -> action`. After action, go to `escalate` when `escalated` is true, otherwise `approval` when `proposal["requires_approval"]` is true, otherwise `executor`. Then `approval -> executor`, and `executor -> END`. `escalate -> END`.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_graph.py -v`
Expected: 3 passed.

```bash
git add mesh/nodes/approval.py mesh/nodes/executor.py mesh/nodes/escalate.py mesh/graph.py tests/test_graph.py
git commit -m "feat: add approval gate, executor, and LangGraph mesh"
```

---

### Task 13: Deterministic scoring and stateless baseline

**Files:**
- Create: `eval/__init__.py`, `eval/scoring.py`, `eval/baseline.py`, `tests/test_scoring.py`, `tests/test_baseline.py`

**Interfaces:**
- Consumes: `ModelRouter`, ground truth, extraction and proposal dicts.
- Produces: `normalize_text(text) -> str`, `token_overlap(a, b) -> float`, `matches(a, b) -> bool`, `score_extraction(expected, actual) -> dict`, `score_memory(expected_prior, hits, negative=False) -> dict`, `score_action(expected_calls, actual_calls) -> dict`, `diagnose(mismatch) -> str`, `run_baseline(transcript, customer_directory, tool_catalog, router, provider, model) -> dict`.

- [ ] **Step 1: Write the failing scoring tests**

```python
# tests/test_scoring.py
from eval.scoring import matches, score_action, score_extraction, score_memory

def test_exact_and_fuzzy_matching():
    assert matches("Acme Logistics", "acme logistics")
    assert matches("refund the duplicate charge", "refund duplicate charge")
    assert not matches("refund duplicate charge", "cancel subscription")

def test_extraction_scoring():
    expected = {"customer_id": "acme", "intent": "billing_dispute", "action_items": [{"type": "refund", "target": "invoice_8841", "amount": 120.0, "due": None}]}
    actual = {"customer_id": "acme", "intent": "billing_dispute", "action_items": [{"type": "refund", "target": "invoice_8841", "amount": 120.0, "due": None}]}
    result = score_extraction(expected, actual)
    assert result["customer_correct"] is True
    assert result["intent_correct"] is True
    assert result["action_f1"] == 1.0

def test_memory_positive_and_negative():
    assert score_memory("int_acme_002", [{"interaction_id": "int_acme_002"}])["recall_at_1"] is True
    assert score_memory("int_acme_002", [{"interaction_id": "int_acme_001"}, {"interaction_id": "int_acme_002"}])["recall_at_3"] is True
    assert score_memory(None, [{"interaction_id": "int_x", "similarity": 0.1}], negative=True)["correct_empty"] is True

def test_action_scoring():
    expected = [{"tool": "create_ticket", "arguments": {"customer_id": "acme", "subject": "Duplicate charge", "priority": "high"}}]
    actual = [{"tool": "create_ticket", "arguments": {"customer_id": "acme", "subject": "Duplicate charge", "priority": "high"}}]
    result = score_action(expected, actual)
    assert result["tool_correct"] is True
    assert result["args_correct"] is True
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_scoring.py -v`
Expected: FAIL because `eval.scoring` does not exist.

- [ ] **Step 3: Implement scoring and the baseline**

Normalize case, whitespace, and punctuation. Use set token overlap `len(a & b) / len(a | b)` and accept at least `0.8`. Exact matches pass regardless of case. `score_extraction` returns customer and intent booleans, action-item precision, recall, F1, and JSON validity. `score_memory` returns `recall_at_1`, `recall_at_3`, `correct_empty`, and `top_similarity`. `score_action` pairs expected and actual calls by tool name and returns tool correctness and field-level argument correctness. `diagnose` returns one of the fixed categories from the design doc and defaults to `"other"`.

`run_baseline` uses one prompt with the transcript, customer directory, tool catalog, and the same output schema. It calls `router.complete_fixed` with the pinned provider and model. It has no memory, no routing, and no tool execution. Its output shape is `{"extraction": ..., "proposal": ...}`.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_scoring.py tests/test_baseline.py -v`
Expected: 5 passed.

```bash
git add eval/__init__.py eval/scoring.py eval/baseline.py tests/test_scoring.py tests/test_baseline.py
git commit -m "feat: add deterministic scoring and stateless baseline"
```

---

### Task 14: Evaluation harness and reports

**Files:**
- Create: `eval/harness.py`, `tests/test_eval_harness.py`
- Generated: `eval/results.json`, `eval/report.md`, `eval/charts/*.png`

**Interfaces:**
- Consumes: `MeshDeps`, node functions, `run_baseline`, scoring functions, ground truth.
- Produces: `run_system(deps, transcripts, provider, model) -> list[dict]`, `run_baseline_all(...) -> list[dict]`, `summarize(system, baseline, routed) -> dict`, `write_report(results, path)`, `write_charts(results, charts_dir)`, `main(argv=None) -> None`.

- [ ] **Step 1: Write the failing harness test**

```python
# tests/test_eval_harness.py
import json
from pathlib import Path
from eval.harness import run_system, summarize

def test_harness_runs_two_transcripts_with_fakes(tmp_path):
    from tests.conftest import make_deps, fake_router
    deps = make_deps(router=fake_router(
        extract='{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        memory_note='{"note":"No prior interactions."}',
        propose='{"tool_calls":[],"rationale":"No action","risk_level":"low","requires_approval":false}',
    ))
    transcripts = [
        {"id": "acme_003", "customer_id": "acme", "occurred_at": "2026-09-02T10:00:00Z", "transcript": "Charged twice.", "intent": "billing_dispute", "action_items": [], "expected_tool_calls": [], "prior_interaction_id": "int_acme_002", "memory_probe": True, "tags": [], "summary": "x", "resolution": "x"},
        {"id": "cedar_001", "customer_id": "cedar", "occurred_at": "2026-09-10T10:00:00Z", "transcript": "Pricing question.", "intent": "upgrade", "action_items": [], "expected_tool_calls": [], "prior_interaction_id": None, "memory_probe": True, "tags": [], "summary": "x", "resolution": None},
    ]
    rows = run_system(deps, transcripts, provider="groq", model="strong")
    assert len(rows) == 2
    assert rows[0]["transcript_id"] == "acme_003"
    assert "extraction" in rows[0]
    summary = summarize(rows, [], [])
    assert "system" in summary
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_eval_harness.py -v`
Expected: FAIL because `eval.harness` does not exist.

- [ ] **Step 3: Implement the harness**

The system run calls `extractor_node`, `memory_node`, and `action_node` directly with the same dependencies. It does not invoke the graph, so no approval interrupt occurs and no writes execute. It forces the pinned provider and model through the router. The baseline run uses `run_baseline`. The routed run uses the production router policy.

`summarize` computes means and per-group metrics for `memory_positive`, `memory_negative`, `guardrail`, and `standard`. It includes median and p95 latency, total tokens, total cost, and escalation rate. `write_report` writes a markdown report with a summary table, a per-transcript table, a failure section with expected and actual values, and a limitations note that n=15 is directional. `write_charts` writes accuracy, latency, and cost bar charts with matplotlib.

The CLI accepts `--provider`, `--model`, `--limit`, and `--output-dir`. The default run uses the pinned primary model. Print the report path when finished.

- [ ] **Step 4: Run the tests and the live evaluation**

Run: `.venv/bin/pytest tests/test_eval_harness.py -v`
Expected: 1 passed.

Run: `.venv/bin/python -m eval.harness`
Expected: `eval/results.json`, `eval/report.md`, and at least two charts are written. The report includes at least one failure or limitation row.

- [ ] **Step 5: Commit**

```bash
git add eval/harness.py tests/test_eval_harness.py eval/results.json eval/report.md eval/charts
git commit -m "feat: add evaluation harness and first results"
```

---

### Task 15: Router benchmark and policy

**Files:**
- Create: `eval/router_benchmark.py`, `tests/test_router_benchmark.py`
- Generated: `eval/router_results.json`, `eval/router_policy.json`

**Interfaces:**
- Consumes: extraction and action prompts, `LLMClient`, scoring functions, ground truth.
- Produces: `benchmark_task(transcripts, clients, provider, model, task) -> dict`, `choose_policy(results) -> dict`, `main(argv=None) -> None`.

- [ ] **Step 1: Write the failing benchmark test**

```python
# tests/test_router_benchmark.py
from eval.router_benchmark import choose_policy

def test_policy_prefers_quality_then_cost():
    results = {
        "extract": [
            {"provider": "groq", "model": "a", "f1": 0.8, "cost_usd": 0.01, "latency_ms": 100},
            {"provider": "gemini", "model": "b", "f1": 0.9, "cost_usd": 0.02, "latency_ms": 200},
            {"provider": "groq", "model": "c", "f1": 0.7, "cost_usd": 0.001, "latency_ms": 50},
        ],
        "propose": [
            {"provider": "groq", "model": "a", "f1": 0.7, "cost_usd": 0.01, "latency_ms": 100},
            {"provider": "gemini", "model": "b", "f1": 0.7, "cost_usd": 0.02, "latency_ms": 200},
        ],
    }
    policy = choose_policy(results)
    assert policy["tasks"]["extract"]["model"] == "b"
    assert policy["tasks"]["propose"]["model"] == "a"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_router_benchmark.py -v`
Expected: FAIL because `eval.router_benchmark` does not exist.

- [ ] **Step 3: Implement the benchmark**

Run three models on both routed tasks. Default candidates are `("groq", settings.groq_model)`, `("groq", settings.groq_fast_model)`, and `("gemini", settings.gemini_model)`. Allow `--provider-model` overrides. Use the same 15 transcripts and the same scoring functions. Add a delay between requests and retry on rate limits. Write raw results to `eval/router_results.json` and the chosen policy to `eval/router_policy.json`. The policy format is:

```json
{
  "generated_at": "2026-11-14T00:00:00Z",
  "tasks": {
    "extract": {"provider": "groq", "model": "openai/gpt-oss-120b", "fallback_provider": "gemini", "fallback_model": "gemini-2.5-flash", "reason": "best action F1 at acceptable cost"},
    "propose": {"provider": "groq", "model": "openai/gpt-oss-120b", "fallback_provider": "gemini", "fallback_model": "gemini-2.5-flash", "reason": "best tool selection accuracy"}
  }
}
```

`choose_policy` ranks by F1 descending, then cost ascending, then latency ascending. The fallback is the second-best model from a different provider when possible.

- [ ] **Step 4: Run the tests and the live benchmark**

Run: `.venv/bin/pytest tests/test_router_benchmark.py -v`
Expected: 1 passed.

Run: `.venv/bin/python -m eval.router_benchmark`
Expected: `eval/router_results.json` and `eval/router_policy.json` are written, and the printed table shows three models per task without rate-limit errors.

- [ ] **Step 5: Commit**

```bash
git add eval/router_benchmark.py tests/test_router_benchmark.py eval/router_results.json eval/router_policy.json
git commit -m "feat: add model router benchmark and policy"
```

---

### Task 16: Streamlit run view and live graph

**Files:**
- Create: `app.py`, `ui/__init__.py`, `ui/graph.py`, `ui/trace.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `Settings`, `SqliteStore`, `ChromaMemory`, `FastEmbedder`, `ToolGateway`, `SimulatedCompanyAPI`, `ModelRouter`, `build_graph`, seed data.
- Produces: `render_graph(node_states: dict[str, str]) -> None`, `render_trace(trace: list[dict]) -> None`, `main() -> None`.

- [ ] **Step 1: Write the failing app test**

```python
# tests/test_app.py
from streamlit.testing.v1 import AppTest

def test_app_loads_without_api_calls():
    at = AppTest.from_file("app.py").run()
    assert not at.exception
    assert at.title[0].value == "Agent Mesh"
    assert at.selectbox[0].label == "Transcript"
    assert at.button[0].label == "Run"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: FAIL because `app.py` does not exist.

- [ ] **Step 3: Implement the run view and graph component**

`app.py` sets the page title to "Agent Mesh", creates the three tabs "Run", "Evaluation", and "Method", and defines a `st.cache_resource` dependency builder. Do not call that builder at import time. The transcript selector reads `data/transcripts` directly, so the app loads without API keys or a database. The first Run press builds dependencies and seeds data if the customer table is empty. It generates a session ID with `uuid4().hex` and stores it in `st.session_state`.

The Run tab lists the 15 transcripts by title. It does not call a model until the Run button is pressed. On run, it invokes the graph with `thread_id=f"{session_id}:{run_id}"` and consumes `graph.stream(initial_state, config, stream_mode="updates")`. It updates the node state map from `pending` to `running` to `done`. When the stream yields `__interrupt__`, it stores the payload in `st.session_state["pending"]` and renders the approval view from Task 17. The trace table shows node, event, model, tokens, cost, and latency.

`render_graph` draws an inline SVG with six nodes and directed edges. Node colors represent `pending`, `running`, `done`, `waiting`, and `error`. Use `st.components.v1.html` with a fixed height of 260 pixels. The graph must remain readable at mobile width.

`render_trace` renders a dataframe sorted by trace order.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: 1 passed.

```bash
git add app.py ui/__init__.py ui/graph.py ui/trace.py tests/test_app.py
git commit -m "feat: add Streamlit run view and live agent graph"
```

---

### Task 17: Approval, audit, evaluation, and method views

**Files:**
- Create: `ui/approval.py`, `ui/audit.py`, `ui/evaluation.py`, `tests/test_ui.py`
- Modify: `app.py`

**Interfaces:**
- Consumes: pending proposal, `SqliteStore` audit events, `eval/results.json`, `eval/router_results.json`, `eval/charts`.
- Produces: `render_approval(pending) -> str | None`, `group_audit_events(events) -> list[dict]`, `render_audit(events) -> None`, `render_evaluation(results_path, router_results_path, charts_dir) -> None`.

- [ ] **Step 1: Write the failing UI helper tests**

```python
# tests/test_ui.py
from ui.audit import group_audit_events

def test_audit_events_group_by_tool_call():
    events = [
        {"tool_call_id": "c1", "event_type": "proposed", "tool": "create_ticket", "agent": "action", "created_at": "1", "detail": {}},
        {"tool_call_id": "c1", "event_type": "approved", "tool": "create_ticket", "agent": "approval", "created_at": "2", "detail": {}},
        {"tool_call_id": "c1", "event_type": "executed", "tool": "create_ticket", "agent": "executor", "created_at": "3", "detail": {}},
        {"tool_call_id": "c2", "event_type": "denied", "tool": "issue_refund", "agent": "executor", "created_at": "4", "detail": {}},
    ]
    groups = group_audit_events(events)
    assert len(groups) == 2
    assert [e["event_type"] for e in groups[0]["events"]] == ["proposed", "approved", "executed"]
    assert groups[1]["status"] == "denied"
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `.venv/bin/pytest tests/test_ui.py -v`
Expected: FAIL because `ui.audit` does not exist.

- [ ] **Step 3: Implement the approval, audit, and evaluation views**

The approval view shows the proposal rationale, risk level, and a card per tool call with tool name, arguments, and idempotency ID. It renders "Approve and execute" and "Reject" buttons. It returns `"approve"`, `"reject"`, or `None`. On a decision, call `graph.invoke(Command(resume={"approved": decision == "approve", "approved_by": "demo_user"}), config)`, clear the pending payload, and rerun the page.

The audit view groups events by tool call ID and shows a status badge per group. It renders below the approval controls and updates after execution.

The evaluation view reads `eval/results.json`, `eval/router_results.json`, and `eval/charts`. It shows the system, baseline, and routed summary table, the router benchmark table, the charts, and a failure explorer with transcript ID, expected, actual, and diagnosis. If a file is missing, it shows a short message with the command to generate it.

The Method tab shows the architecture diagram, the scope matrix, the data note, the limitations, and the cost model. Keep it as markdown in `app.py` or a constant in `ui/evaluation.py`.

- [ ] **Step 4: Run the tests and commit**

Run: `.venv/bin/pytest tests/test_ui.py tests/test_app.py -v`
Expected: 2 passed.

```bash
git add app.py ui/approval.py ui/audit.py ui/evaluation.py tests/test_ui.py
git commit -m "feat: add approval, audit, and evaluation views"
```

---

### Task 18: Docker, deployment, README, and demo assets

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `scripts/deploy_hf.py`, `docs/demo-script.md`, `README.md`, `assets/architecture.png`, `assets/run-view.png`, `assets/demo.gif`
- Modify: `.env.example` if the deployment variables are missing

**Interfaces:**
- Consumes: the working app, `HF_TOKEN` from `.env`, runtime secrets.
- Produces: a public Hugging Face Space and a README with the live link, evaluation table, and demo assets.

- [ ] **Step 1: Write the Dockerfile and build locally**

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/app \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

RUN useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')" \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
```

If fastembed does not recognize the model name, replace it with `BAAI/bge-small-en-v1.5` in the Dockerfile, `.env.example`, and `Settings` defaults before building.

Add `.dockerignore` with `.git`, `.env`, `.venv`, `__pycache__`, `data/*.db`, `data/chroma`, `tests`, `.pytest_cache`, and `*.log`.

Verification.

```bash
docker build -t agent-mesh .
docker run --rm -p 7860:7860 --env-file .env agent-mesh
```

Expected: the app loads at `http://localhost:7860` and the transcript selector is visible.

- [ ] **Step 2: Write the deployment script**

`scripts/deploy_hf.py` reads `HF_TOKEN` and the runtime secrets from the environment and uses `huggingface_hub.HfApi`. It creates the Space `abdulhadihub/multi-agent-call-eval` with `space_sdk="docker"` and `private=False` if it does not exist, uploads the folder while ignoring `.env`, `.git`, `.venv`, `data/*.db`, and `data/chroma`, and adds the secrets `GROQ_API_KEY`, `GEMINI_API_KEY`, `DEMO_MODE`, `CONFIDENCE_THRESHOLD`, `SESSION_RUN_LIMIT`, and `DAILY_RUN_LIMIT`. It never prints secret values.

Verification.

```bash
.venv/bin/python scripts/deploy_hf.py
```

Expected: the script prints the Space URL and the upload completes. The Space builds and the app is reachable.

- [ ] **Step 3: Capture screenshots and write the README**

Use the Playwright tool to open the local app, run a transcript, approve the action, and capture the run view with the graph, trace, and audit timeline. Save `assets/run-view.png`. Capture the evaluation tab as `assets/evaluation.png`. If ffmpeg is available, turn a short Playwright recording into `assets/demo.gif`, otherwise use the screenshots only.

Write `README.md` with these sections.

- Title and one-sentence pitch.
- Live demo link and a 3-minute video link placeholder.
- Demo GIF or screenshot.
- What the demo shows, in five bullets.
- Architecture diagram with Mermaid.
- The three agents and their scopes.
- Tool permission matrix.
- Evaluation table with system, baseline, and routed columns.
- Router benchmark table and the policy rationale.
- Failure cases with transcript IDs.
- Cost and free-tier limits.
- Run locally, test, evaluate, and deploy commands.
- Data note and license note.
- Design decisions and limitations.

Write `docs/demo-script.md` with a 3-minute outline. Use 0:00 to 0:15 for the problem, 0:15 to 0:45 for the architecture, 0:45 to 2:15 for the live run and memory recall, 2:15 to 2:45 for the guardrail failure cases and evaluation, and 2:45 to 3:00 for the close.

- [ ] **Step 4: Run the full verification**

```bash
.venv/bin/pytest -q
.venv/bin/python -m eval.harness --limit 3
docker build -t agent-mesh .
docker run --rm -d -p 7860:7860 --env-file .env --name agent-mesh-check agent-mesh
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860
docker stop agent-mesh-check
```

Expected: all tests pass, the harness writes results, the Docker build succeeds, and curl returns 200. Fix any failures before committing.

- [ ] **Step 5: Commit and deploy**

```bash
git add Dockerfile .dockerignore scripts/deploy_hf.py docs/demo-script.md README.md assets
git commit -m "feat: add container deployment and project documentation"
```

Run the deployment script once the user confirms the public Space name. After deployment, add the live link and video link to the README, commit the README update, and push the repository to GitHub.

```bash
git add README.md
git commit -m "docs: add live demo links"
git push origin main
```

Expected: the GitHub repository is public, the Space URL loads, and a run completes end to end.

---

## Self-Review

Spec coverage check.

- Three agents with scopes and context: Tasks 9, 10, 11, 12.
- Persistent memory: Tasks 3, 5, 10.
- Tool gateway and audit: Tasks 6, 7, 12.
- Benchmark router: Tasks 8, 15.
- Evaluation harness and baseline: Tasks 13, 14.
- Guardrails and escalation: Tasks 7, 9, 12, 17.
- UI and visual presentation: Tasks 16, 17, 18.
- Deployment and public link: Task 18.
- README, video script, charts: Tasks 14, 15, 18.

Type consistency check.

- `MeshDeps` fields are used by all nodes.
- `ToolResult.status` values are `executed`, `denied`, and `failed`. `duplicate` is a separate boolean.
- `memory.search` returns `interaction_id`, `summary`, `resolution`, `similarity`, `occurred_at` consistently.
- `tool_call_id` is generated by `ToolGateway.make_tool_call_id` and reused in proposals, audit events, and execution.
- `ApprovalDecision` is the resume payload with `approved` and `approved_by`.

Review Focus check.

- Unknown customer test is in Task 9.
- Empty and cross-customer memory tests are in Tasks 3 and 10.
- Provider failure tests are in Tasks 8 and 12.
- Adversarial transcript denial is tested through the gateway and graph guardrail tests in Tasks 7 and 12.
- Duplicate execution and session isolation tests are in Tasks 3 and 7.

Proportion check.

The plan is longer than the spec because the project has several independent subsystems. Test code is limited to the assertions that pin behavior, and implementation steps name signatures rather than transcribing bodies.
