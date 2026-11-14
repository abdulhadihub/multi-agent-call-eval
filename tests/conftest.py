import hashlib

from mesh.config import Settings
from mesh.deps import MeshDeps
from mesh.llm import LLMResponse
from tools.gateway import ToolResult


class FakeEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest] * 12
        return values[: self.dim]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


TEST_CUSTOMERS = [
    {
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
    },
    {
        "id": "northwind", "name": "Northwind Health", "aliases": ["Northwind"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
    },
    {
        "id": "blueharbor", "name": "Blue Harbor Retail", "aliases": ["Blue Harbor"],
        "consent_scopes": ["crm:read", "crm:write"],
    },
    {
        "id": "cedar", "name": "Cedar Analytics", "aliases": ["Cedar"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
    },
    {
        "id": "lumen", "name": "Lumen Studio", "aliases": ["Lumen"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
    },
    {
        "id": "atlas", "name": "Atlas Freight", "aliases": ["Atlas"],
        "consent_scopes": ["crm:read", "crm:write", "refund"],
    },
]

TEST_SETTINGS = Settings(
    groq_api_key="test",
    groq_model="fake-strong",
    groq_fast_model="fake-fast",
    gemini_api_key="test",
    gemini_model="fake-gemini",
    embedding_model="fake",
    embedding_dim=384,
    demo_mode=True,
    confidence_threshold=0.6,
    session_run_limit=20,
    daily_run_limit=100,
    log_level="INFO",
)


class _FakeRouter:
    def __init__(self, responses, error, by_task):
        self.responses = list(responses)
        self.error = error
        self.by_task = by_task

    def _text(self, task: str) -> str:
        if self.error is not None:
            raise self.error
        if task in self.by_task:
            return self.by_task[task]
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError(f"No fake response for task {task}")

    def complete_json(
        self,
        *,
        task,
        messages,
        schema,
        force_provider=None,
        force_model=None,
    ):
        raw = self._text(task)
        response = LLMResponse(
            text=raw,
            model=force_model or "fake-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=5,
        )
        return schema.model_validate_json(raw), response, 0.0

    def complete_fixed(self, *, provider, model, messages, schema):
        raw = self._text("memory_note")
        response = LLMResponse(
            text=raw,
            model=model,
            input_tokens=5,
            output_tokens=3,
            latency_ms=5,
        )
        return schema.model_validate_json(raw), response, 0.0


def fake_router(*responses, error=None, **by_task):
    return _FakeRouter(responses, error, by_task)


class _FakeMemory:
    def __init__(self, hits):
        self.hits = hits

    def search(self, **kwargs):
        return list(self.hits)

    def add(self, **kwargs):
        return None


def fake_memory(hits):
    return _FakeMemory(hits)


class _FakeStore:
    def __init__(self, customers=None):
        self.customers = list(customers or TEST_CUSTOMERS)
        self.audit_events = []
        self.tool_calls = {}

    def list_customers(self):
        return list(self.customers)

    def get_customer(self, customer_id):
        return next(
            (customer for customer in self.customers if customer["id"] == customer_id),
            None,
        )

    def get_ticket(self, ticket_id):
        return None

    def add_audit_event(self, event):
        self.audit_events.append({**event, "id": len(self.audit_events) + 1})

    def list_audit_events(self, run_id):
        return [
            event for event in self.audit_events if event["run_id"] == run_id
        ]

    def get_tool_call(self, tool_call_id):
        return self.tool_calls.get(tool_call_id)

    def record_tool_call(self, call):
        if call["id"] in self.tool_calls:
            return False
        self.tool_calls[call["id"]] = call
        return True


class _FakeGateway:
    def __init__(self, store):
        self.store = store
        self.calls = []

    def execute(
        self,
        *,
        agent,
        tool,
        args,
        run_id,
        session_id,
        approved,
        customer=None,
    ):
        self.calls.append({
            "agent": agent, "tool": tool, "args": args,
            "run_id": run_id, "session_id": session_id,
            "approved": approved, "customer": customer,
        })
        if tool == "lookup_customer":
            needle = str(args.get("name", "")).strip().lower()
            for candidate in self.store.customers:
                if candidate["name"].lower() == needle:
                    return ToolResult(
                        tool_call_id="fake", tool=tool,
                        status="executed", result=candidate,
                    )
                if any(alias.lower() == needle for alias in candidate["aliases"]):
                    return ToolResult(
                        tool_call_id="fake", tool=tool,
                        status="executed", result=candidate,
                    )
            return ToolResult(
                tool_call_id="fake", tool=tool,
                status="executed", result=None,
            )
        if tool == "get_open_tickets":
            return ToolResult(
                tool_call_id="fake", tool=tool,
                status="executed",
                result=[{
                    "id": "T-1001",
                    "customer_id": args.get("customer_id"),
                    "subject": "Demo ticket",
                    "status": "open",
                }],
            )
        return ToolResult(
            tool_call_id="fake", tool=tool,
            status="executed", result={"ok": True},
        )

    def validate(self, agent, tool, args, customer):
        return None

    def record_proposed(self, *, run_id, session_id, agent, tool_calls):
        for call in tool_calls:
            self.store.add_audit_event({
                "tool_call_id": call.get("tool_call_id", "fake"),
                "run_id": run_id,
                "session_id": session_id,
                "agent": agent,
                "tool": call.get("tool", "unknown"),
                "args_hash": "fake",
                "event_type": "proposed",
                "detail": {},
            })

    def record_decision(
        self,
        *,
        run_id,
        session_id,
        approved,
        approved_by,
        tool_calls,
    ):
        for call in tool_calls:
            self.store.add_audit_event({
                "tool_call_id": call.get("tool_call_id", "fake"),
                "run_id": run_id,
                "session_id": session_id,
                "agent": "approval",
                "tool": call.get("tool", "unknown"),
                "args_hash": "fake",
                "event_type": "approved" if approved else "rejected",
                "detail": {"approved_by": approved_by},
            })

    def make_tool_call_id(self, run_id, tool, args):
        return f"{run_id}:{tool}"


def make_deps(router=None, memory=None, gateway=None, store=None):
    store = store or _FakeStore()
    gateway = gateway or _FakeGateway(store)
    default_memory = [{
        "interaction_id": "int_acme_002",
        "summary": "Refund confirmed",
        "resolution": "Refund issued",
        "similarity": 0.91,
        "occurred_at": "2026-08-05T00:00:00Z",
    }]
    default_router = fake_router(
        extract='{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        memory_note='{"note":"No prior interactions."}',
        propose='{"tool_calls":[],"rationale":"No action","risk_level":"low","requires_approval":false}',
    )
    return MeshDeps(
        router=router or default_router,
        gateway=gateway,
        store=store,
        memory=memory or fake_memory(default_memory),
        embedder=FakeEmbedder(),
        api=None,
        settings=TEST_SETTINGS,
    )
