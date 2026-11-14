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


def test_lookup_customer_matches_alias(tmp_path):
    store = SqliteStore(tmp_path / "test.db")
    store.init_schema()
    store.upsert_customer({
        "id": "acme", "name": "Acme Logistics", "aliases": ["Acme"],
        "consent_scopes": ["crm:read", "crm:write"], "metadata": {},
        "created_at": "2026-08-01T00:00:00Z",
    })
    api = SimulatedCompanyAPI(store)
    assert api.lookup_customer("acme")["id"] == "acme"
    assert api.lookup_customer("ACME LOGISTICS")["id"] == "acme"
    assert api.lookup_customer("Unknown") is None
