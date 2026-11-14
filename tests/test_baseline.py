from eval.baseline import run_baseline
from tests.conftest import fake_router


def test_baseline_uses_pinned_model_and_no_tools():
    router = fake_router(
        '{"extraction":{"customer_id":"acme","customer_name":"Acme Logistics",'
        '"intent":"billing_dispute","action_items":[],"confidence":0.8},'
        '"proposal":{"tool_calls":[],"rationale":"none","risk_level":"low",'
        '"requires_approval":false}}'
    )
    result = run_baseline(
        transcript="Charged twice.",
        customer_directory=[{"id": "acme", "name": "Acme Logistics"}],
        tool_catalog=[],
        router=router,
        provider="groq",
        model="strong",
    )
    assert result["extraction"]["customer_id"] == "acme"
    assert result["proposal"]["tool_calls"] == []
    assert result["meta"]["model"] == "strong"
