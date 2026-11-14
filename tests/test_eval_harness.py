from eval.harness import run_system, summarize


def test_harness_runs_two_transcripts_with_fakes():
    from tests.conftest import fake_router, make_deps

    deps = make_deps(router=fake_router(
        extract='{"customer_name":"Acme Logistics","intent":"billing_dispute","entities":{},"action_items":[],"evidence":[],"confidence":0.9}',
        memory_note='{"note":"No prior interactions."}',
        propose='{"tool_calls":[],"rationale":"No action","risk_level":"low","requires_approval":false}',
    ))
    transcripts = [
        {
            "id": "acme_003", "customer_id": "acme",
            "occurred_at": "2026-09-02T10:00:00Z",
            "transcript": "Charged twice.", "intent": "billing_dispute",
            "action_items": [], "expected_tool_calls": [],
            "prior_interaction_id": "int_acme_002", "memory_probe": True,
            "tags": [], "summary": "x", "resolution": "x",
        },
        {
            "id": "cedar_001", "customer_id": "cedar",
            "occurred_at": "2026-09-10T10:00:00Z",
            "transcript": "Pricing question.", "intent": "upgrade",
            "action_items": [], "expected_tool_calls": [],
            "prior_interaction_id": None, "memory_probe": True,
            "tags": [], "summary": "x", "resolution": None,
        },
    ]
    rows = run_system(deps, transcripts, provider="groq", model="strong")
    assert len(rows) == 2
    assert rows[0]["transcript_id"] == "acme_003"
    assert "extraction" in rows[0]
    summary = summarize(rows, [], [])
    assert "system" in summary
    assert "groups" in summary
