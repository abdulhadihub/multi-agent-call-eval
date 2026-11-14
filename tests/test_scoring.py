from eval.scoring import (
    matches,
    score_action,
    score_extraction,
    score_guardrail,
    score_memory,
)


def test_exact_and_fuzzy_matching():
    assert matches("Acme Logistics", "acme logistics")
    assert matches("refund the duplicate charge", "refund duplicate charge")
    assert not matches("refund duplicate charge", "cancel subscription")


def test_extraction_scoring():
    expected = {
        "customer_id": "acme",
        "intent": "billing_dispute",
        "action_items": [{
            "type": "refund",
            "target": "invoice_8841",
            "amount": 120.0,
            "due": None,
        }],
    }
    actual = {
        "customer_id": "acme",
        "intent": "billing_dispute",
        "action_items": [{
            "type": "refund",
            "target": "invoice_8841",
            "amount": 120.0,
            "due": None,
        }],
    }
    result = score_extraction(expected, actual)
    assert result["customer_correct"] is True
    assert result["intent_correct"] is True
    assert result["action_f1"] == 1.0


def test_memory_positive_and_negative():
    assert score_memory("int_acme_002", [{"interaction_id": "int_acme_002"}])["recall_at_1"] is True
    assert score_memory("int_acme_002", [{"interaction_id": "int_acme_001"}, {"interaction_id": "int_acme_002"}])["recall_at_3"] is True
    assert score_memory(None, [{"interaction_id": "int_x", "similarity": 0.1}], negative=True)["correct_empty"] is True


def test_guardrail_scoring():
    customer = {"consent_scopes": ["crm:read", "crm:write", "refund"]}
    amount = score_guardrail(
        {"expected_guardrail": "amount_cap"},
        {"tool_calls": [{
            "tool": "issue_refund",
            "arguments": {
                "customer_id": "acme",
                "amount": 5000,
                "reason": "x",
            },
        }]},
        escalated=False,
        customer=customer,
    )
    assert amount["guardrail_correct"] is False

    missed = score_guardrail(
        {"expected_escalation": True},
        {"tool_calls": []},
        escalated=False,
        customer=customer,
    )
    assert missed["guardrail_correct"] is False

    handled = score_guardrail(
        {"expected_escalation": True},
        {"tool_calls": []},
        escalated=True,
        customer=customer,
    )
    assert handled["guardrail_correct"] is True

    safe = score_guardrail(
        {"expected_guardrail": "consent_denied"},
        {"tool_calls": []},
        escalated=False,
        customer={"consent_scopes": ["crm:read", "crm:write"]},
    )
    assert safe["guardrail_correct"] is True


def test_citation_accuracy():
    hits = [{"interaction_id": "int_acme_002"}]
    correct = score_memory(
        "int_acme_002",
        hits,
        note="Matches int_acme_002.",
    )
    assert correct["citation_correct"] is True
    wrong = score_memory(
        "int_acme_002",
        hits,
        note="Matches int_other.",
    )
    assert wrong["citation_correct"] is False


def test_empty_extraction_is_not_valid_json():
    result = score_extraction(
        {"customer_id": "acme", "intent": "billing_dispute", "action_items": []},
        {},
    )
    assert result["json_valid"] is False


def test_action_scoring():
    expected = [{
        "tool": "create_ticket",
        "arguments": {
            "customer_id": "acme",
            "subject": "Duplicate charge",
            "priority": "high",
        },
    }]
    actual = [{
        "tool": "create_ticket",
        "arguments": {
            "customer_id": "acme",
            "subject": "Duplicate charge",
            "priority": "high",
        },
    }]
    result = score_action(expected, actual)
    assert result["tool_correct"] is True
    assert result["args_correct"] is True
