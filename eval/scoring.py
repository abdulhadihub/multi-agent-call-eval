from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return " ".join(cleaned.split())


def token_overlap(a: str, b: str) -> float:
    left = set(normalize_text(a).split())
    right = set(normalize_text(b).split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def matches(a: str, b: str) -> bool:
    if normalize_text(a) == normalize_text(b):
        return True
    return token_overlap(a, b) >= 0.8


def _item_matches(expected: dict, actual: dict) -> bool:
    if expected.get("type") != actual.get("type"):
        return False
    if not matches(
        str(expected.get("target", "")),
        str(actual.get("target", "")),
    ):
        return False
    if expected.get("amount") is not None and actual.get("amount") != expected.get("amount"):
        return False
    if expected.get("due") is not None and actual.get("due") != expected.get("due"):
        return False
    return True


def _match_items(expected_items: list[dict], actual_items: list[dict]) -> int:
    unmatched = list(actual_items)
    true_positives = 0
    for expected in expected_items:
        for index, actual in enumerate(unmatched):
            if _item_matches(expected, actual):
                true_positives += 1
                unmatched.pop(index)
                break
    return true_positives


def score_extraction(expected: dict, actual: dict) -> dict:
    expected = expected or {}
    actual = actual or {}
    expected_items = expected.get("action_items") or []
    actual_items = actual.get("action_items") or []

    true_positives = _match_items(expected_items, actual_items)
    precision = (
        true_positives / len(actual_items)
        if actual_items
        else (1.0 if not expected_items else 0.0)
    )
    recall = (
        true_positives / len(expected_items)
        if expected_items
        else (1.0 if not actual_items else 0.0)
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "customer_correct": expected.get("customer_id") == actual.get("customer_id"),
        "intent_correct": expected.get("intent") == actual.get("intent"),
        "action_precision": precision,
        "action_recall": recall,
        "action_f1": f1,
        "json_valid": isinstance(actual, dict),
    }


def score_memory(
    expected_prior: str | None,
    hits: list[dict],
    negative: bool = False,
) -> dict:
    hits = hits or []
    ids = [hit.get("interaction_id") for hit in hits]
    top_similarity = hits[0].get("similarity", 0.0) if hits else 0.0

    if negative or expected_prior is None:
        return {
            "recall_at_1": False,
            "recall_at_3": False,
            "correct_empty": bool(not hits or top_similarity < 0.35),
            "top_similarity": top_similarity,
        }

    return {
        "recall_at_1": expected_prior in ids[:1],
        "recall_at_3": expected_prior in ids[:3],
        "correct_empty": False,
        "top_similarity": top_similarity,
    }


def _values_match(expected, actual) -> bool:
    if isinstance(expected, str):
        return matches(expected, str(actual))
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            key in actual and _values_match(value, actual[key])
            for key, value in expected.items()
        )
    return expected == actual


def score_action(
    expected_calls: list[dict],
    actual_calls: list[dict],
) -> dict:
    expected_calls = expected_calls or []
    actual_calls = actual_calls or []
    expected_tools = [call.get("tool") for call in expected_calls]
    actual_tools = [call.get("tool") for call in actual_calls]
    tool_correct = set(expected_tools) == set(actual_tools)

    expected_by_tool: dict[str, list[dict]] = {}
    for call in expected_calls:
        expected_by_tool.setdefault(call.get("tool"), []).append(
            call.get("arguments", {})
        )
    actual_by_tool: dict[str, list[dict]] = {}
    for call in actual_calls:
        actual_by_tool.setdefault(call.get("tool"), []).append(
            call.get("arguments", {})
        )

    args_correct = True
    for tool, expected_args_list in expected_by_tool.items():
        actual_args_list = actual_by_tool.get(tool, [])
        if len(actual_args_list) < len(expected_args_list):
            args_correct = False
            break
        for expected_args, actual_args in zip(expected_args_list, actual_args_list):
            if not _values_match(expected_args, actual_args):
                args_correct = False
                break
        if not args_correct:
            break

    return {
        "tool_correct": tool_correct,
        "args_correct": args_correct,
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
    }


def diagnose(mismatch: dict) -> str:
    if mismatch.get("escalated"):
        return "escalated"
    if mismatch.get("json_valid") is False:
        return "invalid_json"
    if mismatch.get("customer_correct") is False:
        return "wrong_customer"
    if mismatch.get("negative") and mismatch.get("correct_empty") is False:
        return "false_retrieval"
    if mismatch.get("expected_prior") and mismatch.get("recall_at_3") is False:
        return "retrieval_miss"
    if mismatch.get("action_recall", 1.0) < 1.0:
        return "missed_action_item"
    if mismatch.get("action_precision", 1.0) < 1.0:
        return "extra_action_item"
    if mismatch.get("tool_correct") is False:
        return "wrong_tool"
    if mismatch.get("args_correct") is False:
        return "wrong_argument"
    return "other"
