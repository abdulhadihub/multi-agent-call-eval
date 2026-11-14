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
