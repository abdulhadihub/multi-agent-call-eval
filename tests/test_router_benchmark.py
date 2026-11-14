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
