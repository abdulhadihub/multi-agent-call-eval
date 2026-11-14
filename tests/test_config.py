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
