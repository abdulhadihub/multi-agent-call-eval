from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel


class Settings(BaseModel):
    groq_api_key: str
    groq_model: str
    groq_fast_model: str
    gemini_api_key: str | None = None
    gemini_model: str
    embedding_model: str
    embedding_dim: int
    demo_mode: bool
    confidence_threshold: float
    session_run_limit: int
    daily_run_limit: int
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if env is None else env
        return cls(
            groq_api_key=source.get("GROQ_API_KEY"),
            groq_model=source.get("GROQ_MODEL"),
            groq_fast_model=source.get("GROQ_FAST_MODEL"),
            gemini_api_key=source.get("GEMINI_API_KEY") or None,
            gemini_model=source.get("GEMINI_MODEL"),
            embedding_model=source.get("EMBEDDING_MODEL"),
            embedding_dim=source.get("EMBEDDING_DIM"),
            demo_mode=source.get("DEMO_MODE"),
            confidence_threshold=source.get("CONFIDENCE_THRESHOLD"),
            session_run_limit=source.get("SESSION_RUN_LIMIT"),
            daily_run_limit=source.get("DAILY_RUN_LIMIT"),
            log_level=source.get("LOG_LEVEL") or "INFO",
        )

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls.from_env()
