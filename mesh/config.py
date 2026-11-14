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
            groq_model=source.get("GROQ_MODEL") or "openai/gpt-oss-120b",
            groq_fast_model=(
                source.get("GROQ_FAST_MODEL") or "openai/gpt-oss-20b"
            ),
            gemini_api_key=source.get("GEMINI_API_KEY") or None,
            gemini_model=source.get("GEMINI_MODEL") or "gemini-2.5-flash",
            embedding_model=(
                source.get("EMBEDDING_MODEL")
                or "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=source.get("EMBEDDING_DIM") or "384",
            demo_mode=source.get("DEMO_MODE") or "true",
            confidence_threshold=(
                source.get("CONFIDENCE_THRESHOLD") or "0.6"
            ),
            session_run_limit=source.get("SESSION_RUN_LIMIT") or "20",
            daily_run_limit=source.get("DAILY_RUN_LIMIT") or "100",
            log_level=source.get("LOG_LEVEL") or "INFO",
        )

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        env = dict(os.environ)
        try:
            import streamlit as st

            for key in (
                "GROQ_API_KEY",
                "GROQ_MODEL",
                "GROQ_FAST_MODEL",
                "GEMINI_API_KEY",
                "GEMINI_MODEL",
                "EMBEDDING_MODEL",
                "EMBEDDING_DIM",
                "DEMO_MODE",
                "CONFIDENCE_THRESHOLD",
                "SESSION_RUN_LIMIT",
                "DAILY_RUN_LIMIT",
                "LOG_LEVEL",
            ):
                if key in st.secrets:
                    env[key] = str(st.secrets[key])
        except Exception:
            pass
        return cls.from_env(env)
