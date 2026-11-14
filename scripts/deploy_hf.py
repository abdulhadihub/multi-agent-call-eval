from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ID = "abdulhadihub/multi-agent-call-eval"
SECRET_KEYS = (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "DEMO_MODE",
    "CONFIDENCE_THRESHOLD",
    "SESSION_RUN_LIMIT",
    "DAILY_RUN_LIMIT",
)
IGNORE_PATTERNS = [
    ".env",
    ".git*",
    ".venv*",
    "__pycache__*",
    "*.pyc",
    ".pytest_cache*",
    "data/*.db",
    "data/chroma*",
    "tests*",
    ".superpowers*",
    "*.log",
]


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type="space",
        folder_path=".",
        ignore_patterns=IGNORE_PATTERNS,
    )

    added = []
    for key in SECRET_KEYS:
        value = os.getenv(key)
        if value:
            api.add_space_secret(
                repo_id=REPO_ID,
                key=key,
                value=value,
            )
            added.append(key)

    print(f"Space: https://huggingface.co/spaces/{REPO_ID}")
    print(f"Secrets set: {', '.join(added)}")


if __name__ == "__main__":
    main()
