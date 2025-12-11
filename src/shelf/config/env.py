"""Load API keys from env.json and hydrate os.environ."""

from __future__ import annotations

import json
import os
from pathlib import Path

import openai

_loaded = False


def load_env(env_file: Path | str | None = None) -> dict[str, str]:
    """Load API keys from env.json into os.environ.

    Args:
        env_file: Path to env.json. Defaults to ./env.json or LOCBENCH_ENV_FILE.

    Returns:
        Dict of loaded secrets.
    """
    global _loaded

    if env_file is None:
        env_file = Path(os.environ.get("LOCBENCH_ENV_FILE", "env.json"))
    else:
        env_file = Path(env_file)

    if not env_file.exists():
        raise FileNotFoundError(
            f"env.json not found at {env_file}. "
            "Expected OPENAI_API_KEY and other keys."
        )

    with env_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    secrets = {k: str(v) for k, v in raw.items() if v}

    # Hydrate os.environ
    for key, value in secrets.items():
        os.environ[key] = value

    _loaded = True
    return secrets


def get_openai_client() -> openai.OpenAI:
    """Get an OpenAI client, loading env.json if needed."""
    if not _loaded and "OPENAI_API_KEY" not in os.environ:
        load_env()
    return openai.OpenAI()


def get_async_openai_client() -> openai.AsyncOpenAI:
    """Get an async OpenAI client, loading env.json if needed."""
    if not _loaded and "OPENAI_API_KEY" not in os.environ:
        load_env()
    return openai.AsyncOpenAI()
