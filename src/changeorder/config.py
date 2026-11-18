"""Settings: processing limits, and the model's address read from the environment.

Nothing about a provider or a model is written into the code. `LLM_API_KEY`, `LLM_BASE_URL` and
`LLM_MODEL` name any OpenAI-compatible chat-completions endpoint; the rest have defaults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

StructuredOutput = Literal["auto", "json_schema", "json_object", "none"]


class ConfigError(Exception):
    """A setting is missing or cannot be read. The message names variables, never their values."""


@dataclass(frozen=True)
class Limits:
    """What one document may cost. A document over a limit is refused, never cut to fit."""

    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 40
    max_chars: int = 150_000  # about 40,000 tokens of text: far above any real change order
    min_chars: int = 40  # fewer visible characters than this is not a readable document


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str
    timeout: float = 120.0  # one wall for the whole request, retries included
    max_output_tokens: int = 4000
    temperature: float | None = None  # sent only when set: some models refuse the parameter
    structured_output: StructuredOutput = "auto"

    def describe(self) -> dict[str, object]:
        """The settings worth recording with a run. The key is never among them."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "structured_output": self.structured_output,
        }


def load_env_file(path: Path = Path(".env")) -> None:
    """Adds NAME=value lines from a local settings file to the environment.

    A variable that is already set wins, so the shell and CI stay in charge. Quotes around a value
    and an `export ` before a name are not part of them.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        name, separator, value = text.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name.isidentifier():
            continue
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def llm_settings(env: Mapping[str, str] | None = None) -> LLMSettings:
    """The model settings from the environment, or ConfigError naming what is missing."""
    env = os.environ if env is None else env
    missing = [
        name
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
        if not env.get(name, "").strip()
    ]
    if missing:
        raise ConfigError(
            f"{', '.join(missing)} must be set (see .env.example); no model call was made"
        )
    mode = env.get("LLM_STRUCTURED_OUTPUT", "auto").strip().lower() or "auto"
    if mode not in get_args(StructuredOutput):
        raise ConfigError(
            "LLM_STRUCTURED_OUTPUT must be one of " + ", ".join(get_args(StructuredOutput))
        )
    temperature = env.get("LLM_TEMPERATURE", "").strip()
    try:
        return LLMSettings(
            api_key=env["LLM_API_KEY"].strip(),
            base_url=env["LLM_BASE_URL"].strip(),
            model=env["LLM_MODEL"].strip(),
            timeout=float(env.get("LLM_TIMEOUT_SECONDS", "").strip() or 120.0),
            max_output_tokens=int(env.get("LLM_MAX_OUTPUT_TOKENS", "").strip() or 4000),
            temperature=float(temperature) if temperature else None,
            structured_output=mode,  # type: ignore[arg-type]
        )
    except ValueError:
        raise ConfigError(
            "LLM_TIMEOUT_SECONDS, LLM_MAX_OUTPUT_TOKENS and LLM_TEMPERATURE must be numbers"
        ) from None
