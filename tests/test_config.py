"""Settings: read from the environment, never guessed, and never echoed."""

from __future__ import annotations

import pytest

from changeorder import config
from changeorder.config import ConfigError, LLMSettings, llm_settings, load_env_file

REQUIRED = {
    "LLM_API_KEY": "sk-test-0123456789",
    "LLM_BASE_URL": "https://llm.test/v1",
    "LLM_MODEL": "test-model",
}


def test_the_three_required_settings_are_enough_and_the_rest_have_defaults():
    settings = llm_settings(REQUIRED)

    assert settings == LLMSettings(
        api_key="sk-test-0123456789", base_url="https://llm.test/v1", model="test-model"
    )
    assert (settings.timeout, settings.max_output_tokens) == (120.0, 4000)
    assert settings.temperature is None  # not sent unless asked for
    assert settings.structured_output == "auto"


def test_optional_settings_are_read_and_surrounding_space_is_ignored():
    settings = llm_settings(
        {
            **REQUIRED,
            "LLM_MODEL": "  test-model  ",
            "LLM_TIMEOUT_SECONDS": "45",
            "LLM_MAX_OUTPUT_TOKENS": "900",
            "LLM_TEMPERATURE": "0",
            "LLM_STRUCTURED_OUTPUT": "JSON_OBJECT",
        }
    )

    assert settings.model == "test-model"
    assert (settings.timeout, settings.max_output_tokens, settings.temperature) == (45.0, 900, 0.0)
    assert settings.structured_output == "json_object"


@pytest.mark.parametrize(
    "missing", [["LLM_API_KEY"], ["LLM_BASE_URL", "LLM_MODEL"], list(REQUIRED)]
)
def test_a_missing_setting_is_named_and_no_value_is_echoed(missing):
    env = {name: value for name, value in REQUIRED.items() if name not in missing}
    env.update(dict.fromkeys(missing[:1], "   "))  # blank counts as missing

    with pytest.raises(ConfigError) as raised:
        llm_settings(env)

    message = str(raised.value)
    assert all(name in message for name in missing)
    assert "sk-test" not in message and "llm.test" not in message


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_STRUCTURED_OUTPUT", "xml"),
        ("LLM_TIMEOUT_SECONDS", "soon"),
        ("LLM_MAX_OUTPUT_TOKENS", "4k"),
        ("LLM_TEMPERATURE", "warm"),
    ],
)
def test_a_setting_that_cannot_be_read_is_refused_not_defaulted(name, value):
    with pytest.raises(ConfigError) as raised:
        llm_settings({**REQUIRED, name: value})

    assert value not in str(raised.value)


def test_what_is_recorded_with_a_run_never_includes_the_key():
    recorded = llm_settings(REQUIRED).describe()

    assert recorded["model"] == "test-model" and recorded["base_url"] == "https://llm.test/v1"
    assert "sk-test-0123456789" not in str(recorded) and "api_key" not in recorded


def test_a_local_settings_file_fills_what_the_environment_does_not_set(tmp_path, monkeypatch):
    environment = {"LLM_MODEL": "from-the-shell"}
    monkeypatch.setattr(config.os, "environ", environment)
    path = tmp_path / ".env"
    path.write_text(
        "﻿# local settings\n"
        "\n"
        "LLM_API_KEY = sk-from-file\n"
        'export LLM_BASE_URL="https://llm.test/v1"\n'
        "LLM_MODEL=from-the-file\n"
        "LLM_TEMPERATURE='0'\n"
        "not a setting line\n"
        "1BAD=value\n",
        encoding="utf-8",
    )

    load_env_file(path)

    assert environment == {
        "LLM_MODEL": "from-the-shell",  # what is already set wins
        "LLM_API_KEY": "sk-from-file",
        "LLM_BASE_URL": "https://llm.test/v1",  # quotes and `export` are not part of it
        "LLM_TEMPERATURE": "0",
    }
    load_env_file(tmp_path / "no-such-file")  # a missing file is not an error
