from __future__ import annotations

import json
from collections.abc import Iterator

import pytest

from client_research_agent.observability import logging as cra_logging
from client_research_agent.observability.logging import (
    bind_context,
    configure_logging,
    current_context,
    get_logger,
    log_context,
    scrub,
)


@pytest.fixture(autouse=True)
def _reset_context() -> Iterator[None]:
    token = cra_logging._context.set(None)
    yield
    cra_logging._context.reset(token)


def test_scrub_patterns_and_structures() -> None:
    assert scrub("key " + "dapi" + "0123456789abcdef" * 2 + "-2 used") == "key [REDACTED] used"
    assert scrub("ghp_" + "a" * 30) == "[REDACTED]"
    assert scrub("sk-" + "b" * 24) == "[REDACTED]"
    assert scrub("Authorization: Bearer abcdefghijklmnop1234") == "Authorization: [REDACTED]"
    assert scrub("AKIA" + "A" * 16) == "[REDACTED]"
    assert scrub({"Token": "x", "nested": ["sk-" + "c" * 24, ("ok",)]}) == {
        "Token": "[REDACTED]",
        "nested": ["[REDACTED]", ("ok",)],
    }
    assert scrub(42) == 42


def test_context_helpers() -> None:
    bind_context(run_id="r1")
    with log_context(step="retrieve"):
        assert current_context() == {"run_id": "r1", "step": "retrieve"}
    assert current_context() == {"run_id": "r1"}


def test_configure_json_logging_redacts_and_includes_context(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("debug", json_logs=True, service_name="cra-test")
    with log_context(run_id="run-9"):
        get_logger("t").info("llm_call", password="hunter2", note="Bearer abcdefghijklmnopqrstu")
    record = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert record["event"] == "llm_call"
    assert record["password"] == "[REDACTED]"
    assert record["note"] == "[REDACTED]"
    assert record["run_id"] == "run-9"
    assert record["service"] == "cra-test"
    assert record["level"] == "info"


def test_console_logging_and_level_filtering(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("WARNING", json_logs=False)
    log = get_logger("t")
    log.info("hidden_event")
    log.warning("shown_event")
    out = capsys.readouterr().out
    assert "shown_event" in out
    assert "hidden_event" not in out


def test_invalid_level() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging("LOUD")
