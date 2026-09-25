from __future__ import annotations

import pytest

from client_research_agent.config.settings import AppSettings, build_settings
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import enable_mlflow_tracing
from tests.support.doubles import ScriptedLLM, StaticFetcher


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("CRA_ENVIRONMENT", "DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    enable_mlflow_tracing(False)
    get_metrics().reset()


@pytest.fixture
def settings() -> AppSettings:
    return build_settings("local")


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def static_fetcher() -> StaticFetcher:
    return StaticFetcher()
