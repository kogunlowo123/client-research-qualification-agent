from __future__ import annotations

import io
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.metrics._internal import _ProxyMeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from client_research_agent.config.settings import ObservabilitySettings
from client_research_agent.observability import setup as obs_setup
from client_research_agent.observability.setup import (
    configure_observability,
    current_state,
    shutdown_observability,
)
from client_research_agent.observability.tracing import mlflow_tracing_enabled

_TRACE_MOD = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
_METRIC_MOD = "opentelemetry.exporter.otlp.proto.http.metric_exporter"


class RecordingSpanExporter(InMemorySpanExporter):
    def __init__(self, endpoint: str) -> None:
        super().__init__()
        self.endpoint = endpoint


class RecordingMetricExporter(ConsoleMetricExporter):
    def __init__(self, endpoint: str) -> None:
        super().__init__(out=io.StringIO())
        self.endpoint = endpoint


@pytest.fixture(autouse=True)
def _reset_state() -> Iterator[None]:
    shutdown_observability()
    yield
    shutdown_observability()


@pytest.fixture
def fake_otlp(monkeypatch: pytest.MonkeyPatch) -> None:
    trace_mod = types.ModuleType(_TRACE_MOD)
    trace_mod.OTLPSpanExporter = RecordingSpanExporter  # type: ignore[attr-defined]
    metric_mod = types.ModuleType(_METRIC_MOD)
    metric_mod.OTLPMetricExporter = RecordingMetricExporter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _TRACE_MOD, trace_mod)
    monkeypatch.setitem(sys.modules, _METRIC_MOD, metric_mod)


def _settings(**overrides: Any) -> ObservabilitySettings:
    return ObservabilitySettings(mlflow_tracing=False, **overrides)


def test_local_defaults_no_exporter_and_idempotent() -> None:
    settings = _settings()
    first = configure_observability(settings, set_global=False, configure_logs=False)
    second = configure_observability(settings, set_global=False, configure_logs=False)
    assert first is second
    assert first.exporter == "none"
    assert not first.installed_globally
    assert not first.mlflow_tracing
    assert current_state() is first
    resource = first.tracer_provider.resource.attributes
    assert resource["service.name"] == "client-research-agent"
    assert resource["deployment.environment"] == "local"


def test_reconfigure_with_new_settings_replaces_state() -> None:
    first = configure_observability(_settings(), set_global=False, configure_logs=False)
    second = configure_observability(_settings(service_name="other"), set_global=False, configure_logs=False)
    assert first is not second
    assert second.tracer_provider.resource.attributes["service.name"] == "other"


def test_console_exporter() -> None:
    state = configure_observability(_settings(), console=True, set_global=False, configure_logs=False)
    assert state.exporter == "console"


def test_otlp_exporter_when_available(fake_otlp: None) -> None:
    state = configure_observability(
        _settings(otlp_endpoint="http://collector:4318/"),
        environment="prod",
        set_global=False,
        configure_logs=False,
    )
    assert state.exporter == "otlp"
    processor = state.tracer_provider._active_span_processor._span_processors[0]
    assert processor.span_exporter.endpoint == "http://collector:4318/v1/traces"  # type: ignore[attr-defined]


def test_otlp_missing_package_falls_back(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, _TRACE_MOD, None)
    state = configure_observability(_settings(otlp_endpoint="http://collector:4318"), set_global=False)
    assert state.exporter == "none"
    assert "otlp_exporter_unavailable" in capsys.readouterr().out


def test_global_install_only_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    installed: dict[str, Any] = {}
    monkeypatch.setattr(trace, "get_tracer_provider", trace.ProxyTracerProvider)
    monkeypatch.setattr(trace, "set_tracer_provider", lambda p: installed.setdefault("trace", p))
    monkeypatch.setattr(otel_metrics, "get_meter_provider", _ProxyMeterProvider)
    monkeypatch.setattr(otel_metrics, "set_meter_provider", lambda p: installed.setdefault("metrics", p))
    state = configure_observability(_settings(), configure_logs=False)
    assert state.installed_globally
    assert installed["trace"] is state.tracer_provider
    assert installed["metrics"] is state.meter_provider


def test_global_already_set_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace, "get_tracer_provider", object)
    monkeypatch.setattr(otel_metrics, "get_meter_provider", object)
    assert not obs_setup._install_globals(
        configure_observability(_settings(), set_global=False, configure_logs=False).tracer_provider,
        current_state().meter_provider,  # type: ignore[union-attr]
    )


def test_mlflow_tracing_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    state = configure_observability(
        ObservabilitySettings(mlflow_tracing=True), set_global=False, configure_logs=False
    )
    assert state.mlflow_tracing
    assert mlflow_tracing_enabled()
    shutdown_observability()
    assert not mlflow_tracing_enabled()
    assert current_state() is None
