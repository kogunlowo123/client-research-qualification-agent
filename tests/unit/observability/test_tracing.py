from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from client_research_agent.observability import tracing
from client_research_agent.observability.logging import log_context


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(tracing, "get_tracer", lambda: provider.get_tracer("test"))
    return memory


class FakeMlflowSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}

    def set_attributes(self, attributes: dict[str, str]) -> None:
        self.attributes.update(attributes)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, FakeMlflowSpan]]:
    spans: list[tuple[str, str, FakeMlflowSpan]] = []
    module = types.ModuleType("mlflow")

    @contextmanager
    def start_span(name: str, span_type: str) -> Iterator[FakeMlflowSpan]:
        span = FakeMlflowSpan()
        spans.append((name, span_type, span))
        yield span

    module.start_span = start_span  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return spans


def test_span_records_attributes_and_context(exporter: InMemorySpanExporter) -> None:
    with (
        log_context(run_id="r1"),
        tracing.span("retrieve", tracing.SpanType.RETRIEVER, k=8, extra={"a": 1}, none=None),
    ):
        pass
    (finished,) = exporter.get_finished_spans()
    attrs: Any = finished.attributes
    assert finished.name == "retrieve"
    assert attrs["cra.span_type"] == "RETRIEVER"
    assert attrs["run_id"] == "r1"
    assert attrs["k"] == 8
    assert attrs["extra"] == "{'a': 1}"
    assert "none" not in attrs
    assert attrs["cra.duration_ms"] >= 0


def test_span_records_errors(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), tracing.span("boom"):
        raise RuntimeError("x")
    (finished,) = exporter.get_finished_spans()
    assert finished.status.status_code is StatusCode.ERROR
    assert finished.events[0].name == "exception"


def test_traced_decorator_names(exporter: InMemorySpanExporter) -> None:
    @tracing.traced()
    def compute(x: int) -> int:
        return x * 2

    @tracing.traced("custom", tracing.SpanType.TOOL)
    def other() -> str:
        return "ok"

    assert compute(2) == 4
    assert other() == "ok"
    names = [s.name for s in exporter.get_finished_spans()]
    assert names[0].endswith("compute")
    assert names[1] == "custom"


def test_mlflow_mirroring(exporter: InMemorySpanExporter, fake_mlflow: list[Any]) -> None:
    assert tracing.enable_mlflow_tracing(True)
    assert tracing.mlflow_tracing_enabled()
    with tracing.span("llm", tracing.SpanType.LLM, model="m", skip=None):
        pass
    name, span_type, span = fake_mlflow[0]
    assert (name, span_type) == ("llm", "LLM")
    assert span.attributes == {"model": "m"}
    assert not tracing.enable_mlflow_tracing(False)
    assert not tracing.mlflow_tracing_enabled()


def test_mlflow_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert not tracing.enable_mlflow_tracing(True)
    assert not tracing.mlflow_tracing_enabled()


def test_default_tracer_is_available() -> None:
    assert tracing.get_tracer() is not None
