"""Unified tracing: OpenTelemetry spans plus MLflow traces when available.

``traced`` wraps a function in an OTel span and, when MLflow tracing is
enabled (Databricks Model Serving / notebooks), in an MLflow span of the given
``span_type`` so the agent's reasoning tree renders in the MLflow Trace UI.
MLflow is an optional dependency; its absence degrades to OTel only.
"""

from __future__ import annotations

import functools
import importlib
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from client_research_agent.observability.logging import current_context

P = ParamSpec("P")
T = TypeVar("T")

_TRACER_NAME = "client_research_agent"
_mlflow_enabled = False


class SpanType:
    AGENT = "AGENT"
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    TOOL = "TOOL"
    PARSER = "PARSER"
    RERANKER = "RERANKER"
    UNKNOWN = "UNKNOWN"


def enable_mlflow_tracing(enabled: bool) -> bool:
    """Turn MLflow span mirroring on; returns whether MLflow is actually importable."""
    global _mlflow_enabled  # noqa: PLW0603 - process-wide switch set once at startup
    if not enabled:
        _mlflow_enabled = False
        return False
    try:
        importlib.import_module("mlflow")
    except ImportError:
        _mlflow_enabled = False
        return False
    _mlflow_enabled = True
    return True


def mlflow_tracing_enabled() -> bool:
    return _mlflow_enabled


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME)


def _set_attributes(span: Span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if isinstance(value, str | bool | int | float):
            span.set_attribute(key, value)
        elif value is not None:
            span.set_attribute(key, str(value)[:1024])


@contextmanager
def span(name: str, span_type: str = SpanType.UNKNOWN, **attributes: Any) -> Iterator[Span]:
    tracer = get_tracer()
    started = time.perf_counter()
    with tracer.start_as_current_span(name) as otel_span:
        _set_attributes(otel_span, {"cra.span_type": span_type, **current_context(), **attributes})
        mlflow_cm = _mlflow_span(name, span_type, attributes)
        with mlflow_cm:
            try:
                yield otel_span
            except Exception as exc:
                otel_span.record_exception(exc)
                otel_span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                raise
            finally:
                otel_span.set_attribute("cra.duration_ms", (time.perf_counter() - started) * 1000)


@contextmanager
def _mlflow_span(name: str, span_type: str, attributes: dict[str, Any]) -> Iterator[None]:
    if not _mlflow_enabled:
        yield
        return
    mlflow = importlib.import_module("mlflow")
    with mlflow.start_span(name=name, span_type=span_type) as ml_span:
        ml_span.set_attributes({k: str(v)[:1024] for k, v in attributes.items() if v is not None})
        yield


def traced(
    name: str | None = None, span_type: str = SpanType.UNKNOWN
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        span_name = name or f"{func.__module__.rsplit('.', 1)[-1]}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with span(span_name, span_type):
                return func(*args, **kwargs)

        return wrapper

    return decorator
