"""One-call, idempotent observability bootstrap.

``configure_observability`` wires structured logging, an OpenTelemetry
``TracerProvider`` and ``MeterProvider`` (with OTLP/HTTP exporters when an
endpoint is configured and ``opentelemetry-exporter-otlp-proto-http`` is
installed, or console exporters when explicitly requested) and toggles MLflow
span mirroring. Calling it again with the same settings returns the existing
state; different settings reconfigure after flushing the previous providers.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

from client_research_agent import __version__
from client_research_agent.config.settings import ObservabilitySettings
from client_research_agent.observability.logging import configure_logging, get_logger
from client_research_agent.observability.tracing import enable_mlflow_tracing


@dataclass(frozen=True, slots=True)
class ObservabilityState:
    tracer_provider: TracerProvider
    meter_provider: MeterProvider
    exporter: str
    mlflow_tracing: bool
    installed_globally: bool
    fingerprint: tuple[Any, ...]


_lock = threading.Lock()
_state: ObservabilityState | None = None


def _otlp_exporters(endpoint: str) -> tuple[SpanExporter, Any] | None:
    try:
        trace_mod = importlib.import_module("opentelemetry.exporter.otlp.proto.http.trace_exporter")
        metric_mod = importlib.import_module("opentelemetry.exporter.otlp.proto.http.metric_exporter")
    except ImportError:
        return None
    base = endpoint.rstrip("/")
    span_exporter: SpanExporter = trace_mod.OTLPSpanExporter(endpoint=f"{base}/v1/traces")
    metric_exporter = metric_mod.OTLPMetricExporter(endpoint=f"{base}/v1/metrics")
    return span_exporter, metric_exporter


def _install_globals(tracer_provider: TracerProvider, meter_provider: MeterProvider) -> bool:
    """Install providers globally only if nobody has done so yet (OTel forbids overriding)."""
    installed = False
    if isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        trace.set_tracer_provider(tracer_provider)
        installed = True
    if type(otel_metrics.get_meter_provider()).__name__ == "_ProxyMeterProvider":
        otel_metrics.set_meter_provider(meter_provider)
        installed = True
    return installed


def configure_observability(
    settings: ObservabilitySettings,
    *,
    environment: str = "local",
    console: bool = False,
    set_global: bool = True,
    configure_logs: bool = True,
    metric_export_interval_ms: int = 60_000,
) -> ObservabilityState:
    """Configure logging, tracing, metrics and MLflow tracing; safe to call repeatedly."""
    global _state  # noqa: PLW0603 - process-wide singleton guarded by _lock
    fingerprint = (settings.model_dump_json(), environment, console, set_global, configure_logs)
    with _lock:
        if _state is not None and _state.fingerprint == fingerprint:
            return _state
        if _state is not None:
            _shutdown(_state)

        if configure_logs:
            configure_logging(
                settings.log_level, json_logs=settings.json_logs, service_name=settings.service_name
            )

        resource = Resource.create(
            {
                "service.name": settings.service_name,
                "service.version": __version__,
                "deployment.environment": environment,
            }
        )
        tracer_provider = TracerProvider(resource=resource)
        readers: list[MetricReader] = []
        exporter = "none"
        otlp = _otlp_exporters(settings.otlp_endpoint) if settings.otlp_endpoint else None
        if otlp is not None:
            span_exporter, metric_exporter = otlp
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
            readers.append(
                PeriodicExportingMetricReader(
                    metric_exporter, export_interval_millis=metric_export_interval_ms
                )
            )
            exporter = "otlp"
        elif console:
            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            readers.append(
                PeriodicExportingMetricReader(
                    ConsoleMetricExporter(), export_interval_millis=metric_export_interval_ms
                )
            )
            exporter = "console"
        meter_provider = MeterProvider(resource=resource, metric_readers=readers)

        installed = _install_globals(tracer_provider, meter_provider) if set_global else False
        mlflow_active = enable_mlflow_tracing(settings.mlflow_tracing)

        _state = ObservabilityState(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            exporter=exporter,
            mlflow_tracing=mlflow_active,
            installed_globally=installed,
            fingerprint=fingerprint,
        )
        if settings.otlp_endpoint and otlp is None:
            get_logger(__name__).warning(
                "otlp_exporter_unavailable", hint="install the 'otlp' extra to export telemetry"
            )
        return _state


def _shutdown(state: ObservabilityState) -> None:
    state.tracer_provider.force_flush()
    state.tracer_provider.shutdown()
    state.meter_provider.shutdown()


def current_state() -> ObservabilityState | None:
    return _state


def shutdown_observability() -> None:
    """Flush and release providers (end of a Databricks job task); next configure starts fresh."""
    global _state  # noqa: PLW0603
    with _lock:
        if _state is not None:
            _shutdown(_state)
            _state = None
        enable_mlflow_tracing(False)
