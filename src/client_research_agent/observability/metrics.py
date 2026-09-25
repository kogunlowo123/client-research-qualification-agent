"""Metrics facade.

Emits to OpenTelemetry instruments (exported via OTLP when configured) and
keeps an in-process snapshot so a run can attach its own counters to the
MLflow run and the audit record without querying a metrics backend.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics as otel_metrics

_METER_NAME = "client_research_agent"


def _key(name: str, attributes: dict[str, Any]) -> str:
    if not attributes:
        return name
    labels = ",".join(f"{k}={attributes[k]}" for k in sorted(attributes))
    return f"{name}{{{labels}}}"


@dataclass
class HistogramSummary:
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if len(self.values) < 10_000:
            self.values.append(value)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def percentile(self, pct: float) -> float:
        if not self.values:
            return 0.0
        ordered = sorted(self.values)
        index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
        return ordered[index]


class Metrics:
    def __init__(self) -> None:
        self._meter = otel_metrics.get_meter(_METER_NAME)
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, HistogramSummary] = defaultdict(HistogramSummary)
        self._otel_counters: dict[str, Any] = {}
        self._otel_histograms: dict[str, Any] = {}

    def increment(self, name: str, value: float = 1.0, **attributes: Any) -> None:
        with self._lock:
            self._counters[_key(name, attributes)] += value
            instrument = self._otel_counters.get(name)
            if instrument is None:
                instrument = self._meter.create_counter(name)
                self._otel_counters[name] = instrument
        instrument.add(value, attributes=attributes)

    def observe(self, name: str, value: float, **attributes: Any) -> None:
        with self._lock:
            self._histograms[_key(name, attributes)].add(value)
            instrument = self._otel_histograms.get(name)
            if instrument is None:
                instrument = self._meter.create_histogram(name)
                self._otel_histograms[name] = instrument
        instrument.record(value, attributes=attributes)

    def counter(self, name: str, **attributes: Any) -> float:
        with self._lock:
            return self._counters.get(_key(name, attributes), 0.0)

    def histogram(self, name: str, **attributes: Any) -> HistogramSummary:
        with self._lock:
            return self._histograms.get(_key(name, attributes), HistogramSummary())

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            data = dict(self._counters)
            for key, hist in self._histograms.items():
                data[f"{key}.count"] = hist.count
                data[f"{key}.mean"] = hist.mean
                data[f"{key}.p95"] = hist.percentile(95)
            return data

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


_metrics = Metrics()


def get_metrics() -> Metrics:
    return _metrics
