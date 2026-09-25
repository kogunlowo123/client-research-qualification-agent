from __future__ import annotations

import threading

import pytest

from client_research_agent.observability.metrics import HistogramSummary, Metrics, get_metrics


def test_counters_with_attributes() -> None:
    metrics = Metrics()
    metrics.increment("docs")
    metrics.increment("docs", 2, source="sec")
    metrics.increment("docs", 3, source="sec")
    assert metrics.counter("docs") == 1
    assert metrics.counter("docs", source="sec") == 5
    assert metrics.counter("missing") == 0.0


def test_histograms_and_snapshot() -> None:
    metrics = Metrics()
    for value in (10.0, 20.0, 30.0):
        metrics.observe("latency_ms", value, step="rerank")
    summary = metrics.histogram("latency_ms", step="rerank")
    assert summary.count == 3
    assert summary.mean == 20.0
    assert (summary.minimum, summary.maximum) == (10.0, 30.0)
    assert summary.percentile(95) == 30.0
    assert summary.percentile(0) == 10.0
    snapshot = metrics.snapshot()
    assert snapshot["latency_ms{step=rerank}.count"] == 3
    assert snapshot["latency_ms{step=rerank}.p95"] == 30.0
    assert metrics.histogram("none").count == 0
    metrics.reset()
    assert metrics.snapshot() == {}


def test_histogram_summary_edges() -> None:
    empty = HistogramSummary()
    assert empty.mean == 0.0
    assert empty.percentile(50) == 0.0
    capped = HistogramSummary()
    for i in range(10_005):
        capped.add(float(i))
    assert capped.count == 10_005
    assert len(capped.values) == 10_000


def test_thread_safety_and_singleton() -> None:
    metrics = Metrics()

    def work() -> None:
        for _ in range(500):
            metrics.increment("n")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert metrics.counter("n") == pytest.approx(2000)
    assert get_metrics() is get_metrics()
