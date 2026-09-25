"""Evaluation harness: produce a brief per example, score it, aggregate, apply the quality gate.

A *brief producer* turns an :class:`EvalExample` into a :class:`ClientBrief`:

* :func:`offline_producer` runs the full orchestrator against a fresh local
  runtime whose ingestion replays the example's snapshot (no network);
* :func:`model_producer` calls a logged/served ResponsesAgent (for example the
  UC ``champion``) and parses ``custom_outputs["brief_json"]``;
* any callable with the same signature (for example a live orchestrator).

:class:`QualityGate` decides per example (pass/fail with reasons) and for the
run (pass rate and mean-metric thresholds).
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from client_research_agent.config.settings import AppSettings
from client_research_agent.evaluation.dataset import EvalExample, EvalExpectations
from client_research_agent.evaluation.metrics import (
    DEFAULT_SUPPORT_THRESHOLD,
    BriefMetrics,
    compute_metrics,
    summarize,
)
from client_research_agent.evaluation.snapshot import SnapshotIngestion
from client_research_agent.models import ClientBrief
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.security.rbac import Principal, Role
from client_research_agent.services.ports import LLMClient

BriefProducer = Callable[[EvalExample], ClientBrief]
EVALUATION_PRINCIPAL = Principal(id="evaluation-harness", roles=frozenset({Role.SERVICE}))
_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QualityGate:
    min_pass_rate: float = 0.85
    min_citation_coverage: float = 0.9
    min_grounded_fact_ratio: float = 0.8
    min_verdict_accuracy: float | None = None
    max_criterion_mae: float | None = None

    def __post_init__(self) -> None:
        for name in ("min_pass_rate", "min_citation_coverage", "min_grounded_fact_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

    def example_failures(self, metrics: BriefMetrics, expectations: EvalExpectations) -> list[str]:
        failures: list[str] = []
        coverage_floor = (
            expectations.min_citation_coverage
            if expectations.min_citation_coverage is not None
            else self.min_citation_coverage
        )
        if metrics.citation_coverage < coverage_floor:
            failures.append(f"citation_coverage {metrics.citation_coverage:.2f} < {coverage_floor:.2f}")
        if metrics.grounded_fact_ratio < self.min_grounded_fact_ratio:
            failures.append(
                f"grounded_fact_ratio {metrics.grounded_fact_ratio:.2f} < {self.min_grounded_fact_ratio:.2f}"
            )
        if metrics.discovery_questions_ok < 1.0:
            failures.append("brief does not have exactly five discovery questions")
        if metrics.verdict_accuracy is not None and metrics.verdict_accuracy < 1.0:
            failures.append("verdict does not match the label")
        return failures

    def run_failures(self, pass_rate: float, aggregates: Mapping[str, float]) -> list[str]:
        failures: list[str] = []
        if pass_rate < self.min_pass_rate:
            failures.append(f"pass_rate {pass_rate:.2f} < {self.min_pass_rate:.2f}")
        coverage = aggregates.get("citation_coverage")
        if coverage is not None and coverage < self.min_citation_coverage:
            failures.append(f"mean citation_coverage {coverage:.2f} < {self.min_citation_coverage:.2f}")
        accuracy = aggregates.get("verdict_accuracy")
        if (
            self.min_verdict_accuracy is not None
            and accuracy is not None
            and accuracy < self.min_verdict_accuracy
        ):
            failures.append(f"verdict_accuracy {accuracy:.2f} < {self.min_verdict_accuracy:.2f}")
        mae = aggregates.get("criterion_score_mae")
        if self.max_criterion_mae is not None and mae is not None and mae > self.max_criterion_mae:
            failures.append(f"criterion_score_mae {mae:.2f} > {self.max_criterion_mae:.2f}")
        return failures

    def thresholds(self) -> dict[str, float]:
        values: dict[str, float] = {
            "pass_rate": self.min_pass_rate,
            "citation_coverage": self.min_citation_coverage,
            "grounded_fact_ratio": self.min_grounded_fact_ratio,
        }
        if self.min_verdict_accuracy is not None:
            values["verdict_accuracy"] = self.min_verdict_accuracy
        if self.max_criterion_mae is not None:
            values["criterion_score_mae"] = self.max_criterion_mae
        return values


@dataclass(frozen=True)
class ExampleResult:
    example: EvalExample
    passed: bool
    failures: tuple[str, ...]
    metrics: BriefMetrics | None = None
    brief: ClientBrief | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.example.eval_id,
            "company": self.example.inputs.company_name,
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics.as_dict() if self.metrics else {},
            "verdict": self.brief.qualification.verdict.value if self.brief else None,
            "run_id": self.brief.run_id if self.brief else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[ExampleResult, ...]
    aggregates: dict[str, float]
    pass_rate: float
    gate_passed: bool
    gate_failures: tuple[str, ...]
    thresholds: dict[str, float] = field(default_factory=dict)

    def metric_rows(self) -> list[dict[str, Any]]:
        """One row per aggregate metric (``metric_name``, ``metric_value``, ``threshold``, ``passed``)."""
        values = {"pass_rate": self.pass_rate, **self.aggregates}
        rows: list[dict[str, Any]] = []
        for name, value in values.items():
            threshold = self.thresholds.get(name)
            if threshold is None:
                passed = True
            elif name == "criterion_score_mae":
                passed = value <= threshold
            else:
                passed = value >= threshold
            rows.append(
                {"metric_name": name, "metric_value": value, "threshold": threshold, "passed": passed}
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_passed": self.gate_passed,
            "gate_failures": list(self.gate_failures),
            "pass_rate": self.pass_rate,
            "aggregates": dict(self.aggregates),
            "thresholds": dict(self.thresholds),
            "results": [r.to_dict() for r in self.results],
        }


class EvaluationHarness:
    def __init__(
        self,
        producer: BriefProducer,
        gate: QualityGate | None = None,
        *,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    ) -> None:
        self._producer = producer
        self._gate = gate or QualityGate()
        self._threshold = support_threshold

    @property
    def gate(self) -> QualityGate:
        return self._gate

    def evaluate_brief(self, brief: ClientBrief, example: EvalExample) -> ExampleResult:
        metrics = compute_metrics(brief, example.expectations, support_threshold=self._threshold)
        failures = self._gate.example_failures(metrics, example.expectations)
        return ExampleResult(example, not failures, tuple(failures), metrics, brief)

    def run_example(self, example: EvalExample) -> ExampleResult:
        try:
            brief = self._producer(example)
        except Exception as exc:  # one broken example must not abort the evaluation run
            _log.warning("evaluation.example_failed", eval_id=example.eval_id, error=type(exc).__name__)
            detail = f"{type(exc).__name__}: {exc}"[:300]
            return ExampleResult(example, False, (f"producer error: {detail}",), error=detail)
        return self.evaluate_brief(brief, example)

    def run(self, examples: Sequence[EvalExample]) -> EvaluationReport:
        if not examples:
            raise ValueError("evaluation requires at least one example")
        results = tuple(self.run_example(example) for example in examples)
        return self.report(results)

    def report(self, results: Sequence[ExampleResult]) -> EvaluationReport:
        pass_rate = sum(1 for r in results if r.passed) / len(results) if results else 0.0
        aggregates = summarize([r.metrics.as_dict() for r in results if r.metrics is not None])
        failures = self._gate.run_failures(pass_rate, aggregates)
        metrics = get_metrics()
        metrics.observe("evaluation.pass_rate", pass_rate)
        metrics.increment("evaluation.runs", gate="pass" if not failures else "fail")
        return EvaluationReport(
            results=tuple(results),
            aggregates=aggregates,
            pass_rate=pass_rate,
            gate_passed=not failures,
            gate_failures=tuple(failures),
            thresholds=self._gate.thresholds(),
        )


def offline_producer(
    settings: AppSettings,
    *,
    llm: LLMClient | None = None,
    var_dir: str | None = None,
) -> BriefProducer:
    """Run the orchestrator per example on a fresh local runtime that replays the example's snapshot."""
    from client_research_agent.agent.factory import build_runtime  # noqa: PLC0415 - avoids an import cycle
    from client_research_agent.orchestration.orchestrator import ClientResearchOrchestrator  # noqa: PLC0415
    from client_research_agent.services.local import InMemoryDocumentStore  # noqa: PLC0415

    root = var_dir or tempfile.mkdtemp(prefix="cra-eval-")

    def produce(example: EvalExample) -> ClientBrief:
        store = InMemoryDocumentStore()
        runtime = build_runtime(
            settings,
            llm=llm,
            document_store=store,
            ingestion=SnapshotIngestion.from_examples([example], document_store=store),
            refresh_ingestion=None,
            var_dir=root,
            track_runs=False,
        )
        try:
            result = ClientResearchOrchestrator(runtime).run(
                example.inputs.to_request(), principal=EVALUATION_PRINCIPAL
            )
        finally:
            runtime.close()
        return result.brief

    return produce


def responses_request(example: EvalExample) -> dict[str, Any]:
    inputs = example.inputs
    return {
        "input": [
            {"role": "user", "content": f"Research {inputs.company_name} and return a qualification brief."}
        ],
        "custom_inputs": {**inputs.model_dump(mode="json", exclude_none=True), "requested_by": "evaluation"},
    }


def brief_from_response(response: Any) -> ClientBrief:
    """Extract the brief from a ResponsesAgent response (pydantic object or dict)."""
    payload = response.model_dump() if hasattr(response, "model_dump") else response
    if not isinstance(payload, Mapping):
        raise ValueError(f"unexpected model response type {type(response).__name__}")
    custom = payload.get("custom_outputs") or {}
    raw = custom.get("brief_json") if isinstance(custom, Mapping) else None
    if raw is None:
        raise ValueError("model response has no custom_outputs.brief_json")
    return ClientBrief.model_validate_json(raw if isinstance(raw, str) else json.dumps(raw))


def model_producer(model: Any) -> BriefProducer:
    """Brief producer over a pyfunc ResponsesAgent (``mlflow.pyfunc.load_model(uri)``)."""

    def produce(example: EvalExample) -> ClientBrief:
        return brief_from_response(model.predict(responses_request(example)))

    return produce
