"""Optional MLflow GenAI evaluation over the deterministic metrics.

:func:`log_to_mlflow_genai` hands an :class:`EvaluationReport`'s briefs to
``mlflow.genai.evaluate`` with custom scorers (``mlflow.genai.scorers.scorer``)
that wrap the deterministic metrics, so every evaluation run is browsable in
the MLflow experiment UI alongside judge-based scorers. Outputs are passed
pre-computed (no ``predict_fn``), so the agent is not re-run. MLflow is
imported lazily; without it (or on any tracking failure) the function returns
``None`` and the caller keeps its deterministic report.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from client_research_agent.evaluation.dataset import EvalExpectations
from client_research_agent.evaluation.harness import EvaluationReport
from client_research_agent.evaluation.metrics import (
    DEFAULT_SUPPORT_THRESHOLD,
    citation_coverage,
    criterion_score_mae,
    discovery_questions_ok,
    expected_fact_recall,
    grounded_fact_ratio,
    section_completeness,
    verdict_accuracy,
)
from client_research_agent.models import ClientBrief
from client_research_agent.observability.logging import get_logger

_log = get_logger(__name__)
MetricFn = Callable[[ClientBrief, EvalExpectations], float | None]


def metric_functions(support_threshold: float = DEFAULT_SUPPORT_THRESHOLD) -> dict[str, MetricFn]:
    return {
        "citation_coverage": lambda b, _e: citation_coverage(b),
        "grounded_fact_ratio": lambda b, _e: grounded_fact_ratio(b, threshold=support_threshold),
        "section_completeness": lambda b, _e: section_completeness(b),
        "discovery_questions_ok": lambda b, _e: discovery_questions_ok(b),
        "verdict_accuracy": lambda b, e: verdict_accuracy(b, e.expected_verdict),
        "criterion_score_mae": lambda b, e: criterion_score_mae(b, e.expected_scores),
        "expected_fact_recall": lambda b, e: expected_fact_recall(
            b, e.expected_facts, threshold=support_threshold
        ),
    }


def _brief(outputs: Any) -> ClientBrief:
    raw = outputs.get("brief_json") if isinstance(outputs, Mapping) else outputs
    return ClientBrief.model_validate_json(str(raw))


def _expectations(expectations: Any) -> EvalExpectations:
    return EvalExpectations.model_validate(dict(expectations or {}))


def build_scorers(
    scorer_decorator: Callable[..., Any], *, support_threshold: float = DEFAULT_SUPPORT_THRESHOLD
) -> list[Any]:
    """Wrap each deterministic metric as an MLflow custom scorer (``outputs`` + ``expectations``)."""
    scorers: list[Any] = []
    for name, metric in metric_functions(support_threshold).items():

        def score(outputs: Any, expectations: Any = None, _metric: MetricFn = metric) -> float | None:
            return _metric(_brief(outputs), _expectations(expectations))

        scorers.append(scorer_decorator(score, name=name))
    return scorers


def evaluation_records(report: EvaluationReport) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in report.results:
        if result.brief is None:
            continue
        record = result.example.to_mlflow_record()
        record["outputs"] = {
            "brief_json": result.brief.model_dump_json(),
            "verdict": result.brief.qualification.verdict.value,
        }
        records.append(record)
    return records


def log_to_mlflow_genai(
    report: EvaluationReport,
    *,
    experiment: str | None = None,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    mlflow_module: Any | None = None,
) -> str | None:
    """Run ``mlflow.genai.evaluate`` over the report's briefs; returns the MLflow run id or ``None``."""
    records = evaluation_records(report)
    if not records:
        return None
    try:
        mlflow = mlflow_module or importlib.import_module("mlflow")
        genai = importlib.import_module("mlflow.genai") if mlflow_module is None else mlflow.genai
        if experiment:
            mlflow.set_experiment(experiment)
        scorers = build_scorers(genai.scorer, support_threshold=support_threshold)
        result = genai.evaluate(data=records, scorers=scorers)
    except Exception as exc:  # optional integration: never fail the deterministic evaluation
        _log.warning("mlflow_genai_evaluate_failed", error=f"{type(exc).__name__}: {exc}"[:300])
        return None
    run_id = getattr(result, "run_id", None)
    return str(run_id) if run_id else None
