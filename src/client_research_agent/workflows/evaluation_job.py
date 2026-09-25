"""``cra-evaluate``: quality gate for one brief or for the agent over an evaluation set.

``--mode brief --brief-id RUN_ID``
    Loads the persisted brief and gates it on citation coverage, grounded-fact
    ratio and the five-discovery-question contract.

``--mode dataset --eval-table T [--model-uri URI] [--results-table R]``
    Loads examples from a Unity Catalog ``eval_set`` table, a ``.jsonl`` file or
    the packaged golden set (``--eval-table golden``). Briefs come from the
    model at ``--model-uri`` (e.g. ``models:/cat.sch.agent@champion``) or, when
    omitted, from the orchestrator (examples with a snapshot replay offline).
    Per-metric results go to ``--results-table`` and to MLflow.

With ``--fail-on-gate`` the process exits non-zero when the gate fails.
"""

from __future__ import annotations

import argparse
import importlib
import json
import uuid
from collections.abc import Sequence
from typing import Any

from client_research_agent.agent.factory import AgentRuntime, build_runtime
from client_research_agent.evaluation.dataset import (
    EvalExample,
    examples_from_rows,
    load_golden_set,
    load_jsonl,
)
from client_research_agent.evaluation.harness import (
    EVALUATION_PRINCIPAL,
    BriefProducer,
    EvaluationHarness,
    EvaluationReport,
    QualityGate,
    model_producer,
    offline_producer,
)
from client_research_agent.evaluation.mlflow_eval import log_to_mlflow_genai
from client_research_agent.models import ClientBrief
from client_research_agent.models.domain import utc_now
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.mlflow_tracking import RunTracker
from client_research_agent.orchestration.orchestrator import ClientResearchOrchestrator
from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows.common import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    base_parser,
    configure_job_observability,
    exit_with,
    optional_float,
    optional_str,
    run_main,
    set_task_values,
    settings_from_args,
    split_table_name,
    statement_executor,
)

_log = get_logger(__name__)
JOB = "evaluate"
GOLDEN = "golden"


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("Evaluate briefs against the quality gate.", vs_endpoint=False, experiment=True)
    parser.add_argument("--mode", required=True, choices=["brief", "dataset"])
    parser.add_argument("--brief-id", type=optional_str, default=None)
    parser.add_argument("--eval-table", type=optional_str, default=None)
    parser.add_argument("--results-table", type=optional_str, default=None)
    parser.add_argument("--model-uri", type=optional_str, default=None)
    parser.add_argument("--min-pass-rate", type=optional_float, default=None)
    parser.add_argument("--min-citation-coverage", type=float, required=True)
    parser.add_argument("--min-grounded-fact-ratio", type=float, default=0.8)
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser


def load_examples(runtime: AgentRuntime, source: str) -> list[EvalExample]:
    if source == GOLDEN:
        return load_golden_set()
    if source.endswith(".jsonl"):
        return load_jsonl(source)
    executor = statement_executor(runtime.settings, runtime.workspace_client)
    rows = executor.execute(
        f"SELECT * FROM {split_table_name(source)} WHERE active = true ORDER BY eval_id",  # noqa: S608 - validated identifiers  # nosec B608
        op="read_eval_set",
    )
    return examples_from_rows(rows)


def orchestrator_producer(runtime: AgentRuntime) -> BriefProducer:
    """Snapshot examples replay offline; the others run live against the runtime's adapters."""
    offline = offline_producer(runtime.settings, llm=runtime.llm)
    orchestrator = ClientResearchOrchestrator(runtime)

    def produce(example: EvalExample) -> ClientBrief:
        if example.snapshot:
            return offline(example)
        return orchestrator.run(example.inputs.to_request(), principal=EVALUATION_PRINCIPAL).brief

    return produce


def load_model(model_uri: str) -> Any:
    mlflow = importlib.import_module("mlflow")
    if model_uri.startswith("models:/") and model_uri.count(".") >= 2:
        mlflow.set_registry_uri("databricks-uc")
    return mlflow.pyfunc.load_model(model_uri)


def model_version_of(model_uri: str | None) -> str | None:
    if not model_uri or "@" in model_uri:
        return None
    tail = model_uri.rstrip("/").rsplit("/", 1)[-1]
    return tail if tail.isdigit() else None


def write_results(
    runtime: AgentRuntime,
    table: str,
    report: EvaluationReport,
    *,
    mlflow_run_id: str | None,
    model_uri: str | None,
    dataset_table: str,
) -> int:
    executor = statement_executor(runtime.settings, runtime.workspace_client)
    target = split_table_name(table)
    eval_run_id = uuid.uuid4().hex
    evaluated_at = utc_now()
    statement = (
        f"INSERT INTO {target} (eval_run_id, mlflow_run_id, model_uri, model_version, dataset_table, "  # noqa: S608 - validated identifiers  # nosec B608
        "dataset_version, metric_name, metric_value, threshold, passed, gate_passed, environment, "
        "evaluated_at) "
        "VALUES (:eval_run_id, :mlflow_run_id, :model_uri, :model_version, :dataset_table, NULL, "
        ":metric_name, "
        ":metric_value, :threshold, :passed, :gate_passed, :environment, :evaluated_at)"
    )
    rows = report.metric_rows()
    for row in rows:
        executor.execute(
            statement,
            {
                "eval_run_id": eval_run_id,
                "mlflow_run_id": mlflow_run_id or "none",
                "model_uri": model_uri or "orchestrator",
                "model_version": model_version_of(model_uri),
                "dataset_table": dataset_table,
                "metric_name": row["metric_name"],
                "metric_value": float(row["metric_value"]),
                "threshold": row["threshold"],
                "passed": bool(row["passed"]),
                "gate_passed": report.gate_passed,
                "environment": runtime.settings.environment.value,
                "evaluated_at": evaluated_at,
            },
            op="insert_eval_result",
        )
    return len(rows)


def track(runtime: AgentRuntime, report: EvaluationReport, *, mode: str, tags: dict[str, str]) -> str | None:
    with RunTracker(
        enabled=runtime.track_runs,
        experiment=runtime.settings.observability.mlflow_experiment,
        run_name=f"evaluate-{mode}",
        tags={"cra.job": "evaluate", "cra.mode": mode, **tags},
    ) as tracker:
        tracker.log_metrics(
            {"pass_rate": report.pass_rate, **report.aggregates, "gate_passed": float(report.gate_passed)}
        )
        tracker.log_dict(report.to_dict(), "evaluation_report.json")
        return tracker.run_id


def run(args: argparse.Namespace) -> int:
    settings = settings_from_args(args)
    configure_job_observability(settings)
    gate = QualityGate(
        min_pass_rate=args.min_pass_rate if args.min_pass_rate is not None else 0.85,
        min_citation_coverage=args.min_citation_coverage,
        min_grounded_fact_ratio=args.min_grounded_fact_ratio,
    )
    runtime = build_runtime(settings)
    try:
        if args.mode == "brief":
            report = evaluate_brief(runtime, gate, args.brief_id)
            if report is None:
                return EXIT_USAGE
            track(runtime, report, mode="brief", tags={"cra.run_id": args.brief_id or ""})
        else:
            report = evaluate_dataset(runtime, gate, args)
        set_task_values(
            {
                "gate_passed": report.gate_passed,
                "pass_rate": round(report.pass_rate, 4),
                "examples": len(report.results),
            }
        )
        _log.info(
            "evaluate.done", **{"gate_passed": report.gate_passed, "failures": list(report.gate_failures)}
        )
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return EXIT_FAILURE if (args.fail_on_gate and not report.gate_passed) else EXIT_OK
    finally:
        runtime.close()


def evaluate_brief(runtime: AgentRuntime, gate: QualityGate, brief_id: str | None) -> EvaluationReport | None:
    if not brief_id:
        raise ConfigurationError("--brief-id is required in brief mode")
    brief = runtime.brief_repository.get(brief_id)
    if brief is None:
        _log.error("evaluate.brief_not_found", brief_id=brief_id)
        return None
    example = EvalExample.model_validate(
        {
            "eval_id": f"brief-{brief_id}",
            "inputs": {"company_name": brief.company},
            "tags": {"run_id": brief_id},
        }
    )
    harness = EvaluationHarness(lambda _example: brief, gate)
    return harness.report([harness.evaluate_brief(brief, example)])


def evaluate_dataset(runtime: AgentRuntime, gate: QualityGate, args: argparse.Namespace) -> EvaluationReport:
    if not args.eval_table:
        raise ConfigurationError("--eval-table is required in dataset mode")
    examples = load_examples(runtime, args.eval_table)
    if not examples:
        raise ConfigurationError(f"no active evaluation examples in {args.eval_table}")
    producer = (
        model_producer(load_model(args.model_uri)) if args.model_uri else orchestrator_producer(runtime)
    )
    report = EvaluationHarness(producer, gate).run(examples)
    mlflow_run_id = track(
        runtime, report, mode="dataset", tags={"cra.model_uri": args.model_uri or "orchestrator"}
    )
    if runtime.track_runs:
        genai_run = log_to_mlflow_genai(report, experiment=runtime.settings.observability.mlflow_experiment)
        mlflow_run_id = mlflow_run_id or genai_run
    if args.results_table:
        write_results(
            runtime,
            args.results_table,
            report,
            mlflow_run_id=mlflow_run_id,
            model_uri=args.model_uri,
            dataset_table=args.eval_table,
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    exit_with(run_main(JOB, run, build_parser(), argv))
