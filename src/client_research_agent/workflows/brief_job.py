"""``cra-brief``: research one company end to end and persist the client brief.

Runs the full orchestrator with ``--run-id`` from the ingest task (so the brief,
lineage and audit trail share one id). Evidence ingested by the upstream task
is reused (``IngestMode.IF_MISSING``). Evidence lineage rows are written to the
Unity Catalog ``lineage`` table when running against Databricks.
Task values: ``brief_id``, ``verdict``, ``citation_coverage``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any

from client_research_agent.agent.factory import AgentRuntime, build_runtime
from client_research_agent.models import ResearchRequest
from client_research_agent.observability.logging import get_logger
from client_research_agent.orchestration.orchestrator import (
    ClientResearchOrchestrator,
    IngestMode,
    RunOptions,
)
from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows.common import (
    EXIT_FAILURE,
    EXIT_OK,
    base_parser,
    configure_job_observability,
    exit_with,
    job_principal,
    optional_str,
    run_main,
    set_task_values,
    settings_from_args,
    statement_executor,
)

_log = get_logger(__name__)
JOB = "brief"
LINEAGE_COLUMNS = (
    "lineage_id",
    "run_id",
    "evidence_id",
    "chunk_id",
    "doc_id",
    "url",
    "content_hash",
    "statement_provenance",
    "supported",
    "support_score",
    "created_at",
)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("Generate a cited, scored client brief for one company.", experiment=True)
    parser.add_argument("--run-id", type=optional_str, default=None)
    parser.add_argument("--company", type=optional_str, default=None)
    parser.add_argument("--domain", type=optional_str, default=None)
    parser.add_argument("--ticker", type=optional_str, default=None)
    parser.add_argument("--requested-by", type=optional_str, default=None)
    parser.add_argument("--max-documents", type=int, default=40)
    return parser


def write_lineage(runtime: AgentRuntime, rows: Sequence[Mapping[str, Any]]) -> int:
    """MERGE evidence lineage rows into ``<catalog>.<schema>.lineage`` (idempotent on ``lineage_id``)."""
    if not rows or runtime.workspace_client is None:
        return 0
    from client_research_agent.databricks.unity_catalog import qualified_name  # noqa: PLC0415

    settings = runtime.settings
    table = qualified_name(settings.databricks.catalog, settings.databricks.schema_, "lineage")
    executor = statement_executor(settings, runtime.workspace_client)
    source = ", ".join(f":{c} AS {c}" for c in LINEAGE_COLUMNS)
    columns = ", ".join(LINEAGE_COLUMNS)
    values = ", ".join(f"s.{c}" for c in LINEAGE_COLUMNS)
    statement = (
        f"MERGE INTO {table} AS t USING (SELECT {source}) AS s ON t.lineage_id = s.lineage_id "  # noqa: S608 - validated identifiers
        f"WHEN NOT MATCHED THEN INSERT ({columns}) VALUES ({values})"
    )
    for row in rows:
        executor.execute(statement, {c: row.get(c) for c in LINEAGE_COLUMNS}, op="merge_lineage")
    return len(rows)


def run(args: argparse.Namespace) -> int:
    if not args.company:
        raise ConfigurationError("--company is required")
    settings = settings_from_args(args)
    configure_job_observability(settings)
    runtime = build_runtime(settings)
    try:
        request = ResearchRequest(
            company_name=args.company,
            domain=args.domain,
            ticker=args.ticker,
            max_documents=args.max_documents,
            requested_by=args.requested_by or "workflow",
        )
        orchestrator = ClientResearchOrchestrator(runtime, options=RunOptions(ingest=IngestMode.IF_MISSING))
        result = orchestrator.run(request, principal=job_principal(settings, JOB), run_id=args.run_id)
        try:
            written = write_lineage(runtime, result.lineage_rows)
        except Exception as exc:  # lineage export is optional; the brief is already persisted
            _log.warning("brief.lineage_export_failed", error=type(exc).__name__)
            written = 0
        brief = result.brief
        set_task_values(
            {
                "brief_id": brief.run_id,
                "verdict": brief.qualification.verdict.value,
                "citation_coverage": round(brief.citation_report.coverage, 4),
                "needs_review": result.review.needs_review,
                "lineage_rows": written,
            }
        )
        _log.info(
            "brief.done", run_id=brief.run_id, status=result.state.status.value, persisted=result.persisted
        )
        return EXIT_OK if result.persisted else EXIT_FAILURE
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    exit_with(run_main(JOB, run, build_parser(), argv))
