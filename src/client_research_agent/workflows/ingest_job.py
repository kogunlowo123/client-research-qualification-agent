"""``cra-ingest``: refresh public evidence for one company or the whole watchlist.

Modes (exactly one)::

    --watchlist-table catalog.schema.companies_watchlist   every active company, by priority
    --company NAME [--domain D] [--ticker T]                one company (job parameters)
    --sync-index-only                                       only trigger the Vector Search sync

``--sync-index`` triggers one Delta Sync of the chunks index after ingesting.
Each company goes through the same Evidence Gathering Agent as an interactive
run (sanitisation, injection/poisoning screening, PII redaction, indexing).
Task values: ``run_id``, ``documents_ingested``, ``chunks_written``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from client_research_agent.agent.factory import AgentRuntime, build_runtime
from client_research_agent.governance.lineage import LineageRecorder
from client_research_agent.models import ResearchRequest
from client_research_agent.observability.logging import get_logger
from client_research_agent.orchestration.orchestrator import new_run_id
from client_research_agent.orchestration.steps import EvidenceGatherer
from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows.common import (
    EXIT_FAILURE,
    EXIT_OK,
    base_parser,
    configure_job_observability,
    empty_to_none,
    exit_with,
    optional_str,
    run_main,
    set_task_values,
    settings_from_args,
    split_table_name,
    statement_executor,
)

_log = get_logger(__name__)
JOB = "ingest"


@dataclass(frozen=True, slots=True)
class IngestTotals:
    companies: int = 0
    failed: int = 0
    documents: int = 0
    chunks: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("Refresh public-source evidence and the Vector Search index.")
    # Exactly one mode is required, but it is validated in ``validate_mode`` rather than with
    # an argparse mutually exclusive group: jobs pass empty strings for unused parameters and
    # argparse's handling of "" in required groups differs across Python 3.12 patch releases.
    parser.add_argument("--watchlist-table", type=optional_str, default=None)
    parser.add_argument("--company", type=optional_str, default=None)
    parser.add_argument("--sync-index-only", action="store_true")
    parser.add_argument("--domain", type=optional_str, default=None)
    parser.add_argument("--ticker", type=optional_str, default=None)
    parser.add_argument("--max-documents", type=int, default=40)
    parser.add_argument("--sync-index", action="store_true")
    return parser


def load_watchlist(runtime: AgentRuntime, table: str) -> list[ResearchRequest]:
    executor = statement_executor(runtime.settings, runtime.workspace_client)
    rows = executor.execute(
        f"SELECT company_name, domain, ticker, cik, industry FROM {split_table_name(table)} "  # noqa: S608 - validated identifiers  # nosec B608
        "WHERE active = true ORDER BY priority, company_name",
        op="read_watchlist",
    )
    requests: list[ResearchRequest] = []
    for row in rows:
        name = empty_to_none(row.get("company_name"))
        if name is None:
            continue
        requests.append(
            ResearchRequest(
                company_name=name,
                domain=empty_to_none(row.get("domain")),
                ticker=empty_to_none(row.get("ticker")),
                cik=empty_to_none(row.get("cik")),
                industry=empty_to_none(row.get("industry")),
                requested_by="job:ingest-watchlist",
            )
        )
    return requests


def mark_ingested(runtime: AgentRuntime, table: str, company: str) -> None:
    executor = statement_executor(runtime.settings, runtime.workspace_client)
    executor.execute(
        f"UPDATE {split_table_name(table)} SET last_ingested_at = current_timestamp() "  # noqa: S608 - validated identifiers  # nosec B608
        "WHERE company_name = :company",
        {"company": company},
        op="mark_ingested",
    )


def sync_index(runtime: AgentRuntime) -> bool:
    sync = getattr(runtime.vector_index, "sync", None)
    if not callable(sync):
        _log.info("ingest.sync_not_supported", index=type(runtime.vector_index).__name__)
        return False
    sync()
    return True


def ingest(
    runtime: AgentRuntime, requests: Sequence[ResearchRequest], *, run_id: str, table: str | None
) -> IngestTotals:
    gatherer = EvidenceGatherer(runtime)
    lineage = LineageRecorder(run_id)
    totals = IngestTotals()
    for request in requests:
        try:
            result = gatherer.gather(request, lineage=lineage)
        except Exception as exc:  # one company's outage must not stop the watchlist
            _log.error("ingest.company_failed", company=request.company_name, error=type(exc).__name__)
            totals = IngestTotals(totals.companies + 1, totals.failed + 1, totals.documents, totals.chunks)
            continue
        _log.info("ingest.company_done", company=request.company_name, **result.summary())
        runtime.audit.append(
            "ingest.company", {"company": request.company_name, **result.summary()}, run_id=run_id
        )
        if table is not None:
            mark_ingested(runtime, table, request.company_name)
        totals = IngestTotals(
            totals.companies + 1,
            totals.failed,
            totals.documents + len(result.documents),
            totals.chunks + result.chunks_written,
        )
    return totals


def validate_mode(args: argparse.Namespace) -> None:
    """Require exactly one of --watchlist-table, --company or --sync-index-only (after ""->None)."""
    selected = [bool(args.watchlist_table), bool(args.company), bool(args.sync_index_only)]
    if sum(selected) != 1:
        raise ConfigurationError("one of --watchlist-table, --company or --sync-index-only is required")


def run(args: argparse.Namespace) -> int:
    validate_mode(args)
    settings = settings_from_args(args)
    configure_job_observability(settings)
    runtime = build_runtime(settings, trigger_index_sync=False, track_runs=False)
    try:
        if args.sync_index_only:
            synced = sync_index(runtime)
            set_task_values(
                {"run_id": "", "documents_ingested": 0, "chunks_written": 0, "index_synced": synced}
            )
            return EXIT_OK
        if args.watchlist_table:
            requests = load_watchlist(runtime, args.watchlist_table)
            run_id = new_run_id("ingest-watchlist")
        elif args.company:
            requests = [
                ResearchRequest(
                    company_name=args.company,
                    domain=args.domain,
                    ticker=args.ticker,
                    max_documents=args.max_documents,
                    requested_by="job:ingest",
                )
            ]
            run_id = new_run_id(args.company)
        else:
            raise ConfigurationError("one of --watchlist-table, --company or --sync-index-only is required")
        totals = ingest(runtime, requests, run_id=run_id, table=args.watchlist_table)
        synced = sync_index(runtime) if args.sync_index and totals.chunks else False
        values: dict[str, Any] = {
            "run_id": run_id,
            "documents_ingested": totals.documents,
            "chunks_written": totals.chunks,
            "companies": totals.companies,
            "companies_failed": totals.failed,
            "index_synced": synced,
        }
        set_task_values(values)
        if requests and totals.failed == len(requests):
            return EXIT_FAILURE
        return EXIT_OK
    finally:
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    exit_with(run_main(JOB, run, build_parser(), argv))
