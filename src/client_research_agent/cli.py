"""``cra`` command line.

::

    cra research --company NAME [--domain D] [--ticker T] [--format md|json] [--output PATH]
    cra ingest   --company NAME [--domain D] [--ticker T]
    cra evaluate [--dataset PATH|golden] [--min-pass-rate X] [--min-citation-coverage Y] [--output PATH]
    cra serve-check

Logs go to stderr so ``cra research`` output can be piped. The acting
principal comes from ``CRA_PRINCIPAL_ID`` and ``CRA_PRINCIPAL_GROUPS`` (comma
separated workspace groups mapped to roles by
:func:`~client_research_agent.security.rbac.map_groups_to_roles`); locally, with
no groups configured, the operator acts as an analyst.
"""

from __future__ import annotations

import argparse
import contextlib
import getpass
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from client_research_agent import __version__
from client_research_agent.config.settings import AppSettings, Environment, build_settings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.setup import configure_observability
from client_research_agent.security.rbac import Principal, Role, map_groups_to_roles
from client_research_agent.utils.errors import AgentError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
_log = get_logger(__name__)


def _optional(value: str) -> str | None:
    return value.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cra", description="Client Research & Qualification Agent")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--environment", choices=[e.value for e in Environment], default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    research = commands.add_parser("research", help="research one company and print its client brief")
    _company_arguments(research)
    research.add_argument("--industry", type=_optional, default=None)
    research.add_argument("--format", choices=["md", "json"], default="md")
    research.add_argument("--output", type=Path, default=None)

    ingest = commands.add_parser("ingest", help="ingest and index public evidence for one company")
    _company_arguments(ingest)

    evaluate = commands.add_parser("evaluate", help="run the offline evaluation harness and quality gate")
    evaluate.add_argument("--dataset", default="golden", help="JSON Lines eval set, or 'golden'")
    evaluate.add_argument("--min-pass-rate", type=float, default=0.85)
    evaluate.add_argument("--min-citation-coverage", type=float, default=0.9)
    evaluate.add_argument("--min-grounded-fact-ratio", type=float, default=0.8)
    evaluate.add_argument("--output", type=Path, default=None)

    commands.add_parser("serve-check", help="exit 0 when the agent can start (container health check)")
    return parser


def _company_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--company", required=True)
    parser.add_argument("--domain", type=_optional, default=None)
    parser.add_argument("--ticker", type=_optional, default=None)
    parser.add_argument("--cik", type=_optional, default=None)
    parser.add_argument("--max-documents", type=int, default=40)


def cli_principal(environ: Mapping[str, str] | None = None) -> Principal:
    env = os.environ if environ is None else environ
    principal_id = (env.get("CRA_PRINCIPAL_ID") or "").strip() or _local_user()
    groups = [g.strip() for g in (env.get("CRA_PRINCIPAL_GROUPS") or "").split(",") if g.strip()]
    if not groups:
        return Principal(id=principal_id, roles=frozenset({Role.ANALYST}))
    return Principal(id=principal_id, roles=map_groups_to_roles(groups), groups=frozenset(groups))


def _local_user() -> str:
    try:
        return f"cli:{getpass.getuser()}"
    except (OSError, KeyError):
        return "cli:unknown"


@contextlib.contextmanager
def _logs_to_stderr() -> Iterator[None]:
    """Configure logging while stdout points at stderr, so log handlers bind to stderr."""
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _settings(args: argparse.Namespace) -> AppSettings:
    settings = build_settings(args.environment)
    with _logs_to_stderr():
        configure_observability(settings.observability, environment=settings.environment.value)
    return settings


def _request(args: argparse.Namespace, principal: Principal) -> Any:
    from client_research_agent.models import ResearchRequest  # noqa: PLC0415

    return ResearchRequest(
        company_name=args.company,
        domain=args.domain,
        ticker=args.ticker,
        cik=args.cik,
        industry=getattr(args, "industry", None),
        max_documents=args.max_documents,
        requested_by=principal.id,
    )


def _emit(text: str, output: Path | None) -> None:
    if output is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    sys.stderr.write(f"wrote {output}\n")


def cmd_research(args: argparse.Namespace) -> int:
    from client_research_agent.agent.factory import build_runtime  # noqa: PLC0415
    from client_research_agent.briefing.renderer import render_json  # noqa: PLC0415
    from client_research_agent.orchestration.orchestrator import ClientResearchOrchestrator  # noqa: PLC0415

    settings = _settings(args)
    principal = cli_principal()
    runtime = build_runtime(settings)
    try:
        result = ClientResearchOrchestrator(runtime).run(_request(args, principal), principal=principal)
    finally:
        runtime.close()
    text = result.markdown if args.format == "md" else render_json(result.brief)
    _emit(text, args.output)
    sys.stderr.write(
        f"run {result.run_id}: verdict={result.verdict.value} status={result.state.status.value} "
        f"review={'required' if result.review.needs_review else 'not required'}\n"
    )
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    from client_research_agent.agent.factory import build_runtime  # noqa: PLC0415
    from client_research_agent.governance.lineage import LineageRecorder  # noqa: PLC0415
    from client_research_agent.orchestration.orchestrator import new_run_id  # noqa: PLC0415
    from client_research_agent.orchestration.steps import EvidenceGatherer  # noqa: PLC0415
    from client_research_agent.security.rbac import Permission, authorize  # noqa: PLC0415

    settings = _settings(args)
    principal = cli_principal()
    runtime = build_runtime(settings)
    try:
        authorize(principal, Permission.RUN_RESEARCH, audit=runtime.audit)
        run_id = new_run_id(args.company)
        result = EvidenceGatherer(runtime).gather(_request(args, principal), lineage=LineageRecorder(run_id))
        runtime.audit.append("ingest.company", {"company": args.company, **result.summary()}, run_id=run_id)
    finally:
        runtime.close()
    _emit(json.dumps({"run_id": run_id, **result.summary()}, indent=2, default=str), None)
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    from client_research_agent.evaluation.dataset import load_golden_set, load_jsonl  # noqa: PLC0415
    from client_research_agent.evaluation.harness import (  # noqa: PLC0415
        EvaluationHarness,
        QualityGate,
        offline_producer,
    )

    settings = _settings(args)
    examples = load_golden_set() if args.dataset == "golden" else load_jsonl(args.dataset)
    gate = QualityGate(
        min_pass_rate=args.min_pass_rate,
        min_citation_coverage=args.min_citation_coverage,
        min_grounded_fact_ratio=args.min_grounded_fact_ratio,
    )
    with tempfile.TemporaryDirectory(prefix="cra-eval-") as scratch:
        report = EvaluationHarness(offline_producer(settings, var_dir=scratch), gate).run(examples)
    _emit(json.dumps(report.to_dict(), indent=2, default=str), args.output)
    sys.stderr.write(
        f"quality gate {'passed' if report.gate_passed else 'FAILED'}: pass_rate={report.pass_rate:.2f}\n"
    )
    return EXIT_OK if report.gate_passed else EXIT_FAILURE


def cmd_serve_check(args: argparse.Namespace) -> int:
    """Offline readiness: settings resolve, prompts load, the var directory is writable, core imports work."""
    from client_research_agent.agent.factory import default_var_dir  # noqa: PLC0415
    from client_research_agent.orchestration import orchestrator  # noqa: PLC0415
    from client_research_agent.prompts.registry import default_registry  # noqa: PLC0415

    settings = build_settings(args.environment)
    checks: dict[str, Any] = {"environment": settings.environment.value, "version": __version__}
    checks["prompts"] = len(default_registry().names())
    var_dir = default_var_dir(settings)
    var_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=var_dir, prefix=".healthcheck-", delete=True) as handle:
        handle.write(b"ok")
    checks["var_dir"] = str(var_dir)
    checks["orchestrator"] = orchestrator.ClientResearchOrchestrator.__name__
    _emit(json.dumps({"status": "healthy", **checks}), None)
    return EXIT_OK


COMMANDS = {
    "research": cmd_research,
    "ingest": cmd_ingest,
    "evaluate": cmd_evaluate,
    "serve-check": cmd_serve_check,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except AgentError as exc:
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return EXIT_FAILURE
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_USAGE if isinstance(exc, ValueError) else EXIT_FAILURE
