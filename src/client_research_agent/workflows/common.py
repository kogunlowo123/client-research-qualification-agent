"""Shared plumbing for the Databricks job entry points (``cra-ingest``, ``cra-brief``, ...).

* Argument parsing: job parameters arrive as strings and an empty string means
  "not given" (``{{job.parameters.domain}}`` renders as ``""`` when unset).
* Settings: ``--environment``/``--catalog``/``--schema``/``--vs-endpoint``/
  ``--experiment``/``--warehouse-id`` override the layered configuration.
* Task values: set with ``dbutils.jobs.taskValues.set`` when running on a
  Databricks runtime (``DATABRICKS_RUNTIME_VERSION``), otherwise logged.
* Exit codes: ``main`` functions raise ``SystemExit`` with a non-zero code on
  failure, which fails both a console script and a ``python_wheel_task``.
"""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from client_research_agent.config.settings import AppSettings, Environment, build_settings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.setup import configure_observability
from client_research_agent.security.rbac import Principal, Role
from client_research_agent.utils.errors import AgentError, ConfigurationError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
_log = get_logger(__name__)


def empty_to_none(value: str | None) -> str | None:
    """``""`` / whitespace -> ``None``; other values are stripped."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def optional_str(value: str) -> str | None:
    """``argparse`` type for optional job parameters."""
    return empty_to_none(value)


def optional_float(value: str) -> float | None:
    cleaned = empty_to_none(value)
    return float(cleaned) if cleaned is not None else None


def base_parser(
    description: str, *, vs_endpoint: bool = True, experiment: bool = False
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--environment", required=True, choices=[e.value for e in Environment])
    parser.add_argument("--catalog", type=optional_str, default=None)
    parser.add_argument("--schema", type=optional_str, default=None)
    parser.add_argument(
        "--warehouse-id", type=optional_str, default=None, help="SQL warehouse for Unity Catalog"
    )
    parser.add_argument(
        "--contact-email",
        type=optional_str,
        default=None,
        help="SEC fair-access contact sent with crawler requests",
    )
    if vs_endpoint:
        parser.add_argument("--vs-endpoint", type=optional_str, default=None)
    if experiment:
        parser.add_argument("--experiment", type=optional_str, default=None)
    return parser


def resolve_databricks_host(environ: Mapping[str, str] | None = None) -> str | None:
    """``DATABRICKS_HOST``, or the host the SDK resolves from the job's runtime credentials."""
    env = os.environ if environ is None else environ
    host = empty_to_none(env.get("DATABRICKS_HOST"))
    if host or not env.get("DATABRICKS_RUNTIME_VERSION"):
        return host
    try:
        client = importlib.import_module("databricks.sdk").WorkspaceClient()
        resolved = getattr(client.config, "host", None)
    except Exception as exc:  # resolution is best effort; settings validation reports a missing host
        _log.warning("job.host_resolution_failed", error=type(exc).__name__)
        return None
    return str(resolved) if resolved else None


def settings_from_args(args: argparse.Namespace, *, environ: Mapping[str, str] | None = None) -> AppSettings:
    overrides: dict[str, Any] = {}
    databricks: dict[str, Any] = {}
    for key, attr in (("catalog", "catalog"), ("schema", "schema"), ("warehouse_id", "warehouse_id")):
        value = getattr(args, attr, None)
        if value:
            databricks[key] = value
    environment = Environment(args.environment)
    if environment is not Environment.LOCAL:
        host = resolve_databricks_host(environ)
        if host:
            databricks.setdefault("host", host)
    if databricks:
        overrides["databricks"] = databricks
    if getattr(args, "vs_endpoint", None):
        overrides["vector_search"] = {"endpoint_name": args.vs_endpoint}
    if getattr(args, "contact_email", None):
        overrides["crawler"] = {"contact_email": args.contact_email}
    if getattr(args, "experiment", None):
        overrides["observability"] = {"mlflow_experiment": args.experiment}
    try:
        return build_settings(environment, **overrides)
    except ValueError as exc:
        raise ConfigurationError(f"invalid job configuration: {exc}") from exc


def job_principal(settings: AppSettings, job: str) -> Principal:
    return Principal(id=f"job:{job}:{settings.environment.value}", roles=frozenset({Role.SERVICE}))


def on_databricks(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(env.get("DATABRICKS_RUNTIME_VERSION"))


def set_task_values(values: Mapping[str, Any], *, environ: Mapping[str, str] | None = None) -> None:
    """Publish task values for downstream tasks (``{{tasks.<key>.values.<name>}}``)."""
    if on_databricks(environ):
        dbutils = importlib.import_module("databricks.sdk.runtime").dbutils
        for key, value in values.items():
            dbutils.jobs.taskValues.set(key=key, value=value)
    _log.info("job.task_values", **{str(k): v for k, v in values.items()})


def split_table_name(full_name: str) -> str:
    """Validate ``catalog.schema.table`` and return it with quoted identifiers."""
    parts = full_name.strip().split(".")
    if len(parts) != 3:
        raise ConfigurationError(f"table name must be catalog.schema.table, got {full_name!r}")
    from client_research_agent.databricks.unity_catalog import qualified_name  # noqa: PLC0415

    return qualified_name(*parts)


def statement_executor(settings: AppSettings, workspace_client: Any) -> Any:
    """A ``StatementExecutor`` on the resolved SQL warehouse (Databricks environments only)."""
    if workspace_client is None:
        raise ConfigurationError(
            f"Unity Catalog tables are not available in the {settings.environment.value} environment"
        )
    from client_research_agent.agent.factory import resolve_warehouse_id  # noqa: PLC0415
    from client_research_agent.databricks.unity_catalog import StatementExecutor  # noqa: PLC0415

    return StatementExecutor(
        workspace_client.statement_execution,
        resolve_warehouse_id(settings, workspace_client),
        resilience=settings.resilience,
    )


def configure_job_observability(settings: AppSettings) -> None:
    configure_observability(settings.observability, environment=settings.environment.value)


def run_main(
    job: str,
    body: Callable[[argparse.Namespace], int],
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
) -> int:
    """Parse, run ``body`` and translate failures into exit codes."""
    args = parser.parse_args(argv)
    try:
        return body(args)
    except AgentError as exc:
        _log.error("job.failed", job=job, error=f"{type(exc).__name__}: {exc}"[:500])
        return EXIT_FAILURE
    except (ValueError, KeyError, OSError) as exc:
        _log.error("job.failed", job=job, error=f"{type(exc).__name__}: {exc}"[:500])
        return EXIT_FAILURE


def exit_with(code: int) -> None:
    """Entry-point epilogue: non-zero codes fail the job run."""
    if code != EXIT_OK:
        raise SystemExit(code)
