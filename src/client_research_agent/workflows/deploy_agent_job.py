"""``cra-deploy-agent``: log, promote and deploy the agent (MLflow + Unity Catalog + Agent Framework).

``--stage log``
    Logs :mod:`client_research_agent.agent.agent_model` as a models-from-code
    ``ResponsesAgent`` with the package source bundled (``code_paths``), the
    Databricks resources it calls declared for automatic auth passthrough
    (chat/fallback/embedding serving endpoints, the Vector Search index, the SQL
    warehouse and the Unity Catalog tables), registers it to ``--uc-model``
    and points the ``challenger`` alias at the new version.
    Task value: ``model_version``.

``--stage promote``
    Moves ``champion`` to ``previous_champion``, points ``champion`` at the
    evaluated version (``--model-version`` or the ``challenger``), deploys it
    with ``databricks.agents.deploy()`` and applies the endpoint's AI Gateway
    configuration from ``--endpoint-config``.

``--stage deploy-champion``
    Redeploys whatever ``champion`` points to (used by rollback).
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from client_research_agent import __version__
from client_research_agent.agent.factory import resolve_warehouse_id
from client_research_agent.config.settings import AppSettings
from client_research_agent.observability.logging import get_logger
from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows.common import (
    EXIT_OK,
    configure_job_observability,
    exit_with,
    optional_str,
    run_main,
    set_task_values,
    settings_from_args,
)

_log = get_logger(__name__)
JOB = "deploy-agent"
DISTRIBUTION = "client-research-qualification-agent"
MODEL_NAME = "agent"
CHALLENGER = "challenger"
CHAMPION = "champion"
PREVIOUS_CHAMPION = "previous_champion"
UC_TABLES = ("documents", "chunks", "parent_chunks", "briefs", "audit_log", "lineage")
INPUT_EXAMPLE: dict[str, Any] = {
    "input": [
        {"role": "user", "content": "Research Contoso Pharmaceuticals and return a qualification brief."}
    ],
    "custom_inputs": {
        "company_name": "Contoso Pharmaceuticals",
        "max_documents": 10,
        "requested_by": "deploy-job",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Log, promote or deploy the Client Research Agent.")
    parser.add_argument("--environment", required=True, choices=["local", "dev", "staging", "prod"])
    parser.add_argument("--stage", required=True, choices=["log", "promote", "deploy-champion"])
    parser.add_argument("--uc-model", required=True, type=optional_str)
    parser.add_argument("--model-version", type=optional_str, default=None)
    parser.add_argument("--experiment", type=optional_str, default=None)
    parser.add_argument("--vs-endpoint", type=optional_str, default=None)
    parser.add_argument("--vs-index", type=optional_str, default=None)
    parser.add_argument("--endpoint-name", type=optional_str, default=None)
    parser.add_argument("--endpoint-config", type=optional_str, default=None)
    parser.add_argument("--warehouse-id", type=optional_str, default=None)
    parser.add_argument("--catalog", type=optional_str, default=None)
    parser.add_argument("--schema", type=optional_str, default=None)
    parser.add_argument("--contact-email", type=optional_str, default=None)
    return parser


def _mlflow() -> Any:
    return importlib.import_module("mlflow")


def _agents() -> Any:
    try:
        return importlib.import_module("databricks.agents")
    except ImportError as exc:
        raise ConfigurationError(
            "databricks-agents is not installed; install the 'databricks' extra (pip install "
            "'client-research-qualification-agent[databricks]') to deploy the agent"
        ) from exc


def uc_parts(uc_model: str) -> tuple[str, str, str]:
    parts = uc_model.split(".")
    if len(parts) != 3 or not all(parts):
        raise ConfigurationError(f"--uc-model must be catalog.schema.model, got {uc_model!r}")
    return parts[0], parts[1], parts[2]


def pip_requirements() -> list[str]:
    """Runtime requirements of this distribution plus its ``databricks`` extra (from installed metadata)."""
    try:
        declared = metadata.requires(DISTRIBUTION) or []
    except metadata.PackageNotFoundError:
        declared = []
    selected: list[str] = []
    for requirement in declared:
        spec, _, marker = requirement.partition(";")
        marker = marker.strip()
        if not marker or 'extra == "databricks"' in marker.replace("'", '"'):
            selected.append(spec.strip())
    return selected


def model_resources(
    settings: AppSettings, *, vs_index: str | None, warehouse_id: str | None, catalog: str, schema: str
) -> list[Any]:
    """Databricks resources the served agent calls (automatic authentication passthrough)."""
    resources = importlib.import_module("mlflow.models.resources")
    endpoints = dict.fromkeys(
        e
        for e in (
            settings.serving.chat_endpoint,
            settings.serving.fallback_chat_endpoint,
            settings.serving.embedding_endpoint,
        )
        if e
    )
    declared: list[Any] = [resources.DatabricksServingEndpoint(endpoint_name=name) for name in endpoints]
    if vs_index:
        declared.append(resources.DatabricksVectorSearchIndex(index_name=vs_index))
    if warehouse_id:
        declared.append(resources.DatabricksSQLWarehouse(warehouse_id=warehouse_id))
    declared.extend(resources.DatabricksTable(table_name=f"{catalog}.{schema}.{t}") for t in UC_TABLES)
    return declared


def log_stage(args: argparse.Namespace, settings: AppSettings) -> dict[str, Any]:
    from client_research_agent.agent import agent_model  # noqa: PLC0415 - validates the file imports cleanly

    mlflow = _mlflow()
    catalog, schema, _ = uc_parts(args.uc_model)
    warehouse_id = None
    try:
        from client_research_agent.databricks.auth import build_workspace_client  # noqa: PLC0415

        warehouse_id = resolve_warehouse_id(settings, build_workspace_client(settings))
    except Exception as exc:  # the warehouse resource is optional for logging
        _log.warning("deploy.warehouse_unresolved", error=type(exc).__name__)
    package_dir = Path(agent_model.__file__).resolve().parents[1]
    mlflow.set_registry_uri("databricks-uc")
    if args.experiment:
        mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=f"log-agent-{__version__}"):
        mlflow.log_params({"agent_version": __version__, "environment": settings.environment.value})
        info = mlflow.pyfunc.log_model(
            name=MODEL_NAME,
            python_model=str(Path(agent_model.__file__).resolve()),
            code_paths=[str(package_dir)],
            pip_requirements=pip_requirements(),
            resources=model_resources(
                settings, vs_index=args.vs_index, warehouse_id=warehouse_id, catalog=catalog, schema=schema
            ),
            input_example=INPUT_EXAMPLE,
            registered_model_name=args.uc_model,
            metadata={"agent_version": __version__},
        )
    version = str(getattr(info, "registered_model_version", "") or "")
    if not version:
        raise ConfigurationError("model registration did not return a version")
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    client.set_registered_model_alias(args.uc_model, CHALLENGER, version)
    set_task_values({"model_version": version, "model_uri": f"models:/{args.uc_model}/{version}"})
    return {"model_version": version}


def load_endpoint_config(path: str | None, uc_model: str) -> dict[str, Any]:
    """Endpoint JSON with every served entity's ``entity_name`` replaced by ``uc_model``."""
    if not path:
        return {}
    config: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    for entity in config.get("config", {}).get("served_entities", []):
        entity["entity_name"] = uc_model
    return config


def _served_entity(config: Mapping[str, Any]) -> Mapping[str, Any]:
    entities = config.get("config", {}).get("served_entities", [])
    return entities[0] if entities else {}


def deploy(uc_model: str, version: str, *, endpoint_name: str | None, config: Mapping[str, Any]) -> Any:
    agents = _agents()
    entity = _served_entity(config)
    kwargs: dict[str, Any] = {
        "scale_to_zero": bool(entity.get("scale_to_zero_enabled", False)),
        "environment_vars": dict(entity.get("environment_vars", {})),
    }
    if endpoint_name or config.get("name"):
        kwargs["endpoint_name"] = endpoint_name or config["name"]
    if entity.get("workload_size"):
        kwargs["workload_size"] = entity["workload_size"]
    tags = {t["key"]: t["value"] for t in config.get("tags", []) if "key" in t and "value" in t}
    if tags:
        kwargs["tags"] = tags
    return agents.deploy(uc_model, int(version), **kwargs)


def apply_ai_gateway(endpoint_name: str, config: Mapping[str, Any], *, workspace_client: Any = None) -> bool:
    """Apply the ``ai_gateway`` block (falls back to inference tables only if rate limits are rejected)."""
    gateway = config.get("ai_gateway")
    if not gateway:
        return False
    serving = importlib.import_module("databricks.sdk.service.serving")
    if workspace_client is None:
        workspace_client = importlib.import_module("databricks.sdk").WorkspaceClient()
    kwargs: dict[str, Any] = {}
    if "usage_tracking_config" in gateway:
        kwargs["usage_tracking_config"] = serving.AiGatewayUsageTrackingConfig.from_dict(
            gateway["usage_tracking_config"]
        )
    if "inference_table_config" in gateway:
        kwargs["inference_table_config"] = serving.AiGatewayInferenceTableConfig.from_dict(
            gateway["inference_table_config"]
        )
    if "rate_limits" in gateway:
        kwargs["rate_limits"] = [serving.AiGatewayRateLimit.from_dict(r) for r in gateway["rate_limits"]]
    if "guardrails" in gateway:
        kwargs["guardrails"] = serving.AiGatewayGuardrails.from_dict(gateway["guardrails"])
    try:
        workspace_client.serving_endpoints.put_ai_gateway(endpoint_name, **kwargs)
    except Exception as exc:
        if "inference_table_config" not in kwargs:
            raise
        _log.warning(
            "deploy.ai_gateway_partial", endpoint=endpoint_name, error=f"{type(exc).__name__}: {exc}"[:300]
        )
        workspace_client.serving_endpoints.put_ai_gateway(
            endpoint_name, inference_table_config=kwargs["inference_table_config"]
        )
    return True


def promote_stage(args: argparse.Namespace) -> dict[str, Any]:
    mlflow = _mlflow()
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    from client_research_agent.databricks.mlflow_registry import (  # noqa: PLC0415
        promote_challenger,
        set_alias,
    )

    if args.model_version:
        set_alias(args.uc_model, CHALLENGER, args.model_version, client=client)
    promotion = promote_challenger(args.uc_model, client=client, keep_previous_alias=PREVIOUS_CHAMPION)
    return _deploy_version(args, promotion.new_champion) | {
        "previous_champion": promotion.previous_champion or ""
    }


def deploy_champion_stage(args: argparse.Namespace) -> dict[str, Any]:
    mlflow = _mlflow()
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient(registry_uri="databricks-uc")
    from client_research_agent.databricks.mlflow_registry import get_version_by_alias  # noqa: PLC0415

    champion = get_version_by_alias(args.uc_model, CHAMPION, client=client)
    if champion is None:
        raise ConfigurationError(f"model {args.uc_model!r} has no '{CHAMPION}' alias to deploy")
    return _deploy_version(args, champion.version)


def _deploy_version(args: argparse.Namespace, version: str) -> dict[str, Any]:
    config = load_endpoint_config(args.endpoint_config, args.uc_model)
    deployment = deploy(args.uc_model, version, endpoint_name=args.endpoint_name, config=config)
    endpoint = args.endpoint_name or config.get("name") or getattr(deployment, "endpoint_name", None)
    gateway = apply_ai_gateway(str(endpoint), config) if endpoint else False
    values = {"model_version": version, "endpoint_name": str(endpoint or ""), "ai_gateway_applied": gateway}
    set_task_values(values)
    return values


def run(args: argparse.Namespace) -> int:
    catalog, schema, _ = uc_parts(args.uc_model)
    args.catalog = args.catalog or catalog
    args.schema = args.schema or schema
    settings = settings_from_args(args)
    configure_job_observability(settings)
    if args.stage == "log":
        result = log_stage(args, settings)
    elif args.stage == "promote":
        result = promote_stage(args)
    else:
        result = deploy_champion_stage(args)
    _log.info("deploy.done", stage=args.stage, **result)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> None:
    exit_with(run_main(JOB, run, build_parser(), argv))
