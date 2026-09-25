"""Create the MLflow experiment and Unity Catalog registered model for the agent.

Idempotent: existing objects are left in place and only their tags are
reconciled. Optionally registers a logged model (``--register-run-id``) as a
new UC model version and points an alias (default ``challenger``) at it.

    python infrastructure/mlflow/setup_mlflow.py --environment staging
    python infrastructure/mlflow/setup_mlflow.py --environment staging \
        --register-run-id "$RUN_ID" --artifact-path agent --alias challenger

The ``cra_agent_deploy`` job performs the same registration automatically; this
script exists for bootstrap and break-glass use.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALIAS = re.compile(r"^[a-z][a-z0-9_]*$")
MODEL_DESCRIPTION = (
    "MLflow ResponsesAgent that researches a company from public sources (SEC EDGAR, investor "
    "relations, newsroom) and returns a scored, cited client brief."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision MLflow experiment and UC registered model.")
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--catalog", default="client_research")
    parser.add_argument("--schema", help="Defaults to agent_<environment>.")
    parser.add_argument("--model-name", default="client_research_agent")
    parser.add_argument("--experiment", help="Defaults to /Shared/client-research-agent-<environment>.")
    parser.add_argument("--profile", help="Databricks CLI profile (sets DATABRICKS_CONFIG_PROFILE).")
    parser.add_argument("--register-run-id", help="Register runs:/<id>/<artifact-path> as a new version.")
    parser.add_argument("--artifact-path", default="agent")
    parser.add_argument("--alias", default="challenger", help="Alias to point at the registered version.")
    return parser.parse_args(argv)


def log(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def ensure_experiment(client: MlflowClient, name: str, environment: str) -> str:
    tags = {"app": "client-research-agent", "environment": environment}
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = client.create_experiment(name, tags=tags)
        log(f"created experiment {name} ({experiment_id})")
        return experiment_id
    for key, value in tags.items():
        if experiment.tags.get(key) != value:
            client.set_experiment_tag(experiment.experiment_id, key, value)
    log(f"experiment {name} exists ({experiment.experiment_id})")
    return str(experiment.experiment_id)


def ensure_registered_model(client: MlflowClient, full_name: str, environment: str) -> None:
    tags = {"app": "client-research-agent", "environment": environment, "flavor": "ResponsesAgent"}
    try:
        model = client.get_registered_model(full_name)
    except MlflowException as exc:
        if exc.error_code not in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
            raise
        client.create_registered_model(full_name, tags=tags, description=MODEL_DESCRIPTION)
        log(f"created registered model {full_name}")
        return
    for key, value in tags.items():
        if model.tags.get(key) != value:
            client.set_registered_model_tag(full_name, key, value)
    log(f"registered model {full_name} exists")


def register_version(
    client: MlflowClient, full_name: str, run_id: str, artifact_path: str, alias: str
) -> str:
    if not _ALIAS.match(alias) or alias == "latest":
        raise SystemExit(f"invalid alias {alias!r}")
    version = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", full_name)
    client.set_registered_model_alias(full_name, alias, version.version)
    log(f"registered {full_name} version {version.version} as @{alias}")
    return str(version.version)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = args.schema or f"agent_{args.environment}"
    for label, value in (("catalog", args.catalog), ("schema", schema), ("model-name", args.model_name)):
        if not _IDENTIFIER.match(value):
            raise SystemExit(f"invalid {label}: {value!r}")
    if args.profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = args.profile

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(tracking_uri="databricks", registry_uri="databricks-uc")

    experiment = args.experiment or f"/Shared/client-research-agent-{args.environment}"
    full_name = f"{args.catalog}.{schema}.{args.model_name}"
    ensure_experiment(client, experiment, args.environment)
    ensure_registered_model(client, full_name, args.environment)
    if args.register_run_id:
        register_version(client, full_name, args.register_run_id, args.artifact_path, args.alias)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
