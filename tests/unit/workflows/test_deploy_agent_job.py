from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows import common, deploy_agent_job

REPO = Path(__file__).resolve().parents[3]
UC_MODEL = "client_research.agent_dev.client_research_agent"
ENDPOINT_DEV = REPO / "deployment" / "serving" / "agent_endpoint.dev.json"


class MissingAliasError(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


class FakeClient:
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.deleted: list[str] = []

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = str(version)

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        if alias not in self.aliases:
            raise MissingAliasError("alias not found")
        return types.SimpleNamespace(name=name, version=self.aliases[alias], aliases=[alias])

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.deleted.append(alias)
        self.aliases.pop(alias, None)


def resource(kind: str) -> Any:
    def make(**kwargs: Any) -> tuple[str, dict[str, Any]]:
        return (kind, kwargs)

    return make


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    state = types.SimpleNamespace(calls=[], client=FakeClient({}), logged={})
    mlflow = types.ModuleType("mlflow")

    @contextmanager
    def start_run(**kwargs: Any) -> Any:
        state.calls.append(("start_run", kwargs))
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="run-1"))

    def log_model(**kwargs: Any) -> Any:
        state.logged = kwargs
        return types.SimpleNamespace(registered_model_version=7, model_uri="runs:/run-1/agent")

    mlflow.set_registry_uri = lambda uri: state.calls.append(("registry", uri))  # type: ignore[attr-defined]
    mlflow.set_experiment = lambda name: state.calls.append(("experiment", name))  # type: ignore[attr-defined]
    mlflow.start_run = start_run  # type: ignore[attr-defined]
    mlflow.log_params = lambda params: state.calls.append(("params", params))  # type: ignore[attr-defined]
    mlflow.pyfunc = types.SimpleNamespace(log_model=log_model)  # type: ignore[attr-defined]
    mlflow.MlflowClient = lambda registry_uri: state.client  # type: ignore[attr-defined]
    resources = types.ModuleType("mlflow.models.resources")
    for name in (
        "DatabricksServingEndpoint",
        "DatabricksVectorSearchIndex",
        "DatabricksSQLWarehouse",
        "DatabricksTable",
    ):
        setattr(resources, name, resource(name))
    models = types.ModuleType("mlflow.models")
    models.set_model = lambda model: state.calls.append(("set_model", model))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", models)
    monkeypatch.setitem(sys.modules, "mlflow.models.resources", resources)
    monkeypatch.setattr(common, "configure_observability", lambda *a, **k: None)
    return state


@pytest.fixture
def task_values(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        deploy_agent_job, "set_task_values", lambda values, **_k: captured.append(dict(values))
    )
    return captured


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    deployed: list[tuple[Any, ...]] = []
    agents = types.ModuleType("databricks.agents")
    agents.deploy = lambda name, version, **kw: (
        deployed.append((name, version, kw))
        or types.SimpleNamespace(  # type: ignore[attr-defined]
            endpoint_name=kw.get("endpoint_name")
        )
    )
    monkeypatch.setitem(sys.modules, "databricks.agents", agents)
    return deployed


@pytest.fixture
def gateway_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    serving = types.SimpleNamespace(put_ai_gateway=lambda name, **kw: calls.append((name, kw)))
    from databricks import sdk

    monkeypatch.setattr(sdk, "WorkspaceClient", lambda: types.SimpleNamespace(serving_endpoints=serving))
    return calls


def test_log_stage_registers_models_from_code(
    fake_mlflow: types.SimpleNamespace, task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from client_research_agent.databricks import auth
    from tests.support.databricks_fakes import FakeWorkspace

    monkeypatch.setattr(auth, "build_workspace_client", lambda settings: FakeWorkspace())
    deploy_agent_job.main(
        ["--environment", "dev", "--stage", "log", "--uc-model", UC_MODEL, "--experiment", "/Shared/cra",
         "--vs-endpoint", "vs", "--vs-index", "client_research.agent_dev.chunks_index"]
    )  # fmt: skip
    logged = fake_mlflow.logged
    assert logged["name"] == "agent"
    assert logged["python_model"].endswith("agent_model.py")
    assert logged["registered_model_name"] == UC_MODEL
    assert Path(logged["code_paths"][0]).name == "client_research_agent"
    kinds = [kind for kind, _ in logged["resources"]]
    assert kinds.count("DatabricksServingEndpoint") == 3
    assert {"DatabricksVectorSearchIndex", "DatabricksSQLWarehouse", "DatabricksTable"} <= set(kinds)
    assert fake_mlflow.client.aliases == {"challenger": "7"}
    assert task_values[-1]["model_version"] == "7"
    assert ("experiment", "/Shared/cra") in fake_mlflow.calls


def test_log_stage_without_version_fails(
    fake_mlflow: types.SimpleNamespace, task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    from client_research_agent.databricks import auth

    def no_auth(settings: Any) -> Any:
        raise ConfigurationError("no credentials")

    monkeypatch.setattr(auth, "build_workspace_client", no_auth)
    fake_mlflow_module = sys.modules["mlflow"]
    fake_mlflow_module.pyfunc = types.SimpleNamespace(log_model=lambda **kw: types.SimpleNamespace())  # type: ignore[attr-defined]
    with pytest.raises(SystemExit):
        deploy_agent_job.main(["--environment", "dev", "--stage", "log", "--uc-model", UC_MODEL])


def test_promote_moves_aliases_deploys_and_applies_gateway(
    fake_mlflow: types.SimpleNamespace,
    task_values: list[dict[str, Any]],
    fake_agents: list[tuple[Any, ...]],
    gateway_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    fake_mlflow.client.aliases.update({"champion": "5"})
    deploy_agent_job.main(
        ["--environment", "dev", "--stage", "promote", "--uc-model", UC_MODEL, "--model-version", "7",
         "--endpoint-name", "cra-agent-dev", "--endpoint-config", str(ENDPOINT_DEV)]
    )  # fmt: skip
    aliases = fake_mlflow.client.aliases
    assert aliases["champion"] == "7"
    assert aliases["previous_champion"] == "5"
    assert "challenger" not in aliases
    name, version, kwargs = fake_agents[0]
    assert (name, version) == (UC_MODEL, 7)
    assert kwargs["endpoint_name"] == "cra-agent-dev"
    assert kwargs["scale_to_zero"] is True
    assert kwargs["environment_vars"]["CRA_ENVIRONMENT"] == "dev"
    assert kwargs["workload_size"] == "Small"
    assert kwargs["tags"]["app"] == "client-research-agent"
    endpoint, gateway = gateway_calls[0]
    assert endpoint == "cra-agent-dev"
    assert set(gateway) == {"usage_tracking_config", "inference_table_config", "rate_limits"}
    assert task_values[-1]["ai_gateway_applied"] is True


def test_deploy_champion_and_missing_champion(
    fake_mlflow: types.SimpleNamespace,
    task_values: list[dict[str, Any]],
    fake_agents: list[tuple[Any, ...]],
    gateway_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    with pytest.raises(SystemExit):
        deploy_agent_job.main(["--environment", "dev", "--stage", "deploy-champion", "--uc-model", UC_MODEL])
    fake_mlflow.client.aliases["champion"] = "4"
    deploy_agent_job.main(["--environment", "dev", "--stage", "deploy-champion", "--uc-model", UC_MODEL])
    assert fake_agents[-1][:2] == (UC_MODEL, 4)
    assert task_values[-1]["endpoint_name"] == ""
    assert gateway_calls == []


def test_agents_package_missing_raises_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "databricks.agents", None)
    with pytest.raises(ConfigurationError, match="databricks-agents is not installed"):
        deploy_agent_job.deploy(UC_MODEL, "1", endpoint_name=None, config={})


def test_endpoint_config_substitutes_entity_name(tmp_path: Path) -> None:
    config = deploy_agent_job.load_endpoint_config(
        str(REPO / "deployment/serving/agent_endpoint.prod.json"), "a.b.c"
    )
    assert all(e["entity_name"] == "a.b.c" for e in config["config"]["served_entities"])
    env = config["config"]["served_entities"][0]["environment_vars"]
    assert env["CRA_CRAWLER__CONTACT_EMAIL"].startswith("{{secrets/")
    assert deploy_agent_job.load_endpoint_config(None, "a.b.c") == {}
    with pytest.raises(ConfigurationError, match=r"catalog\.schema\.model"):
        deploy_agent_job.uc_parts("model")


def test_ai_gateway_falls_back_to_inference_tables() -> None:
    calls: list[dict[str, Any]] = []

    def put(name: str, **kwargs: Any) -> None:
        calls.append(kwargs)
        if "rate_limits" in kwargs:
            raise RuntimeError("agent endpoints only support inference tables")

    workspace = types.SimpleNamespace(serving_endpoints=types.SimpleNamespace(put_ai_gateway=put))
    config = json.loads((REPO / "deployment/serving/agent_endpoint.dev.json").read_text(encoding="utf-8"))
    assert deploy_agent_job.apply_ai_gateway("ep", config, workspace_client=workspace)
    assert list(calls[-1]) == ["inference_table_config"]
    assert not deploy_agent_job.apply_ai_gateway("ep", {}, workspace_client=workspace)

    def always_fail(name: str, **kwargs: Any) -> None:
        raise RuntimeError("denied")

    failing = types.SimpleNamespace(serving_endpoints=types.SimpleNamespace(put_ai_gateway=always_fail))
    guard_only = {"ai_gateway": {"guardrails": {"input": {"pii": {"behavior": "BLOCK"}}}}}
    with pytest.raises(RuntimeError):
        deploy_agent_job.apply_ai_gateway("ep", guard_only, workspace_client=failing)


def test_pip_requirements_from_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    monkeypatch.setattr(
        metadata,
        "requires",
        lambda dist: ["pydantic>=2.7", 'mlflow>=2.17; extra == "databricks"', 'pytest>=8; extra == "dev"'],
    )
    assert deploy_agent_job.pip_requirements() == ["pydantic>=2.7", "mlflow>=2.17"]

    def missing(dist: str) -> Any:
        raise metadata.PackageNotFoundError(dist)

    monkeypatch.setattr(metadata, "requires", missing)
    assert deploy_agent_job.pip_requirements() == []
