from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from databricks.sdk.errors import platform
from databricks.sdk.service.jobs import RunLifeCycleState, RunResultState

from client_research_agent.databricks import mlflow_registry as reg
from client_research_agent.databricks.jobs import JobRunFailedError, JobRunner, JobRunOutcome
from client_research_agent.utils.errors import ConfigurationError, RateLimitedError, UpstreamTimeoutError
from tests.contract.fakes import FakeJobsAPI

NAME = "client_research.agent_prod.research_agent"

# --------------------------------------------------------------- registry


class _MissingAliasError(Exception):
    error_code = "RESOURCE_DOES_NOT_EXIST"


@dataclass
class FakeRegistryClient:
    aliases: dict[str, str] = field(default_factory=dict)
    registry_uri: str | None = None
    deleted: list[str] = field(default_factory=list)
    explode: bool = False

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.deleted.append(alias)
        self.aliases.pop(alias, None)

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        if self.explode:
            raise RuntimeError("tracking server unavailable")
        if alias not in self.aliases:
            raise _MissingAliasError(f"alias {alias} not found")
        version = self.aliases[alias]
        return SimpleNamespace(
            name=name,
            version=version,
            source=f"runs:/r{version}/agent",
            run_id=f"r{version}",
            aliases=[alias],
            tags={"k": "v"},
        )


@dataclass
class FakeMlflow:
    registry_uris: list[str] = field(default_factory=list)
    registered: list[dict[str, Any]] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pyfunc = SimpleNamespace(load_model=self._load)

    def set_registry_uri(self, uri: str) -> None:
        self.registry_uris.append(uri)

    def register_model(self, *, model_uri: str, name: str, tags: dict[str, str]) -> Any:
        self.registered.append({"model_uri": model_uri, "name": name, "tags": tags})
        return SimpleNamespace(name=name, version=3, source=model_uri, run_id="abc", aliases=None, tags=None)

    def MlflowClient(self, *, registry_uri: str) -> FakeRegistryClient:  # noqa: N802 - mirrors mlflow API
        return FakeRegistryClient(registry_uri=registry_uri)

    def _load(self, uri: str) -> str:
        self.loaded.append(uri)
        return f"model<{uri}>"


def test_names_and_uris() -> None:
    assert reg.uc_model_name("client_research", "agent_prod", "research_agent") == NAME
    assert reg.model_uri_for_alias(NAME, "champion") == f"models:/{NAME}@champion"
    with pytest.raises(ConfigurationError):
        reg.uc_model_name("a", "b-c", "d")
    with pytest.raises(ConfigurationError, match=r"catalog\.schema\.model"):
        reg.model_uri_for_alias("just_a_name", "champion")
    with pytest.raises(ConfigurationError, match="alias"):
        reg.model_uri_for_alias(NAME, "Champion!")


def test_register_model_targets_unity_catalog() -> None:
    mlflow = FakeMlflow()
    info = reg.register_model("runs:/abc/agent", NAME, tags={"git_sha": "deadbeef"}, mlflow_module=mlflow)
    assert mlflow.registry_uris == ["databricks-uc"]
    assert mlflow.registered == [
        {"model_uri": "runs:/abc/agent", "name": NAME, "tags": {"git_sha": "deadbeef"}}
    ]
    assert (info.version, info.aliases, dict(info.tags)) == ("3", (), {})
    client = reg.registry_client(mlflow)
    assert client.registry_uri == "databricks-uc"


def test_default_mlflow_module_is_real_mlflow() -> None:
    assert reg._mlflow(None).__name__ == "mlflow"


def test_alias_lifecycle_and_promotion() -> None:
    client = FakeRegistryClient()
    assert reg.get_version_by_alias(NAME, reg.CHAMPION, client=client) is None
    with pytest.raises(ConfigurationError, match="challenger"):
        reg.promote_challenger(NAME, client=client)

    reg.set_alias(NAME, reg.CHALLENGER, 1, client=client)
    first = reg.promote_challenger(NAME, client=client)
    assert (first.new_champion, first.previous_champion) == ("1", None)
    assert client.aliases == {"champion": "1"}

    reg.set_alias(NAME, reg.CHALLENGER, "2", client=client)
    second = reg.promote_challenger(NAME, client=client)
    assert (second.new_champion, second.previous_champion) == ("2", "1")
    assert client.aliases == {"champion": "2", "previous_champion": "1"}
    info = reg.get_version_by_alias(NAME, "champion", client=client)
    assert info is not None
    assert info.run_id == "r2"
    assert info.aliases == ("champion",)

    client.explode = True
    with pytest.raises(RuntimeError, match="unavailable"):
        reg.get_version_by_alias(NAME, "champion", client=client)


def test_load_model_by_alias() -> None:
    mlflow = FakeMlflow()
    assert reg.load_model_by_alias(NAME, mlflow_module=mlflow) == f"model<models:/{NAME}@champion>"
    assert mlflow.registry_uris == ["databricks-uc"]


# ------------------------------------------------------------------- jobs


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _runner(api: FakeJobsAPI, clock: _Clock | None = None) -> JobRunner:
    clock = clock or _Clock()

    def sleep(seconds: float) -> None:
        clock.now += seconds

    return JobRunner(api, poll_interval_seconds=5, sleep=sleep, clock=clock)


def test_run_and_wait_success() -> None:
    api = FakeJobsAPI(
        [
            (RunLifeCycleState.PENDING, None),
            (RunLifeCycleState.RUNNING, None),
            (RunLifeCycleState.TERMINATED, RunResultState.SUCCESS),
        ]
    )
    outcome = _runner(api).run_and_wait(
        7, timeout_seconds=60, job_parameters={"env": "staging"}, idempotency_token="t1"
    )
    assert outcome.succeeded
    assert outcome.run_id == 4242
    assert outcome.run_page_url
    assert outcome.run_page_url.endswith("/4242")
    assert api.run_now_calls == [
        {"job_id": 7, "job_parameters": {"env": "staging"}, "idempotency_token": "t1"}
    ]


def test_failed_run_raises_or_returns() -> None:
    api = FakeJobsAPI([(RunLifeCycleState.TERMINATED, RunResultState.FAILED)])
    with pytest.raises(JobRunFailedError, match="FAILED") as excinfo:
        _runner(api).run_and_wait(7, timeout_seconds=60)
    assert excinfo.value.outcome.result_state == "FAILED"
    outcome = _runner(api).run_and_wait(7, timeout_seconds=60, raise_on_failure=False)
    assert outcome.terminal
    assert not outcome.succeeded
    assert api.run_now_calls[0]["job_parameters"] is None


def test_timeout_cancels_run() -> None:
    api = FakeJobsAPI([(RunLifeCycleState.RUNNING, None)])
    with pytest.raises(UpstreamTimeoutError, match="still RUNNING"):
        _runner(api).wait(99, timeout_seconds=12)
    assert api.cancelled == [99]
    with pytest.raises(UpstreamTimeoutError):
        _runner(api).wait(100, timeout_seconds=0, cancel_on_timeout=False)
    assert api.cancelled == [99]


def test_sdk_errors_are_mapped() -> None:
    api = FakeJobsAPI([(RunLifeCycleState.RUNNING, None)])
    api.fail_with = [platform.TooManyRequests("slow")]
    with pytest.raises(RateLimitedError):
        _runner(api).trigger(1)
    api.fail_with = [platform.NotFound("no run")]
    with pytest.raises(Exception, match="get_run"):
        _runner(api).status(5)

    class _CancelFails(FakeJobsAPI):
        def cancel_run(self, run_id: int) -> None:
            raise platform.PermissionDenied("nope")

    with pytest.raises(ConfigurationError, match="cancel_run"):
        _runner(_CancelFails([(RunLifeCycleState.RUNNING, None)])).wait(1, timeout_seconds=0)


def test_trigger_falls_back_to_bound_run_id() -> None:
    class _Api:
        def run_now(self, **_kwargs: Any) -> Any:
            return SimpleNamespace(response=None, run_id=77)

    assert JobRunner(_Api()).trigger(1) == 77


def test_outcome_and_runner_validation() -> None:
    skipped = JobRunOutcome(
        run_id=1, life_cycle_state="SKIPPED", result_state=None, state_message=None, run_page_url=None
    )
    assert skipped.terminal
    assert not skipped.succeeded
    with pytest.raises(JobRunFailedError, match="no message"):
        skipped.raise_for_failure()
    with pytest.raises(ValueError, match="poll_interval"):
        JobRunner(object(), poll_interval_seconds=0)
