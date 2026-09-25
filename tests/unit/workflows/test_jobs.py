from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.agent.factory import build_runtime
from client_research_agent.config.settings import AppSettings, Environment
from client_research_agent.orchestration import ClientResearchOrchestrator
from client_research_agent.utils.errors import ConfigurationError
from client_research_agent.workflows import brief_job, common, evaluation_job, ingest_job
from tests.support.databricks_fakes import FakeExecutor, FakeWorkspace
from tests.support.world import ANALYST, northwind_fetcher, northwind_request
from tests.unit.orchestration.test_review_state_rendering import make_brief


@pytest.fixture
def patched_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Jobs build their runtime with the offline Northwind world and a temp var dir."""
    built: dict[str, Any] = {}

    def factory(settings: AppSettings, **kwargs: Any) -> Any:
        kwargs.setdefault("fetcher", northwind_fetcher())
        kwargs.setdefault("llm", None)
        runtime = build_runtime(settings, var_dir=tmp_path / "var", **kwargs)
        built["runtime"] = runtime
        return runtime

    for module in (ingest_job, brief_job, evaluation_job):
        monkeypatch.setattr(module, "build_runtime", factory)
    monkeypatch.setattr(common, "configure_observability", lambda *a, **k: None)
    return built


@pytest.fixture
def task_values(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    for module in (ingest_job, brief_job, evaluation_job):
        monkeypatch.setattr(module, "set_task_values", lambda values, **_k: captured.append(dict(values)))
    return captured


# ------------------------------------------------------------------ common
def test_empty_parameters_mean_not_given() -> None:
    assert common.empty_to_none("  ") is None
    assert common.empty_to_none(None) is None
    assert common.empty_to_none(" x ") == "x"
    assert common.optional_float("") is None
    assert common.optional_float("0.9") == 0.9
    parser = common.base_parser("t", experiment=True)
    args = parser.parse_args(
        ["--environment", "local", "--catalog", "", "--vs-endpoint", "vs", "--experiment", "/e"]
    )
    assert args.catalog is None
    assert args.vs_endpoint == "vs"


def test_settings_from_args_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        environment="dev",
        catalog="cat",
        schema="sch",
        warehouse_id="wh",
        vs_endpoint="vs-1",
        experiment="/Shared/x",
        contact_email="ops@corp.test",
    )
    settings = common.settings_from_args(args, environ={"DATABRICKS_HOST": "https://h.example"})
    assert settings.environment is Environment.DEV
    assert settings.databricks.table("t") == "cat.sch.t"
    assert settings.databricks.warehouse_id == "wh"
    assert settings.databricks.host == "https://h.example"
    assert settings.vector_search.endpoint_name == "vs-1"
    assert settings.observability.mlflow_experiment == "/Shared/x"
    assert settings.crawler.contact_email == "ops@corp.test"
    prod = argparse.Namespace(environment="prod")
    with pytest.raises(ConfigurationError, match="invalid job configuration"):
        common.settings_from_args(prod, environ={})


def test_prod_requires_real_contact_email() -> None:
    prod = argparse.Namespace(environment="prod", contact_email=None)
    with pytest.raises(ConfigurationError, match="SEC contact email"):
        common.settings_from_args(prod, environ={"DATABRICKS_HOST": "https://h"})
    ok = argparse.Namespace(environment="prod", contact_email="research-ops@corp.test")
    assert common.settings_from_args(ok, environ={"DATABRICKS_HOST": "https://h"}).crawler.contact_email


def test_host_resolution_on_databricks_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = types.ModuleType("databricks.sdk")
    sdk.WorkspaceClient = lambda: types.SimpleNamespace(config=types.SimpleNamespace(host="https://dbr.host"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databricks.sdk", sdk)
    assert common.resolve_databricks_host({"DATABRICKS_RUNTIME_VERSION": "16.4"}) == "https://dbr.host"
    assert common.resolve_databricks_host({}) is None

    def broken() -> Any:
        raise ValueError("no auth")

    sdk.WorkspaceClient = broken  # type: ignore[attr-defined]
    assert common.resolve_databricks_host({"DATABRICKS_RUNTIME_VERSION": "16.4"}) is None


def test_task_values_use_dbutils_on_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    stored: dict[str, Any] = {}
    runtime_module = types.ModuleType("databricks.sdk.runtime")
    runtime_module.dbutils = types.SimpleNamespace(  # type: ignore[attr-defined]
        jobs=types.SimpleNamespace(
            taskValues=types.SimpleNamespace(set=lambda key, value: stored.update({key: value}))
        )
    )
    monkeypatch.setitem(sys.modules, "databricks.sdk.runtime", runtime_module)
    common.set_task_values(
        {"run_id": "r1", "chunks_written": 3}, environ={"DATABRICKS_RUNTIME_VERSION": "16.4"}
    )
    assert stored == {"run_id": "r1", "chunks_written": 3}
    common.set_task_values({"ignored": 1}, environ={})
    assert "ignored" not in stored


def test_table_names_and_executor() -> None:
    assert common.split_table_name("cat.sch.tbl") == "`cat`.`sch`.`tbl`"
    with pytest.raises(ConfigurationError):
        common.split_table_name("tbl")
    with pytest.raises(ConfigurationError, match="not available"):
        common.statement_executor(common.settings_from_args(argparse.Namespace(environment="local")), None)
    executor = common.statement_executor(
        common.settings_from_args(argparse.Namespace(environment="dev"), environ={}), FakeWorkspace()
    )
    assert executor.warehouse_id == "wh-dev"


def test_run_main_maps_errors_to_exit_codes() -> None:
    parser = argparse.ArgumentParser()

    def agent_error(_args: argparse.Namespace) -> int:
        raise ConfigurationError("bad")

    def value_error(_args: argparse.Namespace) -> int:
        raise ValueError("bad")

    assert common.run_main("j", agent_error, parser, []) == common.EXIT_FAILURE
    assert common.run_main("j", value_error, parser, []) == common.EXIT_FAILURE
    assert common.run_main("j", lambda _a: 0, parser, []) == common.EXIT_OK
    common.exit_with(0)
    with pytest.raises(SystemExit) as exc:
        common.exit_with(3)
    assert exc.value.code == 3
    principal = common.job_principal(
        common.settings_from_args(argparse.Namespace(environment="local")), "brief"
    )
    assert principal.id == "job:brief:local"


# ------------------------------------------------------------------ ingest
def test_ingest_single_company(patched_runtime: dict[str, Any], task_values: list[dict[str, Any]]) -> None:
    ingest_job.main(
        ["--environment", "local", "--company", "Northwind Industries", "--domain", "northwind.example",
         "--ticker", "NWND", "--catalog", "", "--vs-endpoint", "", "--sync-index"]
    )  # fmt: skip
    values = task_values[-1]
    assert values["documents_ingested"] > 0
    assert values["chunks_written"] > 0
    assert values["run_id"].startswith("northwind-industries-")
    assert values["index_synced"] is False  # the in-memory index has no sync
    events = [r.event_type for r in patched_runtime["runtime"].audit.records()]
    assert "ingest.company" in events


def test_ingest_watchlist(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = FakeExecutor(
        {
            "read_watchlist": [
                {
                    "company_name": "Northwind Industries",
                    "domain": "northwind.example",
                    "ticker": "NWND",
                    "cik": "",
                    "industry": None,
                },
                {"company_name": "", "domain": None},
            ]
        }
    )
    monkeypatch.setattr(ingest_job, "statement_executor", lambda *_a: executor)
    ingest_job.main(["--environment", "local", "--watchlist-table", "cat.sch.companies_watchlist"])
    assert task_values[-1]["companies"] == 1
    ops = [op for _s, _p, op in executor.calls]
    assert ops == ["read_watchlist", "mark_ingested"]
    assert executor.calls[1][1] == {"company": "Northwind Industries"}


def test_ingest_all_failures_exit_non_zero(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        def gather(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("down")

    monkeypatch.setattr(ingest_job, "EvidenceGatherer", lambda _rt: Boom())
    with pytest.raises(SystemExit) as exc:
        ingest_job.main(["--environment", "local", "--company", "Northwind Industries"])
    assert exc.value.code == 1
    assert task_values[-1]["companies_failed"] == 1


def test_sync_index_only(patched_runtime: dict[str, Any], task_values: list[dict[str, Any]]) -> None:
    ingest_job.main(["--environment", "local", "--sync-index-only"])
    assert task_values[-1]["index_synced"] is False

    synced: list[bool] = []
    runtime = types.SimpleNamespace(vector_index=types.SimpleNamespace(sync=lambda: synced.append(True)))
    assert ingest_job.sync_index(runtime) is True
    assert synced == [True]


def test_ingest_requires_a_mode(patched_runtime: dict[str, Any]) -> None:
    with pytest.raises(SystemExit):
        ingest_job.main(["--environment", "local"])
    args = ingest_job.build_parser().parse_args(["--environment", "local", "--company", ""])
    with pytest.raises(ConfigurationError, match="one of"):
        ingest_job.run(args)
    both = ingest_job.build_parser().parse_args(
        ["--environment", "local", "--company", "Acme", "--watchlist-table", "c.s.watchlist"]
    )
    with pytest.raises(ConfigurationError, match="one of"):
        ingest_job.run(both)
    blank_extra = ingest_job.build_parser().parse_args(
        ["--environment", "local", "--sync-index-only", "--company", "", "--watchlist-table", ""]
    )
    ingest_job.validate_mode(blank_extra)


# ------------------------------------------------------------------ brief
def test_brief_job_sets_task_values(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]]
) -> None:
    brief_job.main(
        ["--environment", "local", "--run-id", "run-from-ingest", "--company", "Northwind Industries",
         "--domain", "northwind.example", "--ticker", "", "--requested-by", "", "--experiment", "/Shared/x"]
    )  # fmt: skip
    values = task_values[-1]
    assert values["brief_id"] == "run-from-ingest"
    assert values["verdict"] in {"good_fit", "potential_fit", "not_enough_evidence"}
    assert 0.0 <= values["citation_coverage"] <= 1.0
    assert values["lineage_rows"] == 0  # no workspace locally
    assert patched_runtime["runtime"].brief_repository.get("run-from-ingest") is not None


def test_brief_job_requires_company() -> None:
    with pytest.raises(SystemExit) as exc:
        brief_job.main(["--environment", "local", "--company", ""])
    assert exc.value.code == 1


def test_brief_job_fails_when_not_persisted(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        def save(self, brief: Any) -> str:
            raise OSError("warehouse down")

        def get(self, run_id: str) -> None:
            return None

    original = brief_job.build_runtime
    monkeypatch.setattr(
        brief_job, "build_runtime", lambda s, **k: original(s, brief_repository=Broken(), **k)
    )
    with pytest.raises(SystemExit):
        brief_job.main(["--environment", "local", "--company", "Northwind Industries"])


def test_write_lineage_merges_rows(monkeypatch: pytest.MonkeyPatch, settings: AppSettings) -> None:
    executor = FakeExecutor()
    monkeypatch.setattr(brief_job, "statement_executor", lambda *_a: executor)
    runtime = types.SimpleNamespace(settings=settings, workspace_client=object())
    rows = [{"lineage_id": "l1", "run_id": "r", "supported": True}]
    assert brief_job.write_lineage(runtime, rows) == 1  # type: ignore[arg-type]
    statement, params, op = executor.calls[0]
    assert op == "merge_lineage"
    assert "MERGE INTO `client_research`.`agent_local`.`lineage`" in statement
    assert params["lineage_id"] == "l1"
    assert params["url"] is None
    assert brief_job.write_lineage(runtime, []) == 0  # type: ignore[arg-type]


# ------------------------------------------------------------------ evaluate
def test_evaluate_brief_mode(
    patched_runtime: dict[str, Any],
    task_values: list[dict[str, Any]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings_runtime = build_runtime(
        common.settings_from_args(argparse.Namespace(environment="local")),
        var_dir=tmp_path / "var",
        fetcher=northwind_fetcher(),
        llm=None,
    )
    result = ClientResearchOrchestrator(settings_runtime).run(
        northwind_request(), principal=ANALYST, run_id="brief-1"
    )
    assert result.persisted
    evaluation_job.main(
        ["--environment", "local", "--mode", "brief", "--brief-id", "brief-1",
         "--min-citation-coverage", "0.9",
         "--fail-on-gate"]
    )  # fmt: skip
    assert task_values[-1]["gate_passed"] is True
    out = capsys.readouterr().out
    assert json.loads(out[out.index('{\n  "gate_passed"') :])["gate_passed"] is True


def test_evaluate_brief_mode_missing_brief(patched_runtime: dict[str, Any]) -> None:
    with pytest.raises(SystemExit) as exc:
        evaluation_job.main(
            [
                "--environment",
                "local",
                "--mode",
                "brief",
                "--brief-id",
                "nope",
                "--min-citation-coverage",
                "0.9",
            ]
        )
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as missing:
        evaluation_job.main(["--environment", "local", "--mode", "brief", "--min-citation-coverage", "0.9"])
    assert missing.value.code == 1


def test_evaluate_dataset_golden_writes_results(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = FakeExecutor()
    monkeypatch.setattr(evaluation_job, "statement_executor", lambda *_a: executor)
    evaluation_job.main(
        ["--environment", "local", "--mode", "dataset", "--eval-table", "golden", "--results-table",
         "cat.sch.eval_results", "--min-pass-rate", "0.8", "--min-citation-coverage", "0.9", "--fail-on-gate"]
    )  # fmt: skip
    assert task_values[-1]["gate_passed"] is True
    assert task_values[-1]["examples"] >= 5
    params = [p for _s, p, op in executor.calls if op == "insert_eval_result"]
    assert {p["metric_name"] for p in params} >= {"pass_rate", "citation_coverage", "verdict_accuracy"}
    assert all(p["model_uri"] == "orchestrator" and p["environment"] == "local" for p in params)


def test_evaluate_dataset_from_table_with_model(
    patched_runtime: dict[str, Any], task_values: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    brief = make_brief(run_id="m1")
    rows = [
        {
            "eval_id": "t1",
            "company": "Acme Corp",
            "request": "{}",
            "expected_verdict": "potential_fit",
            "active": "true",
        }
    ]
    executor = FakeExecutor({"read_eval_set": rows})
    monkeypatch.setattr(evaluation_job, "statement_executor", lambda *_a: executor)

    class Model:
        def predict(self, request: Any) -> Any:
            return {"custom_outputs": {"brief_json": brief.model_dump_json()}}

    monkeypatch.setattr(evaluation_job, "load_model", lambda uri: Model())
    with pytest.raises(SystemExit) as exc:
        evaluation_job.main(
            ["--environment", "local", "--mode", "dataset", "--eval-table", "cat.sch.eval_set", "--model-uri",
             "models:/cat.sch.agent/3", "--min-citation-coverage", "0.9", "--fail-on-gate"]
        )  # fmt: skip
    assert exc.value.code == 1  # verdict label mismatch fails the gate
    assert task_values[-1]["gate_passed"] is False


def test_evaluate_dataset_errors_and_helpers(
    patched_runtime: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        evaluation_job.main(["--environment", "local", "--mode", "dataset", "--min-citation-coverage", "0.9"])
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluation_job.main(
            [
                "--environment",
                "local",
                "--mode",
                "dataset",
                "--eval-table",
                str(empty),
                "--min-citation-coverage",
                "0.9",
            ]
        )
    assert evaluation_job.model_version_of("models:/a.b.c/12") == "12"
    assert evaluation_job.model_version_of("models:/a.b.c@champion") is None
    assert evaluation_job.model_version_of(None) is None
    assert evaluation_job.model_version_of("runs:/abc/agent") is None


def test_load_model_uses_uc_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    fake = types.ModuleType("mlflow")
    fake.set_registry_uri = calls.append  # type: ignore[attr-defined]
    fake.pyfunc = types.SimpleNamespace(load_model=lambda uri: f"model:{uri}")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    assert (
        evaluation_job.load_model("models:/cat.sch.agent@champion") == "model:models:/cat.sch.agent@champion"
    )
    assert calls == ["databricks-uc"]


def test_orchestrator_producer_runs_live_examples(settings: AppSettings, tmp_path: Path) -> None:
    from client_research_agent.evaluation.dataset import EvalExample

    runtime = build_runtime(settings, var_dir=tmp_path, fetcher=northwind_fetcher(), llm=None)
    produce = evaluation_job.orchestrator_producer(runtime)
    live = EvalExample.model_validate(
        {"eval_id": "live", "inputs": {"company_name": "Northwind Industries", "domain": "northwind.example"}}
    )
    assert produce(live).company == "Northwind Industries"
