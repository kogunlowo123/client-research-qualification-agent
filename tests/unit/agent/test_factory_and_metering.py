from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.agent.factory import (
    LOCAL_EMBEDDING_DIMENSION,
    build_runtime,
    databricks_host_configured,
    default_var_dir,
    research_entity_extractor,
    resolve_warehouse_id,
)
from client_research_agent.agent.metering import (
    LLMBudgetExhaustedError,
    MeteredLLMClient,
    RunAccounting,
    current_accounting,
    estimate_prompt_tokens,
    run_accounting,
)
from client_research_agent.config.settings import AppSettings, Environment, build_settings
from client_research_agent.databricks.audit_sink import FanOutAuditLogger
from client_research_agent.databricks.model_serving import FallbackLLMClient
from client_research_agent.databricks.unity_catalog import (
    DeltaBriefRepository,
    DeltaDocumentStore,
    StatementExecutor,
)
from client_research_agent.databricks.vector_search import DatabricksVectorIndex
from client_research_agent.governance.audit import AuditLogger
from client_research_agent.orchestration import ClientResearchOrchestrator, StepName
from client_research_agent.research.fetcher import PolicyEnforcingFetcher
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.security.rate_limiter import RunBudget
from client_research_agent.services.local import (
    InMemoryDocumentStore,
    InMemoryVectorIndex,
    JsonlBriefRepository,
)
from client_research_agent.services.ports import ChatMessage
from client_research_agent.utils.errors import ConfigurationError
from tests.contract.fakes import FakeVectorSearchIndex
from tests.support.databricks_fakes import FakeWarehouse, FakeWorkspace
from tests.support.doubles import ScriptedLLM
from tests.support.world import ANALYST, local_runtime, northwind_fetcher, northwind_request


# ------------------------------------------------------------------ metering
def test_metered_client_passes_through_without_accounting() -> None:
    inner = ScriptedLLM(default="hello")
    metered = MeteredLLMClient(inner)
    assert metered.inner is inner
    assert metered.model_name == "scripted-llm"
    assert metered.complete([ChatMessage("user", "hi")]).content == "hello"
    assert current_accounting() is None


def test_metered_client_records_costs_and_enforces_budget() -> None:
    metered = MeteredLLMClient(ScriptedLLM(default="x" * 400))
    accounting = RunAccounting(budget=RunBudget(max_tokens=100_000, max_llm_calls=2))
    with run_accounting(accounting):
        accounting.step = "qualification"
        metered.complete([ChatMessage("user", "a" * 400)], max_tokens=100)
        metered.complete([ChatMessage("user", "b")], max_tokens=100)
        with pytest.raises(LLMBudgetExhaustedError):
            metered.complete([ChatMessage("user", "c")], max_tokens=100)
        with pytest.raises(LLMBudgetExhaustedError, match="llm_budget"):
            metered.complete([ChatMessage("user", "d")])
    assert accounting.exhausted
    assert accounting.costs.by_step()["qualification"].calls == 2
    assert estimate_prompt_tokens([ChatMessage("user", "abcd" * 10)]) == 10


def test_charge_overrun_keeps_answer_but_blocks_next_call() -> None:
    metered = MeteredLLMClient(ScriptedLLM(default="y" * 4000))
    accounting = RunAccounting(budget=RunBudget(max_tokens=600))
    with run_accounting(accounting):
        assert metered.complete([ChatMessage("user", "q")], max_tokens=10).content
        assert accounting.exhausted
        with pytest.raises(LLMBudgetExhaustedError):
            metered.complete([ChatMessage("user", "q")], max_tokens=10)


# ------------------------------------------------------------------ local wiring
def test_local_runtime_defaults(settings: AppSettings, tmp_path: Path) -> None:
    runtime = build_runtime(settings, var_dir=tmp_path, environ={})
    assert runtime.llm is None
    assert isinstance(runtime.embedder, HashingEmbeddingClient)
    assert runtime.embedder.dimension == LOCAL_EMBEDDING_DIMENSION
    assert isinstance(runtime.vector_index, InMemoryVectorIndex)
    assert isinstance(runtime.document_store, InMemoryDocumentStore)
    assert isinstance(runtime.brief_repository, JsonlBriefRepository)
    assert isinstance(runtime.fetcher, PolicyEnforcingFetcher)
    assert isinstance(runtime.audit, AuditLogger)
    assert runtime.audit.path == tmp_path / "audit" / "audit.jsonl"
    assert runtime.refresh_ingestion is not None
    assert runtime.track_runs is False
    assert runtime.environment is Environment.LOCAL
    assert runtime.review_policy.min_citation_coverage == settings.guardrails.review_min_citation_coverage
    assert runtime.model_versions()["llm"] == "none"
    runtime.close()
    runtime.close()


def test_runtime_close_swallows_closer_errors(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)

    def boom() -> None:
        raise RuntimeError("close failed")

    runtime._closers.append(boom)
    runtime.close()
    assert runtime._closers == []


def test_local_llm_when_host_configured(settings: AppSettings, tmp_path: Path) -> None:
    runtime = build_runtime(
        settings, var_dir=tmp_path, workspace_client=FakeWorkspace(), environ={"DATABRICKS_HOST": "https://h"}
    )
    assert runtime.llm is not None
    assert isinstance(runtime.llm.inner, FallbackLLMClient)
    assert runtime.model_versions()["llm"] == settings.serving.chat_endpoint


def test_local_llm_failure_degrades_to_offline(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from client_research_agent.databricks import auth

    def fail(*_a: Any, **_k: Any) -> Any:
        raise ConfigurationError("no credentials")

    monkeypatch.setattr(auth, "build_workspace_client", fail)
    runtime = build_runtime(settings, var_dir=tmp_path, environ={"DATABRICKS_HOST": "https://h"})
    assert runtime.llm is None


def test_single_endpoint_llm(settings: AppSettings, tmp_path: Path) -> None:
    from client_research_agent.agent.factory import build_databricks_llm
    from client_research_agent.databricks.model_serving import DatabricksChatClient

    same = settings.model_copy(
        update={
            "serving": settings.serving.model_copy(
                update={"fallback_chat_endpoint": settings.serving.chat_endpoint}
            )
        }
    )
    assert isinstance(build_databricks_llm(same, FakeWorkspace()), DatabricksChatClient)


def test_helpers(settings: AppSettings) -> None:
    extract = research_entity_extractor()
    entities = extract(
        "Acme Corp appointed Jane Smith as Chief Information Officer to lead Databricks adoption."
    )
    assert "Databricks" in entities
    assert default_var_dir(settings, {"CRA_VAR_DIR": "/data/cra"}) == Path("/data/cra")
    assert default_var_dir(settings, {}) == Path("var")
    assert default_var_dir(build_settings("dev"), {}).name == "client-research-agent"
    assert databricks_host_configured(settings, {"DATABRICKS_HOST": "x"})
    assert not databricks_host_configured(settings, {})


def test_resolve_warehouse_id_order() -> None:
    dev = build_settings("dev")
    configured = dev.model_copy(
        update={"databricks": dev.databricks.model_copy(update={"warehouse_id": "wh-set"})}
    )
    assert resolve_warehouse_id(configured, FakeWorkspace(), {}) == "wh-set"
    assert resolve_warehouse_id(dev, FakeWorkspace(), {"DATABRICKS_WAREHOUSE_ID": "wh-env"}) == "wh-env"
    assert resolve_warehouse_id(dev, FakeWorkspace(), {}) == "wh-dev"
    other = FakeWorkspace(
        warehouse_list=[FakeWarehouse("wh-x", "shared", False), FakeWarehouse("wh-s", "serverless")]
    )
    assert resolve_warehouse_id(dev, other, {}) == "wh-s"
    with pytest.raises(ConfigurationError, match="no SQL warehouse"):
        resolve_warehouse_id(dev, FakeWorkspace(warehouse_list=[FakeWarehouse("wh-c", "classic", False)]), {})

    class Broken:
        warehouses = types.SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("403")))

    with pytest.raises(ConfigurationError, match="cannot list"):
        resolve_warehouse_id(dev, Broken(), {})


# ------------------------------------------------------------------ Databricks wiring
def databricks_runtime(tmp_path: Path, **overrides: Any) -> tuple[Any, FakeWorkspace]:
    settings = build_settings("dev")
    workspace = FakeWorkspace()
    sql = workspace.statement_execution
    writer = DeltaDocumentStore(
        StatementExecutor(sql, "wh-dev"), catalog="client_research", schema="agent_dev"
    )
    overrides.setdefault(
        "vector_index", DatabricksVectorIndex(FakeVectorSearchIndex(sql), writer, trigger_sync=False)
    )
    overrides.setdefault("llm", None)
    runtime = build_runtime(
        settings,
        workspace_client=workspace,
        embedder=HashingEmbeddingClient(64),
        var_dir=tmp_path,
        environ={},
        fetcher=northwind_fetcher(),
        **overrides,
    )
    return runtime, workspace


def test_databricks_runtime_uses_delta_adapters_and_replicates_audit(tmp_path: Path) -> None:
    runtime, workspace = databricks_runtime(tmp_path)
    assert isinstance(runtime.document_store, DeltaDocumentStore)
    assert isinstance(runtime.brief_repository, DeltaBriefRepository)
    assert isinstance(runtime.audit, FanOutAuditLogger)
    assert runtime.track_runs is True
    runtime.track_runs = False  # no real MLflow tracking server in unit tests

    result = ClientResearchOrchestrator(runtime).run(northwind_request(), principal=ANALYST)
    runtime.close()
    assert result.persisted
    assert runtime.brief_repository.get(result.run_id) == result.brief
    sql = workspace.statement_execution
    audit_rows = sql.platform_rows["insert_audit"]
    assert audit_rows
    assert all(
        key.startswith(
            ("event_id_", "sequence_", "event_type_", "payload_json_", "recorded_at_", "prev_hash_", "hash_")
        )
        for key in audit_rows[0]
    )
    retrieval = result.state.step(StepName.RETRIEVAL)
    assert retrieval is not None
    assert retrieval.detail["child_chunks"] > 0


def test_databricks_runtime_builds_embedder_and_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import client_research_agent.databricks.vector_search as vs

    workspace = FakeWorkspace()
    created: dict[str, Any] = {}

    def fake_client(ws: Any) -> Any:
        created["ws"] = ws
        return types.SimpleNamespace(
            get_index=lambda **kw: FakeVectorSearchIndex(workspace.statement_execution)
        )

    monkeypatch.setattr(vs, "build_vector_search_client", fake_client)
    runtime = build_runtime(
        build_settings("dev"),
        workspace_client=workspace,
        llm=None,
        var_dir=tmp_path,
        environ={},
        trigger_index_sync=False,
    )
    from client_research_agent.databricks.embeddings import DatabricksEmbeddingClient

    assert isinstance(runtime.embedder, DatabricksEmbeddingClient)
    assert isinstance(runtime.vector_index, DatabricksVectorIndex)
    assert created["ws"] is workspace


def test_databricks_audit_falls_back_when_warehouse_unresolvable(tmp_path: Path) -> None:
    settings = build_settings("dev")
    workspace = FakeWorkspace(warehouse_list=[])
    runtime = build_runtime(
        settings,
        workspace_client=workspace,
        llm=None,
        embedder=HashingEmbeddingClient(64),
        vector_index=InMemoryVectorIndex(),
        document_store=InMemoryDocumentStore(),
        brief_repository=JsonlBriefRepository(tmp_path / "briefs"),
        var_dir=tmp_path,
        environ={},
    )
    assert type(runtime.audit) is AuditLogger
