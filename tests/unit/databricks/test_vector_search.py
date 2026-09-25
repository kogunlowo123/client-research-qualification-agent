from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from databricks.ai_search.exceptions import TooManyRequests
from databricks.sdk.errors import platform

from client_research_agent.config.settings import AppSettings, ResilienceSettings, build_settings
from client_research_agent.databricks.unity_catalog import DeltaDocumentStore, StatementExecutor
from client_research_agent.databricks.vector_search import (
    DatabricksVectorIndex,
    build_vector_search_client,
    ensure_endpoint_and_index,
    ensure_from_settings,
    full_index_name,
    full_source_table,
    to_vector_search_filters,
)
from client_research_agent.models import DocumentType
from client_research_agent.observability.metrics import Metrics
from client_research_agent.utils.errors import (
    CircuitOpenError,
    ConfigurationError,
    RateLimitedError,
    UpstreamServiceError,
)
from client_research_agent.utils.resilience import CircuitBreaker
from tests.contract.fakes import FakeStatementExecution, FakeVectorSearchClient, FakeVectorSearchIndex
from tests.support.doubles import make_chunk

FAST = ResilienceSettings(max_attempts=2, initial_backoff_seconds=0.0, max_backoff_seconds=0.0)


def _wire(**kwargs: Any) -> tuple[DatabricksVectorIndex, FakeVectorSearchIndex, FakeStatementExecution]:
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(StatementExecutor(sql, "wh", resilience=FAST), catalog="c", schema="s")
    fake_index = FakeVectorSearchIndex(sql)
    kwargs.setdefault("resilience", FAST)
    kwargs.setdefault("metrics", Metrics())
    index = DatabricksVectorIndex(fake_index, store, sleep=lambda _s: None, **kwargs)
    return index, fake_index, sql


def test_filter_rendering() -> None:
    rendered = to_vector_search_filters(
        {
            "company": "Acme",
            "document_type": (DocumentType.SEC_FILING, DocumentType.PRESS_RELEASE),
            "publication_date >=": date(2026, 1, 1),
            "NOT source_domain": "spam.example",
        }
    )
    assert rendered == {
        "company": "Acme",
        "document_type": ["sec_filing", "press_release"],
        "publication_date >=": "2026-01-01",
        "source_domain NOT": "spam.example",
    }
    assert to_vector_search_filters(None) == {}
    with pytest.raises(ValueError, match="duplicate"):
        to_vector_search_filters({"company NOT": "a", "company !=": "b"})


def test_search_request_shape() -> None:
    index, fake, _ = _wire()
    index.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    index.search([1, 0], k=3, filters={"company": "Acme Corp"})
    query = fake.queries[-1]
    assert query["query_vector"] == [1.0, 0.0]
    assert query["num_results"] == 3
    assert query["query_type"] == "ANN"
    assert query["filters"] == {"company": "Acme Corp"}
    assert query["total_retries"] == 0
    assert "chunk_id" in query["columns"]
    assert "embedding" not in query["columns"]
    assert "query_text" not in query
    index.search([1, 0], k=1)
    assert "filters" not in fake.queries[-1]


def test_hybrid_search() -> None:
    index, fake, _ = _wire()
    index.upsert([make_chunk("a", "lakehouse"), make_chunk("b", "other")], [[1.0, 0.0], [0.0, 1.0]])
    results = index.hybrid_search("lakehouse", [1.0, 0.0], k=1)
    assert [r.chunk.chunk_id for r in results] == ["a"]
    assert results[0].retriever == "hybrid"
    assert fake.queries[-1]["query_type"] == "HYBRID"
    assert fake.queries[-1]["query_text"] == "lakehouse"
    with pytest.raises(ValueError, match="query_text"):
        index.hybrid_search("  ", [1.0, 0.0], k=1)
    with pytest.raises(ValueError, match="query_vector"):
        index.search([], k=1)


def test_upsert_writes_delta_then_syncs() -> None:
    metrics = Metrics()
    index, fake, sql = _wire(metrics=metrics)
    assert index.upsert([], []) == 0
    assert fake.sync_calls == 0
    index.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    assert sql.tables["chunks"]["a"]["embedding"] == [1.0, 0.0]
    assert fake.sync_calls == 1
    assert metrics.counter("vector_index.upserted") == 1


def test_wait_for_sync_and_continuous_pipelines() -> None:
    index, fake, _ = _wire(wait_for_sync=True, sync_timeout=timedelta(minutes=5))
    index.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    assert fake.wait_calls == [{"wait_for_updates": True, "timeout": timedelta(minutes=5)}]
    continuous, fake2, _ = _wire(trigger_sync=False)
    continuous.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    continuous.delete_company("Acme Corp")
    assert fake2.sync_calls == 0


def test_remote_errors_are_retried_and_mapped() -> None:
    index, fake, _ = _wire()
    index.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    fake.fail_with = [TooManyRequests("slow", status_code=429)]
    assert len(index.search([1.0, 0.0], k=1)) == 1
    fake.fail_with = [TooManyRequests("slow", status_code=429), TooManyRequests("slow", status_code=429)]
    with pytest.raises(RateLimitedError):
        index.search([1.0, 0.0], k=1)
    fake.fail_with = [platform.InternalError("sync broke")] * 2
    with pytest.raises(UpstreamServiceError):
        index.sync()


def test_breaker_short_circuits() -> None:
    breaker = CircuitBreaker("vs", failure_threshold=1, reset_timeout_seconds=60)
    index, fake, _ = _wire(breaker=breaker, resilience=ResilienceSettings(max_attempts=1))
    fake.fail_with = [platform.TemporarilyUnavailable("down")]
    with pytest.raises(UpstreamServiceError):
        index.search([1.0], k=1)
    with pytest.raises(CircuitOpenError):
        index.search([1.0], k=1)


@pytest.mark.parametrize(
    "response",
    [
        ["not", "a", "mapping"],
        {"manifest": {"columns": [{"name": "chunk_id"}]}, "result": {"data_array": [["a"]]}},
    ],
)
def test_malformed_responses(response: Any) -> None:
    index, fake, _ = _wire(resilience=ResilienceSettings(max_attempts=1))
    fake.response_override = response
    with pytest.raises(UpstreamServiceError):
        index.search([1.0], k=1)


def test_empty_response_and_truncation() -> None:
    index, fake, _ = _wire()
    fake.response_override = {"manifest": {}, "result": {}}
    assert index.search([1.0], k=1) == []


def test_columns_must_include_primary_key() -> None:
    with pytest.raises(ConfigurationError):
        DatabricksVectorIndex(object(), DeltaDocumentStore.__new__(DeltaDocumentStore), columns=("text",))


def _settings(pipeline: str = "TRIGGERED") -> AppSettings:
    return build_settings("local").model_copy(
        update={
            "vector_search": build_settings("local").vector_search.model_copy(
                update={"pipeline_type": pipeline}
            )
        }
    )


def test_names_and_from_settings() -> None:
    settings = _settings()
    assert (
        full_index_name(settings.databricks, settings.vector_search)
        == "client_research.agent_local.chunks_index"
    )
    assert (
        full_source_table(settings.databricks, settings.vector_search) == "client_research.agent_local.chunks"
    )
    sql = FakeStatementExecution()
    client = FakeVectorSearchClient()
    fake_index = FakeVectorSearchIndex(sql)
    client.index_objects["client_research.agent_local.chunks_index"] = fake_index
    store = DeltaDocumentStore(StatementExecutor(sql, "wh"), catalog="c", schema="s")
    index = DatabricksVectorIndex.from_settings(client, settings, store)
    index.upsert([make_chunk("a", "t")], [[1.0]])
    assert fake_index.sync_calls == 1
    continuous = DatabricksVectorIndex.from_settings(client, _settings("CONTINUOUS"), store)
    continuous.upsert([make_chunk("b", "t")], [[1.0]])
    assert fake_index.sync_calls == 1
    assert client.calls[0] == (
        "get_index",
        {"endpoint_name": "cra-vs-endpoint", "index_name": "client_research.agent_local.chunks_index"},
    )


# ------------------------------------------------------------- provisioning


def test_ensure_creates_missing_resources_and_waits() -> None:
    client = FakeVectorSearchClient()
    result = ensure_endpoint_and_index(
        client,
        endpoint_name="ep",
        index_name="cat.sch.chunks_index",
        source_table="cat.sch.chunks",
        timeout=timedelta(minutes=10),
    )
    assert result.endpoint_created
    assert result.index_created
    names = [name for name, _ in client.calls]
    assert names == ["create_endpoint_and_wait", "create_delta_sync_index_and_wait"]
    spec = client.calls[1][1]
    assert spec["embedding_vector_column"] == "embedding"
    assert spec["embedding_dimension"] == 1024
    assert spec["primary_key"] == "chunk_id"
    assert spec["pipeline_type"] == "TRIGGERED"
    assert "chunk_id" not in spec["columns_to_sync"]
    assert "company" in spec["columns_to_sync"]
    assert spec["timeout"] == timedelta(minutes=10)
    again = ensure_endpoint_and_index(
        client, endpoint_name="ep", index_name="cat.sch.chunks_index", source_table="cat.sch.chunks"
    )
    assert not again.endpoint_created
    assert not again.index_created
    assert len(client.calls) == 2


def test_ensure_without_wait_and_from_settings() -> None:
    client = FakeVectorSearchClient()
    result = ensure_from_settings(client, _settings("continuous"), wait=False)
    assert result.index_name == "client_research.agent_local.chunks_index"
    assert [name for name, _ in client.calls] == ["create_endpoint", "create_delta_sync_index"]
    assert client.calls[1][1]["pipeline_type"] == "CONTINUOUS"
    assert client.calls[1][1]["source_table_name"] == "client_research.agent_local.chunks"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pipeline_type": "HOURLY"}, "pipeline_type"),
        ({"embedding_dimension": 0}, "embedding_dimension"),
        ({"index_name": "chunks_index"}, "three-level"),
        ({"source_table": "cat.sch.bad-name"}, "invalid SQL identifier"),
    ],
)
def test_ensure_validation(kwargs: dict[str, Any], message: str) -> None:
    params: dict[str, Any] = {
        "endpoint_name": "ep",
        "index_name": "cat.sch.idx",
        "source_table": "cat.sch.chunks",
        **kwargs,
    }
    with pytest.raises(ConfigurationError, match=message):
        ensure_endpoint_and_index(FakeVectorSearchClient(), **params)


def test_build_vector_search_client_credential_selection() -> None:
    captured: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> str:
        captured.append(kwargs)
        return "client"

    def ws(**config: Any) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                **{"host": None, "client_id": None, "client_secret": None, "token": None, **config}
            )
        )

    assert (
        build_vector_search_client(ws(host="https://h", client_id="id", client_secret="s"), factory=factory)
        == "client"
    )
    build_vector_search_client(ws(host="https://h", token="dapi"), factory=factory)
    build_vector_search_client(ws(), factory=factory)
    assert captured[0] == {
        "disable_notice": True,
        "workspace_url": "https://h",
        "service_principal_client_id": "id",
        "service_principal_client_secret": "s",
    }
    assert captured[1] == {
        "disable_notice": True,
        "workspace_url": "https://h",
        "personal_access_token": "dapi",
    }
    assert captured[2] == {"disable_notice": True}


def test_default_vector_search_factory_uses_installed_client() -> None:
    config = SimpleNamespace(host="https://h.example", client_id=None, client_secret=None, token="dapi-test")
    client = build_vector_search_client(SimpleNamespace(config=config))
    assert type(client).__name__ in {"VectorSearchClient", "AISearchClient"}
    assert client.workspace_url == "https://h.example"


def test_default_factory_falls_back_to_ai_search(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "databricks.vector_search.client", None)
    config = SimpleNamespace(host="https://h.example", client_id=None, client_secret=None, token="dapi-test")
    client = build_vector_search_client(SimpleNamespace(config=config))
    assert type(client).__name__ == "AISearchClient"
