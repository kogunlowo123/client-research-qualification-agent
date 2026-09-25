"""Databricks Vector Search (AI Search) adapter for the ``VectorIndex`` port.

Index type decision: **Delta Sync with self-managed embeddings**
----------------------------------------------------------------
Production uses a Delta Sync index whose source is the Unity Catalog ``chunks``
table (primary key ``chunk_id``, vector column ``embedding`` populated by our
own embedding client), not a Direct Vector Access index:

* The Delta table is the single source of truth. Evidence is governed by Unity
  Catalog grants, tags and lineage, and can be audited, time-travelled and
  re-indexed from scratch; a Direct Access index would be an ungoverned second
  copy that must be kept consistent by hand.
* Deletions (right-to-erasure via ``delete_company``) are a single ``DELETE``
  on the table; Change Data Feed propagates them to the index on the next sync.
* Embeddings are computed once by the ingestion pipeline and persisted, so a
  re-sync never re-bills the embedding endpoint.

Consequently ``upsert`` writes rows (with vectors) to the Delta table through a
:class:`ChunkVectorWriter` (``DeltaDocumentStore``) and then triggers
``index.sync()`` for ``TRIGGERED`` pipelines. Sync is asynchronous; pass
``wait_for_sync=True`` when read-after-write visibility is required (tests,
smoke jobs), at the cost of blocking until the index reports ``ONLINE``.

``search`` issues an ANN query with ``query_vector``; :meth:`hybrid_search`
additionally sends ``query_text`` with ``query_type="HYBRID"`` so the service
fuses vector and keyword relevance natively.
"""

from __future__ import annotations

import importlib
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from client_research_agent.config.settings import (
    AppSettings,
    DatabricksSettings,
    ResilienceSettings,
    VectorSearchSettings,
)
from client_research_agent.databricks.errors import map_databricks_error
from client_research_agent.databricks.model_serving import retry_policy
from client_research_agent.databricks.unity_catalog import CHUNK_COLUMNS, chunk_from_row, validate_identifier
from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import Metrics, get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.services.local import DENSE_RETRIEVER, parse_filters
from client_research_agent.utils.errors import ConfigurationError, UpstreamServiceError
from client_research_agent.utils.resilience import CircuitBreaker, call_with_retry

HYBRID_RETRIEVER = "hybrid"
SCORE_COLUMN = "score"


class ChunkVectorWriter(Protocol):
    """Writes chunk rows (with vectors) to the index's Delta source table."""

    def save_chunks_with_embeddings(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int: ...

    def delete_company_chunks(self, company: str) -> int: ...


def to_vector_search_filters(filters: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate port-level filters and render them in the Vector Search dictionary dialect."""
    rendered: dict[str, Any] = {}
    for clause in parse_filters(filters):
        if clause.operator == "=":
            key = clause.field
        elif clause.operator == "NOT":
            key = f"{clause.field} NOT"
        else:
            key = f"{clause.field} {clause.operator}"
        if key in rendered:
            raise ValueError(f"duplicate filter on {key!r}")
        rendered[key] = clause.value
    return rendered


def full_index_name(databricks: DatabricksSettings, vector_search: VectorSearchSettings) -> str:
    return ".".join(
        validate_identifier(part)
        for part in (databricks.catalog, databricks.schema_, vector_search.index_name)
    )


def full_source_table(databricks: DatabricksSettings, vector_search: VectorSearchSettings) -> str:
    return ".".join(
        validate_identifier(part)
        for part in (databricks.catalog, databricks.schema_, vector_search.source_table)
    )


class DatabricksVectorIndex:
    """``VectorIndex`` over a Delta Sync Vector Search index with self-managed embeddings.

    ``index`` is the object returned by ``VectorSearchClient.get_index(...)``.
    """

    def __init__(
        self,
        index: Any,
        writer: ChunkVectorWriter,
        *,
        trigger_sync: bool = True,
        wait_for_sync: bool = False,
        sync_timeout: timedelta = timedelta(minutes=30),
        columns: Sequence[str] = CHUNK_COLUMNS,
        resilience: ResilienceSettings | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        metrics: Metrics | None = None,
    ) -> None:
        if "chunk_id" not in columns:
            raise ConfigurationError("columns must include the primary key 'chunk_id'")
        settings = resilience or ResilienceSettings()
        self._index = index
        self._writer = writer
        self._trigger_sync = trigger_sync
        self._wait_for_sync = wait_for_sync
        self._sync_timeout = sync_timeout
        self._columns = list(columns)
        self._policy = retry_policy(settings)
        self._breaker = breaker or CircuitBreaker(
            f"vector_search:{getattr(index, 'name', 'index')}",
            failure_threshold=settings.breaker_failure_threshold,
            reset_timeout_seconds=settings.breaker_reset_seconds,
        )
        self._sleep = sleep
        self._metrics = metrics or get_metrics()

    @classmethod
    def from_settings(
        cls,
        vector_search_client: Any,
        settings: AppSettings,
        writer: ChunkVectorWriter,
        **kwargs: Any,
    ) -> DatabricksVectorIndex:
        index = vector_search_client.get_index(
            endpoint_name=settings.vector_search.endpoint_name,
            index_name=full_index_name(settings.databricks, settings.vector_search),
        )
        kwargs.setdefault("trigger_sync", settings.vector_search.pipeline_type.upper() == "TRIGGERED")
        kwargs.setdefault("resilience", settings.resilience)
        return cls(index, writer, **kwargs)

    def _remote(self, context: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        def attempt() -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                raise map_databricks_error(exc, f"vector search {context}") from exc

        return call_with_retry(attempt, policy=self._policy, breaker=self._breaker, sleep=self._sleep)

    def sync(self) -> None:
        """Trigger an incremental sync from the Delta source table (TRIGGERED pipelines)."""
        if not self._trigger_sync:
            return
        self._remote("sync", self._index.sync)
        if self._wait_for_sync:
            self._remote(
                "wait_until_ready",
                self._index.wait_until_ready,
                wait_for_updates=True,
                timeout=self._sync_timeout,
            )

    @traced("vector_search.upsert", span_type=SpanType.RETRIEVER)
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0
        written = self._writer.save_chunks_with_embeddings(chunks, embeddings)
        self.sync()
        self._metrics.increment("vector_index.upserted", written)
        return written

    def _query(
        self,
        *,
        k: int,
        filters: Mapping[str, Any] | None,
        query_vector: Sequence[float],
        query_text: str | None,
        query_type: str,
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        vector = [float(value) for value in query_vector]
        if not vector:
            raise ValueError("query_vector must be non-empty")
        request: dict[str, Any] = {
            "columns": self._columns,
            "query_vector": vector,
            "num_results": k,
            "query_type": query_type,
            "disable_notice": True,
            "total_retries": 0,
        }
        rendered = to_vector_search_filters(filters)
        if rendered:
            request["filters"] = rendered
        if query_text is not None:
            request["query_text"] = query_text
        started = time.perf_counter()
        response = self._remote("similarity_search", self._index.similarity_search, **request)
        self._metrics.observe(
            "vector_index.search_ms", (time.perf_counter() - started) * 1000, query_type=query_type
        )
        retriever = HYBRID_RETRIEVER if query_type == "HYBRID" else DENSE_RETRIEVER
        return self._to_results(response, retriever)[:k]

    @staticmethod
    def _to_results(response: Any, retriever: str) -> list[RetrievedChunk]:
        if not isinstance(response, Mapping):
            raise UpstreamServiceError("vector search returned a non-JSON response")
        manifest = response.get("manifest") or {}
        columns = [str(column.get("name")) for column in manifest.get("columns") or []]
        rows = (response.get("result") or {}).get("data_array") or []
        if rows and SCORE_COLUMN not in columns:
            raise UpstreamServiceError("vector search response is missing the score column")
        results: list[RetrievedChunk] = []
        for rank, values in enumerate(rows):
            row = dict(zip(columns, values, strict=False))
            results.append(
                RetrievedChunk(
                    chunk=chunk_from_row(row),
                    score=float(row[SCORE_COLUMN]),
                    retriever=retriever,
                    rank=rank,
                )
            )
        return results

    @traced("vector_search.search", span_type=SpanType.RETRIEVER)
    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        return self._query(k=k, filters=filters, query_vector=query_vector, query_text=None, query_type="ANN")

    @traced("vector_search.hybrid_search", span_type=SpanType.RETRIEVER)
    def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Native Databricks hybrid (vector + keyword) retrieval."""
        if not query_text.strip():
            raise ValueError("query_text must be non-empty for hybrid search")
        return self._query(
            k=k, filters=filters, query_vector=query_vector, query_text=query_text, query_type="HYBRID"
        )

    @traced("vector_search.delete_company", span_type=SpanType.RETRIEVER)
    def delete_company(self, company: str) -> int:
        removed = self._writer.delete_company_chunks(company)
        self.sync()
        return removed


# ---------------------------------------------------------------- provisioning


def _default_client_factory(**kwargs: Any) -> Any:
    """Prefer the legacy ``databricks.vector_search`` path; fall back to ``databricks.ai_search``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            client_cls = importlib.import_module("databricks.vector_search.client").VectorSearchClient
    except (ImportError, AttributeError):
        client_cls = importlib.import_module("databricks.ai_search.client").AISearchClient
    return client_cls(**kwargs)


def build_vector_search_client(workspace_client: Any, *, factory: Callable[..., Any] | None = None) -> Any:
    """Create a Vector Search client that reuses the workspace's resolved credentials.

    OAuth M2M service principals are passed through explicitly; PAT auth (dev only)
    reuses the token; otherwise the client falls back to ambient notebook / Model
    Serving credentials.
    """
    config = workspace_client.config
    kwargs: dict[str, Any] = {"disable_notice": True}
    host = getattr(config, "host", None)
    client_id = getattr(config, "client_id", None)
    client_secret = getattr(config, "client_secret", None)
    token = getattr(config, "token", None)
    if host and client_id and client_secret:
        kwargs.update(
            workspace_url=host,
            service_principal_client_id=client_id,
            service_principal_client_secret=client_secret,
        )
    elif host and token:
        kwargs.update(workspace_url=host, personal_access_token=token)
    return (factory or _default_client_factory)(**kwargs)


@dataclass(frozen=True, slots=True)
class ProvisioningResult:
    endpoint_name: str
    index_name: str
    endpoint_created: bool
    index_created: bool


def ensure_endpoint_and_index(
    client: Any,
    *,
    endpoint_name: str,
    index_name: str,
    source_table: str,
    primary_key: str = "chunk_id",
    embedding_column: str = "embedding",
    embedding_dimension: int = 1024,
    pipeline_type: str = "TRIGGERED",
    endpoint_type: str = "STANDARD",
    columns_to_sync: Sequence[str] = CHUNK_COLUMNS,
    wait: bool = True,
    timeout: timedelta = timedelta(hours=1),
) -> ProvisioningResult:
    """Idempotently create the endpoint and the Delta Sync index (self-managed embeddings).

    Safe to re-run from infrastructure scripts: existing resources are left untouched.
    The source table must already exist with Change Data Feed enabled
    (see ``unity_catalog.DDL``).
    """
    if pipeline_type.upper() not in ("TRIGGERED", "CONTINUOUS"):
        raise ConfigurationError(f"pipeline_type must be TRIGGERED or CONTINUOUS, got {pipeline_type!r}")
    if embedding_dimension < 1:
        raise ConfigurationError("embedding_dimension must be positive")
    for qualified in (index_name, source_table):
        parts = qualified.split(".")
        if len(parts) != 3:
            raise ConfigurationError(f"{qualified!r} must be a three-level catalog.schema.name")
        for part in parts:
            validate_identifier(part)

    endpoint_created = False
    if not client.endpoint_exists(endpoint_name):
        if wait:
            client.create_endpoint_and_wait(name=endpoint_name, endpoint_type=endpoint_type, timeout=timeout)
        else:
            client.create_endpoint(name=endpoint_name, endpoint_type=endpoint_type)
        endpoint_created = True

    index_created = False
    if not client.index_exists(endpoint_name=endpoint_name, index_name=index_name):
        spec: dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "index_name": index_name,
            "primary_key": primary_key,
            "source_table_name": source_table,
            "pipeline_type": pipeline_type.upper(),
            "embedding_dimension": embedding_dimension,
            "embedding_vector_column": embedding_column,
            "columns_to_sync": [c for c in columns_to_sync if c not in (primary_key, embedding_column)],
        }
        if wait:
            client.create_delta_sync_index_and_wait(**spec, timeout=timeout)
        else:
            client.create_delta_sync_index(**spec)
        index_created = True

    return ProvisioningResult(
        endpoint_name=endpoint_name,
        index_name=index_name,
        endpoint_created=endpoint_created,
        index_created=index_created,
    )


def ensure_from_settings(client: Any, settings: AppSettings, *, wait: bool = True) -> ProvisioningResult:
    """:func:`ensure_endpoint_and_index` driven by ``AppSettings``."""
    return ensure_endpoint_and_index(
        client,
        endpoint_name=settings.vector_search.endpoint_name,
        index_name=full_index_name(settings.databricks, settings.vector_search),
        source_table=full_source_table(settings.databricks, settings.vector_search),
        primary_key=settings.vector_search.primary_key,
        embedding_column=settings.vector_search.embedding_column,
        embedding_dimension=settings.serving.embedding_dimension,
        pipeline_type=settings.vector_search.pipeline_type,
        wait=wait,
    )
