"""Unity Catalog Delta tables as the system of record, via the SQL Statement Execution API.

Every statement goes through ``WorkspaceClient.statement_execution.execute_statement``
against a SQL warehouse. Values are *always* bound as named parameter markers
(``:company``); nothing user-controlled is interpolated into SQL text. The only
interpolated tokens are catalog/schema/table identifiers, which are validated
against ``^[A-Za-z0-9_]+$`` and back-quoted.

Batch writes serialise the rows to one JSON string parameter and expand it
server-side with ``inline(from_json(:rows, '<schema>'))`` inside a ``MERGE INTO``.
That keeps writes idempotent (safe to retry), atomic per batch and immune to
injection, and it sidesteps the API's lack of ``ARRAY`` parameter types (the
``embedding ARRAY<FLOAT>`` column is carried inside the JSON payload).

Tables (Delta; Change Data Feed enabled wherever a downstream consumer needs
incremental changes):

* ``documents`` - one row per fetched source document
* ``chunks`` - **embedded child chunks only**; the source table of the Vector
  Search Delta Sync index. Invariant: every row has a non-empty ``embedding``
  and ``strategy <> 'parent'``, enforced by ``NOT NULL`` plus the
  ``chunks_embedded_children_only`` CHECK constraint, because a Delta Sync index
  with self-managed embeddings fails or skips rows whose vector is NULL.
* ``parent_chunks`` - parent chunks (same columns minus ``embedding``). They are
  the LLM context unit for small-to-big expansion and are only read by id, so
  they never enter the index source table and need no Change Data Feed.
* ``briefs`` - one row per generated client brief (full JSON plus query columns)
* ``audit_log`` - append-only, hash-chained governance events

Write routing: :meth:`DeltaDocumentStore.save_chunks` sends ``PARENT`` chunks to
``parent_chunks``; for child chunks (which have no vector yet) it only refreshes
metadata of rows already present in ``chunks``. Children become readable once
they are written with their vectors through
:meth:`DeltaDocumentStore.save_chunks_with_embeddings` (i.e. ``VectorIndex.upsert``),
which is exactly the order the indexing pipeline uses.

Each statement is tagged with a ``cra_op`` query tag so query history can be
attributed to the operation that issued it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from databricks.sdk.service.sql import (
    Disposition,
    ExecuteStatementRequestOnWaitTimeout,
    Format,
    QueryTag,
    StatementParameterListItem,
    StatementState,
)

from client_research_agent.config.settings import DatabricksSettings, ResilienceSettings
from client_research_agent.databricks.errors import DatabricksRequestError, map_databricks_error
from client_research_agent.databricks.model_serving import retry_policy
from client_research_agent.models import Chunk, ChunkStrategy, ClientBrief, SourceDocument
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.services.local import validate_run_id
from client_research_agent.utils.errors import (
    ConfigurationError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import CircuitBreaker, call_with_retry

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
TABLES: tuple[str, ...] = ("documents", "chunks", "parent_chunks", "briefs", "audit_log")
QUERY_TAG_KEY = "cra_op"
WRITE_BATCH_SIZE = 200
READ_BATCH_SIZE = 1000


def validate_identifier(name: str) -> str:
    """Return ``name`` if it is a safe SQL identifier, else raise ``ConfigurationError``."""
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
        raise ConfigurationError(f"invalid SQL identifier {name!r}; must match ^[A-Za-z0-9_]+$")
    return name


def qualified_name(catalog: str, schema: str, table: str) -> str:
    """Back-quoted three-level name after validating every part."""
    return ".".join(f"`{validate_identifier(part)}`" for part in (catalog, schema, table))


# --------------------------------------------------------------------------- DDL

DDL: dict[str, str] = {
    "documents": """CREATE TABLE IF NOT EXISTS {table} (
  doc_id STRING NOT NULL COMMENT 'Stable document identifier',
  company STRING NOT NULL COMMENT 'Company the document is about',
  url STRING NOT NULL COMMENT 'Canonical source URL',
  title STRING COMMENT 'Document title',
  text STRING COMMENT 'Extracted plain text',
  document_type STRING NOT NULL COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  content_hash STRING NOT NULL COMMENT 'SHA-256 of normalised text, used for dedupe',
  publication_date DATE COMMENT 'Publication date when known',
  retrieved_at TIMESTAMP COMMENT 'When the crawler fetched the document',
  industry STRING COMMENT 'Industry label',
  language STRING COMMENT 'ISO language code',
  trust_score DOUBLE COMMENT 'Source trust score in [0,1]',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT documents_pk PRIMARY KEY (doc_id)
) USING DELTA
COMMENT 'Public company evidence documents collected by the research agent'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')""",
    "chunks": """CREATE TABLE IF NOT EXISTS {table} (
  chunk_id STRING NOT NULL COMMENT 'Stable chunk identifier (Vector Search primary key)',
  doc_id STRING NOT NULL COMMENT 'Parent document id',
  text STRING COMMENT 'Chunk text',
  company STRING NOT NULL COMMENT 'Company the chunk is about',
  url STRING COMMENT 'Source URL',
  title STRING COMMENT 'Source title',
  document_type STRING COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  chunk_index INT COMMENT 'Position of the chunk within its document',
  strategy STRING COMMENT 'ChunkStrategy enum value',
  parent_id STRING COMMENT 'Parent chunk id for child chunks',
  publication_date DATE COMMENT 'Publication date of the source',
  industry STRING COMMENT 'Industry label',
  confidence DOUBLE COMMENT 'Extraction confidence in [0,1]',
  token_count INT COMMENT 'Token count of the chunk text',
  entities_json STRING COMMENT 'Extracted entities as a JSON array',
  contextual_header STRING COMMENT 'Contextual-retrieval header',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  embedding ARRAY<FLOAT> NOT NULL COMMENT 'Self-managed embedding vector (never NULL)',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT chunks_pk PRIMARY KEY (chunk_id)
) USING DELTA
COMMENT 'Embedded child chunks only; source table of the Vector Search Delta Sync index'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')""",
    "parent_chunks": """CREATE TABLE IF NOT EXISTS {table} (
  chunk_id STRING NOT NULL COMMENT 'Stable parent chunk identifier',
  doc_id STRING NOT NULL COMMENT 'Parent document id',
  text STRING COMMENT 'Parent chunk text (LLM context unit)',
  company STRING NOT NULL COMMENT 'Company the chunk is about',
  url STRING COMMENT 'Source URL',
  title STRING COMMENT 'Source title',
  document_type STRING COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  chunk_index INT COMMENT 'Position of the chunk within its document',
  strategy STRING COMMENT 'ChunkStrategy enum value (always parent)',
  parent_id STRING COMMENT 'Unused for parents; kept for a uniform chunk schema',
  publication_date DATE COMMENT 'Publication date of the source',
  industry STRING COMMENT 'Industry label',
  confidence DOUBLE COMMENT 'Extraction confidence in [0,1]',
  token_count INT COMMENT 'Token count of the chunk text',
  entities_json STRING COMMENT 'Extracted entities as a JSON array',
  contextual_header STRING COMMENT 'Contextual-retrieval header',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT parent_chunks_pk PRIMARY KEY (chunk_id)
) USING DELTA
COMMENT 'Parent chunks for small-to-big expansion; never indexed'""",
    "briefs": """CREATE TABLE IF NOT EXISTS {table} (
  run_id STRING NOT NULL COMMENT 'Agent run identifier',
  company STRING NOT NULL COMMENT 'Company researched',
  generated_at TIMESTAMP COMMENT 'When the brief was generated',
  verdict STRING COMMENT 'FitVerdict enum value',
  weighted_score DOUBLE COMMENT 'Weighted qualification score in [0,5]',
  brief_json STRING NOT NULL COMMENT 'Full ClientBrief as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT briefs_pk PRIMARY KEY (run_id)
) USING DELTA
COMMENT 'Cited, scored client briefs produced by the agent'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')""",
    "audit_log": """CREATE TABLE IF NOT EXISTS {table} (
  event_id STRING NOT NULL COMMENT 'Unique event identifier',
  sequence BIGINT NOT NULL COMMENT 'Monotonic sequence within the chain',
  event_type STRING NOT NULL COMMENT 'Audit event type',
  payload_json STRING COMMENT 'Event payload as JSON (PII-scrubbed)',
  recorded_at TIMESTAMP NOT NULL COMMENT 'Event time (UTC)',
  prev_hash STRING COMMENT 'Hash of the previous event',
  hash STRING NOT NULL COMMENT 'SHA-256 over prev_hash and this event',
  CONSTRAINT audit_log_pk PRIMARY KEY (event_id)
) USING DELTA
COMMENT 'Append-only, hash-chained audit trail of agent actions'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true', 'delta.appendOnly' = 'true')""",
}

TABLE_TAGS: dict[str, dict[str, str]] = {
    "documents": {"domain": "client_research", "data_classification": "public", "layer": "silver"},
    "chunks": {"domain": "client_research", "data_classification": "public", "layer": "gold"},
    "parent_chunks": {"domain": "client_research", "data_classification": "public", "layer": "gold"},
    "briefs": {"domain": "client_research", "data_classification": "internal", "layer": "gold"},
    "audit_log": {"domain": "client_research", "data_classification": "confidential", "layer": "audit"},
}

_TAG_VALUE = re.compile(r"^[A-Za-z0-9_.\- ]+$")

# Delta CHECK constraints can only be added with ALTER TABLE and have no
# ``IF NOT EXISTS`` form, so they are dropped-if-present then re-added to keep
# ``render_ddl`` idempotent (re-adding re-validates existing rows).
CHECK_CONSTRAINTS: dict[str, dict[str, str]] = {
    "chunks": {
        "chunks_embedded_children_only": "strategy <> 'parent' AND size(embedding) > 0",
    },
    "parent_chunks": {
        "parent_chunks_parents_only": "strategy = 'parent'",
    },
}


def render_ddl(catalog: str, schema: str) -> list[str]:
    """``CREATE TABLE IF NOT EXISTS``, CHECK constraints and ``SET TAGS`` for every table."""
    statements: list[str] = []
    for table, template in DDL.items():
        name = qualified_name(catalog, schema, table)
        statements.append(template.format(table=name))
        for constraint, predicate in CHECK_CONSTRAINTS.get(table, {}).items():
            validate_identifier(constraint)
            statements.append(f"ALTER TABLE {name} DROP CONSTRAINT IF EXISTS {constraint}")
            statements.append(f"ALTER TABLE {name} ADD CONSTRAINT {constraint} CHECK ({predicate})")
        tags = TABLE_TAGS[table]
        for key, value in tags.items():
            if not _TAG_VALUE.fullmatch(key) or not _TAG_VALUE.fullmatch(value):
                raise ConfigurationError(f"invalid tag {key}={value}")
        rendered = ", ".join(f"'{key}' = '{value}'" for key, value in tags.items())
        statements.append(f"ALTER TABLE {name} SET TAGS ({rendered})")
    return statements


# ------------------------------------------------------------------ execution


@dataclass(frozen=True, slots=True)
class SqlParam:
    """A typed named-parameter value (``None`` binds SQL ``NULL``)."""

    value: str | None
    type: str = "STRING"


def to_param(value: Any) -> SqlParam:
    """Convert a Python value into a typed statement parameter."""
    if isinstance(value, SqlParam):
        return value
    if value is None:
        return SqlParam(None)
    if isinstance(value, bool):
        return SqlParam("true" if value else "false", "BOOLEAN")
    if isinstance(value, int):
        return SqlParam(str(value), "BIGINT")
    if isinstance(value, float):
        return SqlParam(repr(value), "DOUBLE")
    if isinstance(value, datetime):
        return SqlParam(value.isoformat(), "TIMESTAMP")
    if isinstance(value, date):
        return SqlParam(value.isoformat(), "DATE")
    if isinstance(value, str):
        return SqlParam(value)
    raise TypeError(f"unsupported SQL parameter type {type(value).__name__}")


Row = dict[str, Any]
_TERMINAL_FAILURES = {StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}


class StatementExecutor:
    """Runs parameterised SQL on a warehouse with polling, retries and a circuit breaker.

    ``statement_api`` is ``WorkspaceClient.statement_execution``.
    """

    def __init__(
        self,
        statement_api: Any,
        warehouse_id: str,
        *,
        wait_timeout: str = "30s",
        poll_interval_seconds: float = 1.0,
        max_wait_seconds: float = 600.0,
        resilience: ResilienceSettings | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not warehouse_id:
            raise ConfigurationError("a SQL warehouse_id is required for Unity Catalog access")
        settings = resilience or ResilienceSettings()
        self._api = statement_api
        self._warehouse_id = warehouse_id
        self._wait_timeout = wait_timeout
        self._poll_interval = poll_interval_seconds
        self._max_wait = max_wait_seconds
        self._policy = retry_policy(settings)
        self._breaker = breaker or CircuitBreaker(
            f"sql:{warehouse_id}",
            failure_threshold=settings.breaker_failure_threshold,
            reset_timeout_seconds=settings.breaker_reset_seconds,
        )
        self._sleep = sleep
        self._clock = clock

    @property
    def warehouse_id(self) -> str:
        return self._warehouse_id

    def execute(self, statement: str, params: Mapping[str, Any] | None = None, *, op: str) -> list[Row]:
        """Execute ``statement`` binding ``params`` by name; return rows as column->string dicts."""
        parameters = [
            StatementParameterListItem(name=name, value=param.value, type=param.type)
            for name, param in ((n, to_param(v)) for n, v in (params or {}).items())
        ]
        return call_with_retry(
            lambda: self._execute_once(statement, parameters, op),
            policy=self._policy,
            breaker=self._breaker,
            sleep=self._sleep,
        )

    def _call(self, context: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            raise map_databricks_error(exc, context) from exc

    def _execute_once(
        self, statement: str, parameters: list[StatementParameterListItem], op: str
    ) -> list[Row]:
        response = self._call(
            f"sql {op}",
            self._api.execute_statement,
            statement=statement,
            warehouse_id=self._warehouse_id,
            parameters=parameters or None,
            wait_timeout=self._wait_timeout,
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
            disposition=Disposition.INLINE,
            format=Format.JSON_ARRAY,
            query_tags=[QueryTag(key=QUERY_TAG_KEY, value=op)],
        )
        response = self._await_completion(response, op)
        return list(self._rows(response, op))

    def _state(self, response: Any) -> StatementState | None:
        status = getattr(response, "status", None)
        return getattr(status, "state", None)

    def _await_completion(self, response: Any, op: str) -> Any:
        deadline = self._clock() + self._max_wait
        while self._state(response) in (StatementState.PENDING, StatementState.RUNNING):
            statement_id = response.statement_id
            if self._clock() >= deadline:
                self._call(f"sql {op} cancel", self._api.cancel_execution, statement_id)
                raise UpstreamTimeoutError(
                    f"sql {op}: statement {statement_id} exceeded {self._max_wait:.0f}s"
                )
            self._sleep(self._poll_interval)
            response = self._call(f"sql {op} poll", self._api.get_statement, statement_id)
        state = self._state(response)
        if state is StatementState.SUCCEEDED:
            return response
        error = getattr(getattr(response, "status", None), "error", None)
        message = getattr(error, "message", None) or "no error detail"
        if state in _TERMINAL_FAILURES and state is not StatementState.FAILED:
            raise UpstreamServiceError(f"sql {op}: statement {state.value}: {message}")
        raise DatabricksRequestError(f"sql {op}: statement failed: {message}")

    def _rows(self, response: Any, op: str) -> Iterator[Row]:
        manifest = getattr(response, "manifest", None)
        schema = getattr(manifest, "schema", None)
        columns = [column.name for column in (getattr(schema, "columns", None) or [])]
        result = getattr(response, "result", None)
        while result is not None:
            for values in result.data_array or []:
                yield dict(zip(columns, values, strict=False))
            next_index = getattr(result, "next_chunk_index", None)
            if next_index is None:
                return
            result = self._call(
                f"sql {op} fetch",
                self._api.get_statement_result_chunk_n,
                response.statement_id,
                next_index,
            )


def affected_rows(rows: Sequence[Row], fallback: int) -> int:
    """Read ``num_affected_rows`` from a DML result, falling back when absent."""
    if rows and rows[0].get("num_affected_rows") is not None:
        return int(rows[0]["num_affected_rows"])
    return fallback


def apply_ddl(executor: StatementExecutor, catalog: str, schema: str) -> int:
    """Idempotently create all tables and apply governance tags; returns statements run."""
    statements = render_ddl(catalog, schema)
    for statement in statements:
        executor.execute(statement, op="ddl")
    return len(statements)


# --------------------------------------------------------------- row mapping

_CHUNK_STRUCT = (
    "chunk_id: STRING, doc_id: STRING, text: STRING, company: STRING, url: STRING, title: STRING, "
    "document_type: STRING, source_domain: STRING, chunk_index: INT, strategy: STRING, "
    "parent_id: STRING, publication_date: DATE, industry: STRING, confidence: DOUBLE, "
    "token_count: INT, entities_json: STRING, contextual_header: STRING, metadata_json: STRING"
)
CHUNK_COLUMNS: tuple[str, ...] = tuple(part.split(":")[0].strip() for part in _CHUNK_STRUCT.split(","))
_DOCUMENT_STRUCT = (
    "doc_id: STRING, company: STRING, url: STRING, title: STRING, text: STRING, document_type: STRING, "
    "source_domain: STRING, content_hash: STRING, publication_date: DATE, retrieved_at: TIMESTAMP, "
    "industry: STRING, language: STRING, trust_score: DOUBLE, metadata_json: STRING"
)
DOCUMENT_COLUMNS: tuple[str, ...] = tuple(part.split(":")[0].strip() for part in _DOCUMENT_STRUCT.split(","))
_EPOCH = date(1970, 1, 1)


def chunk_to_row(chunk: Chunk) -> Row:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text,
        "company": chunk.company,
        "url": chunk.url,
        "title": chunk.title,
        "document_type": chunk.document_type.value,
        "source_domain": chunk.source_domain,
        "chunk_index": chunk.chunk_index,
        "strategy": chunk.strategy.value,
        "parent_id": chunk.parent_id,
        "publication_date": chunk.publication_date.isoformat() if chunk.publication_date else None,
        "industry": chunk.industry,
        "confidence": chunk.confidence,
        "token_count": chunk.token_count,
        "entities_json": json.dumps(list(chunk.entities)),
        "contextual_header": chunk.contextual_header,
        "metadata_json": json.dumps(chunk.metadata, default=str, sort_keys=True),
    }


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, int | float):
        return _EPOCH + timedelta(days=int(value))
    return date.fromisoformat(str(value)[:10])


def _as_float(value: Any, default: float) -> float:
    return default if value is None or value == "" else float(value)


def _parse_json(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, list | dict):
        return value
    return json.loads(str(value))


def chunk_from_row(row: Mapping[str, Any]) -> Chunk:
    """Rebuild a ``Chunk`` from a SQL or Vector Search row (values may arrive as strings)."""
    return Chunk.model_validate(
        {
            "chunk_id": row["chunk_id"],
            "doc_id": row["doc_id"],
            "text": row.get("text") or "",
            "company": row["company"],
            "url": row.get("url") or "",
            "title": row.get("title") or "",
            "document_type": row["document_type"],
            "source_domain": row.get("source_domain") or "",
            "chunk_index": int(float(row.get("chunk_index") or 0)),
            "strategy": row["strategy"],
            "parent_id": row.get("parent_id") or None,
            "publication_date": _parse_date(row.get("publication_date")),
            "industry": row.get("industry") or None,
            "confidence": _as_float(row.get("confidence"), 0.5),
            "token_count": int(float(row.get("token_count") or 0)),
            "entities": tuple(_parse_json(row.get("entities_json"), [])),
            "contextual_header": row.get("contextual_header") or "",
            "metadata": _parse_json(row.get("metadata_json"), {}),
        }
    )


def document_to_row(document: SourceDocument) -> Row:
    return {
        "doc_id": document.doc_id,
        "company": document.company,
        "url": document.url,
        "title": document.title,
        "text": document.text,
        "document_type": document.document_type.value,
        "source_domain": document.source_domain,
        "content_hash": document.content_hash,
        "publication_date": document.publication_date.isoformat() if document.publication_date else None,
        "retrieved_at": document.retrieved_at.isoformat(),
        "industry": document.industry,
        "language": document.language,
        "trust_score": document.trust_score,
        "metadata_json": json.dumps(document.metadata, default=str, sort_keys=True),
    }


def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


def _merge_statement(
    table: str,
    struct: str,
    columns: Sequence[str],
    key: str,
    *,
    extra_struct: str = "",
    extra_columns: Sequence[str] = (),
    insert: bool = True,
) -> str:
    all_struct = f"{struct}, {extra_struct}" if extra_struct else struct
    all_columns = [*columns, *extra_columns]
    updates = ", ".join(f"t.{c} = s.{c}" for c in all_columns if c != key)
    inserts = ", ".join(all_columns)
    values = ", ".join(f"s.{c}" for c in all_columns)
    statement = (
        f"MERGE INTO {table} AS t\n"  # nosec B608 - validated identifiers only
        f"USING (SELECT inline(from_json(:rows, 'ARRAY<STRUCT<{all_struct}>>'))) AS s\n"
        f"ON t.{key} = s.{key}\n"
        f"WHEN MATCHED THEN UPDATE SET {updates}, t.updated_at = current_timestamp()"
    )
    if insert:
        statement += (
            f"\nWHEN NOT MATCHED THEN INSERT ({inserts}, updated_at) VALUES ({values}, current_timestamp())"  # noqa: S608
        )
    return statement


def _is_parent(chunk: Chunk) -> bool:
    return chunk.strategy is ChunkStrategy.PARENT


# ------------------------------------------------------------ document store


class DeltaDocumentStore:
    """``DocumentStore`` over Unity Catalog Delta tables.

    ``documents`` holds source documents, ``chunks`` holds embedded child chunks
    (the Vector Search source table) and ``parent_chunks`` holds parent chunks.
    See the module docstring for the write-routing invariant.
    """

    def __init__(self, executor: StatementExecutor, *, catalog: str, schema: str) -> None:
        self._executor = executor
        self._documents = qualified_name(catalog, schema, "documents")
        self._chunks = qualified_name(catalog, schema, "chunks")
        self._parents = qualified_name(catalog, schema, "parent_chunks")
        cols = ", ".join(CHUNK_COLUMNS)
        by_ids = "WHERE array_contains(from_json(:ids, 'ARRAY<STRING>'), chunk_id)"
        self._sql = {
            "merge_documents": _merge_statement(
                self._documents, _DOCUMENT_STRUCT, DOCUMENT_COLUMNS, "doc_id"
            ),
            # Children without vectors may only refresh metadata of already-embedded rows:
            # inserting them would violate chunks.embedding NOT NULL.
            "merge_chunk_metadata": _merge_statement(
                self._chunks, _CHUNK_STRUCT, CHUNK_COLUMNS, "chunk_id", insert=False
            ),
            "merge_chunk_vectors": _merge_statement(
                self._chunks,
                _CHUNK_STRUCT,
                CHUNK_COLUMNS,
                "chunk_id",
                extra_struct="embedding: ARRAY<FLOAT>",
                extra_columns=("embedding",),
            ),
            "merge_parent_chunks": _merge_statement(self._parents, _CHUNK_STRUCT, CHUNK_COLUMNS, "chunk_id"),
            "list_chunks": (
                f"SELECT {cols} FROM {self._chunks} WHERE company = :company "  # noqa: S608 - validated identifiers only  # nosec B608
                f"UNION ALL SELECT {cols} FROM {self._parents} WHERE company = :company "
                "ORDER BY doc_id, chunk_index, chunk_id"
            ),
            "get_chunks": (
                f"SELECT {cols} FROM {self._chunks} {by_ids} "  # noqa: S608 - validated identifiers only  # nosec B608
                f"UNION ALL SELECT {cols} FROM {self._parents} {by_ids}"
            ),
            "get_parent_chunks": f"SELECT {cols} FROM {self._parents} {by_ids}",  # noqa: S608 - validated identifiers only  # nosec B608
            "known_hashes": f"SELECT DISTINCT content_hash FROM {self._documents} WHERE company = :company",  # noqa: S608 - validated identifiers only  # nosec B608
            "delete_chunks": f"DELETE FROM {self._chunks} WHERE company = :company",  # noqa: S608 - validated identifiers only  # nosec B608
            "delete_parent_chunks": f"DELETE FROM {self._parents} WHERE company = :company",  # noqa: S608 - validated identifiers only  # nosec B608
            "delete_documents": f"DELETE FROM {self._documents} WHERE company = :company",  # noqa: S608 - validated identifiers only  # nosec B608
        }

    @classmethod
    def from_settings(
        cls, statement_api: Any, settings: DatabricksSettings, **executor_kwargs: Any
    ) -> DeltaDocumentStore:
        if not settings.warehouse_id:
            raise ConfigurationError("databricks.warehouse_id must be set for the Delta document store")
        executor = StatementExecutor(statement_api, settings.warehouse_id, **executor_kwargs)
        return cls(executor, catalog=settings.catalog, schema=settings.schema_)

    @property
    def executor(self) -> StatementExecutor:
        return self._executor

    def _merge(self, op: str, rows: Sequence[Row]) -> int:
        written = 0
        for batch in _batches(rows, WRITE_BATCH_SIZE):
            result = self._executor.execute(self._sql[op], {"rows": json.dumps(list(batch))}, op=op)
            written += affected_rows(result, len(batch))
        return written

    @staticmethod
    def _dedupe(rows: Sequence[Row], key: str) -> list[Row]:
        return list({row[key]: row for row in rows}.values())

    @traced("delta.save_documents", span_type=SpanType.TOOL)
    def save_documents(self, documents: Sequence[SourceDocument]) -> int:
        rows = self._dedupe([document_to_row(d) for d in documents], "doc_id")
        self._merge("merge_documents", rows)
        return len(rows)

    @traced("delta.save_chunks", span_type=SpanType.TOOL)
    def save_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Persist parents to ``parent_chunks``; refresh metadata of already-embedded children.

        Child chunks are *inserted* only together with their vectors via
        :meth:`save_chunks_with_embeddings`; returns the number of distinct chunks accepted.
        """
        parents = self._dedupe([chunk_to_row(c) for c in chunks if _is_parent(c)], "chunk_id")
        children = self._dedupe([chunk_to_row(c) for c in chunks if not _is_parent(c)], "chunk_id")
        if parents:
            self._merge("merge_parent_chunks", parents)
        if children:
            self._merge("merge_chunk_metadata", children)
        return len(parents) + len(children)

    @traced("delta.save_chunk_vectors", span_type=SpanType.TOOL)
    def save_chunks_with_embeddings(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> int:
        """Upsert child chunks with their self-managed vectors into the index source table."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        parents = [c.chunk_id for c in chunks if _is_parent(c)]
        if parents:
            raise ValueError(f"parent chunks are never indexed; got {parents[:5]}")
        if any(len(vector) == 0 for vector in embeddings):
            raise ValueError("embeddings must be non-empty vectors")
        rows = [
            {**chunk_to_row(chunk), "embedding": [float(v) for v in vector]}
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        rows = self._dedupe(rows, "chunk_id")
        self._merge("merge_chunk_vectors", rows)
        return len(rows)

    def list_chunks(self, company: str) -> list[Chunk]:
        rows = self._executor.execute(self._sql["list_chunks"], {"company": company}, op="list_chunks")
        return [chunk_from_row(row) for row in rows]

    def _by_ids(self, op: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        wanted = list(dict.fromkeys(chunk_ids))
        found: dict[str, Chunk] = {}
        for batch in _batches(wanted, READ_BATCH_SIZE):
            rows = self._executor.execute(self._sql[op], {"ids": json.dumps(list(batch))}, op=op)
            for row in rows:
                chunk = chunk_from_row(row)
                found[chunk.chunk_id] = chunk
        return [found[cid] for cid in wanted if cid in found]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch children and parents by id (both tables), preserving request order."""
        return self._by_ids("get_chunks", chunk_ids)

    def get_parents(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Resolve the distinct parent chunks of ``chunks`` from ``parent_chunks``."""
        return self._by_ids("get_parent_chunks", [c.parent_id for c in chunks if c.parent_id is not None])

    def known_hashes(self, company: str) -> set[str]:
        rows = self._executor.execute(self._sql["known_hashes"], {"company": company}, op="known_hashes")
        return {str(row["content_hash"]) for row in rows if row.get("content_hash")}

    def delete_company_chunks(self, company: str) -> int:
        """Delete ``company``'s indexed (child) chunk rows; returns the number removed."""
        rows = self._executor.execute(self._sql["delete_chunks"], {"company": company}, op="delete_chunks")
        return affected_rows(rows, 0)

    def delete_company(self, company: str) -> int:
        """Right-to-erasure: remove child chunks, parent chunks and documents; returns chunks removed."""
        removed = self.delete_company_chunks(company)
        parents = self._executor.execute(
            self._sql["delete_parent_chunks"], {"company": company}, op="delete_parent_chunks"
        )
        self._executor.execute(self._sql["delete_documents"], {"company": company}, op="delete_documents")
        return removed + affected_rows(parents, 0)


# ------------------------------------------------------------ brief repository


class DeltaBriefRepository:
    """``BriefRepository`` over the Unity Catalog ``briefs`` table."""

    def __init__(self, executor: StatementExecutor, *, catalog: str, schema: str) -> None:
        self._executor = executor
        table = qualified_name(catalog, schema, "briefs")
        self._merge_sql = (
            f"MERGE INTO {table} AS t\n"  # noqa: S608 - validated identifiers only  # nosec B608
            "USING (SELECT :run_id AS run_id, :company AS company, :generated_at AS generated_at, "
            ":verdict AS verdict, :weighted_score AS weighted_score, :brief_json AS brief_json) AS s\n"
            "ON t.run_id = s.run_id\n"
            "WHEN MATCHED THEN UPDATE SET t.company = s.company, t.generated_at = s.generated_at, "
            "t.verdict = s.verdict, t.weighted_score = s.weighted_score, t.brief_json = s.brief_json, "
            "t.updated_at = current_timestamp()\n"
            "WHEN NOT MATCHED THEN INSERT (run_id, company, generated_at, verdict, weighted_score, "
            "brief_json, updated_at) VALUES (s.run_id, s.company, s.generated_at, s.verdict, "
            "s.weighted_score, s.brief_json, current_timestamp())"
        )
        self._get_sql = f"SELECT brief_json FROM {table} WHERE run_id = :run_id"  # noqa: S608 - validated identifiers only  # nosec B608

    @classmethod
    def from_settings(
        cls, statement_api: Any, settings: DatabricksSettings, **executor_kwargs: Any
    ) -> DeltaBriefRepository:
        if not settings.warehouse_id:
            raise ConfigurationError("databricks.warehouse_id must be set for the Delta brief repository")
        executor = StatementExecutor(statement_api, settings.warehouse_id, **executor_kwargs)
        return cls(executor, catalog=settings.catalog, schema=settings.schema_)

    @traced("delta.save_brief", span_type=SpanType.TOOL)
    def save(self, brief: ClientBrief) -> str:
        validate_run_id(brief.run_id)
        params: dict[str, Any] = {
            "run_id": brief.run_id,
            "company": brief.company,
            "generated_at": brief.generated_at,
            "verdict": brief.qualification.verdict.value,
            "weighted_score": float(brief.qualification.weighted_score),
            "brief_json": brief.model_dump_json(),
        }
        self._executor.execute(self._merge_sql, params, op="merge_brief")
        return brief.run_id

    def get(self, run_id: str) -> ClientBrief | None:
        validate_run_id(run_id)
        rows = self._executor.execute(self._get_sql, {"run_id": run_id}, op="get_brief")
        if not rows or rows[0].get("brief_json") is None:
            return None
        return ClientBrief.model_validate_json(str(rows[0]["brief_json"]))
