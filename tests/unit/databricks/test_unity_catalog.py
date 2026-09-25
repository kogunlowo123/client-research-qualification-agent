from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime

import pytest
from databricks.sdk.errors import platform
from databricks.sdk.service.sql import StatementState

from client_research_agent.config.settings import DatabricksSettings, ResilienceSettings
from client_research_agent.databricks.errors import DatabricksRequestError
from client_research_agent.databricks.unity_catalog import (
    CHECK_CONSTRAINTS,
    CHUNK_COLUMNS,
    DDL,
    TABLES,
    WRITE_BATCH_SIZE,
    DeltaBriefRepository,
    DeltaDocumentStore,
    SqlParam,
    StatementExecutor,
    affected_rows,
    apply_ddl,
    chunk_from_row,
    chunk_to_row,
    qualified_name,
    render_ddl,
    to_param,
    validate_identifier,
)
from client_research_agent.models import ChunkStrategy
from client_research_agent.utils.errors import (
    ConfigurationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from tests.contract.fakes import FakeStatementExecution, make_brief
from tests.support.doubles import make_chunk

FAST = ResilienceSettings(max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=0.0)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _executor(sql: FakeStatementExecution, **kwargs: object) -> StatementExecutor:
    kwargs.setdefault("resilience", FAST)
    kwargs.setdefault("sleep", lambda _s: None)
    return StatementExecutor(sql, "wh-1", **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------ identifiers/DDL


@pytest.mark.parametrize("bad", ["", "a-b", "a.b", "x; DROP TABLE y", "`q`", "naïve", 3])
def test_validate_identifier_rejects(bad: object) -> None:
    with pytest.raises(ConfigurationError):
        validate_identifier(bad)  # type: ignore[arg-type]


def test_qualified_name_and_ddl() -> None:
    assert qualified_name("cat", "sch", "chunks") == "`cat`.`sch`.`chunks`"
    statements = render_ddl("cat", "agent_prod")
    constraints = sum(len(v) for v in CHECK_CONSTRAINTS.values())
    assert len(statements) == 2 * len(TABLES) + 2 * constraints
    creates = [s for s in statements if s.startswith("CREATE TABLE IF NOT EXISTS")]
    assert len(creates) == len(DDL) == len(TABLES) == 5
    assert all("COMMENT '" in s for s in creates)
    cdf = [s for s in creates if "'delta.enableChangeDataFeed' = 'true'" in s]
    assert len(cdf) == 4
    chunk_ddl = next(s for s in creates if "`chunks` (" in s)
    assert "embedding ARRAY<FLOAT> NOT NULL" in chunk_ddl
    assert "PRIMARY KEY (chunk_id)" in chunk_ddl
    assert "delta.enableChangeDataFeed" in chunk_ddl
    parent_ddl = next(s for s in creates if "`parent_chunks` (" in s)
    assert "embedding" not in parent_ddl
    assert "enableChangeDataFeed" not in parent_ddl
    # CHECK constraints are dropped-if-present then re-added so the script stays idempotent.
    drop = statements.index(
        "ALTER TABLE `cat`.`agent_prod`.`chunks` DROP CONSTRAINT IF EXISTS chunks_embedded_children_only"
    )
    assert statements[drop + 1] == (
        "ALTER TABLE `cat`.`agent_prod`.`chunks` ADD CONSTRAINT chunks_embedded_children_only "
        "CHECK (strategy <> 'parent' AND size(embedding) > 0)"
    )
    assert any("parent_chunks_parents_only CHECK (strategy = 'parent')" in s for s in statements)
    tags = [s for s in statements if "SET TAGS" in s]
    assert any("`audit_log` SET TAGS" in s and "'data_classification' = 'confidential'" in s for s in tags)
    with pytest.raises(ConfigurationError):
        render_ddl("cat", "bad-schema")


def test_render_ddl_rejects_unsafe_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    from client_research_agent.databricks import unity_catalog

    monkeypatch.setitem(unity_catalog.TABLE_TAGS, "briefs", {"owner": "x'); DROP"})
    with pytest.raises(ConfigurationError, match="invalid tag"):
        render_ddl("cat", "sch")


def test_apply_ddl_runs_every_statement() -> None:
    sql = FakeStatementExecution()
    assert apply_ddl(_executor(sql), "cat", "sch") == len(render_ddl("cat", "sch")) == 14
    assert {e.op for e in sql.executed} == {"ddl"}
    assert all(e.params == {} for e in sql.executed)


# --------------------------------------------------------------- parameters


def test_to_param_types() -> None:
    assert to_param(None) == SqlParam(None)
    assert to_param(True) == SqlParam("true", "BOOLEAN")
    assert to_param(False) == SqlParam("false", "BOOLEAN")
    assert to_param(7) == SqlParam("7", "BIGINT")
    assert to_param(0.25) == SqlParam("0.25", "DOUBLE")
    assert to_param(datetime(2026, 1, 2, 3, 4, tzinfo=UTC)) == SqlParam(
        "2026-01-02T03:04:00+00:00", "TIMESTAMP"
    )
    assert to_param(date(2026, 1, 2)) == SqlParam("2026-01-02", "DATE")
    assert to_param("x") == SqlParam("x")
    explicit = SqlParam("1", "INT")
    assert to_param(explicit) is explicit
    with pytest.raises(TypeError):
        to_param(object())


def test_values_are_never_interpolated() -> None:
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(_executor(sql), catalog="cat", schema="sch")
    hostile = "Acme'; DROP TABLE chunks; --"
    store.save_chunks_with_embeddings([make_chunk("c1", "text", company=hostile)], [[1.0]])
    store.list_chunks(hostile)
    store.known_hashes(hostile)
    for executed in sql.executed:
        assert hostile not in executed.statement
        assert executed.warehouse_id == "wh-1"
        assert executed.kwargs["format"].value == "JSON_ARRAY"
        assert executed.kwargs["disposition"].value == "INLINE"
        assert re.search(r":(rows|company|ids)\b", executed.statement)
    assert [c.company for c in store.list_chunks(hostile)] == [hostile]
    assert sql.executed[1].params["company"].type == "STRING"


def test_affected_rows_fallback() -> None:
    assert affected_rows([{"num_affected_rows": "4"}], 1) == 4
    assert affected_rows([], 3) == 3
    assert affected_rows([{"num_affected_rows": None}], 2) == 2


# ----------------------------------------------------------------- executor


def test_executor_requires_warehouse() -> None:
    with pytest.raises(ConfigurationError):
        StatementExecutor(FakeStatementExecution(), "")


def test_executor_polls_pending_statements() -> None:
    sql = FakeStatementExecution()
    sql.pending_polls = 3
    sleeps: list[float] = []
    executor = StatementExecutor(sql, "wh-1", resilience=FAST, sleep=sleeps.append, poll_interval_seconds=0.5)
    store = DeltaDocumentStore(executor, catalog="c", schema="s")
    store.save_chunks([make_chunk("c1", "t")])
    assert sleeps == [0.5, 0.5, 0.5]
    assert executor.warehouse_id == "wh-1"


def test_executor_cancels_on_timeout() -> None:
    sql = FakeStatementExecution()
    sql.pending_polls = 100
    clock = _Clock()
    executor = StatementExecutor(
        sql,
        "wh-1",
        resilience=ResilienceSettings(max_attempts=1),
        sleep=clock.advance,
        clock=clock,
        poll_interval_seconds=10,
        max_wait_seconds=25,
    )
    with pytest.raises(UpstreamTimeoutError, match="exceeded"):
        executor.execute("SELECT 1", op="ddl")
    assert sql.cancelled == ["stmt-1"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (StatementState.FAILED, DatabricksRequestError),
        (StatementState.CANCELED, UpstreamServiceError),
        (StatementState.CLOSED, UpstreamServiceError),
    ],
)
def test_executor_terminal_states(state: StatementState, expected: type[Exception]) -> None:
    sql = FakeStatementExecution()
    sql.final_state = state
    with pytest.raises(expected, match="warehouse said no"):
        _executor(sql, resilience=ResilienceSettings(max_attempts=1)).execute("SELECT 1", op="ddl")


def test_executor_retries_mapped_sdk_errors() -> None:
    sql = FakeStatementExecution()
    sql.fail_with = [platform.TemporarilyUnavailable("warehouse starting"), platform.TooManyRequests("slow")]
    rows = _executor(sql).execute("SELECT 1", op="known_hashes", params={"company": "x"})
    assert rows == []
    assert len(sql.executed) == 1

    sql.fail_with = [platform.TooManyRequests("slow")] * 3
    with pytest.raises(RateLimitedError):
        _executor(sql).execute("SELECT 1", op="ddl")

    sql.fail_with = [platform.BadRequest("syntax error")]
    with pytest.raises(DatabricksRequestError, match="syntax error"):
        _executor(sql).execute("SELECT", op="ddl")


def test_executor_paginates_results() -> None:
    sql = FakeStatementExecution(page_size=3)
    store = DeltaDocumentStore(_executor(sql), catalog="c", schema="s")
    store.save_chunks(
        [make_chunk(f"c{i:02d}", "t", index=i, strategy=ChunkStrategy.PARENT) for i in range(10)]
    )
    assert [c.chunk_id for c in store.list_chunks("Acme Corp")] == [f"c{i:02d}" for i in range(10)]


# -------------------------------------------------------------- row mapping


def test_chunk_row_round_trip_and_wire_formats() -> None:
    chunk = make_chunk("c1", "body", parent_id="p1").model_copy(
        update={"entities": ("Acme",), "metadata": {"k": 1}, "industry": "Retail"}
    )
    row = chunk_to_row(chunk)
    assert tuple(row) == CHUNK_COLUMNS
    assert chunk_from_row(row) == chunk
    wire = {k: (None if v is None else str(v)) for k, v in row.items()}
    assert chunk_from_row(wire) == chunk


def test_chunk_from_row_tolerates_vector_search_types() -> None:
    row = chunk_to_row(make_chunk("c1", "body", publication_date=date(2026, 5, 1)))
    row.update(
        publication_date=20574,  # days since epoch -> 2026-05-01
        entities_json=["A"],
        metadata_json={"x": 1},
        confidence=None,
        chunk_index="2.0",
        parent_id="",
        industry="",
    )
    chunk = chunk_from_row(row)
    assert chunk.publication_date == date(2026, 5, 1)
    assert chunk.entities == ("A",)
    assert chunk.metadata == {"x": 1}
    assert chunk.confidence == 0.5
    assert chunk.chunk_index == 2
    assert chunk.parent_id is None
    assert chunk.industry is None
    row.update(publication_date="2026-05-01T00:00:00Z", entities_json="", metadata_json=None, confidence="")
    chunk = chunk_from_row(row)
    assert chunk.publication_date == date(2026, 5, 1)
    assert (chunk.entities, chunk.metadata, chunk.confidence) == ((), {}, 0.5)
    row["publication_date"] = date(2020, 1, 1)
    assert chunk_from_row(row).publication_date == date(2020, 1, 1)


# ---------------------------------------------------------- document store


def test_store_batches_writes_and_embeddings() -> None:
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(_executor(sql), catalog="c", schema="s")
    count = WRITE_BATCH_SIZE + 5
    chunks = [make_chunk(f"c{i}", "t", index=i) for i in range(count)]
    assert store.save_chunks_with_embeddings(chunks, [[float(i), 1.0] for i in range(count)]) == count
    merges = [e for e in sql.executed if e.op == "merge_chunk_vectors"]
    assert [len(json.loads(e.params["rows"].value or "[]")) for e in merges] == [WRITE_BATCH_SIZE, 5]
    assert "embedding: ARRAY<FLOAT>" in merges[0].statement
    assert sql.tables["chunks"]["c3"]["embedding"] == [3.0, 1.0]
    # Metadata-only upserts keep the existing vector.
    store.save_chunks([make_chunk("c3", "updated")])
    assert sql.tables["chunks"]["c3"]["embedding"] == [3.0, 1.0]
    assert sql.tables["chunks"]["c3"]["text"] == "updated"
    with pytest.raises(ValueError, match="embeddings"):
        store.save_chunks_with_embeddings(chunks[:1], [])


def test_store_routes_parents_and_children() -> None:
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(_executor(sql), catalog="c", schema="s")
    parent = make_chunk("p", "parent", strategy=ChunkStrategy.PARENT)
    child = make_chunk("k", "child", parent_id="p")
    assert store.save_chunks([parent, child]) == 2
    assert set(sql.tables["parent_chunks"]) == {"p"}
    # A child without a vector is never inserted into the index source table.
    assert sql.tables["chunks"] == {}
    ops = {e.op: e.statement for e in sql.executed}
    assert "WHEN NOT MATCHED" not in ops["merge_chunk_metadata"]
    assert "`parent_chunks`" in ops["merge_parent_chunks"]
    assert "embedding" not in ops["merge_parent_chunks"]
    assert store.get_parents([child]) == [parent]
    assert store.get_parents([parent]) == []
    assert store.save_chunks([]) == 0
    with pytest.raises(ValueError, match="never indexed"):
        store.save_chunks_with_embeddings([parent], [[1.0]])
    with pytest.raises(ValueError, match="non-empty"):
        store.save_chunks_with_embeddings([child], [[]])
    assert sql.tables["chunks"] == {}


def test_store_delete_and_affected_rows_fallback() -> None:
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(_executor(sql), catalog="c", schema="s")
    store.save_chunks_with_embeddings(
        [make_chunk("a", "t"), make_chunk("b", "t"), make_chunk("g", "t", company="Globex")], [[1.0]] * 3
    )
    store.save_chunks([make_chunk("pa", "t", strategy=ChunkStrategy.PARENT)])
    assert store.delete_company("Acme Corp") == 3
    assert set(sql.tables["chunks"]) == {"g"}
    assert sql.tables["parent_chunks"] == {}
    sql.report_affected_rows = False
    assert store.delete_company_chunks("Globex") == 0
    assert store.executor.warehouse_id == "wh-1"


def test_store_and_repository_from_settings() -> None:
    sql = FakeStatementExecution()
    settings = DatabricksSettings(catalog="cat", schema="agent_dev", warehouse_id="wh-9")
    store = DeltaDocumentStore.from_settings(sql, settings, resilience=FAST)
    store.save_chunks_with_embeddings([make_chunk("a", "t")], [[1.0]])
    assert "`cat`.`agent_dev`.`chunks`" in sql.executed[-1].statement
    repo = DeltaBriefRepository.from_settings(sql, settings)
    repo.save(make_brief("r1"))
    assert "`cat`.`agent_dev`.`briefs`" in sql.executed[-1].statement
    assert sql.executed[-1].params["generated_at"].type == "TIMESTAMP"
    assert sql.executed[-1].params["weighted_score"].type == "DOUBLE"
    missing = DatabricksSettings()
    with pytest.raises(ConfigurationError):
        DeltaDocumentStore.from_settings(sql, missing)
    with pytest.raises(ConfigurationError):
        DeltaBriefRepository.from_settings(sql, missing)


def test_brief_repository_rejects_bad_run_id_on_save() -> None:
    repo = DeltaBriefRepository(_executor(FakeStatementExecution()), catalog="c", schema="s")
    with pytest.raises(ValueError, match="run_id"):
        repo.save(make_brief("../../etc"))
