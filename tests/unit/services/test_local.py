from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime
from pathlib import Path

import pytest

from client_research_agent.models import ChunkStrategy, DocumentType
from client_research_agent.services.local import (
    FilterClause,
    InMemoryDocumentStore,
    InMemoryVectorIndex,
    JsonlAuditSink,
    JsonlBriefRepository,
    chunk_matches,
    normalize_filter_value,
    parse_filters,
    validate_run_id,
)
from client_research_agent.services.ports import AuditSink
from tests.contract.fakes import make_brief
from tests.support.doubles import make_chunk

# ------------------------------------------------------------------ filters


def test_normalize_filter_value_handles_enums_dates_and_collections() -> None:
    assert normalize_filter_value(DocumentType.SEC_FILING) == "sec_filing"
    assert normalize_filter_value(date(2026, 1, 2)) == "2026-01-02"
    assert normalize_filter_value(datetime(2026, 1, 2, 3, 4)) == "2026-01-02"
    assert normalize_filter_value((DocumentType.SEC_FILING, "x")) == ["sec_filing", "x"]
    assert normalize_filter_value(3) == 3


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("company", FilterClause("company", "=", "A")),
        ("company NOT", FilterClause("company", "NOT", "A")),
        ("NOT company", FilterClause("company", "NOT", "A")),
        ("company !=", FilterClause("company", "NOT", "A")),
        ("company not", FilterClause("company", "NOT", "A")),
    ],
)
def test_parse_filter_keys(key: str, expected: FilterClause) -> None:
    assert parse_filters({key: "A"}) == [expected]


@pytest.mark.parametrize(
    ("filters", "message"),
    [
        ({"a b c": 1}, "malformed"),
        ({"company LIKE": "x"}, "unsupported filter operator"),
        ({"embedding": 1}, "not filterable"),
        ({"chunk_index >": [1, 2]}, "does not accept a list"),
    ],
)
def test_parse_filters_rejects_bad_input(filters: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_filters(filters)


def test_parse_filters_none_is_empty() -> None:
    assert parse_filters(None) == []


def test_chunk_matches_comparisons() -> None:
    chunk = make_chunk("c", "t", index=3)
    assert chunk_matches(chunk, parse_filters({"chunk_index >": 2, "chunk_index <=": 3}))
    assert not chunk_matches(chunk, parse_filters({"chunk_index <": 3}))
    assert not chunk_matches(chunk, parse_filters({"chunk_index >": 3}))
    assert chunk_matches(chunk, parse_filters({"document_type NOT": ["sec_filing"]}))
    assert not chunk_matches(chunk, parse_filters({"document_type NOT": ["press_release"]}))
    assert not chunk_matches(chunk, parse_filters({"industry >=": "a"}))  # NULL never compares
    # Incomparable types never match instead of raising.
    assert not chunk_matches(chunk, parse_filters({"chunk_index >=": "three"}))


# -------------------------------------------------------------- vector index


def test_vector_index_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        InMemoryVectorIndex(dimension=0)
    index = InMemoryVectorIndex(dimension=2)
    assert index.dimension == 2
    with pytest.raises(ValueError, match="dimension"):
        index.upsert([make_chunk("a", "t")], [[1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        index.upsert([make_chunk("a", "t")], [[float("nan"), 0.0]])
    with pytest.raises(ValueError, match="1-D"):
        index.upsert([make_chunk("a", "t")], [[]])
    assert index.upsert([], []) == 0
    assert len(index) == 0


def test_vector_index_infers_dimension_only_on_success() -> None:
    index = InMemoryVectorIndex()
    with pytest.raises(ValueError, match="non-finite"):
        index.upsert([make_chunk("a", "t")], [[float("inf"), 1.0, 0.0]])
    assert index.dimension is None
    index.upsert([make_chunk("a", "t")], [[1.0, 0.0]])
    assert index.dimension == 2


def test_vector_index_zero_vector_and_no_matches() -> None:
    index = InMemoryVectorIndex()
    index.upsert([make_chunk("a", "t"), make_chunk("b", "u")], [[0.0, 0.0], [1.0, 0.0]])
    results = index.search([0.0, 0.0], k=5)
    assert {r.score for r in results} == {0.0}
    assert index.search([1.0, 0.0], k=5, filters={"company": "Nobody"}) == []


def test_vector_index_duplicate_ids_in_one_batch_count_once() -> None:
    index = InMemoryVectorIndex()
    assert index.upsert([make_chunk("a", "t"), make_chunk("a", "t2")], [[1.0, 0.0], [0.0, 1.0]]) == 1
    assert len(index) == 1
    assert index.search([0.0, 1.0], k=1)[0].chunk.text == "t2"


def test_vector_index_is_thread_safe() -> None:
    index = InMemoryVectorIndex()

    def writer(offset: int) -> None:
        for i in range(50):
            index.upsert([make_chunk(f"c{offset}-{i}", "t")], [[1.0, float(i)]])
            index.search([1.0, 0.0], k=3)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(index) == 200


# ------------------------------------------------------------ document store


def test_document_store_extensions() -> None:
    from client_research_agent.models import SourceDocument

    store = InMemoryDocumentStore()
    doc = SourceDocument(
        doc_id="d1",
        company="Acme Corp",
        url="https://a.example/1",
        title="t",
        text="x",
        document_type=DocumentType.PRESS_RELEASE,
        source_domain="a.example",
        content_hash="h1",
    )
    other = doc.model_copy(update={"doc_id": "d0", "content_hash": "h0"})
    store.save_documents([doc, other])
    assert store.get_document("d1") == doc
    assert store.get_document("missing") is None
    assert [d.doc_id for d in store.list_documents("Acme Corp")] == ["d0", "d1"]
    parent = make_chunk("p", "parent", strategy=ChunkStrategy.PARENT)
    not_parent = make_chunk("q", "child used as parent id")
    child = make_chunk("k", "child", parent_id="p")
    bogus = make_chunk("z", "points at non-parent", parent_id="q")
    store.save_chunks([parent, not_parent, child, bogus])
    assert store.get_parents([child, bogus]) == [parent]
    assert store.delete_company("Acme Corp") == 4
    assert store.list_chunks("Acme Corp") == []
    assert store.known_hashes("Acme Corp") == set()


# ---------------------------------------------------------- brief repository


def test_validate_run_id() -> None:
    assert validate_run_id("run_1.2-3") == "run_1.2-3"
    for bad in ("", "../x", "a..b", ".hidden", "x" * 201, "a\\b"):
        with pytest.raises(ValueError, match="run_id"):
            validate_run_id(bad)


def test_brief_repository_writes_atomically(tmp_path: Path) -> None:
    repo = JsonlBriefRepository(tmp_path / "nested" / "briefs")
    assert repo.directory.is_dir()
    repo.save(make_brief("run-b"))
    repo.save(make_brief("run-a"))
    assert repo.list_run_ids() == ["run-a", "run-b"]
    assert not list(repo.directory.glob("*.tmp"))
    stored = json.loads((repo.directory / "run-a.json").read_text(encoding="utf-8"))
    assert stored["run_id"] == "run-a"


def test_brief_repository_cleans_temp_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = JsonlBriefRepository(tmp_path)
    repo.save(make_brief("run-1"))
    original = (tmp_path / "run-1.json").read_text(encoding="utf-8")

    def broken_replace(src: str, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disk full"):
        repo.save(make_brief("run-1", company="Globex"))
    assert not list(tmp_path.glob("*.tmp"))
    assert (tmp_path / "run-1.json").read_text(encoding="utf-8") == original


# ---------------------------------------------------------------- audit sink


def test_audit_sink_appends_json_lines(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path / "audit" / "events.jsonl")
    assert isinstance(sink, AuditSink)
    assert sink.read_all() == []
    sink.record("brief.saved", {"run_id": "r1", "when": date(2026, 1, 1)})
    sink.record("brief.read", {"run_id": "r1"})
    events = sink.read_all()
    assert [e["event_type"] for e in events] == ["brief.saved", "brief.read"]
    assert events[0]["payload"] == {"run_id": "r1", "when": "2026-01-01"}
    assert "ts" in events[0]
    assert sink.path.read_text(encoding="utf-8").count("\n") == 2


def test_audit_sink_rejects_empty_event_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="event_type"):
        JsonlAuditSink(tmp_path / "a.jsonl").record("", {})


def test_audit_sink_concurrent_writes_do_not_interleave(tmp_path: Path) -> None:
    sink = JsonlAuditSink(tmp_path / "a.jsonl")
    threads = [
        threading.Thread(target=lambda n=n: [sink.record("e", {"n": n, "i": i}) for i in range(25)])
        for n in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(sink.read_all()) == 100
