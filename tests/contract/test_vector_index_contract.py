"""Behavioural contract every ``VectorIndex`` adapter must honour."""

from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.models import DocumentType
from client_research_agent.services.ports import VectorIndex
from tests.support.doubles import make_chunk

E1 = [1.0, 0.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0, 0.0]
E3 = [0.7, 0.7, 0.0, 0.0]
E4 = [0.0, 0.0, 1.0, 0.0]


def _seed(index: VectorIndex) -> None:
    chunks = [
        make_chunk("a1", "Acme lakehouse migration", publication_date=date(2026, 5, 1)),
        make_chunk(
            "a2",
            "Acme 10-K risk factors",
            document_type=DocumentType.SEC_FILING,
            publication_date=date(2025, 2, 1),
            doc_id="doc-2",
        ),
        make_chunk("a3", "Acme AI programme", publication_date=None, doc_id="doc-3"),
        make_chunk("g1", "Globex data platform", company="Globex", doc_id="doc-9"),
    ]
    assert index.upsert(chunks, [E1, E2, E3, E4]) == 4


def test_satisfies_protocol(vector_index: VectorIndex) -> None:
    assert isinstance(vector_index, VectorIndex)


def test_empty_index_returns_nothing(vector_index: VectorIndex) -> None:
    assert vector_index.search(E1, k=3) == []


def test_nearest_neighbour_ranks_first(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    results = vector_index.search(E1, k=3)
    assert [r.chunk.chunk_id for r in results] == ["a1", "a3", "a2"]  # a2/g1 tie broken by id
    assert [r.rank for r in results] == list(range(len(results)))
    assert results[0].score >= results[1].score >= results[2].score
    assert all(r.retriever == "dense" for r in results)


def test_k_limits_results(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    assert len(vector_index.search(E3, k=2)) == 2


def test_chunk_metadata_round_trips(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    top = vector_index.search(E2, k=1)[0].chunk
    assert top.chunk_id == "a2"
    assert top.document_type is DocumentType.SEC_FILING
    assert top.publication_date == date(2025, 2, 1)
    assert top.company == "Acme Corp"
    assert top.text == "Acme 10-K risk factors"


def test_company_filter(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    results = vector_index.search(E4, k=10, filters={"company": "Acme Corp"})
    assert {r.chunk.chunk_id for r in results} == {"a1", "a2", "a3"}


def test_document_type_membership_filter(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    results = vector_index.search(
        E1, k=10, filters={"document_type": [DocumentType.SEC_FILING], "company": "Acme Corp"}
    )
    assert [r.chunk.chunk_id for r in results] == ["a2"]


def test_publication_date_lower_bound_excludes_older_and_unknown(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    results = vector_index.search(E1, k=10, filters={"publication_date >=": date(2026, 1, 1)})
    assert {r.chunk.chunk_id for r in results} == {"a1", "g1"}


def test_negation_filter(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    results = vector_index.search(E1, k=10, filters={"company NOT": "Acme Corp"})
    assert [r.chunk.chunk_id for r in results] == ["g1"]


def test_unknown_filter_field_is_rejected(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    with pytest.raises(ValueError, match="not filterable"):
        vector_index.search(E1, k=1, filters={"password": "x"})


def test_upsert_replaces_existing_ids(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    replacement = make_chunk("a1", "Acme lakehouse migration (updated)")
    assert vector_index.upsert([replacement], [E4]) == 1
    results = vector_index.search(E4, k=10, filters={"company": "Acme Corp"})
    assert results[0].chunk.chunk_id == "a1"
    assert results[0].chunk.text.endswith("(updated)")
    assert len([r for r in results if r.chunk.chunk_id == "a1"]) == 1


def test_mismatched_lengths_rejected(vector_index: VectorIndex) -> None:
    with pytest.raises(ValueError, match="embeddings"):
        vector_index.upsert([make_chunk("x", "text")], [])


def test_invalid_k_rejected(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    with pytest.raises(ValueError, match="k must be"):
        vector_index.search(E1, k=0)


def test_delete_company(vector_index: VectorIndex) -> None:
    _seed(vector_index)
    assert vector_index.delete_company("Acme Corp") == 3
    remaining = vector_index.search(E1, k=10)
    assert [r.chunk.company for r in remaining] == ["Globex"]
    assert vector_index.delete_company("Acme Corp") == 0
