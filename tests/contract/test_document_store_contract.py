"""Behavioural contract every ``DocumentStore`` adapter must honour."""

from __future__ import annotations

from datetime import date

from client_research_agent.models import Chunk, ChunkStrategy, DocumentType, SourceDocument
from client_research_agent.services.ports import DocumentStore
from tests.contract.conftest import StoreAndIndex
from tests.support.doubles import make_chunk


def _document(doc_id: str, company: str = "Acme Corp", content_hash: str | None = None) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        company=company,
        url=f"https://example.com/{doc_id}",
        title=f"Title {doc_id}",
        text="Body text",
        document_type=DocumentType.INVESTOR_RELATIONS,
        source_domain="example.com",
        content_hash=content_hash or f"hash-{doc_id}",
        publication_date=date(2026, 3, 1),
        metadata={"lang": "en"},
    )


def _rich_chunk() -> Chunk:
    return make_chunk("rich", "Acme expands AI", doc_id="doc-r").model_copy(
        update={
            "entities": ("Acme", "Databricks"),
            "metadata": {"section": "news", "page": 2},
            "contextual_header": "Acme 2026 press release",
            "industry": "Manufacturing",
        }
    )


def test_satisfies_protocol(document_store: DocumentStore) -> None:
    assert isinstance(document_store, DocumentStore)


def test_save_documents_and_known_hashes(document_store: DocumentStore) -> None:
    written = document_store.save_documents(
        [_document("d1"), _document("d2"), _document("d1"), _document("x1", company="Globex")]
    )
    assert written == 3
    assert document_store.known_hashes("Acme Corp") == {"hash-d1", "hash-d2"}
    assert document_store.known_hashes("Globex") == {"hash-x1"}
    assert document_store.known_hashes("Nobody") == set()


def test_save_documents_is_idempotent(document_store: DocumentStore) -> None:
    document_store.save_documents([_document("d1", content_hash="old")])
    document_store.save_documents([_document("d1", content_hash="new")])
    assert document_store.known_hashes("Acme Corp") == {"new"}


def test_list_chunks_is_ordered_and_scoped(store_and_index: StoreAndIndex) -> None:
    chunks = [
        make_chunk("c3", "third", doc_id="doc-b", index=0),
        make_chunk("c2", "second", doc_id="doc-a", index=1),
        make_chunk("p0", "parent", doc_id="doc-a", index=0, strategy=ChunkStrategy.PARENT),
        make_chunk("c1", "first", doc_id="doc-a", index=0, parent_id="p0"),
        make_chunk("g1", "other", company="Globex", doc_id="doc-z"),
    ]
    store_and_index.ingest(chunks)
    store = store_and_index.store
    assert [c.chunk_id for c in store.list_chunks("Acme Corp")] == ["c1", "p0", "c2", "c3"]
    assert [c.chunk_id for c in store.list_chunks("Globex")] == ["g1"]
    assert store.list_chunks("Nobody") == []


def test_save_chunks_counts_distinct_chunks(document_store: DocumentStore) -> None:
    parent = make_chunk("p1", "parent", strategy=ChunkStrategy.PARENT)
    assert document_store.save_chunks([parent, parent, make_chunk("k1", "child", parent_id="p1")]) == 2


def test_get_chunks_spans_parents_and_children_in_request_order(store_and_index: StoreAndIndex) -> None:
    parent = make_chunk("p", "parent", strategy=ChunkStrategy.PARENT)
    store_and_index.ingest(
        [parent, *(make_chunk(f"c{i}", f"text {i}", index=i, parent_id="p") for i in range(5))]
    )
    found = store_and_index.store.get_chunks(["c4", "missing", "p", "c0", "c2", "c4"])
    assert [c.chunk_id for c in found] == ["c4", "p", "c0", "c2"]
    assert store_and_index.store.get_chunks([]) == []


def test_chunk_fields_round_trip(store_and_index: StoreAndIndex) -> None:
    rich = _rich_chunk()
    parent = rich.model_copy(update={"chunk_id": "rich-parent", "strategy": ChunkStrategy.PARENT})
    store_and_index.ingest([rich, parent])
    assert store_and_index.store.get_chunks(["rich", "rich-parent"]) == [rich, parent]


def test_chunk_upsert_replaces(store_and_index: StoreAndIndex) -> None:
    store_and_index.ingest([make_chunk("c1", "v1"), make_chunk("p1", "p-v1", strategy=ChunkStrategy.PARENT)])
    store_and_index.ingest([make_chunk("c1", "v2"), make_chunk("p1", "p-v2", strategy=ChunkStrategy.PARENT)])
    loaded = {c.chunk_id: c.text for c in store_and_index.store.list_chunks("Acme Corp")}
    assert loaded == {"c1": "v2", "p1": "p-v2"}


def test_metadata_refresh_of_indexed_children(store_and_index: StoreAndIndex) -> None:
    store_and_index.ingest([make_chunk("c1", "original")])
    store_and_index.store.save_chunks([make_chunk("c1", "re-enriched")])
    (loaded,) = store_and_index.store.get_chunks(["c1"])
    assert loaded.text == "re-enriched"
    # The vector survives a metadata-only write, so the chunk stays searchable.
    assert [r.chunk.chunk_id for r in store_and_index.index.search([1.0, 0.0], k=1)] == ["c1"]


def test_parents_are_resolvable_but_never_indexed(store_and_index: StoreAndIndex) -> None:
    parent = make_chunk("p1", "parent text", strategy=ChunkStrategy.PARENT)
    children = [
        make_chunk("k1", "child one", parent_id="p1", index=1),
        make_chunk("k2", "child two", parent_id="p1", index=2),
        make_chunk("k3", "orphan", index=3),
    ]
    store_and_index.ingest([parent, *children])
    get_parents = getattr(store_and_index.store, "get_parents")  # noqa: B009 - adapter extension, not on the port
    assert [c.chunk_id for c in get_parents(children)] == ["p1"]
    indexed = {r.chunk.chunk_id for r in store_and_index.index.search([1.0, 0.0], k=10)}
    assert indexed == {"k1", "k2", "k3"}
