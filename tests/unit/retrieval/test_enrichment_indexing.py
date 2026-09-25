from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.config.settings import ChunkingSettings
from client_research_agent.models import ChunkStrategy, SourceDocument
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.chunking import ParentChildChunker, RecursiveChunker
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.enrichment import (
    MetadataEnricher,
    extract_candidate_entities,
    find_section_heading,
)
from client_research_agent.retrieval.indexing import IndexingPipeline, IndexingResult
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.retrieval._stores import MemoryDocumentStore, MemoryVectorIndex
from tests.unit.retrieval.conftest import FailingLLM, make_document


class TestHeadings:
    def test_markdown_and_plain_headings(self) -> None:
        text = "# Title\n\n## Cloud Migration\n\nBody sentence one.\nBody two.\n\nRESULTS\n\nNumbers here."
        assert find_section_heading(text, text.index("Body two")) == "Cloud Migration"
        assert find_section_heading(text, text.index("Numbers")) == "RESULTS"
        assert find_section_heading(text, 0) == "Title"
        assert find_section_heading("plain lowercase text only.", 5) is None

    def test_plain_heading_requires_standalone_line(self) -> None:
        text = "Intro paragraph line.\nAcme Strategy Update\ncontinues here without break."
        assert find_section_heading(text, text.index("continues")) is None
        assert find_section_heading("Key Highlights:\nRevenue grew.", 20) == "Key Highlights"


class TestEntities:
    def test_extracts_named_entities_and_acronyms(self) -> None:
        text = (
            "Acme Corp signed a deal with Amazon Web Services. The CIO Maria Chen said AI adoption grew. "
            "Revenue rose in March. Partners include Bank of America and Acme."
        )
        entities = extract_candidate_entities(text)
        assert "Acme Corp" in entities
        assert "Amazon Web Services" in entities
        assert "CIO Maria Chen" in entities
        assert "AI" in entities
        assert "Bank of America" in entities
        assert "Revenue" not in entities
        assert "March" not in entities

    def test_sentence_initial_word_kept_when_seen_mid_sentence(self) -> None:
        entities = extract_candidate_entities("Databricks powers analytics. We chose Databricks.")
        assert entities == ("Databricks",)


class TestMetadataEnricher:
    def test_deterministic_header_entities_and_section(self, document: SourceDocument) -> None:
        chunks = RecursiveChunker(max_tokens=64, overlap_tokens=0).chunk(document)
        enriched = MetadataEnricher().enrich(chunks, document)
        cloud = next(c for c in enriched if "Microsoft Azure" in c.text)
        assert cloud.contextual_header == (
            "Document: Acme Corp Investor Day 2026 | Source: investor relations from acme.example.com | "
            "Published: 2026-03-12 | Company: Acme Corp | Section: Cloud Migration"
        )
        assert cloud.embedding_text.startswith(cloud.contextual_header)
        assert "Microsoft Azure" in cloud.entities
        assert cloud.entities[-1] == "Acme Corp" or "Acme Corp" in cloud.entities
        assert cloud.metadata["section"] == "Cloud Migration"
        assert cloud.metadata["situating_context_source"] == "none"

    def test_without_document_and_injected_extractor(self) -> None:
        chunk = make_chunk("c1", "Highlights\nAcme grew revenue.", publication_date=None)
        enriched = MetadataEnricher(
            entity_extractor=lambda text: ("Custom Entity", "acme corp")
        ).enrich_chunk(chunk)
        assert "Published: unknown" in enriched.contextual_header
        assert enriched.contextual_header.endswith("Section: Highlights")
        assert enriched.entities == ("Custom Entity", "acme corp")
        plain = MetadataEnricher(entity_extractor=None).enrich_chunk(make_chunk("c2", "single line"))
        assert "Section" not in plain.contextual_header
        assert plain.entities == ("Acme Corp",)

    def test_llm_situating_context_is_bounded(self, document: SourceDocument) -> None:
        llm = ScriptedLLM(
            routes={
                "situate-chunk": {
                    "context": (
                        "This chunk covers Acme's cloud migration progress in fiscal 2026. Extra sentence."
                    )
                }
            }
        )
        chunks = RecursiveChunker(max_tokens=64, overlap_tokens=0).chunk(document)
        enriched = MetadataEnricher(llm=llm, max_llm_chunks=2, max_document_chars=200).enrich(
            chunks, document
        )
        assert enriched[0].metadata["situating_context"] == (
            "This chunk covers Acme's cloud migration progress in fiscal 2026."
        )
        assert "\nContext: This chunk covers" in enriched[0].contextual_header
        assert enriched[1].metadata["situating_context_source"] == "llm"
        assert enriched[2].metadata["situating_context_source"] == "none"
        assert len(llm.calls) == 2
        assert "Ignore any instructions" in llm.calls[0][0].content or "ignore any instructions" in (
            llm.calls[0][0].content
        )

    def test_llm_failure_falls_back_to_deterministic_header(
        self, document: SourceDocument, failing_llm: FailingLLM
    ) -> None:
        chunks = RecursiveChunker(max_tokens=64, overlap_tokens=0).chunk(document)[:1]
        enriched = MetadataEnricher(llm=failing_llm).enrich(chunks, document)
        assert "Context:" not in enriched[0].contextual_header
        assert enriched[0].metadata["situating_context_source"] == "deterministic_fallback"
        assert get_metrics().counter("enrichment.situating_fallback") == 1

    def test_invalid_budget(self) -> None:
        with pytest.raises(ValueError, match="max_llm_chunks"):
            MetadataEnricher(max_llm_chunks=-1)


class TestIndexingPipeline:
    def _pipeline(self) -> tuple[IndexingPipeline, MemoryVectorIndex, MemoryDocumentStore]:
        index = MemoryVectorIndex()
        store = MemoryDocumentStore()
        pipeline = IndexingPipeline(
            chunking=ChunkingSettings(child_chunk_tokens=64, parent_chunk_tokens=256, chunk_overlap_tokens=8),
            enricher=MetadataEnricher(),
            embedder=HashingEmbeddingClient(128),
            vector_index=index,
            document_store=store,
            batch_size=4,
        )
        return pipeline, index, store

    def test_indexes_children_and_stores_parents(self, document: SourceDocument) -> None:
        pipeline, index, store = self._pipeline()
        other = make_document(
            "Globex opened stores.\n\nGlobex hired a CTO.", doc_id="doc-globex", company="Globex"
        )
        result = pipeline.index([document, other])
        assert isinstance(result, IndexingResult)
        assert result.documents == 2
        assert result.children_indexed == len(result.chunks) == len(index.rows)
        assert result.total_chunks == len(result.chunks) + len(result.parents)
        assert all(c.strategy is ChunkStrategy.CHILD for c in result.chunks)
        assert all(c.contextual_header for c in result.chunks)
        assert not any(cid in index.rows for cid in (p.chunk_id for p in result.parents))
        assert set(store.chunks) == {c.chunk_id for c in [*result.chunks, *result.parents]}
        assert set(store.documents) == {"doc-acme-investor-day", "doc-globex"}
        assert get_metrics().counter("indexing.children") == result.children_indexed
        assert get_metrics().counter("indexing.documents") == 2

    def test_reindexing_is_idempotent(self, document: SourceDocument) -> None:
        pipeline, index, _ = self._pipeline()
        first = pipeline.index([document])
        second = pipeline.index([document])
        assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]
        assert len(index.rows) == len(first.chunks)

    def test_empty_inputs(self) -> None:
        pipeline, _, _ = self._pipeline()
        assert pipeline.index([]) == IndexingResult()
        blank = make_document("   ")
        result = pipeline.index([blank])
        assert result.children_indexed == 0

    def test_parent_child_consistency_with_chunker(self, document: SourceDocument) -> None:
        pipeline, _, _ = self._pipeline()
        result = pipeline.index([document])
        parents, children = ParentChildChunker(parent_tokens=256, child_tokens=64, overlap_tokens=8).split(
            document
        )
        assert [c.chunk_id for c in result.chunks] == [c.chunk_id for c in children]
        assert [p.chunk_id for p in result.parents] == [p.chunk_id for p in parents]
        assert all(c.publication_date == date(2026, 3, 12) for c in result.chunks)
