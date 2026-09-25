from __future__ import annotations

from collections.abc import Sequence

import pytest

from client_research_agent.config.settings import ChunkingSettings
from client_research_agent.models import Chunk, ChunkStrategy, SourceDocument
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.chunking import (
    Chunker,
    ParentChildChunker,
    RecursiveChunker,
    SemanticChunker,
    SpanSplitter,
    build_chunk,
    make_chunk_id,
)
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.tokenization import Tokenizer, get_tokenizer
from client_research_agent.utils.errors import CircuitOpenError
from tests.unit.retrieval.conftest import make_document


class DownEmbedder:
    @property
    def model_name(self) -> str:
        return "down"

    @property
    def dimension(self) -> int:
        return 8

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise CircuitOpenError("embeddings", 30.0)


class TopicEmbedder:
    """Two orthogonal topics: sentences mentioning 'revenue' vs everything else."""

    @property
    def model_name(self) -> str:
        return "topic"

    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "revenue" in t.lower() else [0.0, 1.0] for t in texts]


def _assert_offsets(document: SourceDocument, chunks: Sequence[Chunk]) -> None:
    for chunk in chunks:
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert document.text[start:end] == chunk.text


def test_chunk_id_is_deterministic() -> None:
    first = make_chunk_id("doc", ChunkStrategy.CHILD, 3)
    assert first == make_chunk_id("doc", ChunkStrategy.CHILD, 3)
    assert len(first) == 24
    assert first != make_chunk_id("doc", ChunkStrategy.PARENT, 3)
    assert first != make_chunk_id("doc", ChunkStrategy.CHILD, 4)


def test_build_chunk_carries_document_metadata(document: SourceDocument) -> None:
    chunk = build_chunk(
        document, (0, 40), index=0, strategy=ChunkStrategy.RECURSIVE, tokenizer=get_tokenizer()
    )
    assert chunk is not None
    assert chunk.company == "Acme Corp"
    assert chunk.url == document.url
    assert chunk.title == document.title
    assert chunk.document_type is document.document_type
    assert chunk.source_domain == "acme.example.com"
    assert chunk.publication_date == document.publication_date
    assert chunk.industry == "Industrial Manufacturing"
    assert chunk.confidence == pytest.approx(0.9)
    assert chunk.token_count > 0
    assert (
        build_chunk(document, (0, 0), index=1, strategy=ChunkStrategy.RECURSIVE, tokenizer=get_tokenizer())
        is None
    )


class TestSpanSplitter:
    def test_validation(self) -> None:
        tokenizer = Tokenizer(None)
        with pytest.raises(ValueError, match="max_tokens"):
            SpanSplitter(tokenizer, max_tokens=0)
        with pytest.raises(ValueError, match="overlap"):
            SpanSplitter(tokenizer, max_tokens=4, overlap_tokens=4)

    def test_hard_split_of_separator_free_text(self) -> None:
        splitter = SpanSplitter(Tokenizer(None), max_tokens=3, separators=("\n\n",))
        text = "one two three four five six seven"
        spans = splitter.split(text)
        assert [text[s:e] for s, e in spans] == ["one two three", "four five six", "seven"]
        assert splitter.split("   ") == []

    def test_giant_word_is_character_split(self) -> None:
        splitter = SpanSplitter(get_tokenizer(), max_tokens=8, separators=(" ",))
        text = "short " + "x9q" * 200
        spans = splitter.split(text)
        assert len(spans) > 2
        assert "".join(text[s:e] for s, e in spans).replace(" ", "") == text.replace(" ", "")
        assert all(get_tokenizer().count(text[s:e]) <= 8 for s, e in spans)

    def test_overlap_prefixes_previous_tail(self) -> None:
        splitter = SpanSplitter(Tokenizer(None), max_tokens=6, overlap_tokens=2, separators=(" ",))
        text = "a1 a2 a3 a4 a5 a6 a7 a8 a9"
        chunks = [text[s:e].strip() for s, e in splitter.split(text)]
        assert chunks[0] == "a1 a2 a3 a4"
        assert chunks[1].startswith("a3 a4")


class TestRecursiveChunker:
    def test_respects_budget_and_offsets(self, document: SourceDocument) -> None:
        chunker = RecursiveChunker(max_tokens=64, overlap_tokens=8)
        assert isinstance(chunker, Chunker)
        chunks = chunker.chunk(document)
        assert len(chunks) > 3
        assert all(c.token_count <= 64 for c in chunks)
        assert all(c.strategy is ChunkStrategy.RECURSIVE for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        _assert_offsets(document, chunks)
        assert get_metrics().counter("chunking.chunks", strategy="recursive") == len(chunks)

    def test_rechunking_is_stable(self, document: SourceDocument) -> None:
        chunker = RecursiveChunker.from_settings(
            ChunkingSettings(child_chunk_tokens=64, chunk_overlap_tokens=8)
        )
        assert [c.chunk_id for c in chunker.chunk(document)] == [c.chunk_id for c in chunker.chunk(document)]

    def test_short_document_is_single_chunk(self) -> None:
        document = make_document("Acme opened a new plant.")
        chunks = RecursiveChunker(max_tokens=64, overlap_tokens=0).chunk(document)
        assert [c.text for c in chunks] == ["Acme opened a new plant."]

    def test_whitespace_fallback_tokenizer(self, document: SourceDocument) -> None:
        chunks = RecursiveChunker(max_tokens=40, overlap_tokens=5, tokenizer=Tokenizer(None)).chunk(document)
        assert all(c.token_count <= 40 for c in chunks)


class TestSemanticChunker:
    def test_breaks_on_topic_shift(self) -> None:
        text = (
            "Acme builds robots. Acme sells robots worldwide. "
            "Revenue rose 8 percent. Revenue guidance was raised. "
            "Acme hired engineers. Acme opened a lab."
        )
        document = make_document(text)
        chunker = SemanticChunker(TopicEmbedder(), max_tokens=200, breakpoint_percentile=60, window=0)
        chunks = chunker.chunk(document)
        assert [c.text for c in chunks] == [
            "Acme builds robots. Acme sells robots worldwide.",
            "Revenue rose 8 percent. Revenue guidance was raised.",
            "Acme hired engineers. Acme opened a lab.",
        ]
        assert all(c.strategy is ChunkStrategy.SEMANTIC for c in chunks)
        assert not chunks[0].metadata["semantic_fallback"]
        _assert_offsets(document, chunks)

    def test_respects_max_tokens(self, document: SourceDocument) -> None:
        chunker = SemanticChunker.from_settings(
            HashingEmbeddingClient(64), ChunkingSettings(child_chunk_tokens=64, chunk_overlap_tokens=8)
        )
        chunks = chunker.chunk(document)
        assert chunks
        assert all(c.token_count <= 64 for c in chunks)

    def test_oversized_sentence_is_split(self) -> None:
        document = make_document(" ".join(["word"] * 150) + ". Short tail sentence.")
        chunks = SemanticChunker(TopicEmbedder(), max_tokens=64, window=1).chunk(document)
        assert all(c.token_count <= 64 for c in chunks)
        assert len(chunks) >= 3

    def test_single_sentence_and_empty(self) -> None:
        chunker = SemanticChunker(TopicEmbedder(), max_tokens=64)
        assert [c.text for c in chunker.chunk(make_document("Only one sentence here."))] == [
            "Only one sentence here."
        ]
        assert chunker.chunk(make_document("   ")) == []
        assert chunker.breakpoints("x", [(0, 1)]) == set()

    def test_embedding_outage_falls_back_to_recursive(self, document: SourceDocument) -> None:
        chunks = SemanticChunker(DownEmbedder(), max_tokens=64).chunk(document)
        assert chunks
        assert all(c.metadata["semantic_fallback"] for c in chunks)
        assert get_metrics().counter("chunking.semantic_fallback") == 1

    @pytest.mark.parametrize(("percentile", "window"), [(0.0, 1), (100.0, 1), (90.0, -1)])
    def test_validation(self, percentile: float, window: int) -> None:
        with pytest.raises(ValueError, match="must be"):
            SemanticChunker(TopicEmbedder(), breakpoint_percentile=percentile, window=window)


class TestParentChildChunker:
    def test_children_nest_inside_parents(self, document: SourceDocument) -> None:
        chunker = ParentChildChunker(parent_tokens=128, child_tokens=48, overlap_tokens=8)
        parents, children = chunker.split(document)
        assert len(parents) >= 2
        assert len(children) > len(parents)
        parent_by_id = {p.chunk_id: p for p in parents}
        assert all(p.strategy is ChunkStrategy.PARENT for p in parents)
        for child in children:
            assert child.strategy is ChunkStrategy.CHILD
            assert child.token_count <= 48
            parent = parent_by_id[child.parent_id or ""]
            assert parent.metadata["char_start"] <= child.metadata["char_start"]
            assert child.metadata["char_end"] <= parent.metadata["char_end"]
            assert child.chunk_id in parent.metadata["child_ids"]
        assert [c.chunk_index for c in children] == list(range(len(children)))
        _assert_offsets(document, [*parents, *children])
        assert chunker.chunk(document) == [*parents, *children]

    def test_from_settings_and_validation(self, document: SourceDocument) -> None:
        chunker = ParentChildChunker.from_settings(ChunkingSettings())
        parents, children = chunker.split(document)
        assert len(parents) == 1
        assert children
        with pytest.raises(ValueError, match="parent_tokens"):
            ParentChildChunker(parent_tokens=64, child_tokens=64)
