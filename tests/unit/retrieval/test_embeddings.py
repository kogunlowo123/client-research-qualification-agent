from __future__ import annotations

from collections.abc import Sequence

import pytest

from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.embeddings import (
    CachingEmbeddingClient,
    HashingEmbeddingClient,
    cosine_similarity,
    embed_in_batches,
)
from client_research_agent.services.ports import EmbeddingClient
from client_research_agent.utils.errors import OutputValidationError


class CountingEmbedder:
    def __init__(self, *, drop_one: bool = False) -> None:
        self.inner = HashingEmbeddingClient(64)
        self.batches: list[list[str]] = []
        self.drop_one = drop_one

    @property
    def model_name(self) -> str:
        return "counting"

    @property
    def dimension(self) -> int:
        return 64

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        vectors = self.inner.embed(texts)
        return vectors[:-1] if self.drop_one else vectors


class TestHashingEmbedder:
    def test_satisfies_port_and_is_deterministic(self) -> None:
        embedder = HashingEmbeddingClient(128)
        assert isinstance(embedder, EmbeddingClient)
        assert embedder.dimension == 128
        assert embedder.model_name == "cra-hashing-embedder-v1"
        first = embedder.embed(["Acme migrates to the cloud"])[0]
        assert first == HashingEmbeddingClient(128).embed(["Acme migrates to the cloud"])[0]
        assert len(first) == 128
        assert sum(v * v for v in first) == pytest.approx(1.0)

    def test_similar_texts_are_closer_than_unrelated(self) -> None:
        embedder = HashingEmbeddingClient(512)
        a, b, c = embedder.embed(
            [
                "Acme cloud migration to Azure data centers",
                "cloud migration of Acme data centers to Azure completed",
                "quarterly dividend declared by the board",
            ]
        )
        assert cosine_similarity(a, b) > cosine_similarity(a, c)

    def test_empty_text_is_zero_vector(self) -> None:
        vector = HashingEmbeddingClient(16, use_bigrams=False).embed(["the of and"])[0]
        assert vector == [0.0] * 16

    @pytest.mark.parametrize(("dimension", "weight"), [(4, 0.5), (32, -1.0)])
    def test_invalid_configuration(self, dimension: int, weight: float) -> None:
        with pytest.raises(ValueError, match="must be"):
            HashingEmbeddingClient(dimension, bigram_weight=weight)

    def test_cosine_similarity_edge_cases(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="equal dimension"):
            cosine_similarity([1.0], [1.0, 2.0])


class TestCachingEmbedder:
    def test_caches_and_deduplicates(self) -> None:
        inner = CountingEmbedder()
        cache = CachingEmbeddingClient(inner, max_entries=10)
        first = cache.embed(["alpha", "beta", "alpha"])
        assert inner.batches == [["alpha", "beta"]]
        assert first[0] == first[2]
        second = cache.embed(["beta", "gamma"])
        assert inner.batches[-1] == ["gamma"]
        assert second[0] == first[1]
        assert cache.hits == 1
        assert cache.misses == 3
        assert len(cache) == 3
        assert cache.model_name == "counting"
        assert cache.dimension == 64
        assert get_metrics().counter("embedding.cache.hits") == 1

    def test_lru_eviction_and_clear(self) -> None:
        inner = CountingEmbedder()
        cache = CachingEmbeddingClient(inner, max_entries=2)
        cache.embed(["a"])
        cache.embed(["b"])
        cache.embed(["a"])  # refresh a
        cache.embed(["c"])  # evicts b
        cache.embed(["a"])
        assert inner.batches == [["a"], ["b"], ["c"]]
        cache.embed(["b"])
        assert inner.batches[-1] == ["b"]
        cache.clear()
        assert len(cache) == 0

    def test_rejects_mismatched_inner_output(self) -> None:
        cache = CachingEmbeddingClient(CountingEmbedder(drop_one=True))
        with pytest.raises(OutputValidationError):
            cache.embed(["a", "b"])

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="max_entries"):
            CachingEmbeddingClient(CountingEmbedder(), max_entries=0)


class TestBatching:
    def test_batches_in_order(self) -> None:
        inner = CountingEmbedder()
        texts = [f"text {i}" for i in range(5)]
        vectors = embed_in_batches(inner, texts, batch_size=2)
        assert [len(b) for b in inner.batches] == [2, 2, 1]
        assert vectors == inner.inner.embed(texts)
        assert embed_in_batches(inner, [], batch_size=3) == []

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            embed_in_batches(CountingEmbedder(), ["a"], batch_size=0)
        with pytest.raises(OutputValidationError):
            embed_in_batches(CountingEmbedder(drop_one=True), ["a", "b"], batch_size=2)
