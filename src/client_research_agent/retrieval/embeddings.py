"""Embedding clients for local development and CI, plus caching and batching.

``HashingEmbeddingClient`` is a real (if lexical) embedding model: signed
feature hashing of stemmed unigrams and bigrams with sublinear term frequency,
L2-normalised so cosine similarity equals the dot product. It is fully
deterministic and needs no network, which makes retrieval quality gates in CI
reproducible. Production binds the Databricks model-serving embedder instead.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import Counter, OrderedDict
from collections.abc import Sequence
from itertools import pairwise

import numpy as np

from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.lexical import tokenize
from client_research_agent.services.ports import EmbeddingClient
from client_research_agent.utils.errors import OutputValidationError

_logger = get_logger(__name__)


def _feature_hash(feature: str) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, person=b"cra-hash").digest()
    return int.from_bytes(digest, "big")


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("vectors must be non-empty and of equal dimension")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HashingEmbeddingClient:
    """Deterministic feature-hashing embedder (unigrams + bigrams, signed, sublinear tf)."""

    def __init__(
        self,
        dimension: int = 512,
        *,
        model_name: str = "cra-hashing-embedder-v1",
        use_bigrams: bool = True,
        bigram_weight: float = 0.5,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension must be >= 8")
        if bigram_weight < 0:
            raise ValueError("bigram_weight must be non-negative")
        self._dimension = dimension
        self._model_name = model_name
        self._use_bigrams = use_bigrams
        self._bigram_weight = bigram_weight

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @traced("embeddings.hashing_embed", span_type=SpanType.EMBEDDING)
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        terms = tokenize(text)
        features: Counter[str] = Counter(terms)
        bigrams: Counter[str] = (
            Counter(f"{a} {b}" for a, b in pairwise(terms)) if self._use_bigrams else Counter()
        )
        vector = np.zeros(self._dimension, dtype=np.float64)
        for counts, scale in ((features, 1.0), (bigrams, self._bigram_weight)):
            for feature, tf in counts.items():
                hashed = _feature_hash(feature)
                sign = 1.0 if (hashed >> 63) & 1 else -1.0
                vector[hashed % self._dimension] += sign * scale * (1.0 + math.log(tf))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return [float(v) for v in vector]


class CachingEmbeddingClient:
    """LRU cache in front of any ``EmbeddingClient``, keyed by a hash of the text."""

    def __init__(self, inner: EmbeddingClient, *, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._inner = inner
        self._max_entries = max_entries
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        found: dict[str, tuple[float, ...]] = {}
        missing: dict[str, str] = {}
        with self._lock:
            for key, text in zip(keys, texts, strict=True):
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    found[key] = cached
                elif key not in missing:
                    missing[key] = text
            hit_count = len(texts) - sum(1 for key in keys if key in missing)
            self.hits += hit_count
            self.misses += len(missing)
        metrics = get_metrics()
        metrics.increment("embedding.cache.hits", hit_count)
        metrics.increment("embedding.cache.misses", len(missing))
        if missing:
            vectors = self._inner.embed(list(missing.values()))
            if len(vectors) != len(missing):
                raise OutputValidationError(
                    f"embedding client returned {len(vectors)} vectors for {len(missing)} texts"
                )
            with self._lock:
                for key, vector in zip(missing, vectors, strict=True):
                    frozen = tuple(float(v) for v in vector)
                    found[key] = frozen
                    self._cache[key] = frozen
                    self._cache.move_to_end(key)
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)
        return [list(found[key]) for key in keys]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def embed_in_batches(
    client: EmbeddingClient, texts: Sequence[str], batch_size: int = 64
) -> list[list[float]]:
    """Embed ``texts`` in fixed-size batches, validating the vector count per batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = list(texts[offset : offset + batch_size])
        embedded = client.embed(batch)
        if len(embedded) != len(batch):
            raise OutputValidationError(
                f"embedding client returned {len(embedded)} vectors for {len(batch)} texts"
            )
        vectors.extend(embedded)
    _logger.debug("embedded_batches", texts=len(texts), batch_size=batch_size)
    return vectors
