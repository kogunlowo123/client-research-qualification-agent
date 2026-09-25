"""Embeddings over a Databricks Foundation Model API endpoint.

``databricks-gte-large-en`` produces 1024-dimensional vectors and accepts at most
150 inputs per request, so inputs are split into batches of ``batch_size``.
Each batch is retried independently under the endpoint's circuit breaker, and
the returned vectors are re-ordered by the response ``index`` field and their
dimension verified before they can reach the vector index.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import openai

from client_research_agent.config.settings import ResilienceSettings
from client_research_agent.databricks.model_serving import endpoint_breaker, map_openai_error, retry_policy
from client_research_agent.observability.metrics import Metrics, get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.utils.errors import ConfigurationError, UpstreamServiceError
from client_research_agent.utils.resilience import CircuitBreaker, call_with_retry

MAX_BATCH_SIZE = 150
DEFAULT_EMBEDDING_ENDPOINT = "databricks-gte-large-en"
DEFAULT_EMBEDDING_DIMENSION = 1024


class DatabricksEmbeddingClient:
    """``EmbeddingClient`` over a Foundation Model API embeddings endpoint."""

    def __init__(
        self,
        openai_client: Any,
        *,
        endpoint: str = DEFAULT_EMBEDDING_ENDPOINT,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        batch_size: int = MAX_BATCH_SIZE,
        resilience: ResilienceSettings | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        metrics: Metrics | None = None,
    ) -> None:
        if not endpoint:
            raise ConfigurationError("embedding endpoint name is required")
        if dimension < 1:
            raise ConfigurationError("embedding dimension must be positive")
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ConfigurationError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
        settings = resilience or ResilienceSettings()
        self._client = openai_client
        self._endpoint = endpoint
        self._dimension = dimension
        self._batch_size = batch_size
        self._policy = retry_policy(settings)
        self._breaker = breaker or endpoint_breaker(endpoint, settings)
        self._sleep = sleep
        self._metrics = metrics or get_metrics()

    @property
    def model_name(self) -> str:
        return self._endpoint

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._endpoint, input=batch)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, self._endpoint) from exc
        data = sorted(getattr(response, "data", None) or [], key=lambda item: int(item.index))
        if len(data) != len(batch):
            raise UpstreamServiceError(
                f"embedding endpoint '{self._endpoint}' returned {len(data)} vectors for {len(batch)} inputs"
            )
        vectors = [[float(value) for value in item.embedding] for item in data]
        for vector in vectors:
            if len(vector) != self._dimension:
                raise ConfigurationError(
                    f"embedding endpoint '{self._endpoint}' returned dimension {len(vector)}, "
                    f"expected {self._dimension}"
                )
        return vectors

    @traced("databricks.embeddings.embed", span_type=SpanType.EMBEDDING)
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        inputs = [text if text.strip() else " " for text in texts]
        vectors: list[list[float]] = []
        started = time.perf_counter()
        for offset in range(0, len(inputs), self._batch_size):
            batch = inputs[offset : offset + self._batch_size]
            vectors.extend(
                call_with_retry(
                    partial(self._embed_batch, batch),
                    policy=self._policy,
                    breaker=self._breaker,
                    sleep=self._sleep,
                )
            )
        if inputs:
            self._metrics.observe(
                "embedding.latency_ms", (time.perf_counter() - started) * 1000, endpoint=self._endpoint
            )
            self._metrics.increment("embedding.inputs", len(inputs), endpoint=self._endpoint)
        return vectors
