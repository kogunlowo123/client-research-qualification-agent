from __future__ import annotations

from typing import Any

import openai
import pytest

from client_research_agent.config.settings import ResilienceSettings
from client_research_agent.databricks.embeddings import MAX_BATCH_SIZE, DatabricksEmbeddingClient
from client_research_agent.databricks.model_serving import (
    DatabricksChatClient,
    FallbackLLMClient,
    ModelServingRequestError,
    endpoint_breaker,
    map_openai_error,
)
from client_research_agent.observability.metrics import Metrics
from client_research_agent.services.ports import ChatMessage
from client_research_agent.utils.errors import (
    CircuitOpenError,
    ConfigurationError,
    OutputValidationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import CircuitBreaker
from tests.contract.fakes import (
    FakeOpenAI,
    chat_completion,
    connection_error,
    status_error,
    timeout_error,
)

RETRY3 = ResilienceSettings(max_attempts=3, initial_backoff_seconds=0.0, max_backoff_seconds=0.0)
MESSAGES = [ChatMessage(role="user", content="hello")]


def _client(fake: FakeOpenAI, **kwargs: Any) -> DatabricksChatClient:
    kwargs.setdefault("resilience", RETRY3)
    kwargs.setdefault("breaker", CircuitBreaker("t", failure_threshold=10))
    kwargs.setdefault("metrics", Metrics())
    return DatabricksChatClient("ep", fake, sleep=lambda _s: None, **kwargs)


# ------------------------------------------------------------ error mapping


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (timeout_error(), UpstreamTimeoutError),
        (connection_error(), UpstreamServiceError),
        (status_error(openai.RateLimitError, 429), RateLimitedError),
        (status_error(openai.AuthenticationError, 401), ConfigurationError),
        (status_error(openai.InternalServerError, 500), UpstreamServiceError),
        (status_error(openai.APIStatusError, 408), UpstreamTimeoutError),
        (status_error(openai.APIStatusError, 502), UpstreamServiceError),
        (status_error(openai.NotFoundError, 404), ConfigurationError),
        (status_error(openai.BadRequestError, 400), ModelServingRequestError),
        (openai.OpenAIError("generic"), ModelServingRequestError),
    ],
)
def test_map_openai_error(exc: BaseException, expected: type[Exception]) -> None:
    assert isinstance(map_openai_error(exc, "ep"), expected)


def test_retry_after_header_parsing() -> None:
    ok = map_openai_error(status_error(openai.RateLimitError, 429, {"retry-after": "2.5"}), "ep")
    assert isinstance(ok, RateLimitedError)
    assert ok.retry_after_seconds == 2.5
    for header in ("soon", "-1"):
        bad = map_openai_error(status_error(openai.RateLimitError, 429, {"retry-after": header}), "ep")
        assert isinstance(bad, RateLimitedError)
        assert bad.retry_after_seconds is None
    none = map_openai_error(status_error(openai.RateLimitError, 429), "ep")
    assert isinstance(none, RateLimitedError)
    assert none.retry_after_seconds is None
    typed = OutputValidationError("x")
    assert map_openai_error(typed, "ep") is typed


# --------------------------------------------------------------- chat client


def test_retries_transient_errors_then_succeeds() -> None:
    metrics = Metrics()
    fake = FakeOpenAI(
        chat_script=[timeout_error(), status_error(openai.InternalServerError, 503), chat_completion("ok")]
    )
    client = _client(fake, metrics=metrics)
    assert client.complete(MESSAGES).content == "ok"
    assert len(fake.chat_requests) == 3
    assert metrics.counter("llm.retries", endpoint="ep") == 2
    assert metrics.counter("llm.tokens.prompt", endpoint="ep") == 11
    assert metrics.histogram("llm.latency_ms", endpoint="ep").count == 1


def test_non_transient_errors_are_not_retried() -> None:
    metrics = Metrics()
    fake = FakeOpenAI(chat_script=[status_error(openai.BadRequestError, 400), chat_completion("never")])
    with pytest.raises(ModelServingRequestError):
        _client(fake, metrics=metrics).complete(MESSAGES)
    assert len(fake.chat_requests) == 1
    assert metrics.counter("llm.errors", endpoint="ep", error="ModelServingRequestError") == 1


def test_empty_choices_are_transient() -> None:
    empty = chat_completion("x").model_copy(update={"choices": []})
    fake = FakeOpenAI(chat_script=[empty, chat_completion("second")])
    assert _client(fake).complete(MESSAGES).content == "second"


def test_missing_optional_fields_default() -> None:
    completion = chat_completion(None).model_copy(update={"usage": None, "model": ""})
    completion.choices[0].finish_reason = None  # type: ignore[assignment]
    response = _client(FakeOpenAI(chat_script=[completion])).complete(MESSAGES)
    assert (response.content, response.model, response.finish_reason) == ("", "ep", "stop")
    assert response.usage.total_tokens == 0


def test_breaker_opens_and_short_circuits() -> None:
    breaker = CircuitBreaker("ep", failure_threshold=2, reset_timeout_seconds=60)
    fake = FakeOpenAI(chat_script=[timeout_error(), timeout_error(), chat_completion("unused")])
    client = _client(fake, breaker=breaker, resilience=ResilienceSettings(max_attempts=1))
    for _ in range(2):
        with pytest.raises(UpstreamTimeoutError):
            client.complete(MESSAGES)
    with pytest.raises(CircuitOpenError):
        client.complete(MESSAGES)
    assert len(fake.chat_requests) == 2
    assert client.breaker is breaker


def test_validation() -> None:
    with pytest.raises(ConfigurationError):
        DatabricksChatClient("", FakeOpenAI())
    with pytest.raises(ValueError, match="message"):
        _client(FakeOpenAI()).complete([])


def test_endpoint_breaker_is_shared_per_endpoint() -> None:
    settings = ResilienceSettings()
    assert endpoint_breaker("shared-ep", settings) is endpoint_breaker("shared-ep", settings)
    assert endpoint_breaker("shared-ep", settings) is not endpoint_breaker("other-ep", settings)
    client = DatabricksChatClient("shared-ep", FakeOpenAI())
    assert client.breaker is endpoint_breaker("shared-ep", settings)


def test_fallback_on_open_breaker_and_passthrough() -> None:
    open_breaker = CircuitBreaker("p", failure_threshold=1, reset_timeout_seconds=60)
    open_breaker.on_failure()
    primary = _client(FakeOpenAI(), breaker=open_breaker)
    fallback_fake = FakeOpenAI(chat_script=[chat_completion("fb")])
    fallback = DatabricksChatClient("fb-ep", fallback_fake, breaker=CircuitBreaker("f"), metrics=Metrics())
    metrics = Metrics()
    llm = FallbackLLMClient(primary, fallback, metrics=metrics)
    assert llm.fallback_model_name == "fb-ep"
    assert llm.complete(MESSAGES, json_mode=True, max_tokens=9).content == "fb"
    assert fallback_fake.chat_requests[0]["max_tokens"] == 9
    assert fallback_fake.chat_requests[0]["response_format"] == {"type": "json_object"}
    assert metrics.counter("llm.fallback", primary="ep", fallback="fb-ep", reason="CircuitOpenError") == 1


def test_fallback_not_used_when_primary_succeeds() -> None:
    primary_fake = FakeOpenAI(chat_script=[chat_completion("primary")])
    fallback_fake = FakeOpenAI()
    llm = FallbackLLMClient(_client(primary_fake), _client(fallback_fake))
    assert llm.complete(MESSAGES).content == "primary"
    assert fallback_fake.chat_requests == []


# ---------------------------------------------------------------- embeddings


def _embedder(fake: FakeOpenAI, **kwargs: Any) -> DatabricksEmbeddingClient:
    kwargs.setdefault("resilience", RETRY3)
    kwargs.setdefault("breaker", CircuitBreaker("emb", failure_threshold=10))
    kwargs.setdefault("metrics", Metrics())
    kwargs.setdefault("dimension", fake.dimension)
    return DatabricksEmbeddingClient(fake, sleep=lambda _s: None, **kwargs)


def test_embeddings_batching_order_and_blank_inputs() -> None:
    fake = FakeOpenAI(dimension=4, reverse_embedding_order=True)
    metrics = Metrics()
    client = _embedder(fake, batch_size=2, metrics=metrics)
    texts = ["a", "bb", "   ", "dddd", "e"]
    vectors = client.embed(texts)
    assert [len(r["input"]) for r in fake.embedding_requests] == [2, 2, 1]
    assert fake.embedding_requests[1]["input"] == [" ", "dddd"]
    assert vectors == [fake.vector_for(t if t.strip() else " ") for t in texts]
    assert metrics.counter("embedding.inputs", endpoint="databricks-gte-large-en") == 5
    assert client.model_name == "databricks-gte-large-en"


def test_embeddings_retry_and_errors() -> None:
    fake = FakeOpenAI(dimension=4, embedding_script=[status_error(openai.RateLimitError, 429)])
    assert len(_embedder(fake).embed(["x"])) == 1
    assert len(fake.embedding_requests) == 2

    short = FakeOpenAI(dimension=4)
    short.embedding_script.append(short.embed_default("m", ["only-one"]))
    with pytest.raises(UpstreamServiceError, match="returned 1 vectors for 2"):
        _embedder(short, resilience=ResilienceSettings(max_attempts=1)).embed(["a", "b"])

    with pytest.raises(ConfigurationError, match="dimension"):
        _embedder(FakeOpenAI(dimension=4), dimension=1024).embed(["x"])

    denied = FakeOpenAI(embedding_script=[status_error(openai.PermissionDeniedError, 403)])
    with pytest.raises(ConfigurationError):
        _embedder(denied).embed(["x"])
    assert len(denied.embedding_requests) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"endpoint": ""}, {"dimension": 0}, {"batch_size": 0}, {"batch_size": MAX_BATCH_SIZE + 1}],
)
def test_embedding_client_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        DatabricksEmbeddingClient(FakeOpenAI(), **kwargs)
