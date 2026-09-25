"""Behavioural contract for ``LLMClient`` / ``EmbeddingClient`` Databricks adapters.

The chat suite runs against the bare ``DatabricksChatClient`` and against a
``FallbackLLMClient`` whose primary *and* fallback are Databricks clients, so
the wrapper is proven not to alter response mapping or error semantics.
"""

from __future__ import annotations

import openai
import pytest

from client_research_agent.config.settings import ResilienceSettings
from client_research_agent.databricks.embeddings import DatabricksEmbeddingClient
from client_research_agent.databricks.model_serving import DatabricksChatClient, FallbackLLMClient
from client_research_agent.observability.metrics import Metrics
from client_research_agent.services.ports import ChatMessage, EmbeddingClient, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import (
    ConfigurationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import CircuitBreaker
from tests.contract.fakes import FakeOpenAI, chat_completion, status_error, timeout_error

ONE_SHOT = ResilienceSettings(max_attempts=1, initial_backoff_seconds=0.0, max_backoff_seconds=0.0)
MESSAGES = [ChatMessage(role="system", content="be terse"), ChatMessage(role="user", content="hi")]


def _chat(fake: FakeOpenAI, endpoint: str, *, json_mode_supported: bool = True) -> DatabricksChatClient:
    return DatabricksChatClient(
        endpoint,
        fake,
        supports_json_mode=json_mode_supported,
        resilience=ONE_SHOT,
        breaker=CircuitBreaker(endpoint, failure_threshold=100),
        sleep=lambda _s: None,
        metrics=Metrics(),
    )


@pytest.fixture(params=["direct", "fallback-wrapped"])
def llm_and_fake(request: pytest.FixtureRequest) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI()
    primary = _chat(fake, "databricks-claude-sonnet-4")
    if request.param == "direct":
        return primary, fake
    # A separate fake for the fallback so primary errors must surface unless transient.
    fallback = _chat(FakeOpenAI(chat_script=[chat_completion("fallback")]), "databricks-llama")
    return FallbackLLMClient(primary, fallback, metrics=Metrics()), fake


def test_satisfies_protocol(llm_and_fake: tuple[LLMClient, FakeOpenAI]) -> None:
    llm, _ = llm_and_fake
    assert isinstance(llm, LLMClient)
    assert llm.model_name == "databricks-claude-sonnet-4"


def test_response_and_usage_mapping(llm_and_fake: tuple[LLMClient, FakeOpenAI]) -> None:
    llm, fake = llm_and_fake
    fake.chat_script.append(chat_completion("hello", model="served-model", prompt=21, completion=4))
    response = llm.complete(MESSAGES, temperature=0.2, max_tokens=64)
    assert response.content == "hello"
    assert response.model == "served-model"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (21, 4)
    assert response.usage.total_tokens == 25
    assert response.finish_reason == "stop"
    sent = fake.chat_requests[-1]
    assert sent["model"] == "databricks-claude-sonnet-4"
    assert sent["messages"] == [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
    assert (sent["temperature"], sent["max_tokens"]) == (0.2, 64)
    assert "response_format" not in sent


def test_json_mode_requests_json_object(llm_and_fake: tuple[LLMClient, FakeOpenAI]) -> None:
    llm, fake = llm_and_fake
    fake.chat_script.append(chat_completion('{"ok": true}'))
    llm.complete(MESSAGES, json_mode=True)
    assert fake.chat_requests[-1]["response_format"] == {"type": "json_object"}


def test_works_with_structured_output_layer(llm_and_fake: tuple[LLMClient, FakeOpenAI]) -> None:
    from pydantic import BaseModel

    class Answer(BaseModel):
        ok: bool

    llm, fake = llm_and_fake
    fake.chat_script.extend([chat_completion("not json"), chat_completion('```json\n{"ok": true}\n```')])
    parsed, _ = complete_structured(llm, MESSAGES, Answer)
    assert parsed.ok is True


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.AuthenticationError, 401), ConfigurationError),
        (status_error(openai.PermissionDeniedError, 403), ConfigurationError),
        (status_error(openai.NotFoundError, 404), ConfigurationError),
    ],
)
def test_non_transient_errors_are_not_masked(
    llm_and_fake: tuple[LLMClient, FakeOpenAI], error: Exception, expected: type[Exception]
) -> None:
    llm, fake = llm_and_fake
    fake.chat_script.append(error)
    with pytest.raises(expected):
        llm.complete(MESSAGES)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.RateLimitError, 429, {"retry-after": "3"}), RateLimitedError),
        (timeout_error(), UpstreamTimeoutError),
        (status_error(openai.InternalServerError, 503), UpstreamServiceError),
    ],
)
def test_transient_error_mapping(error: Exception, expected: type[Exception]) -> None:
    llm = _chat(FakeOpenAI(chat_script=[error]), "databricks-claude-sonnet-4")
    with pytest.raises(expected):
        llm.complete(MESSAGES)


def test_transient_errors_are_served_by_fallback() -> None:
    primary = _chat(FakeOpenAI(chat_script=[timeout_error()]), "primary")
    fallback = _chat(FakeOpenAI(chat_script=[chat_completion("from fallback")]), "fallback")
    metrics = Metrics()
    llm = FallbackLLMClient(primary, fallback, metrics=metrics)
    assert llm.complete(MESSAGES).content == "from fallback"
    assert (
        metrics.counter("llm.fallback", primary="primary", fallback="fallback", reason="UpstreamTimeoutError")
        == 1
    )


def test_json_mode_can_be_disabled_per_endpoint() -> None:
    fake = FakeOpenAI(chat_script=[chat_completion("{}")])
    _chat(fake, "no-json-endpoint", json_mode_supported=False).complete(MESSAGES, json_mode=True)
    assert "response_format" not in fake.chat_requests[-1]


def test_embedding_client_contract() -> None:
    fake = FakeOpenAI(dimension=8)
    client: EmbeddingClient = DatabricksEmbeddingClient(
        fake, dimension=8, resilience=ONE_SHOT, breaker=CircuitBreaker("emb"), metrics=Metrics()
    )
    assert isinstance(client, EmbeddingClient)
    assert client.dimension == 8
    vectors = client.embed(["alpha", "beta"])
    assert vectors == [fake.vector_for("alpha"), fake.vector_for("beta")]
    assert client.embed([]) == []
