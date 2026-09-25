"""Chat completions over Databricks Model Serving (Foundation Model API).

Foundation Model API endpoints speak the OpenAI chat-completions protocol at
``{host}/serving-endpoints``; ``model`` is the serving endpoint name. We use the
``openai`` client directly (built by :func:`auth.build_openai_client`) rather
than the SDK's ``serving_endpoints.get_open_ai_client()``, which is deprecated
in the installed ``databricks-sdk`` and pins its own retry behaviour.

Every call runs through a per-endpoint circuit breaker and the configured retry
policy. ``openai`` exceptions are mapped onto the package's typed errors so the
resilience layer can distinguish retryable from fatal failures.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import openai

from client_research_agent.config.settings import ResilienceSettings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import Metrics, get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.services.ports import ChatMessage, LLMClient, LLMResponse, LLMUsage
from client_research_agent.utils.errors import (
    AgentError,
    CircuitOpenError,
    ConfigurationError,
    RateLimitedError,
    TransientError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import CircuitBreaker, RetryPolicy, call_with_retry

_log = get_logger(__name__)


class ModelServingRequestError(AgentError):
    """A non-retryable rejection from a serving endpoint (bad request, unknown endpoint, ...)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _retry_after_seconds(exc: openai.APIStatusError) -> float | None:
    raw = exc.response.headers.get("retry-after")
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def map_openai_error(exc: BaseException, endpoint: str) -> AgentError:
    """Map ``openai`` exceptions onto the typed hierarchy; ``AgentError`` passes through unchanged."""
    if isinstance(exc, AgentError):
        return exc
    prefix = f"serving endpoint '{endpoint}'"
    if isinstance(exc, openai.APITimeoutError):
        return UpstreamTimeoutError(f"{prefix} timed out")
    if isinstance(exc, openai.APIConnectionError):
        return UpstreamServiceError(f"{prefix} connection failed: {exc}")
    if isinstance(exc, openai.RateLimitError):
        return RateLimitedError(f"{prefix} rate limited", retry_after_seconds=_retry_after_seconds(exc))
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return ConfigurationError(
            f"{prefix} rejected credentials ({exc.status_code}); check CAN_QUERY grants"
        )
    if isinstance(exc, openai.InternalServerError):
        return UpstreamServiceError(f"{prefix} failed: {exc}", status_code=exc.status_code)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 408:
            return UpstreamTimeoutError(f"{prefix} timed out ({status})")
        if status >= 500:
            return UpstreamServiceError(f"{prefix} failed: {exc}", status_code=status)
        if status == 404:
            return ConfigurationError(f"{prefix} does not exist or is not visible to this principal")
        return ModelServingRequestError(f"{prefix} rejected the request: {exc}", status_code=status)
    return ModelServingRequestError(f"{prefix}: {type(exc).__name__}: {exc}")


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def endpoint_breaker(endpoint: str, resilience: ResilienceSettings) -> CircuitBreaker:
    """Process-wide breaker per serving endpoint, shared by every client of that endpoint."""
    with _breakers_lock:
        breaker = _breakers.get(endpoint)
        if breaker is None:
            breaker = CircuitBreaker(
                f"serving:{endpoint}",
                failure_threshold=resilience.breaker_failure_threshold,
                reset_timeout_seconds=resilience.breaker_reset_seconds,
            )
            _breakers[endpoint] = breaker
        return breaker


def retry_policy(resilience: ResilienceSettings) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=resilience.max_attempts,
        initial_backoff_seconds=resilience.initial_backoff_seconds,
        max_backoff_seconds=resilience.max_backoff_seconds,
    )


class DatabricksChatClient:
    """``LLMClient`` over a Foundation Model API chat endpoint.

    ``openai_client`` is an ``openai.OpenAI`` instance pointed at
    ``{host}/serving-endpoints`` (see :func:`auth.build_openai_client`).
    ``supports_json_mode`` controls whether ``json_mode=True`` is sent as
    ``response_format={"type": "json_object"}``; set it to ``False`` for endpoints
    that reject ``response_format`` (the structured-output layer still extracts
    and validates JSON from free text).
    """

    def __init__(
        self,
        endpoint: str,
        openai_client: Any,
        *,
        supports_json_mode: bool = True,
        resilience: ResilienceSettings | None = None,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], None] = time.sleep,
        metrics: Metrics | None = None,
    ) -> None:
        if not endpoint:
            raise ConfigurationError("serving endpoint name is required")
        settings = resilience or ResilienceSettings()
        self._endpoint = endpoint
        self._client = openai_client
        self._supports_json_mode = supports_json_mode
        self._policy = retry_policy(settings)
        self._breaker = breaker or endpoint_breaker(endpoint, settings)
        self._sleep = sleep
        self._metrics = metrics or get_metrics()

    @property
    def model_name(self) -> str:
        return self._endpoint

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def _request(
        self, messages: Sequence[ChatMessage], temperature: float, max_tokens: int, json_mode: bool
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._endpoint,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode and self._supports_json_mode:
            request["response_format"] = {"type": "json_object"}
        return request

    def _invoke(self, request: dict[str, Any]) -> LLMResponse:
        try:
            completion = self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise map_openai_error(exc, self._endpoint) from exc
        return self._to_response(completion)

    def _on_retry(self, attempt: int, exc: BaseException) -> None:
        self._metrics.increment("llm.retries", endpoint=self._endpoint)
        _log.warning("llm.retry", endpoint=self._endpoint, attempt=attempt, error=type(exc).__name__)

    def _to_response(self, completion: Any) -> LLMResponse:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise UpstreamServiceError(f"serving endpoint '{self._endpoint}' returned no choices")
        choice = choices[0]
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            content=getattr(choice.message, "content", None) or "",
            model=getattr(completion, "model", None) or self._endpoint,
            usage=LLMUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
            finish_reason=getattr(choice, "finish_reason", None) or "stop",
        )

    @traced("databricks.chat.complete", span_type=SpanType.LLM)
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("at least one message is required")
        request = self._request(messages, temperature, max_tokens, json_mode)
        started = time.perf_counter()
        try:
            response = call_with_retry(
                lambda: self._invoke(request),
                policy=self._policy,
                breaker=self._breaker,
                sleep=self._sleep,
                on_retry=self._on_retry,
            )
        except AgentError as exc:
            self._metrics.increment("llm.errors", endpoint=self._endpoint, error=type(exc).__name__)
            raise
        self._metrics.observe(
            "llm.latency_ms", (time.perf_counter() - started) * 1000, endpoint=self._endpoint
        )
        self._metrics.increment("llm.tokens.prompt", response.usage.prompt_tokens, endpoint=self._endpoint)
        self._metrics.increment(
            "llm.tokens.completion", response.usage.completion_tokens, endpoint=self._endpoint
        )
        return response


class FallbackLLMClient:
    """Try ``primary``; on a transient failure or open breaker, answer from ``fallback``.

    Non-transient errors (bad request, auth, validation) are not masked by the
    fallback: they indicate a defect that the second endpoint would repeat.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient, *, metrics: Metrics | None = None) -> None:
        self._primary = primary
        self._fallback = fallback
        self._metrics = metrics or get_metrics()

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    @property
    def fallback_model_name(self) -> str:
        return self._fallback.model_name

    @traced("databricks.chat.fallback", span_type=SpanType.LLM)
    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        try:
            return self._primary.complete(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
            )
        except (TransientError, CircuitOpenError) as exc:
            self._metrics.increment(
                "llm.fallback",
                primary=self._primary.model_name,
                fallback=self._fallback.model_name,
                reason=type(exc).__name__,
            )
            _log.warning(
                "llm.fallback",
                primary=self._primary.model_name,
                fallback=self._fallback.model_name,
                reason=type(exc).__name__,
            )
            return self._fallback.complete(
                messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
            )
