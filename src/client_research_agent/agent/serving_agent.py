"""MLflow ``ResponsesAgent`` that serves the research orchestrator on Model Serving.

Request (Responses API)::

    {"input": [{"role": "user", "content": "Research Contoso (ticker CTSO, domain contoso.com)"}],
     "custom_inputs": {"company_name": "...", "ticker": "...", "domain": "...",
                       "max_documents": 10, "requested_by": "..."}}

``custom_inputs`` wins; without it the company, ticker and domain are
extracted from the last user message (structured LLM extraction when a model is
configured, then a deterministic parser). The response carries the Markdown
brief as ``output_text`` and ``custom_outputs`` with ``run_id``, ``verdict``,
``weighted_score``, ``confidence``, ``citation_coverage`` and ``brief_json``.
``predict_stream`` streams the same Markdown as ``output_text`` deltas.

Identity and authorisation
    Who may call the endpoint is enforced by Model Serving itself (the
    ``CAN_QUERY`` permission on the serving endpoint). Inside the agent the RBAC
    principal is the endpoint's own service identity (``Role.SERVICE``,
    ``CRA_SERVICE_PRINCIPAL_ID`` or ``model-serving:<environment>``).
    ``custom_inputs.requested_by`` and ``context.user_id`` are *caller-asserted*
    and unauthenticated: they are recorded as ``asserted_requester`` (labelled
    unverified) in the audit trail and on the brief request, and used only as the
    key of a per-requester rate limit (defence in depth against unbounded
    consumption), never for authorisation.

MLflow is imported lazily; without it the class still works on plain dicts
(so the package does not require the ``databricks`` extra).
"""

from __future__ import annotations

import importlib
import os
import re
import threading
import uuid
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from client_research_agent.agent.factory import AgentRuntime, build_runtime
from client_research_agent.config.settings import build_settings
from client_research_agent.models import ResearchRequest
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.orchestration.orchestrator import ClientResearchOrchestrator, RunResult, new_run_id
from client_research_agent.security.rate_limiter import PrincipalRateLimiter, RateLimitExceededError
from client_research_agent.security.rbac import Principal, Role
from client_research_agent.services.ports import ChatMessage, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import CircuitOpenError, OutputValidationError, TransientError

STREAM_CHUNK_CHARS = 400
UNVERIFIED_PREFIX = "unverified:"
#: Research runs per asserted requester: bursts of 5, refilled at 20 per hour.
RATE_LIMIT_CAPACITY = 5.0
RATE_LIMIT_REFILL_PER_SECOND = 20 / 3600
DEFAULT_MAX_DOCUMENTS = 25
EXTRACTION_SYSTEM_PROMPT = (
    "Extract the company the user wants researched. Return JSON with company_name, and ticker and "
    "domain only when the user states them. Never guess a ticker or domain."
)
_TICKER = re.compile(r"\b(?:ticker|symbol|nyse|nasdaq)\s*[:=]?\s*\$?([A-Z][A-Z0-9.\-]{0,9})\b", re.IGNORECASE)
_DOMAIN = re.compile(
    r"\b(?:domain|website|site)\s*[:=]?\s*(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", re.I
)
_BARE_DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+\.(?:com|net|org|io|ai|co|example))\b", re.I)
_COMPANY_PATTERNS = (
    re.compile(
        r"\b(?:research|qualify|analy[sz]e|assess|evaluate|brief(?:\s+me)?\s+on|look\s+into|profile)\s+"
        r"(?P<name>[^\n(,;:]+?)(?=\s*(?:\(|,|;|:|\.\s|\.$|$|\s+and\s+(?:return|give|produce|write|create)|"
        r"\s+(?:with|for|using|including)\s))",
        re.IGNORECASE,
    ),
    re.compile(r"\bcompany\s*[:=]\s*(?P<name>[^\n,;(]+)", re.IGNORECASE),
)
_log = get_logger(__name__)
_MLFLOW_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)


class ServingInputError(ValueError):
    """The request does not identify a company to research."""


class _Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company_name: str = Field(min_length=1, max_length=200)
    ticker: str | None = Field(default=None, max_length=10)
    domain: str | None = Field(default=None, max_length=253)


@dataclass(frozen=True, slots=True)
class ParsedRequest:
    request: ResearchRequest
    asserted_requester: str
    source: str


def _load_base() -> type[Any]:
    try:
        base: type[Any] = importlib.import_module("mlflow.pyfunc").ResponsesAgent
    except (ImportError, AttributeError):
        return object
    return base


def _response_types() -> tuple[Any, Any] | None:
    try:
        module = importlib.import_module("mlflow.types.responses")
    except ImportError:
        return None
    return module.ResponsesAgentResponse, module.ResponsesAgentStreamEvent


def as_mapping(request: Any) -> dict[str, Any]:
    if hasattr(request, "model_dump"):
        dumped: dict[str, Any] = request.model_dump(exclude_none=True)
        return dumped
    if isinstance(request, Mapping):
        return dict(request)
    raise ServingInputError(f"unsupported request type {type(request).__name__}")


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, Mapping)]
        return "\n".join(p for p in parts if p)
    return ""


def last_user_message(items: Sequence[Any]) -> str:
    for item in reversed(list(items)):
        if isinstance(item, Mapping) and item.get("role") == "user":
            return message_text(item.get("content"))
    return ""


def _blank_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_company(text: str) -> _Extraction | None:
    """Deterministic parser for requests such as 'Research Contoso Ltd (ticker CTSO, domain contoso.com)'."""
    name: str | None = None
    for pattern in _COMPANY_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group("name").strip().strip("\"'").strip()
            break
    if not name:
        return None
    ticker = _TICKER.search(text)
    domain = _DOMAIN.search(text) or _BARE_DOMAIN.search(text)
    return _Extraction(
        company_name=name[:200],
        ticker=ticker.group(1).upper() if ticker else None,
        domain=domain.group(1).lower() if domain else None,
    )


def extract_with_llm(llm: LLMClient, text: str) -> _Extraction | None:
    try:
        result, _ = complete_structured(
            llm,
            [
                ChatMessage(role="system", content=EXTRACTION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=text[:4000]),
            ],
            _Extraction,
            max_repairs=1,
            max_tokens=200,
        )
    except _MLFLOW_FAILURES:
        return None
    return result


def parse_request(payload: Mapping[str, Any], *, llm: LLMClient | None = None) -> ParsedRequest:
    custom = payload.get("custom_inputs") or {}
    if not isinstance(custom, Mapping):
        raise ServingInputError("custom_inputs must be an object")
    context = payload.get("context") or {}
    user_id = _blank_to_none(context.get("user_id")) if isinstance(context, Mapping) else None
    company = _blank_to_none(custom.get("company_name"))
    ticker = _blank_to_none(custom.get("ticker"))
    domain = _blank_to_none(custom.get("domain"))
    source = "custom_inputs"
    if company is None:
        text = last_user_message(payload.get("input") or [])
        extracted = (extract_with_llm(llm, text) if llm is not None and text else None) or extract_company(
            text
        )
        if extracted is None:
            raise ServingInputError(
                "no company to research: set custom_inputs.company_name or name it in the message"
            )
        company = extracted.company_name
        ticker = ticker or extracted.ticker
        domain = domain or extracted.domain
        source = "message"
    asserted = _blank_to_none(custom.get("requested_by")) or user_id or "anonymous"
    try:
        max_documents = int(custom.get("max_documents") or DEFAULT_MAX_DOCUMENTS)
        request = ResearchRequest(
            company_name=company,
            ticker=ticker,
            domain=domain,
            cik=_blank_to_none(custom.get("cik")),
            industry=_blank_to_none(custom.get("industry")),
            max_documents=max_documents,
            requested_by=f"{UNVERIFIED_PREFIX}{asserted}"[:200],
        )
    except (ValidationError, ValueError) as exc:
        raise ServingInputError(f"invalid research request: {exc}") from exc
    return ParsedRequest(request=request, asserted_requester=asserted[:200], source=source)


def custom_outputs(result: RunResult) -> dict[str, Any]:
    brief = result.brief
    return {
        "run_id": brief.run_id,
        "verdict": brief.qualification.verdict.value,
        "weighted_score": brief.qualification.weighted_score,
        "confidence": brief.qualification.overall_confidence,
        "citation_coverage": brief.citation_report.coverage,
        "needs_review": result.review.needs_review,
        "brief_json": brief.model_dump_json(),
    }


def rate_limited_outputs(exc: RateLimitExceededError) -> tuple[str, dict[str, Any]]:
    retry = max(1, round(exc.retry_after_seconds))
    text = (
        f"Rate limit exceeded for this requester; retry in about {retry} seconds. No research was performed."
    )
    return text, {"error": "rate_limited", "retry_after_seconds": retry}


def text_output_item(text: str, item_id: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


RuntimeFactory = Callable[[], AgentRuntime]


def default_runtime_factory() -> AgentRuntime:
    """Runtime from environment settings (``CRA_ENVIRONMENT`` etc.); serving never starts MLflow runs."""
    return build_runtime(build_settings(), track_runs=False)


class ClientResearchResponsesAgent(_load_base()):  # type: ignore[misc]
    """Responses API agent: one research run per request."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory | None = None,
        *,
        rate_limiter: PrincipalRateLimiter | None = None,
        service_principal_id: str | None = None,
    ) -> None:
        self._factory = runtime_factory or default_runtime_factory
        self._runtime: AgentRuntime | None = None
        self._orchestrator: ClientResearchOrchestrator | None = None
        self._lock = threading.Lock()
        self._limiter = rate_limiter or PrincipalRateLimiter(
            RATE_LIMIT_CAPACITY, RATE_LIMIT_REFILL_PER_SECOND
        )
        self._service_principal_id = service_principal_id

    def service_principal(self, runtime: AgentRuntime) -> Principal:
        """The endpoint's own identity; callers are authorised by Model Serving (CAN_QUERY)."""
        identity = (
            self._service_principal_id
            or os.environ.get("CRA_SERVICE_PRINCIPAL_ID", "").strip()
            or f"model-serving:{runtime.settings.environment.value}"
        )
        return Principal(id=identity, roles=frozenset({Role.SERVICE}))

    def _get_orchestrator(self) -> ClientResearchOrchestrator:
        with self._lock:
            if self._orchestrator is None:
                self._runtime = self._factory()
                self._orchestrator = ClientResearchOrchestrator(self._runtime)
            return self._orchestrator

    def research(self, request: Any) -> RunResult:
        """Parse, rate-limit per asserted requester, audit the assertion, run the orchestrator."""
        orchestrator = self._get_orchestrator()
        runtime = orchestrator.runtime
        parsed = parse_request(as_mapping(request), llm=runtime.llm)
        principal = self.service_principal(runtime)
        self._limiter.acquire(parsed.asserted_requester)
        run_id = new_run_id(parsed.request.company_name)
        try:
            runtime.audit.append(
                "serving.request",
                {
                    "asserted_requester": parsed.asserted_requester,
                    "asserted_requester_verified": False,
                    "input_source": parsed.source,
                    "company": parsed.request.company_name,
                },
                principal=principal.id,
                run_id=run_id,
            )
        except Exception as exc:  # auditing must not block serving
            get_metrics().increment("audit.write_failures")
            _log.warning("serving.audit_failed", error=type(exc).__name__)
        return orchestrator.run(parsed.request, principal=principal, run_id=run_id)

    def _respond(self, request: Any) -> tuple[str, dict[str, Any]]:
        try:
            result = self.research(request)
        except RateLimitExceededError as exc:
            get_metrics().increment("serving.rate_limited")
            return rate_limited_outputs(exc)
        return result.markdown, custom_outputs(result)

    def predict(self, request: Any) -> Any:
        text, outputs = self._respond(request)
        item = text_output_item(text, f"msg_{uuid.uuid4().hex}")
        types = _response_types()
        if types is None:
            return {"output": [item], "custom_outputs": outputs}
        response_cls, _ = types
        return response_cls(output=[item], custom_outputs=outputs)

    def predict_stream(self, request: Any) -> Generator[Any, None, None]:
        text, outputs = self._respond(request)
        item_id = f"msg_{uuid.uuid4().hex}"
        types = _response_types()
        wrap: Callable[[dict[str, Any]], Any] = (
            (lambda event: event) if types is None else (lambda event: types[1](**event))
        )
        for start in range(0, len(text), STREAM_CHUNK_CHARS):
            yield wrap(
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "delta": text[start : start + STREAM_CHUNK_CHARS],
                }
            )
        yield wrap(
            {
                "type": "response.output_item.done",
                "item": text_output_item(text, item_id),
                "custom_outputs": outputs,
            }
        )
