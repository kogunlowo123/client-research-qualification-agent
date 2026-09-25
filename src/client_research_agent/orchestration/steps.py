"""The building blocks the orchestrator composes, one per agent/pipeline stage.

* :class:`ResearchPlanner` - the Research Agent: focus questions, targeted
  queries and source priorities (``research_planner`` prompt, deterministic
  fallback from the criteria catalogue).
* :class:`EvidenceGatherer` - the Evidence Gathering Agent: ingestion, text
  sanitisation, document-level injection and poisoning screening, PII
  redaction, persistence, indexing and lineage.
* :class:`GuardedRetriever` / :class:`ChunkBlocklist` - retrieval that can
  never surface a quarantined chunk or another company's chunk.
* :class:`KnowledgeRefresher` - the bounded CRAG knowledge-refresh hook
  (targeted re-ingestion from seed and corporate URLs).
* :func:`strip_statements` - removes brief statements flagged by the output
  guard or the responsible-AI policy.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.governance.lineage import LineageRecorder
from client_research_agent.models import (
    Chunk,
    ChunkStrategy,
    ClientBrief,
    Criterion,
    DocumentType,
    ResearchRequest,
    RetrievedChunk,
    SourceDocument,
)
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.prompts.registry import PromptRegistry
from client_research_agent.qualification.criteria import all_definitions
from client_research_agent.research.ingestion import IngestionResult
from client_research_agent.retrieval.crag import RetrievalVerdict
from client_research_agent.retrieval.pipeline import RetrievalOutcome, Retriever
from client_research_agent.security.poisoning import ChunkAction
from client_research_agent.security.prompt_injection import PromptInjectionDetector
from client_research_agent.services.ports import ChatMessage, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import (
    AgentError,
    CircuitOpenError,
    OutputValidationError,
    TransientError,
)

if TYPE_CHECKING:
    from client_research_agent.agent.factory import AgentRuntime, IngestionRunner

LLM_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)
#: Characters per screening window; small enough that one injected paragraph dominates its window.
SCREEN_WINDOW_CHARS = 1500
#: Text handed to the document-level poisoning guard (its duplicate check is quadratic in shingles).
GUARD_TEXT_CHARS = 20_000
REFRESH_MAX_DOCUMENTS = 5
_log = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = (
    "You are the research planning agent of an enterprise client-qualification system. "
    "You only plan searches over public sources; you never state facts about the company."
)


# ----------------------------------------------------------------------------- research plan
class _PlannedQueryDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    criterion: str = Field(default="", max_length=80)
    query: str = Field(min_length=3, max_length=300)
    rationale: str = Field(default="", max_length=600)


class ResearchPlanDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    queries: list[_PlannedQueryDraft] = Field(default_factory=list, max_length=30)
    hypotheses: list[str] = Field(default_factory=list, max_length=30)


@dataclass(frozen=True, slots=True)
class PlannedQuery:
    query: str
    criterion: Criterion | None = None
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    queries: tuple[PlannedQuery, ...]
    focus_questions: tuple[str, ...]
    source_priorities: tuple[DocumentType, ...]
    source: str
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "queries": [
                {
                    "query": q.query,
                    "criterion": q.criterion.value if q.criterion else None,
                    "rationale": q.rationale,
                }
                for q in self.queries
            ],
            "focus_questions": list(self.focus_questions),
            "source_priorities": [p.value for p in self.source_priorities],
        }


def source_priorities(request: ResearchRequest) -> tuple[DocumentType, ...]:
    """Which public sources to trust and read first for this request."""
    regulated = [DocumentType.SEC_FILING, DocumentType.EARNINGS_RELEASE, DocumentType.INVESTOR_RELATIONS]
    company = [DocumentType.PRESS_RELEASE, DocumentType.LEADERSHIP_ANNOUNCEMENT]
    order: list[DocumentType] = (
        [*regulated, *company] if (request.ticker or request.cik) else [*company, *regulated]
    )
    if request.domain:
        order.append(DocumentType.CORPORATE_WEBPAGE)
    if request.seed_urls:
        order.append(DocumentType.ANALYST_PUBLIC)
    return tuple(order)


class ResearchPlanner:
    """Research Agent: plans targeted queries per criterion before evidence is gathered."""

    def __init__(self, llm: LLMClient | None, registry: PromptRegistry, *, max_queries: int = 10) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be >= 1")
        self._llm = llm
        self._template = registry.get("research_planner")
        self._max_queries = max_queries

    @property
    def prompt_version(self) -> str:
        return f"{self._template.version}+{self._template.short_fingerprint}"

    def plan(self, request: ResearchRequest) -> ResearchPlan:
        priorities = source_priorities(request)
        if self._llm is None:
            return self.deterministic(request, priorities)
        try:
            draft = self._ask(request)
        except LLM_FAILURES as exc:
            get_metrics().increment("research_plan.fallback", reason=type(exc).__name__)
            fallback = self.deterministic(request, priorities)
            return dataclasses.replace(
                fallback,
                warnings=(f"research plan: LLM unavailable ({type(exc).__name__}); used criteria catalogue",),
            )
        queries = self._accept(request.company_name, draft)
        if not queries:
            fallback = self.deterministic(request, priorities)
            return dataclasses.replace(
                fallback, warnings=("research plan: LLM proposed no usable queries; used criteria catalogue",)
            )
        questions = tuple(dict.fromkeys(h.strip() for h in draft.hypotheses if h.strip()))[
            : self._max_queries
        ]
        return ResearchPlan(
            queries=queries,
            focus_questions=questions or self.deterministic(request, priorities).focus_questions,
            source_priorities=priorities,
            source="llm",
        )

    def deterministic(
        self, request: ResearchRequest, priorities: tuple[DocumentType, ...] | None = None
    ) -> ResearchPlan:
        company = request.company_name
        queries: list[PlannedQuery] = []
        for definition in all_definitions():
            queries.extend(
                PlannedQuery(query=q, criterion=definition.criterion, rationale=definition.title)
                for q in definition.render_queries(company)[:2]
            )
        return ResearchPlan(
            queries=tuple(queries[: self._max_queries]),
            focus_questions=tuple(d.render_discovery_question(company) for d in all_definitions()),
            source_priorities=priorities or source_priorities(request),
            source="deterministic",
        )

    def _ask(self, request: ResearchRequest) -> ResearchPlanDraft:
        criteria = "\n".join(f"- {d.criterion.value}: {d.title} - {d.description}" for d in all_definitions())
        prompt = self._template.render(
            company=request.company_name,
            industry=request.industry or "unknown",
            criteria=criteria,
            max_queries=self._max_queries,
        )
        if self._llm is None:
            raise AgentError("internal invariant violated: self._llm is unset")
        draft, _ = complete_structured(
            self._llm,
            [
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
            ResearchPlanDraft,
            max_repairs=1,
            max_tokens=1200,
        )
        return draft

    def _accept(self, company: str, draft: ResearchPlanDraft) -> tuple[PlannedQuery, ...]:
        accepted: list[PlannedQuery] = []
        seen: set[str] = set()
        for item in draft.queries:
            query = " ".join(item.query.split())
            if company.casefold() not in query.casefold():
                query = f"{company} {query}"
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            try:
                criterion: Criterion | None = Criterion(item.criterion.strip())
            except ValueError:
                criterion = None
            accepted.append(PlannedQuery(query=query, criterion=criterion, rationale=item.rationale.strip()))
            if len(accepted) >= self._max_queries:
                break
        return tuple(accepted)


# ----------------------------------------------------------------------------- screening
@dataclass(frozen=True, slots=True)
class ScreenResult:
    text: str
    blocked: bool
    score: float
    signals: tuple[str, ...]
    removed_segments: int


def text_windows(text: str, window_chars: int = SCREEN_WINDOW_CHARS) -> list[str]:
    """Split on line boundaries into windows of at most ``window_chars`` (long lines are cut)."""
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.split("\n"):
        pieces = [line[i : i + window_chars] for i in range(0, len(line), window_chars)] or [""]
        for piece in pieces:
            if current and size + len(piece) + 1 > window_chars:
                windows.append("\n".join(current))
                current, size = [], 0
            current.append(piece)
            size += len(piece) + 1
    if current:
        windows.append("\n".join(current))
    return [w for w in windows if w.strip()]


def screen_document(text: str, detector: PromptInjectionDetector) -> ScreenResult:
    """Indirect prompt-injection screening of a whole document.

    A document containing any window at or above the block threshold is *blocked*:
    content planted next to an injection payload cannot be trusted either. Windows
    that are merely suspicious (at least half the threshold) are removed and the
    rest of the document is kept.
    """
    kept: list[str] = []
    signals: set[str] = set()
    blocked = False
    removed = 0
    best = 0.0
    suspicious = detector.threshold / 2
    for window in text_windows(text):
        assessment = detector.assess(window)
        best = max(best, assessment.score)
        if assessment.blocked or assessment.score >= suspicious:
            blocked = blocked or assessment.blocked
            removed += 1
            signals.update(assessment.signal_names)
            continue
        kept.append(window)
    return ScreenResult("\n".join(kept), blocked, round(best, 4), tuple(sorted(signals)), removed)


# ----------------------------------------------------------------------------- evidence gathering
@dataclass(frozen=True, slots=True)
class QuarantinedDocument:
    doc_id: str
    source_domain: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class GatheringResult:
    documents: tuple[SourceDocument, ...] = ()
    chunks: tuple[Chunk, ...] = ()
    quarantined: tuple[QuarantinedDocument, ...] = ()
    skipped: dict[str, int] = field(default_factory=dict)
    pii_redactions: int = 0
    candidates: int = 0
    has_company_facts: bool = False
    cik: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def chunks_written(self) -> int:
        return len(self.chunks)

    def summary(self) -> dict[str, Any]:
        return {
            "documents_ingested": len(self.documents),
            "chunks_written": self.chunks_written,
            "documents_quarantined": len(self.quarantined),
            "pii_redactions": self.pii_redactions,
            "sources_skipped": dict(self.skipped),
            "has_company_facts": self.has_company_facts,
            "cik": self.cik,
        }


def document_probe(document: SourceDocument) -> Chunk:
    """A whole-document pseudo-chunk so the poisoning guard can judge documents before indexing."""
    return Chunk(
        chunk_id=document.doc_id,
        doc_id=document.doc_id,
        text=document.text[:GUARD_TEXT_CHARS],
        company=document.company,
        url=document.url,
        title=document.title,
        document_type=document.document_type,
        source_domain=document.source_domain,
        chunk_index=0,
        strategy=ChunkStrategy.PARENT,
        publication_date=document.publication_date,
        industry=document.industry,
        confidence=document.trust_score,
        metadata={"trust_score": document.trust_score},
    )


class EvidenceGatherer:
    """Evidence Gathering Agent: ingest, screen, redact, persist, index and record lineage."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    def gather(
        self,
        request: ResearchRequest,
        *,
        runner: IngestionRunner | None = None,
        lineage: LineageRecorder | None = None,
    ) -> GatheringResult:
        rt = self._runtime
        ingested: IngestionResult = (runner or rt.ingestion).run(request)
        warnings: list[str] = []
        screened, quarantined, redactions = self._screen(request, ingested.documents)
        if quarantined:
            warnings.append(
                f"evidence gathering: {len(quarantined)} document(s) quarantined by the "
                "injection/poisoning guards"
            )
        chunks: tuple[Chunk, ...] = ()
        if screened:
            rt.document_store.save_documents(screened)
            indexed = rt.indexer.index(screened)
            chunks = (*indexed.parents, *indexed.chunks)
        if lineage is not None:
            for document in screened:
                lineage.record_source(document.url)
                lineage.record_document(document)
            for chunk in chunks:
                lineage.record_chunk(chunk)
        metrics = get_metrics()
        metrics.increment("evidence.documents_accepted", len(screened))
        metrics.increment("evidence.documents_quarantined", len(quarantined))
        metrics.increment("evidence.pii_redactions", redactions)
        return GatheringResult(
            documents=tuple(screened),
            chunks=chunks,
            quarantined=tuple(quarantined),
            skipped=ingested.skipped_by_reason(),
            pii_redactions=redactions,
            candidates=len(ingested.documents) + len(ingested.skipped),
            has_company_facts=ingested.company_facts is not None,
            cik=ingested.cik,
            warnings=tuple(warnings),
        )

    def _screen(
        self, request: ResearchRequest, documents: Sequence[SourceDocument]
    ) -> tuple[list[SourceDocument], list[QuarantinedDocument], int]:
        rt = self._runtime
        cleaned: list[SourceDocument] = []
        quarantined: list[QuarantinedDocument] = []
        for document in documents:
            sanitized = rt.document_sanitizer.sanitize_untrusted(document.text)
            screen = screen_document(sanitized.text, rt.injection_detector)
            if screen.blocked or not screen.text.strip():
                reasons = (
                    ("prompt_injection", *screen.signals) if screen.blocked else ("empty_after_screening",)
                )
                quarantined.append(QuarantinedDocument(document.doc_id, document.source_domain, reasons))
                continue
            text = screen.text
            pii_counts: dict[str, int] = {}
            if rt.settings.guardrails.redact_pii:
                redacted = rt.pii_redactor.redact(text)
                text = redacted.text
                pii_counts = redacted.counts()
            metadata = {
                **document.metadata,
                "sanitizer_removed": dict(sanitized.removed_counts),
                "injection_score": screen.score,
                "injection_signals": list(screen.signals),
                "screened_segments_removed": screen.removed_segments,
                "pii_redactions": pii_counts,
            }
            cleaned.append(document.model_copy(update={"text": text, "metadata": metadata}))

        if cleaned:
            report = rt.poisoning_guard.evaluate(
                [document_probe(d) for d in cleaned], company_domain=request.domain
            )
            accepted: list[SourceDocument] = []
            for document in cleaned:
                decision = report.decision_for(document.doc_id)
                if decision is not None and decision.action is ChunkAction.QUARANTINE:
                    quarantined.append(
                        QuarantinedDocument(document.doc_id, document.source_domain, decision.reasons)
                    )
                    continue
                if decision is not None and decision.action is ChunkAction.FLAG:
                    flagged = {**document.metadata, "poisoning_flags": list(decision.reasons)}
                    document = document.model_copy(update={"metadata": flagged})  # noqa: PLW2901
                accepted.append(document)
            cleaned = accepted
        redactions = sum(sum(d.metadata.get("pii_redactions", {}).values()) for d in cleaned)
        for item in quarantined:
            _log.warning(
                "evidence.quarantined",
                doc_id=item.doc_id,
                domain=item.source_domain,
                reasons=list(item.reasons),
            )
        return cleaned, quarantined, redactions


# ----------------------------------------------------------------------------- guarded retrieval
class ChunkBlocklist:
    """Thread-safe set of quarantined chunk and document ids."""

    def __init__(self, chunk_ids: Iterable[str] = (), doc_ids: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._chunks = set(chunk_ids)
        self._docs = set(doc_ids)

    def add(self, *, chunk_ids: Iterable[str] = (), doc_ids: Iterable[str] = ()) -> None:
        with self._lock:
            self._chunks.update(chunk_ids)
            self._docs.update(doc_ids)

    def blocks(self, chunk: Chunk) -> bool:
        with self._lock:
            return (
                chunk.chunk_id in self._chunks
                or chunk.doc_id in self._docs
                or (chunk.parent_id is not None and chunk.parent_id in self._chunks)
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks) + len(self._docs)


def _reindex(results: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    return [r if r.rank == rank else r.model_copy(update={"rank": rank}) for rank, r in enumerate(results)]


class GuardedRetriever:
    """Filters every retrieval outcome: no quarantined chunk and no other company's chunk ever passes."""

    def __init__(self, inner: Retriever, blocklist: ChunkBlocklist) -> None:
        self._inner = inner
        self._blocklist = blocklist
        self._lock = threading.Lock()
        self._urls: dict[str, None] = {}
        self._removed = 0

    @property
    def retrieved_urls(self) -> list[str]:
        with self._lock:
            return list(self._urls)

    @property
    def removed(self) -> int:
        with self._lock:
            return self._removed

    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> RetrievalOutcome:
        outcome = self._inner.retrieve(query, company=company, top_k=top_k)
        target = company.casefold()
        kept = [
            r
            for r in outcome.chunks
            if r.chunk.company.casefold() == target and not self._blocklist.blocks(r.chunk)
        ]
        with self._lock:
            self._removed += len(outcome.chunks) - len(kept)
            self._urls.update(dict.fromkeys(r.chunk.url for r in kept))
        if len(kept) == len(outcome.chunks):
            return outcome
        return dataclasses.replace(outcome, chunks=_reindex(kept))


class NoEvidenceRetriever:
    """Retriever for a company with no stored evidence: every query returns nothing."""

    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> RetrievalOutcome:
        return RetrievalOutcome(
            query=query,
            company=company,
            chunks=[],
            trace=["no_evidence: the document store holds no chunks for this company"],
            corrective_actions=[],
            mean_relevance=0.0,
            verdict=RetrievalVerdict.INCORRECT,
        )


class KnowledgeRefresher:
    """CRAG knowledge-refresh hook: bounded, targeted re-ingestion of seed and corporate URLs."""

    def __init__(
        self,
        gatherer: EvidenceGatherer,
        runner: IngestionRunner,
        request: ResearchRequest,
        blocklist: ChunkBlocklist,
        runtime: AgentRuntime,
        *,
        max_refreshes: int = 1,
        lineage: LineageRecorder | None = None,
    ) -> None:
        self._gatherer = gatherer
        self._runner = runner
        self._request = request.model_copy(
            update={"max_documents": min(request.max_documents, REFRESH_MAX_DOCUMENTS)}
        )
        self._blocklist = blocklist
        self._runtime = runtime
        self._max = max_refreshes
        self._lineage = lineage
        self._lock = threading.Lock()
        self._used = 0
        self.documents_added = 0

    @property
    def refreshes_used(self) -> int:
        with self._lock:
            return self._used

    def __call__(self, query: str) -> int:
        with self._lock:
            if self._used >= self._max:
                return 0
            self._used += 1
        get_metrics().increment("retrieval.knowledge_refresh")
        _log.info("knowledge_refresh.start", company=self._request.company_name, query=query[:120])
        result = self._gatherer.gather(self._request, runner=self._runner, lineage=self._lineage)
        self._blocklist.add(doc_ids=(q.doc_id for q in result.quarantined))
        children = [c for c in result.chunks if c.strategy is not ChunkStrategy.PARENT]
        if children:
            report = self._runtime.poisoning_guard.evaluate(children, company_domain=self._request.domain)
            self._blocklist.add(chunk_ids=report.quarantined_ids)
        with self._lock:
            self.documents_added += len(result.documents)
        return len(result.documents)


# ----------------------------------------------------------------------------- output clean-up
STATEMENT_SECTIONS: tuple[str, ...] = (
    "company_overview",
    "technology_priorities",
    "gartner_relevant_insights",
    "opportunities",
    "risks",
    "executive_summary",
    "executive_talking_points",
    "recommended_next_actions",
)
_LOCATION = re.compile(r"^([a-z_]+)\[(\d+)\]$")


def strip_statements(brief: ClientBrief, locations: Iterable[str]) -> tuple[ClientBrief, int]:
    """Drop statements at ``section[index]`` locations; other locations are left for review."""
    doomed: dict[str, set[int]] = {}
    for location in locations:
        match = _LOCATION.match(location)
        if match and match.group(1) in STATEMENT_SECTIONS:
            doomed.setdefault(match.group(1), set()).add(int(match.group(2)))
    if not doomed:
        return brief, 0
    update: dict[str, tuple[Any, ...]] = {}
    removed = 0
    for section, indices in doomed.items():
        statements = getattr(brief, section)
        update[section] = tuple(s for i, s in enumerate(statements) if i not in indices)
        removed += len(statements) - len(update[section])
    return brief.model_copy(update=update), removed
