"""Public-source ingestion pipeline.

``IngestionPipeline.run`` turns a :class:`ResearchRequest` into de-duplicated,
classified, trust-scored :class:`SourceDocument` objects:

1. **Collect candidates** concurrently from SEC EDGAR (recent filings + XBRL
   facts) and corporate-site discovery, plus the request's explicit seed URLs
   (analyst-domain seeds go through :class:`PublicAnalystSource` only).
2. **Fetch + parse** candidates in bounded waves on a thread pool, stopping as
   soon as ``request.max_documents`` documents are accepted.
3. **Classify, extract entities, hash and de-duplicate** by SHA-256 of the
   normalized text, also skipping hashes the document store already holds.

Every step degrades gracefully: a failing source or URL becomes a
:class:`SkippedSource` with a reason and never aborts the run. The request's
corporate domain and seed hosts are placed in the crawl scope (see
:func:`~client_research_agent.research.url_guard.crawl_scope`) for the duration
of the run, and the scope is propagated into worker threads.
"""

from __future__ import annotations

import contextvars
import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from itertools import zip_longest
from typing import Any, TypeVar

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import DocumentType, ResearchRequest, SourceDocument
from client_research_agent.observability.logging import get_logger, log_context
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.research.fetcher import media_type
from client_research_agent.research.parsing.classifier import DocumentTypeClassifier
from client_research_agent.research.parsing.entities import EntityExtractor
from client_research_agent.research.parsing.html import HtmlParser, ParsedPage
from client_research_agent.research.sources.analyst_public import PublicAnalystSource
from client_research_agent.research.sources.corporate_site import CorporateSiteDiscoverer
from client_research_agent.research.sources.edgar import CompanyFacts, EdgarClient
from client_research_agent.research.url_guard import crawl_scope, host_matches, host_of, normalize_url
from client_research_agent.research.url_guard import request_scope_domains as scope_domains
from client_research_agent.services.ports import DocumentStore, FetchResult, HttpFetcher
from client_research_agent.utils.errors import (
    AgentError,
    CircuitOpenError,
    CrawlPolicyViolationError,
)

_log = get_logger(__name__)
T = TypeVar("T")

_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml", ""})
_WS = re.compile(r"\s+")

TRUST_SEC = 0.95
TRUST_COMPANY = 0.85
TRUST_ANALYST = 0.7
TRUST_OTHER = 0.5


class SkipReason(StrEnum):
    POLICY = "policy_violation"
    HTTP_STATUS = "http_status"
    FETCH_FAILED = "fetch_failed"
    CIRCUIT_OPEN = "circuit_open"
    UNSUPPORTED_CONTENT = "unsupported_content"
    EMPTY_CONTENT = "empty_content"
    DUPLICATE = "duplicate_content"
    ALREADY_INGESTED = "already_ingested"
    LIMIT_REACHED = "max_documents_reached"
    SOURCE_FAILED = "source_failed"
    PROCESSING_ERROR = "processing_error"


class Origin(StrEnum):
    SEED = "seed"
    ANALYST = "analyst_public"
    EDGAR = "sec_edgar"
    CORPORATE = "corporate_site"


@dataclass(frozen=True, slots=True)
class SkippedSource:
    url: str
    reason: SkipReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    documents: list[SourceDocument]
    skipped: list[SkippedSource]
    company_facts: CompanyFacts | None = None
    cik: str | None = None

    def skipped_by_reason(self) -> dict[str, int]:
        return dict(Counter(s.reason.value for s in self.skipped))


@dataclass(frozen=True, slots=True)
class _Candidate:
    url: str
    origin: Origin
    hint: DocumentType | None = None
    publication_date: date | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class _SourceBatch:
    candidates: list[_Candidate] = field(default_factory=list)
    skipped: list[SkippedSource] = field(default_factory=list)
    facts: CompanyFacts | None = None
    cik: str | None = None


def content_hash(text: str) -> str:
    return hashlib.sha256(_WS.sub(" ", text).strip().lower().encode("utf-8")).hexdigest()


def document_id(url: str) -> str:
    return "doc_" + hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()[:32]


def trust_score(host: str, origin: Origin, company_domain: str | None) -> float:
    if host_matches(host, "sec.gov"):
        return TRUST_SEC
    if company_domain and host_matches(host, company_domain):
        return TRUST_COMPANY
    if origin is Origin.ANALYST:
        return TRUST_ANALYST
    return TRUST_OTHER


def _submit(pool: ThreadPoolExecutor, func: Callable[..., T], *args: Any) -> Future[T]:
    """Submit with a copy of the current context so crawl scope and log context follow the task."""
    return pool.submit(contextvars.copy_context().run, func, *args)


class IngestionPipeline:
    def __init__(  # noqa: PLR0917 - positional order is the published wiring contract
        self,
        fetcher: HttpFetcher,
        settings: AppSettings,
        parser: HtmlParser,
        classifier: DocumentTypeClassifier,
        extractor: EntityExtractor,
        edgar: EdgarClient | None,
        discoverer: CorporateSiteDiscoverer | None,
        analyst_source: PublicAnalystSource | None,
        document_store: DocumentStore | None = None,
        *,
        max_workers: int = 8,
        max_filings: int = 12,
        min_text_chars: int = 200,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._fetcher = fetcher
        self._settings = settings
        self._parser = parser
        self._classifier = classifier
        self._extractor = extractor
        self._edgar = edgar
        self._discoverer = discoverer
        self._analyst = analyst_source
        self._store = document_store
        self._max_workers = max_workers
        self._max_filings = max_filings
        self._min_text_chars = min_text_chars

    @traced(span_type=SpanType.CHAIN)
    def run(self, request: ResearchRequest) -> IngestionResult:
        started = time.perf_counter()
        with log_context(company=request.company_name, step="ingestion"), crawl_scope(scope_domains(request)):
            result = self._run(request)
        metrics = get_metrics()
        metrics.increment("ingestion.documents_fetched", len(result.documents))
        for reason, count in result.skipped_by_reason().items():
            metrics.increment("ingestion.sources_skipped", count, reason=reason)
        metrics.observe("ingestion.duration_ms", (time.perf_counter() - started) * 1000)
        _log.info(
            "ingestion.completed",
            company=request.company_name,
            documents=len(result.documents),
            skipped=result.skipped_by_reason(),
            has_company_facts=result.company_facts is not None,
        )
        return result

    def _run(self, request: ResearchRequest) -> IngestionResult:
        skipped: list[SkippedSource] = []
        seeds = self._seed_candidates(request, skipped)
        edgar_batch, corporate_batch = self._collect_sources(request)
        skipped.extend(edgar_batch.skipped)
        skipped.extend(corporate_batch.skipped)
        candidates = _dedupe_candidates(
            [*seeds, *_interleave(edgar_batch.candidates, corporate_batch.candidates)]
        )
        get_metrics().increment("ingestion.candidates", len(candidates))
        documents = self._fetch_documents(request, candidates, skipped)
        return IngestionResult(documents, skipped, edgar_batch.facts, edgar_batch.cik)

    def _seed_candidates(self, request: ResearchRequest, skipped: list[SkippedSource]) -> list[_Candidate]:
        seed_urls = [str(u) for u in request.seed_urls]
        analyst_urls: set[str] = set()
        candidates: list[_Candidate] = []
        if self._analyst is not None:
            permitted, rejected = self._analyst.select(seed_urls)
            analyst_urls = set(permitted) | set(rejected)
            candidates.extend(_Candidate(u, Origin.ANALYST, DocumentType.ANALYST_PUBLIC) for u in permitted)
            skipped.extend(
                SkippedSource(u, SkipReason.POLICY, "analyst page outside public paths or over limit")
                for u in rejected
            )
        candidates.extend(_Candidate(u, Origin.SEED) for u in seed_urls if u not in analyst_urls)
        return candidates

    def _collect_sources(self, request: ResearchRequest) -> tuple[_SourceBatch, _SourceBatch]:
        jobs: dict[str, Callable[[ResearchRequest], _SourceBatch]] = {}
        if self._edgar is not None:
            jobs["sec_edgar"] = self._collect_edgar
        if self._discoverer is not None and request.domain:
            jobs["corporate_site"] = self._collect_corporate
        batches: dict[str, _SourceBatch] = {}
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="cra-source") as pool:
                futures = {name: _submit(pool, job, request) for name, job in jobs.items()}
                for name, future in futures.items():
                    try:
                        batches[name] = future.result()
                    except Exception as exc:
                        _log.warning("ingestion.source_failed", source=name, error=type(exc).__name__)
                        batches[name] = _SourceBatch(
                            skipped=[
                                SkippedSource(f"source:{name}", SkipReason.SOURCE_FAILED, _describe(exc))
                            ]
                        )
        return batches.get("sec_edgar", _SourceBatch()), batches.get("corporate_site", _SourceBatch())

    def _collect_edgar(self, request: ResearchRequest) -> _SourceBatch:
        if self._edgar is None:
            return _SourceBatch()
        cik = request.cik or self._edgar.resolve_cik(ticker=request.ticker, company_name=request.company_name)
        if cik is None:
            return _SourceBatch(
                skipped=[
                    SkippedSource("source:sec_edgar", SkipReason.SOURCE_FAILED, "no SEC registrant matched")
                ]
            )
        filings = self._edgar.recent_filings(cik, max_filings=self._max_filings)
        candidates = [
            _Candidate(
                url=f.url,
                origin=Origin.EDGAR,
                hint=DocumentType.SEC_FILING,
                publication_date=f.filing_date,
                metadata={
                    "form": f.form,
                    "filing_date": f.filing_date.isoformat(),
                    "accession_number": f.accession_number,
                    "cik": f.cik,
                },
            )
            for f in filings
        ]
        facts: CompanyFacts | None
        try:
            facts = self._edgar.company_facts(cik)
        except AgentError as exc:
            _log.info("ingestion.company_facts_unavailable", cik=cik, error=type(exc).__name__)
            facts = None
        return _SourceBatch(candidates=candidates, facts=facts, cik=cik)

    def _collect_corporate(self, request: ResearchRequest) -> _SourceBatch:
        if self._discoverer is None or not request.domain:
            return _SourceBatch()
        found = self._discoverer.discover(request.domain)
        return _SourceBatch(
            candidates=[
                _Candidate(
                    url=c.url,
                    origin=Origin.CORPORATE,
                    metadata={
                        "discovery_score": c.score,
                        "discovery_reasons": list(c.reasons),
                        "lastmod": c.lastmod.isoformat() if c.lastmod else None,
                    },
                )
                for c in found
            ]
        )

    def _known_hashes(self, company: str) -> set[str]:
        if self._store is None:
            return set()
        try:
            return set(self._store.known_hashes(company))
        except Exception as exc:
            _log.warning("ingestion.known_hashes_unavailable", error=type(exc).__name__)
            return set()

    def _fetch_documents(
        self,
        request: ResearchRequest,
        candidates: list[_Candidate],
        skipped: list[SkippedSource],
    ) -> list[SourceDocument]:
        known = self._known_hashes(request.company_name)
        seen: set[str] = set()
        accepted: list[SourceDocument] = []
        limit = request.max_documents
        index = 0
        workers = max(1, min(self._max_workers, len(candidates)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cra-fetch") as pool:
            while index < len(candidates) and len(accepted) < limit:
                wave = candidates[index : index + (limit - len(accepted))]
                index += len(wave)
                futures = [_submit(pool, self._process, candidate, request) for candidate in wave]
                for future in futures:
                    outcome = future.result()
                    if isinstance(outcome, SkippedSource):
                        skipped.append(outcome)
                    elif outcome.content_hash in known:
                        skipped.append(SkippedSource(outcome.url, SkipReason.ALREADY_INGESTED))
                    elif outcome.content_hash in seen:
                        skipped.append(SkippedSource(outcome.url, SkipReason.DUPLICATE))
                    elif len(accepted) >= limit:
                        skipped.append(SkippedSource(outcome.url, SkipReason.LIMIT_REACHED))
                    else:
                        seen.add(outcome.content_hash)
                        accepted.append(outcome)
        skipped.extend(SkippedSource(c.url, SkipReason.LIMIT_REACHED) for c in candidates[index:])
        return accepted

    def _process(self, candidate: _Candidate, request: ResearchRequest) -> SourceDocument | SkippedSource:
        """Fetch, parse and classify one candidate. Never raises: failures become ``SkippedSource``."""
        try:
            if candidate.origin is Origin.ANALYST and self._analyst is not None:
                result = self._analyst.fetch(candidate.url)
            else:
                result = self._fetcher.fetch(candidate.url)
        except CircuitOpenError as exc:
            return SkippedSource(candidate.url, SkipReason.CIRCUIT_OPEN, str(exc))
        except CrawlPolicyViolationError as exc:
            return SkippedSource(candidate.url, SkipReason.POLICY, str(exc))
        except AgentError as exc:
            return SkippedSource(candidate.url, SkipReason.FETCH_FAILED, _describe(exc))
        if not result.ok:
            return SkippedSource(candidate.url, SkipReason.HTTP_STATUS, f"HTTP {result.status_code}")
        try:
            return self._build_document(candidate, result, request)
        except Exception as exc:
            _log.warning("ingestion.processing_error", url=candidate.url, error=type(exc).__name__)
            return SkippedSource(candidate.url, SkipReason.PROCESSING_ERROR, _describe(exc))

    def _build_document(
        self,
        candidate: _Candidate,
        result: FetchResult,
        request: ResearchRequest,
    ) -> SourceDocument | SkippedSource:
        kind = media_type(result.content_type)
        final_url = result.final_url or candidate.url
        if kind in _HTML_TYPES:
            page = self._parser.parse(result.text, final_url)
        elif kind == "text/plain":
            text = result.text.strip()
            page = ParsedPage(url=final_url, title=text.split("\n", 1)[0][:200], text=text)
        else:
            return SkippedSource(candidate.url, SkipReason.UNSUPPORTED_CONTENT, kind)
        if len(page.text) < self._min_text_chars:
            return SkippedSource(candidate.url, SkipReason.EMPTY_CONTENT, f"{len(page.text)} chars")

        title = page.title or final_url
        classification = self._classifier.classify(final_url, title, page.text, hint=candidate.hint)
        entities = self._extractor.extract(page.text)
        host = host_of(final_url)
        publication_date = (
            candidate.publication_date or page.publication_date
            if candidate.origin is Origin.EDGAR
            else page.publication_date
        )
        metadata: dict[str, Any] = {
            **candidate.metadata,
            "origin": candidate.origin.value,
            "final_url": final_url,
            "canonical_url": page.canonical_url,
            "http_status": result.status_code,
            "content_type": kind,
            "classification_confidence": classification.confidence,
            "classification_signals": list(classification.signals),
            "entities": list(entities.entities),
            "technologies": list(entities.technologies),
            "executives": [{"name": e.name, "title": e.title} for e in entities.executives],
            "fiscal_periods": list(entities.fiscal_periods),
            "headings": list(page.headings[:20]),
            "word_count": page.word_count,
        }
        return SourceDocument(
            doc_id=document_id(candidate.url),
            company=request.company_name,
            url=candidate.url,
            title=title,
            text=page.text,
            document_type=classification.document_type,
            source_domain=host,
            content_hash=content_hash(page.text),
            publication_date=publication_date,
            industry=request.industry,
            language=page.language or "en",
            trust_score=trust_score(host, candidate.origin, request.domain),
            metadata=metadata,
        )


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _interleave(*groups: Iterable[_Candidate]) -> list[_Candidate]:
    """Round-robin across sources so an early document limit still yields a diverse evidence base."""
    merged: list[_Candidate] = []
    for row in zip_longest(*groups):
        merged.extend(c for c in row if c is not None)
    return merged


def _dedupe_candidates(candidates: Iterable[_Candidate]) -> list[_Candidate]:
    unique: dict[str, _Candidate] = {}
    for candidate in candidates:
        key = normalize_url(candidate.url)
        if key not in unique:
            unique[key] = candidate
    return list(unique.values())
