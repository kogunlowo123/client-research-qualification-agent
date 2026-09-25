from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import Chunk, DocumentType, ResearchRequest, SourceDocument
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.research import build_ingestion_pipeline
from client_research_agent.research.ingestion import (
    TRUST_ANALYST,
    TRUST_COMPANY,
    TRUST_OTHER,
    TRUST_SEC,
    IngestionPipeline,
    Origin,
    SkippedSource,
    SkipReason,
    content_hash,
    document_id,
    trust_score,
)
from client_research_agent.research.parsing import DocumentTypeClassifier, EntityExtractor, HtmlParser
from client_research_agent.research.sources import CorporateSiteDiscoverer, EdgarClient, PublicAnalystSource
from client_research_agent.research.sources.edgar import COMPANY_FACTS_URL, SUBMISSIONS_URL, TICKERS_URL
from client_research_agent.research.url_guard import current_scope
from client_research_agent.services.ports import FetchResult
from client_research_agent.utils.errors import (
    CircuitOpenError,
    CrawlPolicyViolationError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from tests.unit.research.helpers import ScriptedFetcher, fixture_text

CIK = "0001234567"
BASE = "https://northwind.example"
EDGAR_BASE = "https://www.sec.gov/Archives/edgar/data/1234567"
TEN_K = f"{EDGAR_BASE}/000123456726000012/nwnd-20251231.htm"
EIGHT_K = f"{EDGAR_BASE}/000123456726000009/nwnd-8k_20260212.htm"
TEN_Q = f"{EDGAR_BASE}/000123456725000044/nwnd-20250930.htm"
PROXY = f"{EDGAR_BASE}/000123456725000020/nwnd-def14a.htm"
Q4 = f"{BASE}/newsroom/press-releases/2026/northwind-q4-fy2025-results"
CIO = f"{BASE}/newsroom/press-releases/2026/northwind-appoints-cio"
AI = f"{BASE}/newsroom/press-releases/2026/northwind-launches-ai-platform"
OLD = f"{BASE}/newsroom/press-releases/2019/old-news"
GARTNER = "https://www.gartner.com/en/newsroom/press-releases/2026-03-01-gartner-survey-cios"
GARTNER_GATED = "https://www.gartner.com/en/documents/4012345"
OTHER_SEED = "https://news.other.example/northwind-profile"
TODAY = date(2026, 9, 24)


def filing_html(form: str, body: str) -> str:
    return (
        f"<html><head><title>nwnd-{form}</title></head><body><div><p>UNITED STATES SECURITIES AND EXCHANGE "
        f"COMMISSION, Washington, D.C. 20549. FORM {form}. Commission File Number 001-12345.</p>"
        f"<p>{body}</p></div></body></html>"
    )


def article(title: str, body: str) -> str:
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><article><h1>{title}</h1><p>{body}</p></article></body></html>"
    )


FILLER = (
    "Northwind continues to invest in plant automation, a cloud data platform and customer analytics "
    "while expanding margins across its industrial segments and maintaining disciplined capital allocation."
)


def register_world(scripted: ScriptedFetcher) -> None:
    scripted.add(TICKERS_URL, fixture_text("company_tickers.json"), content_type="application/json")
    scripted.add(
        SUBMISSIONS_URL.format(cik=CIK), fixture_text("submissions.json"), content_type="application/json"
    )
    scripted.add(
        COMPANY_FACTS_URL.format(cik=CIK), fixture_text("companyfacts.json"), content_type="application/json"
    )
    scripted.add(TEN_K, filing_html("10-K", "Annual report. Revenue of $12.4 billion. " + FILLER))
    scripted.add(EIGHT_K, filing_html("8-K", "Item 2.02 Results of Operations. Exhibit 99.1. " + FILLER))
    scripted.fail(TEN_Q, UpstreamTimeoutError("sec slow"))
    scripted.add(f"{BASE}/robots.txt", "User-agent: *\nAllow: /\n", content_type="text/plain")
    scripted.add(f"{BASE}/sitemap.xml", fixture_text("sitemap_news.xml"), content_type="application/xml")
    scripted.add(f"{BASE}/", fixture_text("homepage.html"))
    scripted.add(f"{BASE}/newsroom/", fixture_text("newsroom.html"))
    scripted.add(Q4, fixture_text("press_release.html"))
    cio_body = "Northwind today announced it has appointed Rahul Mehta as Chief Information Officer. "
    scripted.add(
        CIO, article("Northwind Appoints Rahul Mehta as Chief Information Officer", cio_body + FILLER)
    )
    scripted.add(
        AI, article("Northwind Launches Industrial AI Platform", "Generative AI on Databricks. " + FILLER)
    )
    scripted.add(OLD, fixture_text("press_release.html"))
    scripted.add(f"{BASE}/about/leadership", article("Leadership", "Short."))
    scripted.add(f"{BASE}/investors/", '{"kind": "json"}', content_type="application/json")
    scripted.add(GARTNER, article("Gartner Survey Finds CIOs Prioritize AI", "Gartner survey. " + FILLER))
    scripted.add(OTHER_SEED, "Northwind profile\n" + FILLER + " " + FILLER, content_type="text/plain")


def make_pipeline(
    scripted: ScriptedFetcher,
    settings: AppSettings,
    *,
    store: InMemoryStore | None = None,
    parser: HtmlParser | None = None,
    discoverer: object | None = None,
    edgar: EdgarClient | bool | None = True,
    max_workers: int = 4,
) -> IngestionPipeline:
    html = parser or HtmlParser(today=TODAY)
    return IngestionPipeline(
        scripted,
        settings,
        html,
        DocumentTypeClassifier(),
        EntityExtractor(),
        EdgarClient(scripted, user_agent="TestAgent/1.0", contact_email="ops@agent.example")
        if edgar is True
        else (edgar or None),
        discoverer if discoverer is not None else CorporateSiteDiscoverer(scripted, parser=html, today=TODAY),
        PublicAnalystSource(scripted),
        store,
        max_workers=max_workers,
    )


def request(**overrides: object) -> ResearchRequest:
    values: dict[str, object] = {
        "company_name": "Northwind Industries",
        "domain": "northwind.example",
        "ticker": "NWND",
        "industry": "Industrial Manufacturing",
        "seed_urls": (GARTNER, GARTNER_GATED, OTHER_SEED),
        "max_documents": 40,
    }
    values.update(overrides)
    return ResearchRequest.model_validate(values)


@dataclass
class InMemoryStore:
    hashes: set[str] = field(default_factory=set)
    fail: bool = False
    documents: list[SourceDocument] = field(default_factory=list)

    def save_documents(self, documents: Sequence[SourceDocument]) -> int:
        self.documents.extend(documents)
        return len(documents)

    def save_chunks(self, chunks: Sequence[Chunk]) -> int:
        return len(chunks)

    def list_chunks(self, company: str) -> list[Chunk]:
        return []

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return []

    def known_hashes(self, company: str) -> set[str]:
        if self.fail:
            raise UpstreamServiceError("warehouse unavailable", 503)
        return set(self.hashes)


def by_url(documents: Sequence[SourceDocument]) -> dict[str, SourceDocument]:
    return {d.url: d for d in documents}


def reasons(skipped: Sequence[SkippedSource]) -> dict[str, str]:
    return {s.url: s.reason.value for s in skipped}


class TestFullRun:
    def test_collects_classifies_and_scores_every_source(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        register_world(scripted)
        result = make_pipeline(scripted, settings).run(request())
        docs = by_url(result.documents)

        ten_k = docs[TEN_K]
        assert ten_k.document_type is DocumentType.SEC_FILING
        assert ten_k.trust_score == TRUST_SEC
        assert ten_k.publication_date == date(2026, 2, 13)
        assert ten_k.metadata["form"] == "10-K"
        assert ten_k.metadata["origin"] == Origin.EDGAR.value
        assert ten_k.source_domain == "www.sec.gov"

        q4 = docs[Q4]
        assert q4.document_type is DocumentType.EARNINGS_RELEASE
        assert q4.trust_score == TRUST_COMPANY
        assert q4.publication_date == date(2026, 2, 12)
        assert q4.industry == "Industrial Manufacturing"
        assert "Databricks" in q4.metadata["technologies"]
        assert {"name": "Rahul Mehta", "title": "Chief Information Officer"} in q4.metadata["executives"]
        assert "Ignore all previous instructions" not in q4.text

        assert docs[CIO].document_type is DocumentType.LEADERSHIP_ANNOUNCEMENT
        gartner = docs[GARTNER]
        assert gartner.document_type is DocumentType.ANALYST_PUBLIC
        assert gartner.trust_score == TRUST_ANALYST
        other = docs[OTHER_SEED]
        assert other.trust_score == TRUST_OTHER
        assert other.title == "Northwind profile"

        assert result.cik == CIK
        assert result.company_facts is not None
        assert result.company_facts.revenue is not None
        assert result.company_facts.revenue.value == 12_400_000_000

        skipped = reasons(result.skipped)
        assert skipped[GARTNER_GATED] == SkipReason.POLICY.value
        assert skipped[OLD] == SkipReason.DUPLICATE.value
        assert docs[AI].document_type is DocumentType.PRESS_RELEASE
        assert skipped[TEN_Q] == SkipReason.FETCH_FAILED.value
        assert skipped[PROXY] == SkipReason.HTTP_STATUS.value
        assert skipped[f"{BASE}/about/leadership"] == SkipReason.EMPTY_CONTENT.value
        assert skipped[f"{BASE}/investors/"] == SkipReason.UNSUPPORTED_CONTENT.value

        ids = [d.doc_id for d in result.documents]
        assert len(ids) == len(set(ids))
        assert docs[Q4].doc_id == document_id(Q4)
        hashes = [d.content_hash for d in result.documents]
        assert len(hashes) == len(set(hashes))
        assert GARTNER_GATED not in scripted.urls

        metrics = get_metrics()
        assert metrics.counter("ingestion.documents_fetched") == len(result.documents)
        assert metrics.counter("ingestion.sources_skipped", reason="duplicate_content") == 1
        assert metrics.histogram("ingestion.duration_ms").count == 1
        assert result.skipped_by_reason()["policy_violation"] == 1

    def test_scope_is_propagated_to_worker_threads(self, settings: AppSettings) -> None:
        seen: list[frozenset[str]] = []
        lock = threading.Lock()

        class ScopeRecorder:
            def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
                with lock:
                    seen.append(current_scope())
                return FetchResult(url, url, 404, "text/plain", "")

        pipeline = IngestionPipeline(
            ScopeRecorder(), settings, HtmlParser(), DocumentTypeClassifier(), EntityExtractor(),
            None, CorporateSiteDiscoverer(ScopeRecorder()), None,
        )  # fmt: skip
        result = pipeline.run(request(seed_urls=(OTHER_SEED,)))
        assert result.documents == []
        assert seen
        assert all({"northwind.example", "news.other.example"} <= scope for scope in seen)
        assert current_scope() == frozenset()


class TestDedupeAndLimits:
    def test_known_hashes_are_skipped(self, scripted: ScriptedFetcher, settings: AppSettings) -> None:
        register_world(scripted)
        first = make_pipeline(scripted, settings).run(request())
        store = InMemoryStore(hashes={d.content_hash for d in first.documents if d.url == Q4})
        second = make_pipeline(scripted, settings, store=store).run(request())
        assert Q4 not in by_url(second.documents)
        assert reasons(second.skipped)[Q4] == SkipReason.ALREADY_INGESTED.value

    def test_store_failure_degrades_to_no_known_hashes(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        register_world(scripted)
        result = make_pipeline(scripted, settings, store=InMemoryStore(fail=True)).run(request())
        assert Q4 in by_url(result.documents)

    def test_max_documents_is_enforced_and_bounds_fetching(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        register_world(scripted)
        result = make_pipeline(scripted, settings).run(request(max_documents=2))
        assert len(result.documents) == 2
        assert [d.url for d in result.documents] == [GARTNER, OTHER_SEED]
        limited = [s for s in result.skipped if s.reason is SkipReason.LIMIT_REACHED]
        assert limited
        assert TEN_K not in scripted.urls
        assert Q4 not in scripted.urls

    def test_later_waves_fill_the_quota_after_skips(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        register_world(scripted)
        result = make_pipeline(scripted, settings, max_workers=1).run(request(max_documents=5, seed_urls=()))
        assert [d.url for d in result.documents] == [TEN_K, AI, EIGHT_K, CIO, Q4]
        assert reasons(result.skipped)[TEN_Q] == SkipReason.FETCH_FAILED.value

    def test_duplicate_candidate_urls_are_fetched_once(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        register_world(scripted)
        make_pipeline(scripted, settings, edgar=None).run(
            request(seed_urls=(Q4, Q4.replace("northwind.example", "NORTHWIND.example") + "#top"))
        )
        assert scripted.urls.count(Q4) == 1


class TestGracefulDegradation:
    def test_edgar_failure_does_not_abort(self, scripted: ScriptedFetcher, settings: AppSettings) -> None:
        register_world(scripted)
        scripted.responses[TICKERS_URL] = [UpstreamServiceError("sec down", 503)]
        result = make_pipeline(scripted, settings).run(request())
        skipped = reasons(result.skipped)
        assert skipped["source:sec_edgar"] == SkipReason.SOURCE_FAILED.value
        assert result.company_facts is None
        assert Q4 in by_url(result.documents)

    def test_unmatched_registrant(self, scripted: ScriptedFetcher, settings: AppSettings) -> None:
        register_world(scripted)
        result = make_pipeline(scripted, settings).run(
            request(ticker=None, company_name="Private Widgets LLC", seed_urls=())
        )
        skipped = [s for s in result.skipped if s.url == "source:sec_edgar"]
        assert skipped[0].detail == "no SEC registrant matched"
        assert result.cik is None

    def test_explicit_cik_and_facts_failure(self, scripted: ScriptedFetcher, settings: AppSettings) -> None:
        register_world(scripted)
        scripted.responses.pop(TICKERS_URL)

        class FactsFailingEdgar(EdgarClient):
            def company_facts(self, cik: str) -> None:
                raise UpstreamServiceError("facts down", 500)

        edgar = FactsFailingEdgar(scripted, user_agent="a", contact_email="a@b.example")
        result = make_pipeline(scripted, settings, edgar=edgar).run(request(ticker=None, cik="1234567"))
        assert result.cik == "1234567"
        assert result.company_facts is None
        assert TEN_K in by_url(result.documents)

    def test_discoverer_crash_is_isolated(self, scripted: ScriptedFetcher, settings: AppSettings) -> None:
        register_world(scripted)

        class Exploding:
            def discover(self, domain: str) -> list[object]:
                raise RuntimeError("parser bug")

        result = make_pipeline(scripted, settings, discoverer=Exploding()).run(request())
        detail = next(s.detail for s in result.skipped if s.url == "source:corporate_site")
        assert detail.startswith("RuntimeError")
        assert TEN_K in by_url(result.documents)

    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (CircuitOpenError("crawl:northwind.example", 12.0), SkipReason.CIRCUIT_OPEN),
            (CrawlPolicyViolationError("disallowed by robots.txt"), SkipReason.POLICY),
            (UpstreamServiceError("bad gateway", 502), SkipReason.FETCH_FAILED),
        ],
    )
    def test_fetch_errors_map_to_reasons(
        self, scripted: ScriptedFetcher, settings: AppSettings, error: Exception, reason: SkipReason
    ) -> None:
        scripted.fail(OTHER_SEED, error)
        pipeline = make_pipeline(scripted, settings, edgar=None)
        result = pipeline.run(request(domain=None, ticker=None, seed_urls=(OTHER_SEED,)))
        assert reasons(result.skipped)[OTHER_SEED] == reason.value

    def test_parser_crash_becomes_processing_error(
        self, scripted: ScriptedFetcher, settings: AppSettings
    ) -> None:
        scripted.add(Q4, fixture_text("press_release.html"))

        class BrokenParser(HtmlParser):
            def parse(self, html: str, url: str) -> object:  # type: ignore[override]
                raise ValueError("malformed markup")

        pipeline = make_pipeline(scripted, settings, parser=BrokenParser(), edgar=None)
        result = pipeline.run(request(domain=None, ticker=None, seed_urls=(Q4,)))
        skipped = result.skipped[0]
        assert skipped.reason is SkipReason.PROCESSING_ERROR
        assert "malformed markup" in skipped.detail


def test_helpers() -> None:
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert document_id("https://Northwind.example/a#x") == document_id("https://northwind.example/a")
    assert document_id(Q4).startswith("doc_")
    assert trust_score("data.sec.gov", Origin.SEED, None) == TRUST_SEC
    assert trust_score("ir.northwind.example", Origin.CORPORATE, "northwind.example") == TRUST_COMPANY
    assert trust_score("www.gartner.com", Origin.ANALYST, "northwind.example") == TRUST_ANALYST
    assert trust_score("blog.example", Origin.SEED, None) == TRUST_OTHER


def test_invalid_worker_count(scripted: ScriptedFetcher, settings: AppSettings) -> None:
    with pytest.raises(ValueError, match="max_workers"):
        make_pipeline(scripted, settings, max_workers=0)


def test_build_ingestion_pipeline_wires_defaults(scripted: ScriptedFetcher, settings: AppSettings) -> None:
    register_world(scripted)
    pipeline = build_ingestion_pipeline(settings, scripted, document_store=InMemoryStore(), max_workers=2)
    result = pipeline.run(request(max_documents=3))
    assert len(result.documents) == 3
