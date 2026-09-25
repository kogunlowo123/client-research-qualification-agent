"""Public-source research and ingestion.

Typical wiring::

    fetcher = PolicyEnforcingFetcher.from_settings(settings)
    pipeline = build_ingestion_pipeline(settings, fetcher, document_store=store)
    result = pipeline.run(request)
"""

from __future__ import annotations

from client_research_agent.config.settings import AppSettings
from client_research_agent.research.fetcher import HttpxFetcher, PolicyEnforcingFetcher
from client_research_agent.research.ingestion import (
    IngestionPipeline,
    IngestionResult,
    SkippedSource,
    SkipReason,
)
from client_research_agent.research.parsing import DocumentTypeClassifier, EntityExtractor, HtmlParser
from client_research_agent.research.rate_limit import HostRateLimiter
from client_research_agent.research.robots import RobotsPolicy
from client_research_agent.research.sources import (
    CompanyFacts,
    CorporateSiteDiscoverer,
    EdgarClient,
    PublicAnalystSource,
)
from client_research_agent.research.url_guard import UrlGuard, crawl_scope
from client_research_agent.services.ports import DocumentStore, HttpFetcher


def build_ingestion_pipeline(
    settings: AppSettings,
    fetcher: HttpFetcher,
    *,
    document_store: DocumentStore | None = None,
    max_workers: int = 8,
) -> IngestionPipeline:
    """Wire the default parser, classifier, extractor and sources around ``fetcher``."""
    parser = HtmlParser()
    return IngestionPipeline(
        fetcher,
        settings,
        parser,
        DocumentTypeClassifier(),
        EntityExtractor(),
        EdgarClient.from_settings(fetcher, settings.crawler),
        CorporateSiteDiscoverer(fetcher, parser=parser, max_pages=settings.crawler.max_pages_per_domain),
        PublicAnalystSource(fetcher),
        document_store,
        max_workers=max_workers,
    )


__all__ = [
    "CompanyFacts",
    "CorporateSiteDiscoverer",
    "DocumentTypeClassifier",
    "EdgarClient",
    "EntityExtractor",
    "HostRateLimiter",
    "HtmlParser",
    "HttpxFetcher",
    "IngestionPipeline",
    "IngestionResult",
    "PolicyEnforcingFetcher",
    "PublicAnalystSource",
    "RobotsPolicy",
    "SkipReason",
    "SkippedSource",
    "UrlGuard",
    "build_ingestion_pipeline",
    "crawl_scope",
]
