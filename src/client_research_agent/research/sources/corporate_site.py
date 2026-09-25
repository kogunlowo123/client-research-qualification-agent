"""Corporate website discovery.

Finds the pages on a company's own domain most likely to carry qualification
evidence (investor relations, newsroom/press releases, earnings, leadership)
without crawling the whole site:

1. Sitemaps declared in robots.txt plus ``/sitemap.xml``; sitemap indexes are
   expanded breadth-first, news/press/investor child sitemaps first, bounded
   by ``max_sitemaps`` fetches and ``max_sitemap_urls`` entries.
2. The homepage, and one hop into the highest-scoring hub pages found on it
   (``max_hub_pages``), using anchor text as an extra signal.

All fetches go through the injected :class:`HttpFetcher`, so robots.txt, the
allow-list and rate limits apply. Candidates are scored by keyword signals in
the URL path and anchor text plus recency, and the top ``max_pages`` returned.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Tag

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.research.parsing.html import HtmlParser, parse_date_value
from client_research_agent.research.url_guard import host_matches, normalize_url
from client_research_agent.services.ports import HttpFetcher
from client_research_agent.utils.errors import AgentError

_log = get_logger(__name__)

_KEYWORDS: tuple[tuple[re.Pattern[str], float, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), weight, label)
    for pattern, weight, label in (
        (r"investor|(?:^|[/.])ir(?:[/.-]|$)|shareholder|stockholder", 3.0, "investor"),
        (r"press[-_ ]?releases?|news[-_ ]?releases?", 3.0, "press-release"),
        (r"earnings|quarterly[-_ ]results|financial[-_ ]results|annual[-_ ]report", 3.0, "earnings"),
        (r"newsroom|news|/press(?:[/.-]|$)|/media(?:[/.-]|$)", 2.0, "news"),
        (r"leadership|management|executive|board[-_ ]of[-_ ]directors|our[-_ ]team", 2.0, "leadership"),
        (r"sec[-_ ]filings|10-k|10-q|proxy", 2.0, "filings"),
        (r"strategy|transformation|innovation|technology|digital|artificial[-_ ]intelligence|\bai\b|data"
         r"|cloud", 1.0, "technology"),
        (r"about(?:[-_ ]us)?|company|who[-_ ]we[-_ ]are|overview", 0.5, "about"),
    )
)  # fmt: skip
_EXCLUDED = re.compile(
    r"(?:^|/)(?:careers?|jobs|privacy[\w-]*|terms[\w-]*|cookies?[\w-]*|legal|log-?in|sign-?in|register|cart|"
    r"checkout|shop|store|support|contact[\w-]*|search|tags?|category|author|feed)(?:/|$)|"
    r"\.(?:pdf|jpe?g|png|gif|svg|zip|mp[34]|css|js|ico|xlsx?|docx?|pptx?)$",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_SITEMAP_PRIORITY = re.compile(r"news|press|investor|media|post|article", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CandidateUrl:
    url: str
    score: float
    reasons: tuple[str, ...] = ()
    source: str = "sitemap"
    lastmod: date | None = None


@dataclass
class _Collector:
    domain: str
    found: dict[str, CandidateUrl] = field(default_factory=dict)

    def add(self, candidate: CandidateUrl) -> None:
        existing = self.found.get(candidate.url)
        if existing is None or candidate.score > existing.score:
            self.found[candidate.url] = candidate


class CorporateSiteDiscoverer:
    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        parser: HtmlParser | None = None,
        max_pages: int = 60,
        max_sitemaps: int = 5,
        max_sitemap_urls: int = 5000,
        max_hub_pages: int = 3,
        today: date | None = None,
    ) -> None:
        self._fetcher = fetcher
        self._parser = parser or HtmlParser()
        self._max_pages = max_pages
        self._max_sitemaps = max_sitemaps
        self._max_sitemap_urls = max_sitemap_urls
        self._max_hub_pages = max_hub_pages
        self._today = today

    @classmethod
    def from_settings(cls, fetcher: HttpFetcher, crawler: CrawlerSettings) -> CorporateSiteDiscoverer:
        return cls(fetcher, max_pages=crawler.max_pages_per_domain)

    @traced(span_type=SpanType.RETRIEVER)
    def discover(self, domain: str) -> list[CandidateUrl]:
        domain = domain.lower().strip().removeprefix("www.")
        collector = _Collector(domain)
        self._from_sitemaps(collector)
        self._from_homepage(collector)
        ranked = sorted(
            collector.found.values(),
            key=lambda c: (c.score, c.lastmod or date.min, -len(c.url)),
            reverse=True,
        )
        _log.info("discovery.completed", domain=domain, candidates=len(ranked))
        return ranked[: self._max_pages]

    def score_url(self, url: str, anchor_text: str = "", lastmod: date | None = None) -> CandidateUrl | None:
        """Score one URL; ``None`` when excluded (careers, legal, binaries ...)."""
        parts = urlsplit(url)
        path = parts.path or "/"
        if _EXCLUDED.search(path):
            return None
        haystack = f"{parts.hostname or ''}{path} {anchor_text}"
        score = 0.0
        reasons: list[str] = []
        for pattern, weight, label in _KEYWORDS:
            if pattern.search(haystack):
                score += weight
                reasons.append(label)
        today = self._today or datetime.now(UTC).date()
        if lastmod is not None and 0 <= (today - lastmod).days <= 365:
            score += 1.0
            reasons.append("recent-lastmod")
        years = [int(y) for y in _YEAR.findall(path)]
        if years and max(years) >= today.year - 1:
            score += 1.0
            reasons.append("recent-path")
        elif years and max(years) < today.year - 3:
            score -= 1.0
            reasons.append("stale-path")
        return CandidateUrl(normalize_url(url), round(score, 2), tuple(reasons), lastmod=lastmod)

    def _on_domain(self, url: str, domain: str) -> bool:
        parts = urlsplit(url)
        return parts.scheme in ("http", "https") and host_matches(parts.hostname or "", domain)

    def _fetch_text(self, url: str) -> str | None:
        try:
            result = self._fetcher.fetch(url)
        except AgentError as exc:
            _log.info("discovery.fetch_failed", url=url, error=type(exc).__name__)
            return None
        return result.text if result.ok else None

    def _sitemap_seeds(self, domain: str) -> list[str]:
        seeds: list[str] = []
        robots = self._fetch_text(f"https://{domain}/robots.txt")
        if robots:
            for line in robots.splitlines():
                key, _, value = line.partition(":")
                if key.strip().lower() == "sitemap" and value.strip():
                    seeds.append(value.strip())
        seeds.append(f"https://{domain}/sitemap.xml")
        return [s for s in dict.fromkeys(seeds) if self._on_domain(s, domain)]

    def _from_sitemaps(self, collector: _Collector) -> None:
        queue = self._sitemap_seeds(collector.domain)
        fetched = 0
        seen: set[str] = set()
        entries = 0
        while queue and fetched < self._max_sitemaps and entries < self._max_sitemap_urls:
            sitemap_url = queue.pop(0)
            if sitemap_url in seen:
                continue
            seen.add(sitemap_url)
            fetched += 1
            body = self._fetch_text(sitemap_url)
            if not body:
                continue
            children, urls = parse_sitemap(body)
            children = [c for c in children if self._on_domain(c, collector.domain)]
            children.sort(key=lambda c: 0 if _SITEMAP_PRIORITY.search(c) else 1)
            queue.extend(children)
            for loc, lastmod in urls:
                if entries >= self._max_sitemap_urls:
                    break
                entries += 1
                if not self._on_domain(loc, collector.domain):
                    continue
                candidate = self.score_url(loc, lastmod=lastmod)
                if candidate is not None and candidate.score > 0:
                    collector.add(candidate)

    def _from_homepage(self, collector: _Collector) -> None:
        homepage = f"https://{collector.domain}/"
        page_links = self._page_links(homepage)
        if page_links is None:
            homepage = f"https://www.{collector.domain}/"
            page_links = self._page_links(homepage)
        if page_links is None:
            return
        collector.add(CandidateUrl(normalize_url(homepage), 1.0, ("homepage",), source="homepage"))
        hubs = self._add_links(collector, page_links)
        for hub in hubs[: self._max_hub_pages]:
            hub_links = self._page_links(hub.url)
            if hub_links:
                self._add_links(collector, hub_links, bonus=0.5)

    def _page_links(self, url: str) -> Mapping[str, str] | None:
        body = self._fetch_text(url)
        if body is None:
            return None
        return self._parser.parse(body, url).link_texts

    def _add_links(
        self, collector: _Collector, links: Mapping[str, str], *, bonus: float = 0.0
    ) -> list[CandidateUrl]:
        added: list[CandidateUrl] = []
        for url, text in links.items():
            if not self._on_domain(url, collector.domain):
                continue
            candidate = self.score_url(url, text)
            if candidate is None or candidate.score <= 0:
                continue
            candidate = CandidateUrl(
                candidate.url, candidate.score + bonus, candidate.reasons, "homepage", candidate.lastmod
            )
            collector.add(candidate)
            added.append(candidate)
        added.sort(key=lambda c: c.score, reverse=True)
        return added


def parse_sitemap(body: str) -> tuple[list[str], list[tuple[str, date | None]]]:
    """Return (child sitemap URLs, [(page URL, lastmod)]) from a sitemap or sitemap index."""
    soup = BeautifulSoup(body, "xml")
    children: list[str] = []
    urls: list[tuple[str, date | None]] = []
    for node in soup.find_all("sitemap"):
        loc = node.find("loc") if isinstance(node, Tag) else None
        if isinstance(loc, Tag) and loc.get_text(strip=True):
            children.append(loc.get_text(strip=True))
    for node in soup.find_all("url"):
        if not isinstance(node, Tag):
            continue
        loc = node.find("loc")
        if not isinstance(loc, Tag) or not loc.get_text(strip=True):
            continue
        lastmod_tag = node.find("lastmod")
        lastmod = parse_date_value(lastmod_tag.get_text(strip=True)) if isinstance(lastmod_tag, Tag) else None
        urls.append((loc.get_text(strip=True), lastmod))
    return children, urls
