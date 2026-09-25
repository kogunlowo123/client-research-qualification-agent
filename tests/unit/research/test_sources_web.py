from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.research.sources.analyst_public import (
    DOCUMENT_TYPE,
    AnalystDomainPolicy,
    PublicAnalystSource,
)
from client_research_agent.research.sources.corporate_site import CorporateSiteDiscoverer, parse_sitemap
from client_research_agent.utils.errors import CrawlPolicyViolationError, UpstreamTimeoutError
from tests.unit.research.helpers import ScriptedFetcher, fixture_text

TODAY = date(2026, 9, 24)
BASE = "https://northwind.example"


def register_site(scripted: ScriptedFetcher) -> None:
    scripted.add(
        f"{BASE}/robots.txt",
        f"User-agent: *\nAllow: /\nSitemap: {BASE}/sitemap_index.xml\nSitemap: https://evil.example/s.xml\n",
        content_type="text/plain",
    )
    scripted.add(
        f"{BASE}/sitemap_index.xml", fixture_text("sitemap_index.xml"), content_type="application/xml"
    )
    scripted.add(f"{BASE}/sitemap-news.xml", fixture_text("sitemap_news.xml"), content_type="application/xml")
    scripted.fail(f"{BASE}/sitemap-products.xml", UpstreamTimeoutError("slow"))
    scripted.add(f"{BASE}/", fixture_text("homepage.html"))
    scripted.add(f"{BASE}/newsroom/", fixture_text("newsroom.html"))


class TestCorporateSiteDiscoverer:
    def test_discovers_and_ranks_candidates(self, scripted: ScriptedFetcher) -> None:
        register_site(scripted)
        discoverer = CorporateSiteDiscoverer(scripted, today=TODAY, max_pages=20)
        found = discoverer.discover("www.Northwind.example")
        urls = [c.url for c in found]

        # Found one hop into the newsroom hub; AI anchor text + recency put it first.
        assert urls[0] == f"{BASE}/newsroom/press-releases/2026/northwind-launches-ai-platform"
        assert f"{BASE}/newsroom/press-releases/2026/northwind-q4-fy2025-results" in urls
        assert f"{BASE}/newsroom/press-releases/2026/northwind-appoints-cio" in urls
        assert f"{BASE}/investors/" in urls
        assert f"{BASE}/about/leadership" in urls
        assert f"{BASE}/" in urls
        for excluded in ("careers", "privacy", ".pdf", "twitter.com", "cdn.other.example", "evil.example"):
            assert not any(excluded in u for u in urls)
        scores = [c.score for c in found]
        assert scores == sorted(scores, reverse=True)
        assert "https://evil.example/sitemap.xml" not in scripted.urls
        assert "https://evil.example/s.xml" not in scripted.urls
        assert scripted.urls.index(f"{BASE}/sitemap-news.xml") < scripted.urls.index(
            f"{BASE}/sitemap-products.xml"
        )

    def test_old_items_rank_below_recent(self, scripted: ScriptedFetcher) -> None:
        register_site(scripted)
        found = {
            c.url: c for c in CorporateSiteDiscoverer(scripted, today=TODAY).discover("northwind.example")
        }
        old = found[f"{BASE}/newsroom/press-releases/2019/old-news"]
        recent = found[f"{BASE}/newsroom/press-releases/2026/northwind-q4-fy2025-results"]
        assert "stale-path" in old.reasons
        assert "recent-lastmod" in recent.reasons
        assert recent.score > old.score
        assert recent.lastmod == date(2026, 2, 12)

    def test_max_pages_and_sitemap_budgets(self, scripted: ScriptedFetcher) -> None:
        register_site(scripted)
        discoverer = CorporateSiteDiscoverer(
            scripted, today=TODAY, max_pages=3, max_sitemaps=1, max_hub_pages=0
        )
        found = discoverer.discover("northwind.example")
        assert len(found) == 3
        assert f"{BASE}/sitemap-news.xml" not in scripted.urls
        assert f"{BASE}/newsroom/" not in scripted.urls

    def test_sitemap_url_budget(self, scripted: ScriptedFetcher) -> None:
        scripted.add(f"{BASE}/sitemap.xml", fixture_text("sitemap_news.xml"))
        discoverer = CorporateSiteDiscoverer(scripted, today=TODAY, max_sitemap_urls=1)
        found = discoverer.discover("northwind.example")
        assert [c.url for c in found] == [f"{BASE}/newsroom/press-releases/2026/northwind-q4-fy2025-results"]

    def test_www_fallback_for_homepage(self, scripted: ScriptedFetcher) -> None:
        scripted.add("https://www.northwind.example/", fixture_text("homepage.html"))
        found = CorporateSiteDiscoverer(scripted, today=TODAY).discover("northwind.example")
        assert "https://www.northwind.example/" in [c.url for c in found]
        assert "https://www.northwind.example/investors/" in [c.url for c in found]

    def test_unreachable_site_yields_nothing(self, scripted: ScriptedFetcher) -> None:
        assert CorporateSiteDiscoverer(scripted, today=TODAY).discover("northwind.example") == []

    def test_from_settings_uses_page_budget(self, scripted: ScriptedFetcher) -> None:
        discoverer = CorporateSiteDiscoverer.from_settings(scripted, CrawlerSettings(max_pages_per_domain=7))
        assert discoverer._max_pages == 7

    @pytest.mark.parametrize(
        ("url", "anchor", "expected_reason"),
        [
            (f"{BASE}/investors/quarterly-results", "", "earnings"),
            (f"{BASE}/company/strategy", "Our AI strategy", "technology"),
            (f"{BASE}/x", "Board of Directors", "leadership"),
            ("https://ir.northwind.example/sec-filings", "", "investor"),
        ],
    )
    def test_score_url_signals(self, url: str, anchor: str, expected_reason: str) -> None:
        candidate = CorporateSiteDiscoverer(ScriptedFetcher(), today=TODAY).score_url(url, anchor)
        assert candidate is not None
        assert expected_reason in candidate.reasons

    @pytest.mark.parametrize(
        "url",
        [f"{BASE}/careers/", f"{BASE}/legal", f"{BASE}/search", f"{BASE}/logo.png", f"{BASE}/tag/ai/"],
    )
    def test_score_url_exclusions(self, url: str) -> None:
        assert CorporateSiteDiscoverer(ScriptedFetcher(), today=TODAY).score_url(url) is None

    def test_parse_sitemap(self) -> None:
        children, urls = parse_sitemap(fixture_text("sitemap_index.xml"))
        assert children[1] == f"{BASE}/sitemap-news.xml"
        assert urls == []
        _children, entries = parse_sitemap(fixture_text("sitemap_news.xml"))
        assert len(entries) == 6
        assert entries[0][1] == date(2026, 2, 12)
        assert entries[3][1] is None


class TestPublicAnalystSource:
    def test_only_public_paths_on_listed_domains(self, scripted: ScriptedFetcher) -> None:
        source = PublicAnalystSource(scripted)
        assert source.domains == ("gartner.com", "forrester.com")
        assert source.is_permitted("https://www.gartner.com/en/newsroom/press-releases/2026-01-10-forecast")
        assert source.is_permitted("https://www.forrester.com/blogs/ai-predictions/")
        assert not source.is_permitted("https://www.gartner.com/en/documents/4012345")
        assert not source.is_permitted("https://www.gartner.com/document/4012345")
        assert not source.is_permitted("https://www.gartner.com/en/newsroom/../account/")
        assert not source.is_permitted("https://www.gartner.com/en/newsroom/../../research/x")
        assert not source.is_permitted("ftp://www.gartner.com/en/newsroom/x")
        assert not source.is_permitted("https://gartner.com.evil.example/en/newsroom/x")

    def test_select_splits_and_caps(self, scripted: ScriptedFetcher) -> None:
        source = PublicAnalystSource(scripted, max_urls=1)
        permitted, rejected = source.select(
            [
                "https://www.gartner.com/en/newsroom/press-releases/a",
                "https://www.gartner.com/en/articles/b",
                "https://www.gartner.com/en/documents/c",
                "https://northwind.example/news",
                "https://www.gartner.com/en/newsroom/press-releases/a",
            ]
        )
        assert permitted == ["https://www.gartner.com/en/newsroom/press-releases/a"]
        assert rejected == ["https://www.gartner.com/en/articles/b", "https://www.gartner.com/en/documents/c"]

    def test_fetch_is_single_request_and_policy_checked(self, scripted: ScriptedFetcher) -> None:
        url = "https://www.gartner.com/en/newsroom/press-releases/2026-01-10-forecast"
        scripted.add(url, "<html><body><p>Gartner forecast</p></body></html>")
        source = PublicAnalystSource(scripted)
        assert source.fetch(url).ok
        assert scripted.urls == [url]
        with pytest.raises(CrawlPolicyViolationError, match="public analyst page"):
            source.fetch("https://www.gartner.com/en/documents/4012345")
        assert DOCUMENT_TYPE.value == "analyst_public"

    def test_policy_without_prefixes_allows_any_ungated_path(self) -> None:
        policy = AnalystDomainPolicy("analyst.example")
        assert policy.permits("https://analyst.example/research-notes/x")
        assert not policy.permits("https://analyst.example/login")
