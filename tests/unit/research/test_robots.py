from __future__ import annotations

import pytest

from client_research_agent.research.robots import (
    RobotsMode,
    RobotsPolicy,
    is_robots_url,
    origin_of,
    product_token,
    robots_url_for,
)
from client_research_agent.utils.errors import CrawlPolicyViolationError, UpstreamTimeoutError
from tests.unit.research.helpers import ScriptedFetcher

UA = "ClientResearchAgent/1.0 (+https://github.com/example/agent)"
ROBOTS = """
# Northwind robots
User-agent: *
Disallow: /private/
Crawl-delay: 2

User-agent: ClientResearchAgent
Disallow: /newsroom/drafts/
Allow: /
Crawl-delay: 3

User-agent: BadBot
Disallow: /

Sitemap: https://northwind.example/sitemap_index.xml
Sitemap: https://northwind.example/sitemap-news.xml
"""


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_rules_are_applied_to_our_product_token(scripted: ScriptedFetcher) -> None:
    scripted.add("https://northwind.example/robots.txt", ROBOTS, content_type="text/plain")
    policy = RobotsPolicy(scripted, UA)
    assert policy.token == "ClientResearchAgent"
    assert policy.is_allowed("https://northwind.example/newsroom/press-releases/q4")
    assert policy.is_allowed("https://northwind.example/private/ok-for-us")
    assert not policy.is_allowed("https://northwind.example/newsroom/drafts/secret")
    with pytest.raises(CrawlPolicyViolationError, match=r"robots\.txt"):
        policy.check("https://northwind.example/newsroom/drafts/secret")
    assert policy.crawl_delay("https://northwind.example/") == 3.0
    assert policy.sitemaps("https://northwind.example/") == (
        "https://northwind.example/sitemap_index.xml",
        "https://northwind.example/sitemap-news.xml",
    )
    assert scripted.urls == ["https://northwind.example/robots.txt"]


def test_wildcard_group_applies_to_unknown_agents(scripted: ScriptedFetcher) -> None:
    scripted.add("https://northwind.example/robots.txt", ROBOTS)
    policy = RobotsPolicy(scripted, "OtherAgent/2.0")
    assert not policy.is_allowed("https://northwind.example/private/x")
    assert policy.crawl_delay("https://northwind.example/x") == 2.0


def test_disallow_all(scripted: ScriptedFetcher) -> None:
    scripted.add("https://blocked.example/robots.txt", "User-agent: *\nDisallow: /\n")
    policy = RobotsPolicy(scripted, UA)
    assert not policy.is_allowed("https://blocked.example/")
    assert policy.is_allowed("https://blocked.example/robots.txt")
    assert policy.crawl_delay("https://blocked.example/") is None


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_4xx_means_allow_all(scripted: ScriptedFetcher, status: int) -> None:
    scripted.add("https://open.example/robots.txt", "nope", status=status)
    policy = RobotsPolicy(scripted, UA)
    assert policy.record_for("https://open.example/").mode is RobotsMode.ALLOW_ALL
    assert policy.is_allowed("https://open.example/anything")
    assert policy.crawl_delay("https://open.example/anything") is None


@pytest.mark.parametrize("status", [429, 500, 503])
def test_unreachable_status_means_disallow_all(scripted: ScriptedFetcher, status: int) -> None:
    scripted.add("https://down.example/robots.txt", "err", status=status)
    policy = RobotsPolicy(scripted, UA)
    assert policy.record_for("https://down.example/").mode is RobotsMode.DISALLOW_ALL
    assert not policy.is_allowed("https://down.example/page")


def test_fetch_errors_fail_closed_and_retry_after_failure_ttl(scripted: ScriptedFetcher) -> None:
    clock = Clock()
    scripted.fail("https://flaky.example/robots.txt", UpstreamTimeoutError("timeout"))
    scripted.add("https://flaky.example/robots.txt", "User-agent: *\nAllow: /\n")
    policy = RobotsPolicy(scripted, UA, clock=clock, failure_ttl_seconds=60, ttl_seconds=3600)
    assert not policy.is_allowed("https://flaky.example/a")
    clock.now = 30
    assert not policy.is_allowed("https://flaky.example/a")
    clock.now = 61
    assert policy.is_allowed("https://flaky.example/a")
    assert scripted.urls.count("https://flaky.example/robots.txt") == 2


def test_cache_is_per_origin_and_expires(scripted: ScriptedFetcher) -> None:
    clock = Clock()
    scripted.add("https://a.example/robots.txt", "User-agent: *\nAllow: /\n")
    scripted.add("http://a.example/robots.txt", "User-agent: *\nDisallow: /\n")
    policy = RobotsPolicy(scripted, UA, clock=clock, ttl_seconds=100)
    assert policy.is_allowed("https://a.example/x")
    assert not policy.is_allowed("http://a.example/x")
    policy.is_allowed("https://a.example/y")
    assert scripted.urls.count("https://a.example/robots.txt") == 1
    clock.now = 101
    policy.is_allowed("https://a.example/z")
    assert scripted.urls.count("https://a.example/robots.txt") == 2


def test_disabled_policy_never_fetches(scripted: ScriptedFetcher) -> None:
    policy = RobotsPolicy(scripted, UA, enabled=False)
    assert not policy.enabled
    assert policy.is_allowed("https://any.example/private")
    assert policy.crawl_delay("https://any.example/") is None
    assert scripted.calls == []


def test_helpers() -> None:
    assert product_token("  ") == "*"
    assert product_token("Bot Name/1.0") == "Bot"
    assert origin_of("HTTPS://Www.Example.com:8443/a/b?c") == "https://www.example.com:8443"
    assert robots_url_for("https://x.example/a/b") == "https://x.example/robots.txt"
    assert is_robots_url("https://x.example/robots.txt")
    assert not is_robots_url("https://x.example/robots.txt.bak")
