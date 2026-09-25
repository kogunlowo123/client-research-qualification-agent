from __future__ import annotations

import threading

import pytest

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.models import ResearchRequest
from client_research_agent.research.url_guard import (
    UrlGuard,
    crawl_scope,
    current_scope,
    host_matches,
    normalize_url,
    request_scope_domains,
    system_resolver,
)
from client_research_agent.utils.errors import CrawlPolicyViolationError
from tests.unit.research.helpers import public_resolver


@pytest.fixture
def guard() -> UrlGuard:
    return UrlGuard(["sec.gov", "northwind.example"], resolver=public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.sec.gov/Archives/edgar/data/1/2/doc.htm",
        "https://data.sec.gov/submissions/CIK0000320193.json",
        "http://northwind.example/newsroom/",
        "https://NorthWind.Example:443/investors",
    ],
)
def test_allows_public_allow_listed_urls(guard: UrlGuard, url: str) -> None:
    assert guard.check(url).startswith(("https://", "http://"))
    assert guard.is_allowed(url)


@pytest.mark.parametrize(
    ("url", "fragment"),
    [
        ("ftp://sec.gov/file", "scheme"),
        ("file:///etc/passwd", "scheme"),
        ("javascript:alert(1)", "scheme"),
        ("https://user:pw@www.sec.gov/", "credentials"),
        ("https://www.sec.gov@evil.example/", "credentials"),
        ("https://www.sec.gov:8443/", "port"),
        ("https://www.sec.gov:99999/", "malformed"),
        ("https://127.0.0.1/admin", "IP-literal"),
        ("http://169.254.169.254/latest/meta-data/", "IP-literal"),
        ("http://[::1]/", "IP-literal"),
        ("http://10.0.0.5/", "private"),
        ("http://8.8.8.8/", "public"),
        ("http://2130706433/", "numeric"),
        ("http://0x7f.0x0.0x0.0x1/", "numeric"),
        ("https:///no-host", "no host"),
        ("https://evil.example/", "allow-list"),
        ("https://notsec.gov/", "allow-list"),
        ("https://sec.gov.evil.example/", "allow-list"),
        ("https://bad_host.sec.gov/", "invalid hostname"),
    ],
)
def test_rejects_unsafe_urls(guard: UrlGuard, url: str, fragment: str) -> None:
    with pytest.raises(CrawlPolicyViolationError, match=fragment):
        guard.check(url)
    assert not guard.is_allowed(url)


@pytest.mark.parametrize(
    "address", ["10.1.2.3", "127.0.0.1", "169.254.169.254", "::ffff:192.168.1.1", "fc00::1"]
)
def test_rejects_hosts_resolving_to_non_public_addresses(address: str) -> None:
    guard = UrlGuard(["sec.gov"], resolver=lambda _h: [address])
    with pytest.raises(CrawlPolicyViolationError, match="non-public"):
        guard.check("https://www.sec.gov/")


def test_rejects_when_any_resolved_address_is_private() -> None:
    guard = UrlGuard(["sec.gov"], resolver=lambda _h: ["93.184.216.34", "192.168.0.10"])
    with pytest.raises(CrawlPolicyViolationError, match=r"192\.168\.0\.10"):
        guard.check("https://www.sec.gov/")


def test_resolution_failures_fail_closed() -> None:
    def broken(_host: str) -> list[str]:
        raise OSError("NXDOMAIN")

    with pytest.raises(CrawlPolicyViolationError, match="could not be resolved"):
        UrlGuard(["sec.gov"], resolver=broken).check("https://www.sec.gov/")
    with pytest.raises(CrawlPolicyViolationError, match="no addresses"):
        UrlGuard(["sec.gov"], resolver=lambda _h: []).check("https://www.sec.gov/")
    with pytest.raises(CrawlPolicyViolationError, match="invalid address"):
        UrlGuard(["sec.gov"], resolver=lambda _h: ["not-an-ip"]).check("https://www.sec.gov/")


def test_resolution_can_be_disabled() -> None:
    assert UrlGuard(["sec.gov"], resolver=None).check("https://www.sec.gov/x") == "https://www.sec.gov/x"


def test_from_settings_uses_static_suffixes() -> None:
    guard = UrlGuard.from_settings(CrawlerSettings(allowed_domain_suffixes=("sec.gov", "*.example.org")))
    assert guard.static_suffixes == frozenset({"sec.gov", "example.org"})


def test_crawl_scope_extends_allow_list_temporarily(guard: UrlGuard) -> None:
    url = "https://www.contoso.example/investors"
    assert not guard.is_allowed(url)
    with crawl_scope(["Contoso.Example", ""]) as scope:
        assert "contoso.example" in scope
        assert guard.is_allowed(url)
    assert not guard.is_allowed(url)
    assert current_scope() == frozenset()
    assert guard.is_allowed(url, extra_suffixes=["contoso.example"])


def test_crawl_scope_is_isolated_per_thread(guard: UrlGuard) -> None:
    seen: list[bool] = []
    with crawl_scope(["contoso.example"]):
        worker = threading.Thread(target=lambda: seen.append(guard.is_allowed("https://contoso.example/")))
        worker.start()
        worker.join()
    assert seen == [False]


def test_request_scope_includes_domain_and_seed_hosts() -> None:
    request = ResearchRequest(
        company_name="Northwind",
        domain="https://www.Northwind.example/about",
        seed_urls=("https://www.gartner.com/en/newsroom/press-releases/x", "https://news.other.example/a"),
    )
    assert request_scope_domains(request) == frozenset(
        {"northwind.example", "www.gartner.com", "news.other.example"}
    )


def test_helpers() -> None:
    assert host_matches("ir.northwind.example", "northwind.example")
    assert host_matches("northwind.example.", "*.northwind.example")
    assert not host_matches("northwind.example", "")
    assert normalize_url("HTTPS://Www.Sec.Gov:443/a?b=1#frag") == "https://www.sec.gov/a?b=1"
    assert normalize_url("http://host.example:8080") == "http://host.example:8080/"


def test_system_resolver_resolves_localhost() -> None:
    assert any(addr in ("127.0.0.1", "::1") for addr in system_resolver("localhost"))
