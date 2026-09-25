from __future__ import annotations

import gzip
from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest
import respx

from client_research_agent.config.settings import AppSettings, CrawlerSettings
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.research.fetcher import (
    HttpxFetcher,
    PolicyEnforcingFetcher,
    content_type_allowed,
    detect_encoding,
    media_type,
    parse_retry_after,
)
from client_research_agent.research.rate_limit import HostRateLimiter
from client_research_agent.research.url_guard import UrlGuard, crawl_scope
from client_research_agent.utils.errors import (
    CircuitOpenError,
    CrawlPolicyViolationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import RetryPolicy
from tests.unit.research.helpers import ScriptedFetcher, public_resolver

CRAWLER = CrawlerSettings(
    user_agent="TestAgent/1.0 (+https://agent.example)", contact_email="ops@agent.example"
)
URL = "https://www.sec.gov/page"


@pytest.fixture
def mock_router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def fetcher() -> Iterator[HttpxFetcher]:
    with HttpxFetcher(CRAWLER, max_response_bytes=1024) as instance:
        yield instance


class TestHttpxFetcher:
    def test_success_sends_declared_identity(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        route = mock_router.get(URL).respond(
            200, text="<html><body>ok</body></html>", headers={"Content-Type": "text/html; charset=utf-8"}
        )
        result = fetcher.fetch(URL, headers={"X-Extra": "1"})
        assert result.ok
        assert result.content_type == "text/html"
        assert "ok" in result.text
        sent = route.calls.last.request.headers
        assert sent["User-Agent"] == CRAWLER.user_agent
        assert sent["From"] == "ops@agent.example"
        assert sent["X-Extra"] == "1"
        assert fetcher.default_headers["From"] == "ops@agent.example"

    def test_429_maps_to_rate_limited_with_retry_after(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        mock_router.get(URL).respond(429, headers={"Retry-After": "7"})
        with pytest.raises(RateLimitedError) as info:
            fetcher.fetch(URL)
        assert info.value.retry_after_seconds == 7.0

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_maps_to_upstream_service_error(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher, status: int
    ) -> None:
        mock_router.get(URL).respond(status)
        with pytest.raises(UpstreamServiceError) as info:
            fetcher.fetch(URL)
        assert info.value.status_code == status

    def test_timeout_maps_to_upstream_timeout(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        mock_router.get(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(UpstreamTimeoutError):
            fetcher.fetch(URL)

    def test_connect_error_maps_to_upstream_service_error(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        mock_router.get(URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(UpstreamServiceError, match="ConnectError"):
            fetcher.fetch(URL)

    def test_invalid_url_is_a_policy_violation(self, fetcher: HttpxFetcher) -> None:
        with pytest.raises(CrawlPolicyViolationError):
            fetcher.fetch("https://exa mple.com:bad/")

    def test_4xx_is_returned_not_raised(self, mock_router: respx.MockRouter, fetcher: HttpxFetcher) -> None:
        mock_router.get(URL).respond(404, text="missing", headers={"Content-Type": "text/plain"})
        result = fetcher.fetch(URL)
        assert result.status_code == 404
        assert not result.ok

    def test_redirect_is_returned_with_location(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        mock_router.get(URL).respond(301, headers={"Location": "https://www.sec.gov/new"})
        result = fetcher.fetch(URL)
        assert result.status_code == 301
        assert result.headers["location"] == "https://www.sec.gov/new"
        assert result.text == ""

    def test_disallowed_content_type(self, mock_router: respx.MockRouter, fetcher: HttpxFetcher) -> None:
        mock_router.get(URL).respond(200, content=b"%PDF-1.7", headers={"Content-Type": "application/pdf"})
        with pytest.raises(CrawlPolicyViolationError, match="application/pdf"):
            fetcher.fetch(URL)

    def test_declared_length_over_cap(self, mock_router: respx.MockRouter, fetcher: HttpxFetcher) -> None:
        mock_router.get(URL).respond(200, content=b"x" * 2048, headers={"Content-Type": "text/plain"})
        with pytest.raises(CrawlPolicyViolationError, match="exceeds cap"):
            fetcher.fetch(URL)

    def test_streamed_body_over_cap_counts_decompressed_bytes(
        self, mock_router: respx.MockRouter, fetcher: HttpxFetcher
    ) -> None:
        bomb = gzip.compress(b"a" * 50_000)
        mock_router.get(URL).respond(
            200, content=bomb, headers={"Content-Type": "text/plain", "Content-Encoding": "gzip"}
        )
        with pytest.raises(CrawlPolicyViolationError, match="exceeds 1024 bytes"):
            fetcher.fetch(URL)

    def test_meta_charset_is_honoured(self, mock_router: respx.MockRouter, fetcher: HttpxFetcher) -> None:
        body = '<html><meta charset="windows-1252"><p>Café</p></html>'.encode("cp1252")
        mock_router.get(URL).respond(200, content=body, headers={"Content-Type": "text/html"})
        assert "Café" in fetcher.fetch(URL).text

    def test_injected_client_is_not_closed(self) -> None:
        client = httpx.Client()
        HttpxFetcher(CRAWLER, client=client).close()
        assert not client.is_closed
        client.close()


def test_helpers() -> None:
    assert media_type("Text/HTML; charset=UTF-8") == "text/html"
    assert content_type_allowed("")
    assert content_type_allowed("application/ld+json")
    assert content_type_allowed("application/vnd.sec+xml")
    assert not content_type_allowed("image/png")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_retry_after(None) is None
    assert parse_retry_after("  ") is None
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after("Thu, 01 Jan 2026 00:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Wed, 31 Dec 2025 00:00:00 GMT", now=now) == 0.0
    assert parse_retry_after("soon") is None
    assert detect_encoding("text/html; charset=ISO-8859-1", b"") == "iso8859-1"
    assert detect_encoding("text/html; charset=bogus", b"\xef\xbb\xbfhi") == "utf-8-sig"
    assert detect_encoding("text/html", b"<html>") == "utf-8"


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def build_policy_fetcher(
    inner: ScriptedFetcher,
    fake: FakeTime,
    *,
    suffixes: tuple[str, ...] = ("northwind.example", "sec.gov"),
    breaker_threshold: int = 5,
    max_attempts: int = 3,
) -> PolicyEnforcingFetcher:
    return PolicyEnforcingFetcher(
        inner,
        guard=UrlGuard(suffixes, resolver=public_resolver),
        user_agent="TestAgent/1.0",
        limiter=HostRateLimiter(1.0, clock=fake.clock, sleep=fake.sleep),
        retry_policy=RetryPolicy(max_attempts=max_attempts, initial_backoff_seconds=0.1, jitter=0.0),
        breaker_failure_threshold=breaker_threshold,
        sleep=fake.sleep,
        clock=fake.clock,
    )


ROBOTS_URL = "https://northwind.example/robots.txt"


class TestPolicyEnforcingFetcher:
    def test_allowed_fetch_consults_robots_first(self, scripted: ScriptedFetcher) -> None:
        fake = FakeTime()
        scripted.add(ROBOTS_URL, "User-agent: *\nDisallow: /private/\n", content_type="text/plain")
        scripted.add("https://northwind.example/news", "<p>news</p>")
        policy = build_policy_fetcher(scripted, fake)
        result = policy.fetch("https://northwind.example/news")
        assert result.ok
        assert scripted.urls == [ROBOTS_URL, "https://northwind.example/news"]
        assert fake.sleeps == [pytest.approx(1.0)]
        assert get_metrics().counter("crawler.fetches", status="2xx") == 2

    def test_robots_disallow_blocks_without_fetching(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "User-agent: *\nDisallow: /private/\n")
        policy = build_policy_fetcher(scripted, FakeTime())
        with pytest.raises(CrawlPolicyViolationError, match="robots"):
            policy.fetch("https://northwind.example/private/x")
        assert "https://northwind.example/private/x" not in scripted.urls

    def test_guard_blocks_before_any_request(self, scripted: ScriptedFetcher) -> None:
        policy = build_policy_fetcher(scripted, FakeTime())
        with pytest.raises(CrawlPolicyViolationError, match="allow-list"):
            policy.fetch("https://evil.example/")
        assert scripted.calls == []

    def test_request_scope_opens_the_guard(self, scripted: ScriptedFetcher) -> None:
        scripted.add("https://contoso.example/robots.txt", "", status=404)
        scripted.add("https://contoso.example/", "<p>home</p>")
        policy = build_policy_fetcher(scripted, FakeTime())
        with crawl_scope(["contoso.example"]):
            assert policy.fetch("https://contoso.example/").ok

    def test_redirects_are_followed_and_rechecked(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.add("https://northwind.example/old", status=301, headers={"Location": "/new"})
        scripted.add("https://northwind.example/new", "<p>moved</p>")
        policy = build_policy_fetcher(scripted, FakeTime())
        result = policy.fetch("https://northwind.example/old")
        assert result.url == "https://northwind.example/old"
        assert result.final_url == "https://northwind.example/new"
        assert result.text == "<p>moved</p>"

    def test_redirect_to_disallowed_host_is_blocked(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.add(
            "https://northwind.example/go",
            status=302,
            headers={"location": "http://169.254.169.254/latest/meta-data/"},
        )
        policy = build_policy_fetcher(scripted, FakeTime())
        with pytest.raises(CrawlPolicyViolationError, match="IP-literal"):
            policy.fetch("https://northwind.example/go")

    def test_redirect_loop_is_bounded(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.add("https://northwind.example/a", status=302, headers={"Location": "/b"})
        scripted.add("https://northwind.example/b", status=302, headers={"Location": "/a"})
        policy = build_policy_fetcher(scripted, FakeTime())
        with pytest.raises(CrawlPolicyViolationError, match="too many redirects"):
            policy.fetch("https://northwind.example/a")

    def test_transient_errors_are_retried(self, scripted: ScriptedFetcher) -> None:
        fake = FakeTime()
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.fail("https://northwind.example/page", UpstreamServiceError("boom", 503))
        scripted.add("https://northwind.example/page", "<p>ok</p>")
        policy = build_policy_fetcher(scripted, fake)
        assert policy.fetch("https://northwind.example/page").ok
        assert scripted.urls.count("https://northwind.example/page") == 2

    def test_retry_exhaustion_raises_and_counts_error(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.fail("https://northwind.example/down", UpstreamTimeoutError("slow"))
        policy = build_policy_fetcher(scripted, FakeTime(), max_attempts=2)
        with pytest.raises(UpstreamTimeoutError):
            policy.fetch("https://northwind.example/down")
        assert get_metrics().counter("crawler.fetch_errors", error="UpstreamTimeoutError") == 1

    def test_breaker_opens_per_host(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "", status=404)
        scripted.fail("https://northwind.example/down", UpstreamServiceError("boom", 500))
        policy = build_policy_fetcher(scripted, FakeTime(), breaker_threshold=2, max_attempts=1)
        for _ in range(2):
            with pytest.raises(UpstreamServiceError):
                policy.fetch("https://northwind.example/down")
        with pytest.raises(CircuitOpenError):
            policy.fetch("https://northwind.example/other")
        assert policy.breaker_for("www.sec.gov").state == "closed"

    def test_crawl_delay_slows_the_host(self, scripted: ScriptedFetcher) -> None:
        fake = FakeTime()
        scripted.add(ROBOTS_URL, "User-agent: *\nCrawl-delay: 4\n")
        scripted.add("https://northwind.example/a", "a")
        scripted.add("https://northwind.example/b", "b")
        policy = build_policy_fetcher(scripted, fake)
        policy.fetch("https://northwind.example/a")
        policy.fetch("https://northwind.example/b")
        assert policy.limiter.rate_for("northwind.example") == pytest.approx(0.25)
        assert fake.sleeps[-1] == pytest.approx(4.0)

    def test_excessive_crawl_delay_is_refused(self, scripted: ScriptedFetcher) -> None:
        scripted.add(ROBOTS_URL, "User-agent: *\nCrawl-delay: 600\n")
        policy = build_policy_fetcher(scripted, FakeTime())
        with pytest.raises(CrawlPolicyViolationError, match="crawl-delay"):
            policy.fetch("https://northwind.example/a")

    def test_robots_unreachable_fails_closed(self, scripted: ScriptedFetcher) -> None:
        scripted.fail(ROBOTS_URL, UpstreamServiceError("down", 503))
        policy = build_policy_fetcher(scripted, FakeTime(), max_attempts=1)
        with pytest.raises(CrawlPolicyViolationError, match="robots"):
            policy.fetch("https://northwind.example/a")
        assert policy.robots.enabled
        assert policy.guard.static_suffixes == frozenset({"northwind.example", "sec.gov"})

    def test_from_settings_wires_a_working_stack(self, mock_router: respx.MockRouter) -> None:
        settings = AppSettings(crawler=CRAWLER)
        mock_router.get("https://www.sec.gov/robots.txt").respond(404)
        mock_router.get(URL).respond(200, text="hello", headers={"Content-Type": "text/plain"})
        fake = FakeTime()
        policy = PolicyEnforcingFetcher.from_settings(
            settings, resolver=public_resolver, sleep=fake.sleep, clock=fake.clock
        )
        result = policy.fetch(URL)
        assert result.text == "hello"
