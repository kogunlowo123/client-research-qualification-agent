"""HTTP fetching for public-source research.

Two layers, both implementing :class:`~client_research_agent.services.ports.HttpFetcher`:

:class:`HttpxFetcher`
    Raw transport. One ``httpx.Client``, declared User-Agent and ``From``
    headers, streamed bodies capped at ``max_response_bytes`` (checked on the
    *decompressed* stream, which also defuses compression bombs), a
    content-type allow-list, and error mapping onto the typed hierarchy:
    429 -> :class:`RateLimitedError` (honouring ``Retry-After``), 5xx ->
    :class:`UpstreamServiceError`, timeouts -> :class:`UpstreamTimeoutError`.
    It never follows redirects itself.

:class:`PolicyEnforcingFetcher`
    Crawl policy around any transport: :class:`UrlGuard` (SSRF + allow-list),
    :class:`RobotsPolicy` (including ``Crawl-delay``), the per-host
    :class:`HostRateLimiter`, retry with backoff and one circuit breaker per
    host. Redirects are followed here, re-running the whole policy on every
    hop, so a redirect can never bypass the guard or robots.txt.
"""

from __future__ import annotations

import codecs
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Self
from urllib.parse import urljoin

import httpx

from client_research_agent.config.settings import AppSettings, CrawlerSettings, ResilienceSettings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.research.rate_limit import HostRateLimiter
from client_research_agent.research.robots import RobotsPolicy, is_robots_url
from client_research_agent.research.url_guard import Resolver, UrlGuard, host_of, system_resolver
from client_research_agent.services.ports import FetchResult, HttpFetcher
from client_research_agent.utils.errors import (
    CrawlPolicyViolationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from client_research_agent.utils.resilience import CircuitBreaker, RetryPolicy, call_with_retry

_log = get_logger(__name__)

DEFAULT_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/xml",
        "application/xml",
        "application/json",
        "text/plain",
        "application/rss+xml",
        "application/atom+xml",
    }
)
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_META_CHARSET = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]+)""", re.IGNORECASE)
_MAX_CRAWL_DELAY_SECONDS = 60.0


def media_type(content_type: str) -> str:
    """``text/html; charset=utf-8`` -> ``text/html``."""
    return content_type.split(";", 1)[0].strip().lower()


def content_type_allowed(content_type: str, allowed: Iterable[str] = DEFAULT_ALLOWED_CONTENT_TYPES) -> bool:
    kind = media_type(content_type)
    if not kind:
        return True
    return kind in set(allowed) or kind.endswith(("+xml", "+json"))


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    """Seconds to wait from a ``Retry-After`` header (delta-seconds or HTTP-date)."""
    if value is None or not value.strip():
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    reference = now or datetime.now(UTC)
    return max(0.0, (when - reference).total_seconds())


def _charset_from_header(content_type: str) -> str | None:
    for param in content_type.split(";")[1:]:
        key, _, value = param.partition("=")
        if key.strip().lower() == "charset" and value.strip():
            return value.strip().strip("\"'")
    return None


def detect_encoding(content_type: str, body: bytes) -> str:
    candidates: list[str] = []
    header_charset = _charset_from_header(content_type)
    if header_charset:
        candidates.append(header_charset)
    if body.startswith(codecs.BOM_UTF8):
        candidates.append("utf-8-sig")
    meta = _META_CHARSET.search(body[:4096])
    if meta:
        candidates.append(meta.group(1).decode("ascii", errors="ignore"))
    for candidate in candidates:
        try:
            return codecs.lookup(candidate).name
        except LookupError:
            continue
    return "utf-8"


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered:
            return value
    return None


class HttpxFetcher:
    """Raw HTTP transport with size/content-type limits and typed error mapping."""

    def __init__(
        self,
        crawler: CrawlerSettings,
        *,
        client: httpx.Client | None = None,
        max_response_bytes: int | None = None,
        allowed_content_types: Iterable[str] = DEFAULT_ALLOWED_CONTENT_TYPES,
    ) -> None:
        self._max_bytes = max_response_bytes or crawler.max_response_bytes
        self._allowed = frozenset(allowed_content_types)
        self._default_headers = {
            "User-Agent": crawler.user_agent,
            "From": crawler.contact_email,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.5"
            ),
            "Accept-Language": "en;q=1.0",
        }
        timeout = httpx.Timeout(
            crawler.request_timeout_seconds, connect=min(10.0, crawler.request_timeout_seconds)
        )
        self._owns_client = client is None
        self._client = client or httpx.Client(follow_redirects=False, timeout=timeout)

    @property
    def default_headers(self) -> Mapping[str, str]:
        return dict(self._default_headers)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        request_headers = {**self._default_headers, **(headers or {})}
        try:
            with self._client.stream("GET", url, headers=request_headers, follow_redirects=False) as response:
                return self._to_result(url, response)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(f"timeout fetching {url}") from exc
        except httpx.InvalidURL as exc:
            raise CrawlPolicyViolationError(f"invalid URL: {url}") from exc
        except httpx.HTTPError as exc:
            raise UpstreamServiceError(f"transport error fetching {url}: {type(exc).__name__}") from exc

    def _to_result(self, url: str, response: httpx.Response) -> FetchResult:
        status = response.status_code
        headers = dict(response.headers.items())
        if status == 429:
            retry_after = parse_retry_after(response.headers.get("retry-after"))
            raise RateLimitedError(f"429 from {host_of(url)}", retry_after_seconds=retry_after)
        if status >= 500:
            raise UpstreamServiceError(f"HTTP {status} from {host_of(url)}", status_code=status)
        content_type = response.headers.get("content-type", "")
        if 300 <= status < 400:
            return FetchResult(url, str(response.url), status, media_type(content_type), "", headers)
        if 200 <= status < 300 and not content_type_allowed(content_type, self._allowed):
            raise CrawlPolicyViolationError(f"content type {media_type(content_type)!r} not allowed: {url}")
        body = self._read_capped(url, response)
        text = body.decode(detect_encoding(content_type, body), errors="replace")
        return FetchResult(url, str(response.url), status, media_type(content_type), text, headers)

    def _read_capped(self, url: str, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max_bytes:
            raise CrawlPolicyViolationError(f"response of {declared} bytes exceeds cap: {url}")
        buffer = bytearray()
        for chunk in response.iter_bytes():
            buffer.extend(chunk)
            if len(buffer) > self._max_bytes:
                raise CrawlPolicyViolationError(f"response exceeds {self._max_bytes} bytes: {url}")
        return bytes(buffer)


class _RobotsTransport:
    """Fetches robots.txt through the full policy except the robots check itself."""

    def __init__(self, owner: PolicyEnforcingFetcher) -> None:
        self._owner = owner

    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        return self._owner.fetch_unchecked_robots(url, headers=headers)


class PolicyEnforcingFetcher:
    """Composes guard, robots.txt, rate limiting, retry and per-host breakers around a transport."""

    def __init__(
        self,
        inner: HttpFetcher,
        *,
        guard: UrlGuard,
        user_agent: str,
        limiter: HostRateLimiter,
        retry_policy: RetryPolicy | None = None,
        respect_robots: bool = True,
        robots: RobotsPolicy | None = None,
        breaker_failure_threshold: int = 5,
        breaker_reset_seconds: float = 30.0,
        max_redirects: int = 5,
        max_crawl_delay_seconds: float = _MAX_CRAWL_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._guard = guard
        self._limiter = limiter
        self._retry = retry_policy or RetryPolicy()
        self._robots = robots or RobotsPolicy(_RobotsTransport(self), user_agent, enabled=respect_robots)
        self._breaker_threshold = breaker_failure_threshold
        self._breaker_reset = breaker_reset_seconds
        self._max_redirects = max_redirects
        self._max_crawl_delay = max_crawl_delay_seconds
        self._sleep = sleep
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        *,
        client: httpx.Client | None = None,
        resolver: Resolver | None = system_resolver,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> PolicyEnforcingFetcher:
        crawler = settings.crawler
        return cls(
            HttpxFetcher(crawler, client=client),
            guard=UrlGuard.from_settings(crawler, resolver=resolver),
            user_agent=crawler.user_agent,
            limiter=HostRateLimiter(crawler.requests_per_second_per_host, clock=clock, sleep=sleep),
            retry_policy=retry_policy_from(settings.resilience),
            respect_robots=crawler.respect_robots_txt,
            breaker_failure_threshold=settings.resilience.breaker_failure_threshold,
            breaker_reset_seconds=settings.resilience.breaker_reset_seconds,
            sleep=sleep,
            clock=clock,
        )

    @property
    def guard(self) -> UrlGuard:
        return self._guard

    @property
    def robots(self) -> RobotsPolicy:
        return self._robots

    @property
    def limiter(self) -> HostRateLimiter:
        return self._limiter

    def breaker_for(self, host: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(host)
            if breaker is None:
                breaker = CircuitBreaker(
                    f"crawl:{host}",
                    failure_threshold=self._breaker_threshold,
                    reset_timeout_seconds=self._breaker_reset,
                    clock=self._clock,
                )
                self._breakers[host] = breaker
            return breaker

    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        return self._follow(url, headers, check_robots=True)

    def fetch_unchecked_robots(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        """Fetch without consulting robots.txt; used only to retrieve robots.txt itself."""
        return self._follow(url, headers, check_robots=False)

    def _follow(self, url: str, headers: Mapping[str, str] | None, *, check_robots: bool) -> FetchResult:
        current = url
        for _ in range(self._max_redirects + 1):
            result = self._fetch_hop(current, headers, check_robots=check_robots)
            location = _header(result.headers, "location")
            if result.status_code not in REDIRECT_STATUSES or not location:
                return FetchResult(
                    url=url,
                    final_url=result.final_url or current,
                    status_code=result.status_code,
                    content_type=result.content_type,
                    text=result.text,
                    headers=result.headers,
                )
            current = urljoin(current, location)
            get_metrics().increment("crawler.redirects")
        raise CrawlPolicyViolationError(f"too many redirects (> {self._max_redirects}) from {url}")

    def _fetch_hop(self, url: str, headers: Mapping[str, str] | None, *, check_robots: bool) -> FetchResult:
        normalized = self._guard.check(url)
        host = host_of(normalized)
        if check_robots and not is_robots_url(normalized):
            self._robots.check(normalized)
            delay = self._robots.crawl_delay(normalized)
            if delay is not None and delay > 0:
                if delay > self._max_crawl_delay:
                    raise CrawlPolicyViolationError(f"crawl-delay {delay:.0f}s for {host} exceeds budget")
                self._limiter.set_min_interval(host, delay)

        def attempt() -> FetchResult:
            self._limiter.acquire(host)
            return self._inner.fetch(normalized, headers=headers)

        started = time.perf_counter()
        metrics = get_metrics()
        try:
            result = call_with_retry(
                attempt,
                policy=self._retry,
                breaker=self.breaker_for(host),
                sleep=self._sleep,
                on_retry=lambda n, exc: _log.info(
                    "crawler.retry", host=host, attempt=n, error=type(exc).__name__
                ),
            )
        except Exception as exc:
            metrics.increment("crawler.fetch_errors", error=type(exc).__name__)
            raise
        metrics.observe("crawler.fetch_latency_ms", (time.perf_counter() - started) * 1000)
        metrics.increment("crawler.fetches", status=f"{result.status_code // 100}xx")
        return result


def retry_policy_from(resilience: ResilienceSettings) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=resilience.max_attempts,
        initial_backoff_seconds=resilience.initial_backoff_seconds,
        max_backoff_seconds=resilience.max_backoff_seconds,
    )
