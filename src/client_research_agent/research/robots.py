"""robots.txt policy (RFC 9309).

Behaviour, per origin (``scheme://host[:port]``), cached for ``ttl_seconds``:

* **2xx** - rules are parsed with :mod:`urllib.robotparser` and applied to our
  product token (the part of the User-Agent before ``/``). Only the first
  500 KiB are parsed, the minimum RFC 9309 requires crawlers to honour.
* **4xx** (except 429) - "unavailable": RFC 9309 section 2.3.1.3 allows
  crawling everything.
* **429, 5xx, network errors, policy errors on redirect** - "unreachable":
  RFC 9309 section 2.3.1.4 requires assuming complete disallow. We fail
  closed, and cache the failure for a shorter ``failure_ttl_seconds`` so a
  transient outage does not block a host for the whole TTL.

``Crawl-delay`` is non-standard but widely used; it is exposed through
:meth:`RobotsPolicy.crawl_delay` so the fetcher can slow the host down.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from client_research_agent.observability.logging import get_logger
from client_research_agent.services.ports import HttpFetcher
from client_research_agent.utils.errors import AgentError, CrawlPolicyViolationError

_log = get_logger(__name__)

MAX_ROBOTS_BYTES = 500 * 1024


class RobotsMode(StrEnum):
    RULES = "rules"
    ALLOW_ALL = "allow_all"
    DISALLOW_ALL = "disallow_all"


@dataclass(frozen=True, slots=True)
class RobotsRecord:
    mode: RobotsMode
    parser: RobotFileParser | None
    fetched_at: float
    expires_at: float
    sitemaps: tuple[str, ...] = ()
    status_code: int | None = None


def product_token(user_agent: str) -> str:
    """``ClientResearchAgent/1.0 (+https://...)`` -> ``ClientResearchAgent``."""
    head = user_agent.strip().split("/", 1)[0].strip()
    return head.split()[0] if head.split() else "*"


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), "", "", ""))


def robots_url_for(url: str) -> str:
    return origin_of(url) + "/robots.txt"


def is_robots_url(url: str) -> bool:
    return urlsplit(url).path == "/robots.txt"


class RobotsPolicy:
    def __init__(
        self,
        fetcher: HttpFetcher,
        user_agent: str,
        *,
        enabled: bool = True,
        ttl_seconds: float = 3600.0,
        failure_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fetcher = fetcher
        self._token = product_token(user_agent)
        self._enabled = enabled
        self._ttl = ttl_seconds
        self._failure_ttl = failure_ttl_seconds
        self._clock = clock
        self._records: dict[str, RobotsRecord] = {}
        self._lock = threading.Lock()
        self._origin_locks: dict[str, threading.Lock] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def token(self) -> str:
        return self._token

    def _origin_lock(self, origin: str) -> threading.Lock:
        with self._lock:
            lock = self._origin_locks.get(origin)
            if lock is None:
                lock = threading.Lock()
                self._origin_locks[origin] = lock
            return lock

    def record_for(self, url: str) -> RobotsRecord:
        origin = origin_of(url)
        with self._origin_lock(origin):
            cached = self._records.get(origin)
            now = self._clock()
            if cached is not None and cached.expires_at > now:
                return cached
            record = self._load(origin, now)
            with self._lock:
                self._records[origin] = record
            return record

    def _load(self, origin: str, now: float) -> RobotsRecord:
        robots_url = origin + "/robots.txt"
        try:
            result = self._fetcher.fetch(robots_url)
        except AgentError as exc:
            _log.warning("robots.unreachable", url=robots_url, error=type(exc).__name__)
            return RobotsRecord(RobotsMode.DISALLOW_ALL, None, now, now + self._failure_ttl)
        status = result.status_code
        if 200 <= status < 300:
            parser = RobotFileParser(robots_url)
            text = result.text[:MAX_ROBOTS_BYTES]
            parser.parse(text.splitlines())
            sitemaps = tuple(parser.site_maps() or ())
            return RobotsRecord(RobotsMode.RULES, parser, now, now + self._ttl, sitemaps, status)
        if 400 <= status < 500 and status != 429:
            return RobotsRecord(RobotsMode.ALLOW_ALL, None, now, now + self._ttl, (), status)
        _log.warning("robots.unreachable", url=robots_url, status=status)
        return RobotsRecord(RobotsMode.DISALLOW_ALL, None, now, now + self._failure_ttl, (), status)

    def is_allowed(self, url: str) -> bool:
        if not self._enabled or is_robots_url(url):
            return True
        record = self.record_for(url)
        if record.mode is RobotsMode.ALLOW_ALL:
            return True
        if record.mode is RobotsMode.DISALLOW_ALL or record.parser is None:
            return False
        return record.parser.can_fetch(self._token, url)

    def check(self, url: str) -> None:
        if not self.is_allowed(url):
            raise CrawlPolicyViolationError(f"disallowed by robots.txt: {url}")

    def crawl_delay(self, url: str) -> float | None:
        if not self._enabled:
            return None
        record = self.record_for(url)
        if record.parser is None:
            return None
        delay = record.parser.crawl_delay(self._token)
        return float(delay) if delay is not None else None

    def sitemaps(self, url: str) -> tuple[str, ...]:
        """Sitemap URLs declared in robots.txt for the origin of ``url``."""
        return self.record_for(url).sitemaps
