"""Outbound URL policy: SSRF protection plus a domain allow-list.

Every URL the crawler touches passes through :class:`UrlGuard` before a socket
is opened, and again for every redirect hop. The guard rejects:

* schemes other than ``http``/``https``;
* credentials embedded in the URL (``https://user:pass@host``);
* ports other than the scheme defaults (80/443);
* IP-literal hosts, and numeric/hex host spellings that resolvers may turn
  into IPs (``2130706433``, ``0x7f.1``);
* hostnames that resolve to any non-globally-routable address (private,
  loopback, link-local, multicast, reserved, IPv4-mapped private ...);
* hosts outside the allow-list.

The allow-list is the static ``CrawlerSettings.allowed_domain_suffixes`` plus a
*request scope*: the corporate domain and explicit seed-URL hosts of the
research request currently being processed. The scope lives in a
``ContextVar`` (see :func:`crawl_scope`) so one shared, thread-safe fetcher can
serve many concurrent requests without leaking one request's permissions into
another.

Known limitation: DNS answers can change between the check and the connect
(DNS rebinding). Re-validating every redirect hop and refusing IP literals
narrows the window; network-level egress controls remain the backstop.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from urllib.parse import urlsplit, urlunsplit

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.models import ResearchRequest
from client_research_agent.utils.errors import CrawlPolicyViolationError

Resolver = Callable[[str], Sequence[str]]

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+))*\.?$", re.IGNORECASE)
_HOST_LABEL = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)$")

_scope: ContextVar[frozenset[str]] = ContextVar("cra_crawl_scope", default=frozenset())


def system_resolver(hostname: str) -> list[str]:
    """Resolve ``hostname`` to every address the OS would connect to."""
    infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    return sorted({str(info[4][0]) for info in infos})


def normalize_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def host_of(url: str) -> str:
    """Lower-cased hostname of ``url`` (empty string when absent)."""
    return normalize_host(urlsplit(url).hostname or "")


def host_matches(host: str, suffix: str) -> bool:
    """True when ``host`` equals ``suffix`` or is a subdomain of it."""
    host = normalize_host(host)
    suffix = normalize_host(suffix).removeprefix("*.")
    return bool(suffix) and (host == suffix or host.endswith("." + suffix))


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication: lower-case scheme/host, no fragment, no default port."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    host = normalize_host(parts.hostname or "")
    port = parts.port
    netloc = host if port is None or port == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def request_scope_domains(request: ResearchRequest) -> frozenset[str]:
    """Domains a research request may crawl beyond the static allow-list."""
    domains: set[str] = set()
    if request.domain:
        domains.add(normalize_host(request.domain))
    for seed in request.seed_urls:
        if seed.host:
            domains.add(normalize_host(seed.host))
    return frozenset(domains)


@contextmanager
def crawl_scope(domains: Iterable[str]) -> Iterator[frozenset[str]]:
    """Temporarily extend the allow-list for the current context (thread/task)."""
    added = frozenset(normalize_host(d) for d in domains if d and d.strip())
    token = _scope.set(_scope.get() | added)
    try:
        yield _scope.get()
    finally:
        _scope.reset(token)


def current_scope() -> frozenset[str]:
    return _scope.get()


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None and not mapped.is_global:
            return False
    return ip.is_global and not ip.is_multicast and not ip.is_reserved


class UrlGuard:
    """Validates outbound URLs; raises :class:`CrawlPolicyViolationError` on any breach."""

    def __init__(
        self,
        allowed_domain_suffixes: Iterable[str],
        *,
        resolver: Resolver | None = system_resolver,
        allowed_ports: Iterable[int] = (80, 443),
    ) -> None:
        self._suffixes = frozenset(normalize_host(s).removeprefix("*.") for s in allowed_domain_suffixes if s)
        self._resolver = resolver
        self._ports = frozenset(allowed_ports)

    @classmethod
    def from_settings(
        cls, crawler: CrawlerSettings, *, resolver: Resolver | None = system_resolver
    ) -> UrlGuard:
        return cls(crawler.allowed_domain_suffixes, resolver=resolver)

    @property
    def static_suffixes(self) -> frozenset[str]:
        return self._suffixes

    def allowed_suffixes(self, extra: Iterable[str] = ()) -> frozenset[str]:
        return self._suffixes | current_scope() | {normalize_host(e) for e in extra}

    def is_host_allowed(self, host: str, extra: Iterable[str] = ()) -> bool:
        return any(host_matches(host, suffix) for suffix in self.allowed_suffixes(extra))

    def check(self, url: str, *, extra_suffixes: Iterable[str] = ()) -> str:
        """Return the normalized URL when it is safe to fetch, else raise."""
        try:
            parts = urlsplit(url.strip())
            port = parts.port
        except ValueError as exc:
            raise CrawlPolicyViolationError(f"malformed URL: {url!r}") from exc
        scheme = parts.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise CrawlPolicyViolationError(f"scheme {scheme or '<none>'!r} is not allowed: {url}")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise CrawlPolicyViolationError(f"credentials in URL are not allowed: {parts.hostname}")
        host = normalize_host(parts.hostname or "")
        if not host:
            raise CrawlPolicyViolationError(f"URL has no host: {url!r}")
        if port is not None and port not in self._ports:
            raise CrawlPolicyViolationError(f"port {port} is not allowed for {host}")
        self._check_host_syntax(host)
        if not self.is_host_allowed(host, extra_suffixes):
            raise CrawlPolicyViolationError(f"host {host!r} is not in the crawl allow-list")
        self._check_resolution(host)
        return normalize_url(url)

    def is_allowed(self, url: str, *, extra_suffixes: Iterable[str] = ()) -> bool:
        try:
            self.check(url, extra_suffixes=extra_suffixes)
        except CrawlPolicyViolationError:
            return False
        return True

    @staticmethod
    def _check_host_syntax(host: str) -> None:
        literal = host.strip("[]")
        try:
            ip = ipaddress.ip_address(literal.split("%", 1)[0])
        except ValueError:
            ip = None
        if ip is not None:
            kind = "public" if _is_public(literal) else "private/reserved"
            raise CrawlPolicyViolationError(f"IP-literal hosts are not allowed ({kind}): {host}")
        if _NUMERIC_HOST.match(host) or host.rsplit(".", 1)[-1].isdigit():
            raise CrawlPolicyViolationError(f"numeric host spelling is not allowed: {host}")
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise CrawlPolicyViolationError(f"invalid hostname: {host!r}") from exc
        if not all(_HOST_LABEL.match(label) for label in ascii_host.split(".")):
            raise CrawlPolicyViolationError(f"invalid hostname: {host!r}")

    def _check_resolution(self, host: str) -> None:
        if self._resolver is None:
            return
        try:
            addresses = list(self._resolver(host))
        except OSError as exc:
            raise CrawlPolicyViolationError(f"host {host!r} could not be resolved") from exc
        if not addresses:
            raise CrawlPolicyViolationError(f"host {host!r} resolved to no addresses")
        for address in addresses:
            try:
                public = _is_public(address)
            except ValueError as exc:
                raise CrawlPolicyViolationError(f"host {host!r} resolved to invalid address") from exc
            if not public:
                raise CrawlPolicyViolationError(f"host {host!r} resolves to non-public address {address}")
