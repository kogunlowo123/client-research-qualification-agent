"""Public analyst-firm pages, fetched only when a human explicitly supplies them.

Compliance reasoning
--------------------
Analyst firms such as Gartner and Forrester publish some material openly
(press releases in their newsrooms, public articles), but their terms of use
prohibit automated scraping, crawling and systematic retrieval of their sites,
and their research documents are licensed, paywalled content. This source is
therefore deliberately narrow:

* It **never searches, crawls, follows links or enumerates sitemaps** on these
  domains. It fetches exactly the URLs a user placed in
  ``ResearchRequest.seed_urls`` - the equivalent of a person opening a link -
  one request each, through the same robots.txt-honouring, rate-limited
  fetcher as everything else.
* Only URLs on a configured analyst domain *and* under that domain's public
  path prefixes (newsroom / public articles) are accepted; anything that looks
  like licensed research (``/document/``, ``/doc/``, login or account pages)
  is refused with :class:`CrawlPolicyViolationError`.
* Documents are labelled ``DocumentType.ANALYST_PUBLIC`` with a lower trust
  score than primary sources, so briefs cite them as third-party commentary,
  not as facts about the prospect.

Operators who hold a research licence should integrate through the vendor's
sanctioned API or data feed instead of widening this list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from client_research_agent.models import DocumentType
from client_research_agent.observability.logging import get_logger
from client_research_agent.research.url_guard import host_matches
from client_research_agent.services.ports import FetchResult, HttpFetcher
from client_research_agent.utils.errors import CrawlPolicyViolationError

_log = get_logger(__name__)

DOCUMENT_TYPE = DocumentType.ANALYST_PUBLIC
_GATED_PATH = re.compile(
    r"/(?:document|doc|documents|login|account|signin|sign-in|subscribe|reprints?)(?:/|$)", re.I
)


@dataclass(frozen=True, slots=True)
class AnalystDomainPolicy:
    domain: str
    public_path_prefixes: tuple[str, ...] = ()

    def permits(self, url: str) -> bool:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not host_matches(parts.hostname or "", self.domain):
            return False
        path = parts.path or "/"
        if _GATED_PATH.search(path) or any(segment in (".", "..") for segment in path.split("/")):
            return False
        if not self.public_path_prefixes:
            return True
        lowered = path.lower()
        return any(lowered.startswith(prefix.lower()) for prefix in self.public_path_prefixes)


DEFAULT_ANALYST_POLICIES: tuple[AnalystDomainPolicy, ...] = (
    AnalystDomainPolicy("gartner.com", ("/en/newsroom/", "/en/articles/")),
    AnalystDomainPolicy("forrester.com", ("/press-newsroom/", "/blogs/")),
)


class PublicAnalystSource:
    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        policies: Sequence[AnalystDomainPolicy] = DEFAULT_ANALYST_POLICIES,
        max_urls: int = 10,
    ) -> None:
        self._fetcher = fetcher
        self._policies = tuple(policies)
        self._max_urls = max_urls

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(p.domain for p in self._policies)

    def is_analyst_domain(self, url: str) -> bool:
        host = urlsplit(url).hostname or ""
        return any(host_matches(host, p.domain) for p in self._policies)

    def is_permitted(self, url: str) -> bool:
        return any(p.permits(url) for p in self._policies)

    def select(self, seed_urls: Iterable[str]) -> tuple[list[str], list[str]]:
        """Split analyst-domain seeds into (permitted, rejected), capped at ``max_urls`` permitted."""
        permitted: list[str] = []
        rejected: list[str] = []
        for url in dict.fromkeys(seed_urls):
            if not self.is_analyst_domain(url):
                continue
            if self.is_permitted(url) and len(permitted) < self._max_urls:
                permitted.append(url)
            else:
                rejected.append(url)
        return permitted, rejected

    def fetch(self, url: str) -> FetchResult:
        """Fetch one explicitly supplied public analyst page (no link following)."""
        if not self.is_permitted(url):
            raise CrawlPolicyViolationError(f"not a permitted public analyst page: {url}")
        _log.info("analyst.fetch", url=url)
        return self._fetcher.fetch(url)
