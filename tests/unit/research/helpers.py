"""Helpers shared by the research unit tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from client_research_agent.services.ports import FetchResult

FIXTURES = Path(__file__).parent / "fixtures"
PUBLIC_IP = "93.184.216.34"


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def public_resolver(_host: str) -> list[str]:
    return [PUBLIC_IP]


Response = FetchResult | BaseException | Callable[[str], FetchResult]


@dataclass
class ScriptedFetcher:
    """HttpFetcher double with headers, redirects and raised errors (research tests only)."""

    responses: dict[str, list[Response]] = field(default_factory=dict)
    calls: list[tuple[str, Mapping[str, str] | None]] = field(default_factory=list)

    def add(
        self,
        url: str,
        body: str = "",
        *,
        status: int = 200,
        content_type: str = "text/html",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        result = FetchResult(url, url, status, content_type, body, dict(headers or {}))
        self.responses.setdefault(url, []).append(result)

    def fail(self, url: str, error: BaseException) -> None:
        self.responses.setdefault(url, []).append(error)

    @property
    def urls(self) -> list[str]:
        return [u for u, _ in self.calls]

    def fetch(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResult:
        self.calls.append((url, headers))
        queue = self.responses.get(url)
        if not queue:
            return FetchResult(url, url, 404, "text/plain", "not found")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(url)
        return item
