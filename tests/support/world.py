"""Offline "public web" for orchestrator-level tests.

Reuses the research slice's fixtures (EDGAR JSON, corporate HTML) through
``tests.unit.research.test_ingestion.register_world`` so integration, e2e,
security and performance tests share one realistic company: Northwind
Industries (ticker NWND, domain northwind.example).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from client_research_agent.agent.factory import AgentRuntime, build_runtime
from client_research_agent.config.settings import AppSettings
from client_research_agent.models import ResearchRequest
from client_research_agent.security.rbac import Principal, Role
from tests.unit.research.helpers import ScriptedFetcher
from tests.unit.research.test_ingestion import BASE, register_world

COMPANY = "Northwind Industries"
DOMAIN = "northwind.example"
TICKER = "NWND"
ANALYST = Principal(id="analyst@example.com", roles=frozenset({Role.ANALYST}))


def northwind_fetcher(extra: Callable[[ScriptedFetcher], None] | None = None) -> ScriptedFetcher:
    fetcher = ScriptedFetcher()
    register_world(fetcher)
    if extra is not None:
        extra(fetcher)
    return fetcher


def northwind_request(**overrides: Any) -> ResearchRequest:
    values: dict[str, Any] = {
        "company_name": COMPANY,
        "domain": DOMAIN,
        "ticker": TICKER,
        "industry": "Industrial Manufacturing",
        "max_documents": 40,
        "requested_by": ANALYST.id,
    }
    values.update(overrides)
    return ResearchRequest.model_validate(values)


def local_runtime(settings: AppSettings, tmp_path: Path, **overrides: Any) -> AgentRuntime:
    overrides.setdefault("fetcher", northwind_fetcher())
    overrides.setdefault("llm", None)
    return build_runtime(settings, var_dir=tmp_path / "var", **overrides)


__all__ = [
    "ANALYST",
    "BASE",
    "COMPANY",
    "DOMAIN",
    "TICKER",
    "local_runtime",
    "northwind_fetcher",
    "northwind_request",
]
