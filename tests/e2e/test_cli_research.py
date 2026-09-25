"""``cra research`` end to end, offline: the Markdown brief must carry every required section."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from client_research_agent import cli
from client_research_agent.agent import factory
from client_research_agent.services.ports import ChatMessage, FetchResult
from client_research_agent.utils.errors import UpstreamServiceError
from tests.support.doubles import ScriptedLLM
from tests.support.world import northwind_fetcher

pytestmark = pytest.mark.e2e

REAL_BUILD = factory.build_runtime
REQUIRED_SECTIONS = (
    "## Executive Summary",
    "## Fit Assessment",
    "## Company Overview",
    "## Technology Priorities",
    "## Gartner-Relevant Insights",
    "## Opportunities",
    "## Risks",
    "## Discovery Questions",
    "## Executive Talking Points",
    "## Recommended Next Actions",
    "## Evidence",
    "## Citation Report",
    "## Confidence",
    "## Sources and Citation Links",
    "## Verified Facts",
    "## AI-Generated Recommendations",
)


def use_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **defaults: Any) -> None:
    def build(settings: Any, **kwargs: Any) -> Any:
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        kwargs.setdefault("var_dir", tmp_path / "var")
        return REAL_BUILD(settings, **kwargs)

    monkeypatch.setattr(factory, "build_runtime", build)


def run_cli(args: Sequence[str], tmp_path: Path) -> str:
    output = tmp_path / "brief.md"
    assert cli.main([*args, "--output", str(output)]) == 0
    return output.read_text(encoding="utf-8")


def assert_complete_brief(markdown: str) -> None:
    positions = [markdown.find(section) for section in REQUIRED_SECTIONS]
    assert all(p >= 0 for p in positions), [
        s for s, p in zip(REQUIRED_SECTIONS, positions, strict=True) if p < 0
    ]
    questions = markdown.split("## Discovery Questions", 1)[1].split("\n## ", 1)[0]
    assert len(re.findall(r"^\d\. ", questions, flags=re.MULTILINE)) == 5
    assert "**Verdict:**" in markdown
    assert "**Overall confidence:**" in markdown


def test_offline_research_with_deterministic_agents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_world(monkeypatch, tmp_path, fetcher=northwind_fetcher(), llm=None)
    markdown = run_cli(
        [
            "research",
            "--company",
            "Northwind Industries",
            "--domain",
            "northwind.example",
            "--ticker",
            "NWND",
        ],
        tmp_path,
    )
    assert_complete_brief(markdown)
    assert "https://www.sec.gov/" in markdown
    assert "https://northwind.example/" in markdown
    assert "**Verified facts**" in markdown
    assert "**AI-generated recommendations**" in markdown


def analyst_llm() -> ScriptedLLM:
    def plan(_messages: Sequence[ChatMessage]) -> dict[str, Any]:
        return {
            "queries": [
                {"criterion": "ai_and_data_focus", "query": "Northwind Industries industrial AI platform"}
            ],
            "hypotheses": ["Is the industrial AI platform in production?"],
        }

    def criterion(messages: Sequence[ChatMessage]) -> dict[str, Any]:
        ids = re.findall(r"\bE\d+\b", messages[-2].content if len(messages) > 1 else messages[-1].content)
        return {
            "score": 4,
            "confidence": 0.7,
            "rationale": "Public filings and press releases describe relevant programmes.",
            "evidence_ids": ids[:2],
        }

    return ScriptedLLM(
        routes={
            "You are planning public-source research": plan,
            "You assess exactly ONE criterion": criterion,
        },
        default="{}",
    )


def test_offline_research_with_scripted_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llm = analyst_llm()
    use_world(monkeypatch, tmp_path, fetcher=northwind_fetcher(), llm=llm)
    markdown = run_cli(
        ["research", "--company", "Northwind Industries", "--domain", "northwind.example"], tmp_path
    )
    assert_complete_brief(markdown)
    assert "scripted-llm" in markdown
    assert llm.calls


class OfflineFetcher:
    """Every request fails as it would with no network."""

    def fetch(self, url: str, *, headers: Any = None) -> FetchResult:
        raise UpstreamServiceError(f"network unreachable: {url}")


def test_no_network_yields_not_enough_evidence_brief(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_world(monkeypatch, tmp_path, fetcher=OfflineFetcher(), llm=None)
    markdown = run_cli(["research", "--company", "Contoso Pharmaceuticals", "--ticker", "CTSO"], tmp_path)
    assert_complete_brief(markdown)
    assert "Not enough evidence of fit" in markdown
    assert "Analyst review required" in markdown
