from __future__ import annotations

import json

from client_research_agent.briefing.generator import BriefGenerationAgent
from client_research_agent.briefing.renderer import (
    EMPTY_SECTION,
    escape_markdown,
    render_json,
    render_markdown,
    render_statements,
    safe_link_target,
    to_json_dict,
)
from client_research_agent.config.settings import AppSettings
from client_research_agent.models import BriefStatement, ClientBrief, Evidence, ProvenanceKind
from client_research_agent.prompts.registry import default_registry
from tests.unit.briefing.conftest import Qualified
from tests.unit.qualification.helpers import COMPANY


def build(settings: AppSettings, qualified: Qualified) -> ClientBrief:
    return BriefGenerationAgent(None, default_registry(), settings).generate(
        run_id="run-md",
        company=COMPANY,
        qualification=qualified.result,
        evidence=qualified.evidence,
        warnings=("check <this> | now",),
    )


def test_markdown_structure(settings: AppSettings, qualified: Qualified) -> None:
    brief = build(settings, qualified)
    markdown = render_markdown(brief)
    for heading in (
        "# Client Brief: Acme Corp",
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
        "## Warnings",
        "## Model and Prompt Versions",
    ):
        assert heading in markdown
    assert "**Verified facts**" in markdown
    assert "**AI-generated recommendations**" in markdown
    assert "| Criterion | Weight | Score (0-5) | Weighted | Confidence | Evidence |" in markdown
    assert "| Company Size & Scale | 20% | 5 |" in markdown
    first = brief.evidence[0]
    assert f"[\\[{first.evidence_id}\\]]({first.url})" in markdown
    assert "not statements by Gartner" in markdown
    assert "5. " in markdown
    assert "check &lt;this&gt; \\| now" in markdown
    assert markdown.endswith("\n")


def test_render_statements_separates_provenance() -> None:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            chunk_id="c",
            url="javascript:alert(1)",
            title="t",
            quote="q",
            document_type="press_release",
        )
    }
    lines = render_statements(
        [
            BriefStatement(text="Rec", provenance=ProvenanceKind.AI_RECOMMENDATION),
            BriefStatement(text="Fact", provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=("E1", "E9")),
        ],
        evidence,
    )
    assert lines.index("**Verified facts**") < lines.index("**AI-generated recommendations**")
    assert "- Fact \\[E1\\] \\[E9\\]" in lines
    assert render_statements([], evidence) == [EMPTY_SECTION, ""]


def test_escaping_and_link_safety() -> None:
    assert escape_markdown("[click](http://x) <script> *bold*") == (
        "\\[click\\](http://x) &lt;script&gt; \\*bold\\*"
    )
    assert safe_link_target("javascript:alert(1)") is None
    assert safe_link_target("https://a.example/x y(1)") == "https://a.example/x%20y%281%29"


def test_json_round_trip(settings: AppSettings, qualified: Qualified) -> None:
    brief = build(settings, qualified)
    payload = json.loads(render_json(brief))
    assert payload == to_json_dict(brief)
    assert payload["qualification"]["verdict"] == brief.qualification.verdict.value
    assert ClientBrief.model_validate(payload) == brief
    assert "\n" not in render_json(brief, indent=None)


def test_markdown_without_evidence_or_warnings(settings: AppSettings, qualified: Qualified) -> None:
    brief = build(settings, qualified).model_copy(
        update={"evidence": (), "warnings": (), "model_versions": {}}
    )
    markdown = render_markdown(brief)
    assert "_No evidence was retrieved._" in markdown
    assert "## Warnings" not in markdown
    assert "## Model and Prompt Versions" not in markdown
