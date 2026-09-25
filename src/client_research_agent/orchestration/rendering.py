"""Markdown for a completed run: the rendered brief plus run-level sections.

``render_markdown`` (briefing slice) renders the brief itself. This module
appends what only the orchestrator knows: a confidence summary with the
verdict's sensitivity, the list of source links, the verified facts and the
AI-generated recommendations as two separate lists, the human-review status
and the responsible-AI notice.
"""

from __future__ import annotations

from collections.abc import Mapping

from client_research_agent.briefing.renderer import (
    escape_markdown,
    render_markdown,
    safe_link_target,
)
from client_research_agent.models import BriefStatement, ClientBrief, Evidence
from client_research_agent.orchestration.review import ReviewDecision
from client_research_agent.qualification.criteria import get_definition
from client_research_agent.scoring.engine import SensitivityReport

NO_SOURCES = "_No public sources were cited._"


def _citations(statement: BriefStatement, evidence: Mapping[str, Evidence]) -> str:
    links: list[str] = []
    for evidence_id in statement.evidence_ids:
        item = evidence.get(evidence_id)
        target = safe_link_target(item.url) if item is not None else None
        links.append(f"[\\[{evidence_id}\\]]({target})" if target else f"\\[{evidence_id}\\]")
    return (" " + " ".join(links)) if links else ""


def _statement_list(
    statements: tuple[BriefStatement, ...], evidence: Mapping[str, Evidence], empty: str
) -> list[str]:
    if not statements:
        return [empty, ""]
    lines = [f"- {escape_markdown(s.text)}{_citations(s, evidence)}" for s in statements]
    return [*lines, ""]


def render_confidence(brief: ClientBrief, sensitivity: SensitivityReport | None) -> list[str]:
    q = brief.qualification
    lines = [
        "## Confidence",
        "",
        f"- Overall confidence: {q.overall_confidence:.0%}",
        f"- Citation coverage: {brief.citation_report.coverage:.0%} of "
        f"{brief.citation_report.total_statements} fact statement(s) supported",
    ]
    lines.extend(
        f"- {escape_markdown(get_definition(s.criterion).title)}: {s.confidence:.0%} "
        f"({len(s.evidence_ids)} evidence item(s))"
        for s in q.scores
    )
    if sensitivity is not None:
        lines.append(f"- Sensitivity: {escape_markdown(sensitivity.summary())}")
    lines.append("")
    return lines


def render_sources(brief: ClientBrief) -> list[str]:
    lines = ["## Sources and Citation Links", ""]
    seen: dict[str, Evidence] = {}
    for item in brief.evidence:
        seen.setdefault(item.url, item)
    if not seen:
        return [*lines, NO_SOURCES, ""]
    for number, (url, item) in enumerate(seen.items(), 1):
        target = safe_link_target(url)
        cited = ", ".join(e.evidence_id for e in brief.evidence if e.url == url)
        title = escape_markdown(item.title)
        link = f"[{title}]({target})" if target else title
        date = item.publication_date.isoformat() if item.publication_date else "undated"
        lines.append(f"{number}. {link} - {item.document_type.value}, {date} (cited as {cited})")
    return [*lines, ""]


def render_provenance(brief: ClientBrief) -> list[str]:
    evidence = {item.evidence_id: item for item in brief.evidence}
    return [
        "## Verified Facts",
        "",
        "_Restated from cited public sources and checked against them automatically._",
        "",
        *_statement_list(brief.verified_facts, evidence, "_No statement met the verified-fact bar._"),
        "## AI-Generated Recommendations",
        "",
        "_Model analysis and suggestions; validate before acting on them._",
        "",
        *_statement_list(brief.recommendations, evidence, "_No recommendations were generated._"),
    ]


def render_review(decision: ReviewDecision) -> list[str]:
    lines = ["## Review Status", ""]
    if not decision.needs_review:
        return [*lines, "No analyst review required by policy.", ""]
    lines.append(f"**Analyst review required** (priority: {decision.priority.value}).")
    lines.append("")
    lines.extend(f"- {escape_markdown(detail)}" for detail in decision.details)
    return [*lines, ""]


def render_run_markdown(
    brief: ClientBrief,
    *,
    review: ReviewDecision,
    sensitivity: SensitivityReport | None = None,
    disclaimer: str | None = None,
) -> str:
    sections = [
        render_markdown(brief).rstrip(),
        "",
        *render_confidence(brief, sensitivity),
        *render_sources(brief),
        *render_provenance(brief),
        *render_review(review),
    ]
    if disclaimer:
        sections.extend(["## Responsible AI Notice", "", escape_markdown(disclaimer), ""])
    return "\n".join(sections).rstrip() + "\n"
