"""Render a ``ClientBrief`` to Markdown (for people) and JSON (for systems).

The Markdown keeps provenance visible everywhere: each section lists
"Verified facts" separately from "AI-generated recommendations", and every
citation marker ``[E1]`` links to the public source it cites. All text that
originates from sources or models is escaped so it cannot inject links,
HTML or table breaks into the rendered brief.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from client_research_agent.citations.validator import is_http_url
from client_research_agent.models import BriefStatement, ClientBrief, Evidence, ProvenanceKind
from client_research_agent.qualification.criteria import get_definition

_MD_SPECIAL = re.compile(r"([\\`*_\[\]|#])")
_VERDICT_TITLES = {
    "good_fit": "Good fit",
    "potential_fit": "Potential fit",
    "not_enough_evidence": "Not enough evidence of fit",
}
EMPTY_SECTION = "_No statements met the evidence bar for this section._"


def escape_markdown(text: str) -> str:
    collapsed = " ".join(text.split())
    escaped = _MD_SPECIAL.sub(r"\\\1", collapsed)
    return escaped.replace("<", "&lt;").replace(">", "&gt;")


def safe_link_target(url: str) -> str | None:
    if not is_http_url(url):
        return None
    return quote(url, safe=":/?#=&%@+,;~.-_!$'*")


def _citation(evidence_id: str, evidence: Mapping[str, Evidence]) -> str:
    item = evidence.get(evidence_id)
    target = safe_link_target(item.url) if item is not None else None
    return f"[\\[{evidence_id}\\]]({target})" if target else f"\\[{evidence_id}\\]"


def _statement_line(statement: BriefStatement, evidence: Mapping[str, Evidence]) -> str:
    markers = " ".join(_citation(i, evidence) for i in statement.evidence_ids)
    return f"- {escape_markdown(statement.text)}" + (f" {markers}" if markers else "")


def render_statements(statements: Sequence[BriefStatement], evidence: Mapping[str, Evidence]) -> list[str]:
    facts = [s for s in statements if s.provenance is ProvenanceKind.VERIFIED_FACT]
    recommendations = [s for s in statements if s.provenance is ProvenanceKind.AI_RECOMMENDATION]
    if not facts and not recommendations:
        return [EMPTY_SECTION, ""]
    lines: list[str] = []
    if facts:
        lines.append("**Verified facts**")
        lines.append("")
        lines.extend(_statement_line(s, evidence) for s in facts)
        lines.append("")
    if recommendations:
        lines.append("**AI-generated recommendations**")
        lines.append("")
        lines.extend(_statement_line(s, evidence) for s in recommendations)
        lines.append("")
    return lines


def _table_cell(text: str, limit: int = 240) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 3].rstrip() + "..."
    return escape_markdown(collapsed)


def render_markdown(brief: ClientBrief) -> str:
    evidence = {item.evidence_id: item for item in brief.evidence}
    q = brief.qualification
    llm = brief.model_versions.get("llm", "unknown")
    lines: list[str] = [
        f"# Client Brief: {escape_markdown(brief.company)}",
        "",
        f"_Run `{brief.run_id}` - generated {brief.generated_at.isoformat(timespec='seconds')} - model "
        f"{escape_markdown(llm)}_",
        "",
        "> **Verified facts** restate cited public evidence and passed automated citation checks. "
        "**AI-generated recommendations** are model analysis and should be validated before use.",
        "",
        "## Executive Summary",
        "",
        *render_statements(brief.executive_summary, evidence),
        "## Fit Assessment",
        "",
        f"**Verdict:** {_VERDICT_TITLES.get(q.verdict.value, q.verdict.value)} - "
        f"**Weighted score:** {q.weighted_score:.2f} / 5 - "
        f"**Overall confidence:** {q.overall_confidence:.0%}",
        "",
        escape_markdown(q.verdict_rationale),
        "",
        "| Criterion | Weight | Score (0-5) | Weighted | Confidence | Evidence |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for score in q.scores:
        cites = " ".join(_citation(i, evidence) for i in score.evidence_ids) or "none"
        lines.append(
            f"| {escape_markdown(get_definition(score.criterion).title)} | {score.weight:.0%} | "
            f"{score.score} | "
            f"{score.weighted:.2f} | {score.confidence:.0%} | {cites} |"
        )
    lines.append("")
    for score in q.scores:
        lines.append(
            f"- **{escape_markdown(get_definition(score.criterion).title)}:** "
            f"{escape_markdown(score.rationale)}"
        )
    lines.append("")

    for title, statements, note in (
        ("Company Overview", brief.company_overview, None),
        ("Technology Priorities", brief.technology_priorities, None),
        (
            "Gartner-Relevant Insights",
            brief.gartner_relevant_insights,
            "_Theme mappings are our analysis of public evidence against publicly known Gartner strategic "
            "technology trend themes; they are not statements by Gartner._",
        ),
        ("Opportunities", brief.opportunities, None),
        ("Risks", brief.risks, None),
    ):
        lines.extend([f"## {title}", ""])
        if note:
            lines.extend([note, ""])
        lines.extend(render_statements(statements, evidence))

    lines.extend(["## Discovery Questions", ""])
    lines.extend(
        f"{index}. {escape_markdown(question)}" for index, question in enumerate(brief.discovery_questions, 1)
    )
    lines.append("")
    for title, statements in (
        ("Executive Talking Points", brief.executive_talking_points),
        ("Recommended Next Actions", brief.recommended_next_actions),
    ):
        lines.extend([f"## {title}", "", *render_statements(statements, evidence)])

    lines.extend(["## Evidence", ""])
    if brief.evidence:
        lines.extend(["| ID | Source | Type | Date | Quote |", "|---|---|---|---|---|"])
        for item in brief.evidence:
            target = safe_link_target(item.url)
            source = f"[{_table_cell(item.title, 80)}]({target})" if target else _table_cell(item.title, 80)
            date_text = item.publication_date.isoformat() if item.publication_date else "undated"
            lines.append(
                f"| {item.evidence_id} | {source} | {item.document_type.value} | {date_text} | "
                f"{_table_cell(item.quote)} |"
            )
    else:
        lines.append("_No evidence was retrieved._")
    lines.append("")

    report = brief.citation_report
    lines.extend(
        [
            "## Citation Report",
            "",
            f"- Fact statements checked: {report.total_statements}",
            f"- Fact statements supported: {report.supported_statements} ({report.coverage:.0%})",
            f"- Citation checks performed: {len(report.checks)}",
            f"- Statements removed as unsupported: {len(report.removed_statements)}",
            "",
        ]
    )
    if brief.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {escape_markdown(w)}" for w in brief.warnings)
        lines.append("")
    if brief.model_versions:
        lines.extend(["## Model and Prompt Versions", ""])
        lines.extend(
            f"- {escape_markdown(key)}: `{value.replace('`', '')}`"
            for key, value in sorted(brief.model_versions.items())
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_json_dict(brief: ClientBrief) -> dict[str, Any]:
    payload: dict[str, Any] = brief.model_dump(mode="json")
    return payload


def render_json(brief: ClientBrief, *, indent: int | None = 2) -> str:
    return json.dumps(to_json_dict(brief), indent=indent, ensure_ascii=False)
