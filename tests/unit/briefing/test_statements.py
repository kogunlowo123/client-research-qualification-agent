from __future__ import annotations

from client_research_agent.briefing.statements import (
    StatementDraft,
    drafts_to_statements,
    matching_sentence,
    qualification_summary,
    statements_summary,
    verbatim_fact,
)
from client_research_agent.models import BriefStatement, ProvenanceKind
from tests.unit.briefing.conftest import Qualified
from tests.unit.qualification.helpers import COMPANY, acme_corpus, registry_for


def test_drafts_to_statements_rules() -> None:
    drafts = [
        StatementDraft(text="Revenue was $12.4 billion [E1].", provenance="verified_fact"),
        StatementDraft(text="An uncited fact.", provenance="verified_fact"),
        StatementDraft(text="Lead with AI.", evidence_ids=[" E2 ", ""]),
        StatementDraft(text="[E3]", provenance="ai_recommendation"),
        StatementDraft(text="Overflow.", provenance="ai_recommendation"),
    ]
    statements, warnings = drafts_to_statements(drafts, section="s", limit=2)
    assert statements[0] == BriefStatement(
        text="Revenue was $12.4 billion.", provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=("E1",)
    )
    assert statements[1].evidence_ids == ("E2",)
    assert len(statements) == 2
    assert any("dropped uncited fact" in w for w in warnings)
    assert any("truncated to 2" in w for w in warnings)


def test_facts_only_sections_drop_recommendations() -> None:
    statements, warnings = drafts_to_statements(
        [StatementDraft(text="Opinion.")], section="company_overview", facts_only=True
    )
    assert statements == []
    assert "facts-only" in warnings[0]


def test_verbatim_fact_and_matching_sentence() -> None:
    registry = registry_for(acme_corpus())
    assert matching_sentence(registry, "E1", ("employees",)) == (
        "The company employs approximately 45,000 employees in 30 countries."
    )
    assert matching_sentence(registry, "E1", ("kubernetes",)) is None
    fact = verbatim_fact(registry, "E1", ("kubernetes",))
    assert fact.text == " ".join(registry.require("E1").quote.split())
    assert fact.provenance is ProvenanceKind.VERIFIED_FACT
    assert fact.evidence_ids == ("E1",)


def test_summaries(qualified: Qualified) -> None:
    summary = qualification_summary(COMPANY, qualified.result)
    assert summary.startswith(f"Company: {COMPANY}")
    assert "Company Size & Scale: 5/5" in summary
    assert statements_summary("Risks", []) == "Risks: none"
    rendered = statements_summary(
        "Risks", [BriefStatement(text="x", provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=("E1",))]
    )
    assert "- [verified_fact] x (E1)" in rendered
