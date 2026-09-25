"""Citation quality gate: every citation in a brief resolves to real, retrieved, verbatim evidence."""

from __future__ import annotations

import re

import pytest

from client_research_agent.briefing.generator import BriefGenerationAgent
from client_research_agent.briefing.renderer import render_markdown
from client_research_agent.citations.validator import CitationValidator, is_http_url
from client_research_agent.config.settings import AppSettings
from client_research_agent.models import BriefStatement, ClientBrief, ProvenanceKind
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.agent import QualificationAgent
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.scoring.engine import ScoringEngine
from tests.support.doubles import make_chunk
from tests.unit.qualification.helpers import COMPANY, TODAY, KeywordRetriever, acme_corpus, registry_for

pytestmark = pytest.mark.rag_eval

CORPUS = acme_corpus()
CORPUS_URLS = {chunk.url for chunk in CORPUS}
CORPUS_TEXT = {chunk.chunk_id: chunk.text for chunk in CORPUS}


@pytest.fixture
def run(settings: AppSettings) -> tuple[ClientBrief, EvidenceRegistry]:
    registry = default_registry()
    output = QualificationAgent(None, KeywordRetriever(CORPUS), registry, settings, today=TODAY).qualify(
        COMPANY
    )
    result = ScoringEngine(settings.scoring).evaluate(output.scores)
    brief = BriefGenerationAgent(None, registry, settings).generate(
        run_id="citation-eval",
        company=COMPANY,
        qualification=result,
        evidence=output.evidence,
        retrieved_urls=CORPUS_URLS,
    )
    return brief, output.evidence


def test_every_cited_id_resolves_to_brief_evidence(run: tuple[ClientBrief, EvidenceRegistry]) -> None:
    brief, _ = run
    known = {e.evidence_id for e in brief.evidence}
    for statement in brief.all_statements():
        assert set(statement.evidence_ids) <= known
    for score in brief.qualification.scores:
        assert set(score.evidence_ids) <= known


def test_evidence_is_retrieved_http_and_verbatim(run: tuple[ClientBrief, EvidenceRegistry]) -> None:
    brief, _ = run
    assert brief.evidence
    for item in brief.evidence:
        assert is_http_url(item.url)
        assert item.url in CORPUS_URLS
        assert item.quote in CORPUS_TEXT[item.chunk_id]
        assert len(item.quote) <= 500
        assert 0.0 <= item.relevance <= 1.0


def test_citation_report_is_consistent(run: tuple[ClientBrief, EvidenceRegistry]) -> None:
    brief, _ = run
    report = brief.citation_report
    assert report.total_statements >= report.supported_statements == len(brief.verified_facts)
    assert report.coverage == 1.0
    supported_ids = {c.evidence_id for c in report.checks if c.supported}
    for statement in brief.verified_facts:
        assert set(statement.evidence_ids) <= supported_ids
    for check in report.checks:
        assert check.url in CORPUS_URLS
        assert 0.0 <= check.support_score <= 1.0


def test_markdown_citation_links_point_to_cited_sources(run: tuple[ClientBrief, EvidenceRegistry]) -> None:
    brief, _ = run
    markdown = render_markdown(brief)
    urls = {e.evidence_id: e.url for e in brief.evidence}
    links = re.findall(r"\[\\\[(E\d+)\\\]\]\(([^)]+)\)", markdown)
    assert links
    for evidence_id, url in links:
        assert urls[evidence_id] == url


def test_citations_to_unretrieved_documents_are_rejected(
    run: tuple[ClientBrief, EvidenceRegistry], settings: AppSettings
) -> None:
    _, evidence = run
    validator = CitationValidator(None, settings.guardrails)
    claim = BriefStatement(
        text=CORPUS[0].text.split(". ")[0] + ".",
        provenance=ProvenanceKind.VERIFIED_FACT,
        evidence_ids=(evidence.ids()[0],),
    )
    outcome = validator.validate({"s": [claim]}, evidence, retrieved_urls={"https://elsewhere.example/doc"})
    assert outcome.sections["s"] == ()
    assert outcome.report.checks[0].reason == "evidence URL is not among retrieved documents"


def test_misattributed_citation_is_not_supported(settings: AppSettings) -> None:
    revenue = make_chunk(
        "rev", "Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.", doc_id="rev"
    )
    unrelated = make_chunk("hr", "Acme opened a new cafeteria at its Ohio campus.", doc_id="hr")
    registry = registry_for([revenue, unrelated])
    claim = BriefStatement(
        text="Acme Corp reported annual revenue of $12.4 billion for fiscal 2025.",
        provenance=ProvenanceKind.VERIFIED_FACT,
        evidence_ids=("E2",),
    )
    outcome = CitationValidator(None, settings.guardrails).validate({"s": [claim]}, registry)
    assert outcome.sections["s"] == ()
    correct = claim.model_copy(update={"evidence_ids": ("E1", "E2")})
    kept = CitationValidator(None, settings.guardrails).validate({"s": [correct]}, registry).sections["s"]
    assert kept[0].evidence_ids == ("E1",)
