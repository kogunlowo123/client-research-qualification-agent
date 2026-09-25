"""Full orchestrator over local adapters with the offline EDGAR + corporate-site world."""

from __future__ import annotations

from pathlib import Path

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import DocumentType, FitVerdict, ProvenanceKind
from client_research_agent.orchestration import ClientResearchOrchestrator, StepName, StepStatus
from tests.support.world import ANALYST, COMPANY, local_runtime, northwind_request

pytestmark = pytest.mark.integration


def test_edgar_and_corporate_evidence_flow_into_a_cited_brief(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    result = ClientResearchOrchestrator(runtime).run(northwind_request(), principal=ANALYST)

    assert result.state.status is StepStatus.OK
    documents = {d.document_type for d in runtime.document_store.list_documents(COMPANY)}  # type: ignore[attr-defined]
    assert DocumentType.SEC_FILING in documents
    sources = {e.url for e in result.brief.evidence}
    assert any("sec.gov" in url for url in sources)
    assert any("northwind.example" in url for url in sources)
    assert result.verdict in (FitVerdict.GOOD_FIT, FitVerdict.POTENTIAL_FIT)
    for fact in result.brief.verified_facts:
        assert fact.provenance is ProvenanceKind.VERIFIED_FACT
        assert fact.evidence_ids
        assert set(fact.evidence_ids) <= {e.evidence_id for e in result.brief.evidence}
    assert result.brief.citation_report.coverage >= 0.9
    assert all(url in result.markdown for url in sources)


def test_second_run_reuses_evidence_and_keeps_audit_chain(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    orchestrator = ClientResearchOrchestrator(runtime)
    first = orchestrator.run(northwind_request(), principal=ANALYST)
    second = orchestrator.run(northwind_request(), principal=ANALYST)

    gathering = second.state.step(StepName.EVIDENCE_GATHERING)
    assert gathering is not None
    assert gathering.detail["documents_ingested"] == 0
    assert gathering.detail["sources_skipped"].get("already_ingested", 0) > 0
    assert gathering.status is StepStatus.OK
    assert second.verdict is first.verdict
    assert second.brief.qualification.weighted_score == first.brief.qualification.weighted_score
    assert runtime.audit.verify().valid
    assert sorted(runtime.brief_repository.list_run_ids()) == sorted([first.run_id, second.run_id])  # type: ignore[attr-defined]
