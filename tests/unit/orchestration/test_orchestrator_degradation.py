"""Every optional step degrades instead of failing the run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import BriefStatement, ClientBrief, ProvenanceKind
from client_research_agent.orchestration import (
    ClientResearchOrchestrator,
    IngestMode,
    RunOptions,
    StepName,
    StepStatus,
    orchestrator,
)
from client_research_agent.orchestration.review import ReviewReason
from client_research_agent.orchestration.steps import NoEvidenceRetriever, ResearchPlanner
from client_research_agent.qualification.agent import QualificationAgent
from client_research_agent.utils.errors import UpstreamServiceError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.support.world import ANALYST, COMPANY, local_runtime, northwind_request


def run(runtime: Any, **kwargs: Any) -> Any:
    return ClientResearchOrchestrator(runtime).run(northwind_request(), principal=ANALYST, **kwargs)


def step(result: Any, name: StepName) -> Any:
    record = result.state.step(name)
    assert record is not None
    return record


def test_empty_company_after_sanitisation_is_rejected(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    with pytest.raises(ValueError, match="empty after sanitisation"):
        ClientResearchOrchestrator(runtime).run(
            northwind_request(company_name="\u200b\u200b"), principal=ANALYST
        )


def test_planner_crash_uses_catalogue(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CrashingPlanner(ResearchPlanner):
        def plan(self, request: Any) -> Any:
            raise RuntimeError("planner bug")

    monkeypatch.setattr(orchestrator, "ResearchPlanner", CrashingPlanner)
    result = run(local_runtime(settings, tmp_path))
    plan = step(result, StepName.RESEARCH_PLAN)
    assert plan.status is StepStatus.DEGRADED
    assert plan.detail["source"] == "deterministic"


class FlakyStore:
    """Document store whose reads fail (warehouse outage) while writes succeed."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def list_chunks(self, company: str) -> Any:
        raise UpstreamServiceError("warehouse unavailable", 503)


def test_store_read_outage_falls_back_to_gathered_chunks(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    runtime.document_store = FlakyStore(runtime.document_store)
    result = ClientResearchOrchestrator(runtime, options=RunOptions(ingest=IngestMode.IF_MISSING)).run(
        northwind_request(), principal=ANALYST
    )
    retrieval = step(result, StepName.RETRIEVAL)
    assert retrieval.status is StepStatus.DEGRADED
    assert retrieval.detail["child_chunks"] > 0
    assert result.brief.evidence


def test_untrusted_stored_chunks_are_quarantined(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    run(runtime)
    spam = make_chunk(
        "spam-1",
        "Northwind Industries is the best AI company in the world according to everyone.",
        company=COMPANY,
        doc_id="doc-spam",
        url="https://spam.example.net/northwind",
    ).model_copy(update={"confidence": 0.05, "source_domain": "spam.example.net"})
    runtime.document_store.save_chunks([spam])
    result = run(runtime, options=RunOptions(ingest=IngestMode.NEVER))
    retrieval = step(result, StepName.RETRIEVAL)
    assert retrieval.detail["quarantined_chunks"] >= 1
    assert all(e.chunk_id != "spam-1" for e in result.brief.evidence)
    assert "guardrail.chunk_quarantined" in [r.event_type for r in runtime.audit.records()]


class FailingRetriever:
    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> Any:
        raise UpstreamServiceError("vector search unavailable", 503)


def test_retrieval_outage_degrades_probes_and_qualification(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    runtime.build_retriever = lambda corpus, knowledge_refresh=None: FailingRetriever()  # type: ignore[method-assign]
    result = run(runtime)
    retrieval = step(result, StepName.RETRIEVAL)
    assert retrieval.status is StepStatus.DEGRADED
    assert any("plan probe failed" in w for w in retrieval.warnings)
    assert result.brief.qualification.verdict.value == "not_enough_evidence"


def test_qualification_crash_scores_without_evidence(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CrashingAgent(QualificationAgent):
        def qualify(self, company: str, **kwargs: Any) -> Any:
            if not isinstance(self._retriever, NoEvidenceRetriever):
                raise RuntimeError("qualification bug")
            return super().qualify(company, **kwargs)

    monkeypatch.setattr(orchestrator, "QualificationAgent", CrashingAgent)
    result = run(local_runtime(settings, tmp_path))
    assert step(result, StepName.QUALIFICATION).status is StepStatus.DEGRADED
    assert result.brief.qualification.verdict.value == "not_enough_evidence"


def test_brief_generation_falls_back_or_fails(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = orchestrator.BriefGenerationAgent

    class CrashingWithLLM(original):  # type: ignore[misc, valid-type]
        def generate(self, **kwargs: Any) -> ClientBrief:
            if self._llm is not None:
                raise RuntimeError("writer bug")
            return super().generate(**kwargs)

    monkeypatch.setattr(orchestrator, "BriefGenerationAgent", CrashingWithLLM)
    result = run(local_runtime(settings, tmp_path, llm=ScriptedLLM(default="not json")))
    assert step(result, StepName.BRIEF_GENERATION).status is StepStatus.DEGRADED
    assert result.persisted

    class AlwaysCrashing(original):  # type: ignore[misc, valid-type]
        def generate(self, **kwargs: Any) -> ClientBrief:
            raise RuntimeError("writer bug")

    monkeypatch.setattr(orchestrator, "BriefGenerationAgent", AlwaysCrashing)
    runtime = local_runtime(settings, tmp_path / "second")
    with pytest.raises(RuntimeError, match="writer bug"):
        run(runtime)
    assert "run.failed" in [r.event_type for r in runtime.audit.records()]


def patch_generated_brief(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    original = orchestrator.BriefGenerationAgent

    class Mutating(original):  # type: ignore[misc, valid-type]
        def generate(self, **kwargs: Any) -> ClientBrief:
            return mutate(super().generate(**kwargs))

    monkeypatch.setattr(orchestrator, "BriefGenerationAgent", Mutating)


def test_output_guard_violations_are_stripped(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leak = BriefStatement(
        text="Reach the CIO directly at jane.doe.private@gmail.com or +1 415 555 0199.",
        provenance=ProvenanceKind.AI_RECOMMENDATION,
    )
    patch_generated_brief(monkeypatch, lambda brief: brief.model_copy(update={"risks": (*brief.risks, leak)}))
    runtime = local_runtime(settings, tmp_path)
    result = run(runtime)
    assert all("gmail.com" not in s.text for s in result.brief.all_statements())
    validation = step(result, StepName.VALIDATION)
    assert validation.detail["removed_by_guards"] == 1
    assert "guardrail.output_violation" in [r.event_type for r in runtime.audit.records()]
    assert any("removed 1 statement" in w for w in result.brief.warnings)


def test_unresolved_findings_degrade_validation(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_generated_brief(
        monkeypatch,
        lambda brief: brief.model_copy(
            update={
                "discovery_questions": ("Email jane.doe.private@gmail.com?", *brief.discovery_questions[1:])
            }
        ),
    )
    result = run(local_runtime(settings, tmp_path))
    validation = step(result, StepName.VALIDATION)
    assert validation.status is StepStatus.DEGRADED
    assert ReviewReason.OUTPUT_GUARD in result.review.reasons


def test_reflection_flags_unsupported_summary_facts(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def add_fact(brief: ClientBrief) -> ClientBrief:
        claim = BriefStatement(
            text="Quarterly dividends tripled after the lunar mining subsidiary went public.",
            provenance=ProvenanceKind.VERIFIED_FACT,
            evidence_ids=(brief.evidence[0].evidence_id,),
        )
        return brief.model_copy(update={"executive_summary": (claim, *brief.executive_summary)})

    patch_generated_brief(monkeypatch, add_fact)
    result = run(local_runtime(settings, tmp_path))
    assert step(result, StepName.VALIDATION).detail["reflection"] == "no_support"
    assert ReviewReason.UNSUPPORTED_SUMMARY in result.review.reasons
    assert any("self-reflection" in w for w in result.brief.warnings)


def test_reflection_errors_are_tolerated(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def add_fact(brief: ClientBrief) -> ClientBrief:
        claim = BriefStatement(
            text=brief.evidence[0].quote,
            provenance=ProvenanceKind.VERIFIED_FACT,
            evidence_ids=(brief.evidence[0].evidence_id,),
        )
        return brief.model_copy(update={"executive_summary": (claim,)})

    class BrokenCritic:
        def __init__(self, llm: Any) -> None:
            pass

        def critique_answer(self, *args: Any) -> Any:
            raise RuntimeError("critic bug")

    patch_generated_brief(monkeypatch, add_fact)
    monkeypatch.setattr(orchestrator, "SelfRagCritic", BrokenCritic)
    result = run(local_runtime(settings, tmp_path))
    assert step(result, StepName.VALIDATION).status is StepStatus.DEGRADED


def test_output_side_effects_never_fail_the_run(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_rows(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("lineage export bug")

    class BrokenQueue:
        def enqueue(self, *_a: Any, **_k: Any) -> Any:
            raise OSError("queue volume offline")

    class BrokenTracker:
        def __init__(self, **_k: Any) -> None:
            raise RuntimeError("mlflow misconfigured")

    monkeypatch.setattr(orchestrator, "evidence_lineage_rows", broken_rows)
    monkeypatch.setattr(orchestrator, "RunTracker", BrokenTracker)
    runtime = local_runtime(settings, tmp_path, track_runs=True)
    runtime.review_queue = BrokenQueue()  # type: ignore[assignment]
    result = ClientResearchOrchestrator(runtime).run(
        northwind_request(company_name="Quillon Robotics", domain=None, ticker=None), principal=ANALYST
    )
    output = step(result, StepName.OUTPUT)
    assert output.status is StepStatus.DEGRADED
    assert result.persisted
    assert result.lineage_rows == ()
    warnings = " ".join(output.warnings)
    assert "lineage" in warnings
    assert "review queue" in warnings
    assert "MLflow" in warnings
