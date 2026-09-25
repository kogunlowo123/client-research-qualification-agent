from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from client_research_agent.config.settings import build_settings
from client_research_agent.governance.responsible_ai import PolicyFinding, PolicyReport, Severity
from client_research_agent.models import (
    BriefStatement,
    CitationReport,
    ClientBrief,
    Criterion,
    CriterionScore,
    DocumentType,
    Evidence,
    FitVerdict,
    ProvenanceKind,
    QualificationResult,
    ResearchRequest,
)
from client_research_agent.orchestration.rendering import (
    NO_SOURCES,
    render_provenance,
    render_review,
    render_run_markdown,
    render_sources,
)
from client_research_agent.orchestration.review import (
    AnalystFeedback,
    FeedbackStore,
    ReviewDecision,
    ReviewPolicy,
    ReviewPriority,
    ReviewQueue,
    ReviewReason,
    ReviewStatus,
    StatementCorrection,
    feedback_to_example,
)
from client_research_agent.orchestration.state import RunState, StepName, StepRecord, StepStatus
from client_research_agent.retrieval.self_rag import AnswerCritique, SupportLevel
from client_research_agent.scoring.engine import ScoringEngine
from client_research_agent.security.output_guard import OutputReport, OutputViolation, ViolationKind


def make_brief(
    *,
    verdict: FitVerdict = FitVerdict.GOOD_FIT,
    total: int = 2,
    supported: int = 2,
    run_id: str = "run-1",
    evidence_url: str = "https://acme.example.com/news/1",
) -> ClientBrief:
    evidence = Evidence(
        evidence_id="E1",
        chunk_id="c1",
        url=evidence_url,
        title="Acme | news [1]",
        quote="Acme Corp employs 12,000 people across 30 countries.",
        document_type=DocumentType.PRESS_RELEASE,
    )
    fact = BriefStatement(
        text="Acme Corp employs 12,000 people across 30 countries.",
        provenance=ProvenanceKind.VERIFIED_FACT,
        evidence_ids=("E1",),
    )
    rec = BriefStatement(
        text="Propose a data platform assessment.", provenance=ProvenanceKind.AI_RECOMMENDATION
    )
    scores = tuple(
        CriterionScore(criterion=c, score=4, weight=0.2, confidence=0.8, rationale="r", evidence_ids=("E1",))
        for c in Criterion
    )
    return ClientBrief(
        run_id=run_id,
        company="Acme Corp",
        company_overview=(fact,),
        qualification=QualificationResult(
            scores=scores, weighted_score=4.0, overall_confidence=0.8, verdict=verdict, verdict_rationale="ok"
        ),
        evidence=(evidence,),
        technology_priorities=(rec,),
        gartner_relevant_insights=(),
        opportunities=(rec,),
        risks=(),
        executive_summary=(fact, rec),
        discovery_questions=tuple(f"Question {i}?" for i in range(5)),
        executive_talking_points=(rec,),
        recommended_next_actions=(rec,),
        citation_report=CitationReport(total_statements=total, supported_statements=supported),
    )


# ------------------------------------------------------------------ policy
def test_clean_brief_needs_no_review() -> None:
    decision = ReviewPolicy().decide(make_brief())
    assert decision == ReviewDecision(False, (), (), ReviewPriority.NONE)
    assert decision.to_dict()["priority"] == "none"


def test_every_review_reason() -> None:
    brief = make_brief(verdict=FitVerdict.NOT_ENOUGH_EVIDENCE, total=4, supported=1)
    sensitivity = ScoringEngine(build_settings("local").scoring).sensitivity(brief.qualification.scores)
    output = OutputReport((OutputViolation(ViolationKind.PII, "email", "risks[0]"),))
    policy = PolicyReport(
        findings=(PolicyFinding("protected_attributes", Severity.ERROR, "m", "risks[0]"),),
        disclaimer="d",
        verified_facts=1,
        recommendations=1,
    )
    critique = AnswerCritique(SupportLevel.NO, 1, ("x",), 0.0, "0/1 grounded", "heuristic")
    decision = ReviewPolicy(min_citation_coverage=0.8).decide(
        brief,
        sensitivity=sensitivity,
        output_report=output,
        policy_report=policy,
        degraded_steps=["retrieval"],
        reflection=critique,
    )
    assert ReviewReason.NOT_ENOUGH_EVIDENCE in decision.reasons
    assert ReviewReason.LOW_CITATION_COVERAGE in decision.reasons
    assert ReviewReason.OUTPUT_GUARD in decision.reasons
    assert ReviewReason.POLICY_FINDING in decision.reasons
    assert ReviewReason.DEGRADED_RUN in decision.reasons
    assert ReviewReason.UNSUPPORTED_SUMMARY in decision.reasons
    assert decision.priority is ReviewPriority.HIGH


@pytest.mark.parametrize(
    ("kwargs", "priority"),
    [
        ({"degraded_steps": ["x"]}, ReviewPriority.LOW),
        (
            {"reflection": AnswerCritique(SupportLevel.NO, 1, (), 0.0, "r", "heuristic")},
            ReviewPriority.MEDIUM,
        ),
        (
            {"policy_report": PolicyReport((PolicyFinding("r", Severity.WARNING, "m", "l"),), "d", 0, 0)},
            ReviewPriority.HIGH,
        ),
    ],
)
def test_priorities(kwargs: dict[str, object], priority: ReviewPriority) -> None:
    assert ReviewPolicy().decide(make_brief(), **kwargs).priority is priority  # type: ignore[arg-type]


def test_policy_validates_threshold() -> None:
    with pytest.raises(ValueError, match="min_citation_coverage"):
        ReviewPolicy(min_citation_coverage=1.5)


# ------------------------------------------------------------------ queue
def test_review_queue_lifecycle(tmp_path: Path) -> None:
    queue = ReviewQueue(tmp_path / "q.jsonl")
    decision = ReviewPolicy().decide(make_brief(verdict=FitVerdict.NOT_ENOUGH_EVIDENCE))
    request = ResearchRequest(
        company_name="Acme Corp", domain="acme.example.com", ticker="acme", requested_by="ana"
    )
    item = queue.enqueue(make_brief(verdict=FitVerdict.NOT_ENOUGH_EVIDENCE), decision, request=request)
    high = ReviewDecision(True, (ReviewReason.OUTPUT_GUARD,), ("x",), ReviewPriority.HIGH)
    queue.enqueue(make_brief(run_id="run-2"), high)
    assert item.ticker == "ACME"
    assert item.requested_by == "ana"
    assert [i.run_id for i in queue.pending()] == ["run-2", "run-1"]
    resolved = queue.resolve("run-1", reviewer="lead", approved=False, notes="needs more sources")
    assert resolved.status is ReviewStatus.REJECTED
    assert resolved.resolved_at is not None
    assert queue.get("run-1") == resolved
    assert [i.run_id for i in queue.pending()] == ["run-2"]
    with pytest.raises(ValueError, match="already rejected"):
        queue.resolve("run-1", reviewer="lead", approved=True)
    with pytest.raises(KeyError):
        queue.resolve("missing", reviewer="lead", approved=True)
    with pytest.raises(ValueError, match="reviewer"):
        queue.resolve("run-2", reviewer=" ", approved=True)
    with pytest.raises(ValueError, match="does not require"):
        queue.enqueue(make_brief(), ReviewDecision(False, (), (), ReviewPriority.NONE))
    assert queue.path.exists()
    assert ReviewQueue(tmp_path / "none.jsonl").items() == []


# ------------------------------------------------------------------ feedback loop
def test_feedback_exports_eval_examples(tmp_path: Path) -> None:
    store = FeedbackStore(tmp_path / "fb.jsonl")
    queue = ReviewQueue(tmp_path / "q.jsonl")
    brief = make_brief()
    queue.enqueue(
        brief,
        ReviewDecision(True, (ReviewReason.DEGRADED_RUN,), (), ReviewPriority.LOW),
        request=ResearchRequest(company_name="Acme Corp", domain="acme.example.com"),
    )
    store.record(AnalystFeedback(run_id="run-1", reviewer="lead", rating=2))
    store.record(
        AnalystFeedback(
            run_id="run-1",
            reviewer="lead",
            rating=5,
            verdict_override=FitVerdict.POTENTIAL_FIT,
            corrections=(StatementCorrection(original="Old claim", corrected="Acme Corp has 12,000 staff."),),
        )
    )
    store.record(AnalystFeedback(run_id="run-unknown", reviewer="lead", rating=5))
    store.record(AnalystFeedback(run_id="run-empty", reviewer="lead"))
    briefs = {"run-1": brief, "run-empty": make_brief(run_id="run-empty")}
    examples = store.export_eval_examples(briefs.get, review_queue=queue)
    assert len(examples) == 1
    example = examples[0]
    assert example.expectations.expected_verdict is FitVerdict.POTENTIAL_FIT
    assert example.expectations.expected_facts[0] == "Acme Corp has 12,000 staff."
    assert example.inputs.domain == "acme.example.com"
    assert example.tags["rating"] == "5"
    assert len(store.entries("run-1")) == 2
    assert store.path.exists()


def test_feedback_without_signal_yields_no_example() -> None:
    assert feedback_to_example(AnalystFeedback(run_id="r", reviewer="x", rating=3), make_brief()) is None
    trusted = feedback_to_example(AnalystFeedback(run_id="r", reviewer="x", rating=4), make_brief())
    assert trusted is not None
    assert trusted.expectations.expected_verdict is FitVerdict.GOOD_FIT


# ------------------------------------------------------------------ state
def test_step_record_transitions() -> None:
    record = StepRecord(name=StepName.RETRIEVAL)
    record.note("info")
    assert record.status is StepStatus.OK
    record.degrade("partial")
    record.skip("nothing to do")
    record.degrade("after skip")
    assert record.status is StepStatus.SKIPPED
    record.fail(RuntimeError("x" * 400))
    assert record.status is StepStatus.FAILED
    assert record.error is not None
    assert len(record.error) == 300
    payload = record.to_dict()
    assert payload["warnings"] == ["info", "partial", "after skip"]


def test_run_state_aggregates() -> None:
    state = RunState(run_id="r", request=ResearchRequest(company_name="Acme Corp"), principal="p")
    ok = StepRecord(name=StepName.COMPANY_INPUT, duration_ms=5.0)
    degraded = StepRecord(name=StepName.RETRIEVAL, duration_ms=7.0)
    degraded.degrade("w")
    state.steps.extend([ok, degraded])
    assert state.status is StepStatus.DEGRADED
    assert state.degraded_steps == ["retrieval"]
    assert state.duration_ms == 12.0
    assert state.warnings == ["w"]
    assert state.step(StepName.OUTPUT) is None
    failed = StepRecord(name=StepName.OUTPUT)
    failed.fail(ValueError("boom"))
    state.steps.append(failed)
    state.finished_at = datetime.now(UTC)
    assert state.status is StepStatus.FAILED
    assert json.loads(json.dumps(state.to_dict()))["status"] == "failed"


# ------------------------------------------------------------------ rendering
def test_render_run_markdown_sections() -> None:
    brief = make_brief()
    decision = ReviewDecision(
        True, (ReviewReason.DEGRADED_RUN,), ("degraded steps: retrieval",), ReviewPriority.LOW
    )
    sensitivity = ScoringEngine(build_settings("local").scoring).sensitivity(brief.qualification.scores)
    markdown = render_run_markdown(
        brief, review=decision, sensitivity=sensitivity, disclaimer="Use with care."
    )
    for heading in (
        "## Confidence",
        "## Sources and Citation Links",
        "## Verified Facts",
        "## AI-Generated Recommendations",
        "## Review Status",
        "## Responsible AI Notice",
    ):
        assert heading in markdown
    assert "[Acme \\| news \\[1\\]](https://acme.example.com/news/1)" in markdown
    assert "**Analyst review required** (priority: low)." in markdown
    assert render_review(ReviewDecision(False, (), (), ReviewPriority.NONE))[2].startswith(
        "No analyst review"
    )


def test_render_sources_and_provenance_edge_cases() -> None:
    bad_url = make_brief(evidence_url="javascript:alert(1)")
    lines = render_sources(bad_url)
    assert "javascript" not in "\n".join(lines).split("(")[0] or "](" not in lines[2]
    empty = bad_url.model_copy(update={"evidence": ()})
    assert NO_SOURCES in render_sources(empty)
    no_statements = empty.model_copy(
        update={
            "company_overview": (),
            "executive_summary": (),
            "technology_priorities": (),
            "opportunities": (),
            "executive_talking_points": (),
            "recommended_next_actions": (),
        }
    )
    rendered = "\n".join(render_provenance(no_statements))
    assert "No statement met the verified-fact bar" in rendered
    assert "No recommendations were generated" in rendered
    citation = "\n".join(render_provenance(bad_url))
    assert "\\[E1\\]" in citation
