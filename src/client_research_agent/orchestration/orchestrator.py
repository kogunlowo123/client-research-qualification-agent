"""End-to-end research run: company in, cited and scored client brief out.

Flow (each step is a traced span, emits metrics and an audit event, and is
recorded in :class:`~client_research_agent.orchestration.state.RunState`)::

    Company Input        validate + sanitise the request, RBAC (RUN_RESEARCH), per-run budget
    Research Agent       focus questions, targeted queries, source priorities
    Evidence Gathering   ingest -> sanitise -> injection/poisoning screen -> PII redaction
                         -> persist -> index -> lineage
    Retrieval Pipeline   corpus from the document store, chunk poisoning guard, guarded
                         retriever with bounded CRAG knowledge refresh, plan probes
    Qualification Agent  five criteria, LLM + deterministic cross-check
    Scoring Engine       weighted score, verdict policy, sensitivity
    Brief Generation     opportunity analysis + narrative + discovery questions
                         (citation validation runs inside the generator)
    Validation           output guard + responsible-AI policy; flagged statements removed
    Output Client Brief  persist, render Markdown, lineage, review queue, MLflow, costs

Graceful degradation: optional steps (planning, ingestion, retrieval probes,
lineage, review queue, MLflow) record a warning and the run continues. With no
evidence at all the run still produces a NOT_ENOUGH_EVIDENCE brief. Only an
invalid or unauthorised request, or a brief that cannot be generated even
deterministically, fails the run.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from client_research_agent.agent.metering import RunAccounting, run_accounting
from client_research_agent.briefing.generator import BriefGenerationAgent
from client_research_agent.governance.lineage import LineageRecorder, statement_id
from client_research_agent.governance.responsible_ai import PolicyReport
from client_research_agent.models import (
    ChunkStrategy,
    ClientBrief,
    FitVerdict,
    ProvenanceKind,
    ResearchRequest,
)
from client_research_agent.models.domain import utc_now
from client_research_agent.observability.logging import get_logger, log_context
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.mlflow_tracking import RunTracker
from client_research_agent.observability.tracing import SpanType, span
from client_research_agent.orchestration.rendering import render_run_markdown
from client_research_agent.orchestration.review import ReviewDecision
from client_research_agent.orchestration.state import RunState, StepName, StepRecord, StepStatus
from client_research_agent.orchestration.steps import (
    ChunkBlocklist,
    EvidenceGatherer,
    GatheringResult,
    GuardedRetriever,
    KnowledgeRefresher,
    NoEvidenceRetriever,
    ResearchPlan,
    ResearchPlanner,
    strip_statements,
)
from client_research_agent.qualification.agent import QualificationAgent, QualificationOutput
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.retrieval.self_rag import AnswerCritique, SelfRagCritic, SupportLevel
from client_research_agent.scoring.engine import ScoreBreakdown, SensitivityReport
from client_research_agent.security.output_guard import OutputReport, ViolationKind
from client_research_agent.security.rbac import Permission, Principal, authorize
from client_research_agent.services.local import validate_run_id
from client_research_agent.utils.errors import AgentError, SecurityViolationError

if TYPE_CHECKING:
    from client_research_agent.agent.factory import AgentRuntime

_log = get_logger(__name__)
_SLUG = re.compile(r"[^a-z0-9]+")
#: Output-guard findings whose statement is removed rather than left for review.
REMOVABLE_VIOLATIONS = frozenset(
    {
        ViolationKind.PII,
        ViolationKind.SECRET,
        ViolationKind.PROMPT_LEAK,
        ViolationKind.UNSAFE_URL,
        ViolationKind.UNCITED_URL,
        ViolationKind.CODE_EXECUTION,
    }
)


class IngestMode(StrEnum):
    ALWAYS = "always"
    IF_MISSING = "if_missing"
    NEVER = "never"


@dataclass(frozen=True, slots=True)
class RunOptions:
    ingest: IngestMode = IngestMode.ALWAYS
    plan_probe_queries: int = 3


@dataclass(frozen=True)
class RunResult:
    brief: ClientBrief
    markdown: str
    state: RunState
    review: ReviewDecision
    lineage_rows: tuple[dict[str, Any], ...] = ()

    @property
    def run_id(self) -> str:
        return self.brief.run_id

    @property
    def verdict(self) -> FitVerdict:
        return self.brief.qualification.verdict

    @property
    def persisted(self) -> bool:
        step = self.state.step(StepName.OUTPUT)
        return bool(step is not None and step.detail.get("persisted"))


@dataclass
class _Run:
    """Mutable intermediate artefacts of one run."""

    request: ResearchRequest
    principal: Principal
    state: RunState
    lineage: LineageRecorder
    options: RunOptions
    accounting: RunAccounting | None = None
    plan: ResearchPlan | None = None
    gathering: GatheringResult | None = None
    gathering_failed: bool = False
    blocklist: ChunkBlocklist = field(default_factory=ChunkBlocklist)
    retriever: GuardedRetriever | None = None
    refresher: KnowledgeRefresher | None = None
    qualification: QualificationOutput | None = None
    breakdown: ScoreBreakdown | None = None
    sensitivity: SensitivityReport | None = None
    brief: ClientBrief | None = None
    output_report: OutputReport | None = None
    policy_report: PolicyReport | None = None
    reflection: AnswerCritique | None = None

    @property
    def company(self) -> str:
        return self.request.company_name


def new_run_id(company: str, *, now: datetime | None = None) -> str:
    slug = _SLUG.sub("-", company.lower()).strip("-")[:40].strip("-") or "run"
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return validate_run_id(f"{slug}-{stamp}-{uuid.uuid4().hex[:8]}")


def evidence_lineage_rows(
    brief: ClientBrief, registry: EvidenceRegistry, *, created_at: datetime | None = None
) -> list[dict[str, Any]]:
    """Unity Catalog ``lineage`` rows: every cited statement -> evidence -> chunk -> document."""
    created = created_at or utc_now()
    support: dict[str, tuple[bool, float]] = {}
    for check in brief.citation_report.checks:
        previous = support.get(check.evidence_id)
        if previous is None or check.support_score > previous[1]:
            support[check.evidence_id] = (check.supported, check.support_score)
    evidence = {e.evidence_id: e for e in brief.evidence}
    sections = {
        "company_overview": brief.company_overview,
        "technology_priorities": brief.technology_priorities,
        "gartner_relevant_insights": brief.gartner_relevant_insights,
        "opportunities": brief.opportunities,
        "risks": brief.risks,
        "executive_summary": brief.executive_summary,
        "executive_talking_points": brief.executive_talking_points,
        "recommended_next_actions": brief.recommended_next_actions,
    }
    rows: list[dict[str, Any]] = []
    for section, statements in sections.items():
        for statement in statements:
            sid = statement_id(section, statement.text)
            for evidence_id in statement.evidence_ids:
                chunk = registry.chunk(evidence_id)
                item = evidence.get(evidence_id)
                if chunk is None or item is None:
                    continue
                checked = support.get(evidence_id)
                key = f"{brief.run_id}|{sid}|{evidence_id}".encode()
                rows.append(
                    {
                        "lineage_id": hashlib.sha256(key).hexdigest()[:32],
                        "run_id": brief.run_id,
                        "evidence_id": evidence_id,
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "url": item.url,
                        "content_hash": str(chunk.metadata.get("content_hash", "")),
                        "statement_provenance": statement.provenance.value,
                        "supported": checked[0] if checked else None,
                        "support_score": checked[1] if checked else None,
                        "created_at": created,
                    }
                )
    return rows


class ClientResearchOrchestrator:
    def __init__(self, runtime: AgentRuntime, *, options: RunOptions | None = None) -> None:
        self._rt = runtime
        self._options = options or RunOptions()
        self._gatherer = EvidenceGatherer(runtime)

    @property
    def runtime(self) -> AgentRuntime:
        return self._rt

    # ------------------------------------------------------------------ public API
    def run(
        self,
        request: ResearchRequest,
        *,
        principal: Principal,
        run_id: str | None = None,
        options: RunOptions | None = None,
    ) -> RunResult:
        resolved_id = validate_run_id(run_id) if run_id else new_run_id(request.company_name)
        state = RunState(run_id=resolved_id, request=request, principal=principal.id)
        run = _Run(
            request=request,
            principal=principal,
            state=state,
            lineage=LineageRecorder(resolved_id),
            options=options or self._options,
        )
        metrics = get_metrics()
        with (
            log_context(run_id=resolved_id, company=request.company_name, principal=principal.id),
            span("orchestrator.run", SpanType.AGENT, run_id=resolved_id, company=request.company_name),
        ):
            self._audit(
                run, "run.started", {"company": request.company_name, "requested_by": request.requested_by}
            )
            try:
                result = self._execute(run)
            except Exception as exc:
                state.finished_at = utc_now()
                metrics.increment("orchestrator.runs", status=StepStatus.FAILED.value)
                self._flush_audit()
                self._audit(
                    run,
                    "run.failed",
                    {"error": type(exc).__name__, "steps": [s.name.value for s in state.steps]},
                )
                raise
        state.finished_at = utc_now()
        self._flush_audit()
        metrics.increment("orchestrator.runs", status=state.status.value, verdict=result.verdict.value)
        metrics.observe("orchestrator.run.duration_ms", state.duration_ms)
        self._audit(
            run,
            "run.completed",
            {
                "status": state.status.value,
                "verdict": result.verdict.value,
                "duration_ms": round(state.duration_ms, 1),
            },
        )
        return result

    # ------------------------------------------------------------------ flow
    def _execute(self, run: _Run) -> RunResult:
        self._company_input(run)
        if run.accounting is None:
            raise AgentError("internal invariant violated: run.accounting is unset")
        with run_accounting(run.accounting):
            self._research_plan(run)
            self._evidence_gathering(run)
            self._retrieval(run)
            self._qualification(run)
            self._scoring(run)
            self._brief_generation(run)
            self._validation(run)
            return self._output(run)

    @contextmanager
    def _step(self, run: _Run, name: StepName, span_type: str) -> Iterator[StepRecord]:
        record = StepRecord(name=name)
        run.state.steps.append(record)
        if run.accounting is not None:
            run.accounting.step = name.value
        started = time.perf_counter()
        try:
            with (
                span(f"orchestrator.{name.value}", span_type, run_id=run.state.run_id),
                log_context(step=name.value),
            ):
                yield record
        except Exception as exc:
            record.fail(exc)
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000
            metrics = get_metrics()
            metrics.observe("orchestrator.step.duration_ms", record.duration_ms, step=name.value)
            metrics.increment("orchestrator.steps", step=name.value, status=record.status.value)
            self._audit(
                run,
                f"step.{name.value}",
                {
                    "status": record.status.value,
                    "duration_ms": round(record.duration_ms, 1),
                    "warnings": len(record.warnings),
                    "error": record.error,
                },
            )

    def _company_input(self, run: _Run) -> None:
        rt = self._rt
        with self._step(run, StepName.COMPANY_INPUT, SpanType.TOOL) as record:
            authorize(run.principal, Permission.RUN_RESEARCH, audit=rt.audit)
            request = run.request
            company = rt.input_sanitizer.sanitize_untrusted(request.company_name).text
            if not company:
                raise ValueError("company_name is empty after sanitisation")
            for field_name, value in (("company_name", company), ("industry", request.industry)):
                if value and rt.injection_detector.assess(value).blocked:
                    self._audit(
                        run, "guardrail.injection_blocked", {"stage": "company_input", "field": field_name}
                    )
                    raise SecurityViolationError(
                        f"request field {field_name!r} rejected: prompt injection detected"
                    )
            if company != request.company_name:
                run.request = request.model_copy(update={"company_name": company})
                run.state.request = run.request
                record.note(
                    "company input: the company name was normalised (control or invisible characters removed)"
                )
            run.accounting = RunAccounting(budget=rt.budget_factory())
            snapshot = run.accounting.budget.snapshot()
            record.detail.update(
                {
                    "company": company,
                    "domain": request.domain,
                    "ticker": request.ticker,
                    "max_documents": request.max_documents,
                    "budget_max_tokens": snapshot.max_tokens,
                    "budget_max_llm_calls": snapshot.max_llm_calls,
                }
            )

    def _research_plan(self, run: _Run) -> None:
        with self._step(run, StepName.RESEARCH_PLAN, SpanType.AGENT) as record:
            planner = ResearchPlanner(self._rt.llm, self._rt.prompts)
            try:
                plan = planner.plan(run.request)
            except Exception as exc:  # the plan is advisory; never fail a run over it
                record.degrade(
                    f"research plan: planner error ({type(exc).__name__}); used criteria catalogue"
                )
                plan = planner.deterministic(run.request)
            for warning in plan.warnings:
                record.degrade(warning)
            run.plan = plan
            record.detail.update(
                {
                    "source": plan.source,
                    "queries": len(plan.queries),
                    "focus_questions": len(plan.focus_questions),
                    "source_priorities": [p.value for p in plan.source_priorities],
                    "prompt_version": planner.prompt_version,
                }
            )

    def _stored_chunks(self, company: str) -> int | None:
        try:
            return len(self._rt.document_store.list_chunks(company))
        except Exception as exc:
            _log.warning("document_store.list_chunks_failed", error=type(exc).__name__)
            return None

    def _evidence_gathering(self, run: _Run) -> None:
        with self._step(run, StepName.EVIDENCE_GATHERING, SpanType.CHAIN) as record:
            mode = run.options.ingest
            if mode is IngestMode.NEVER:
                record.skip("ingestion disabled for this run")
                return
            if mode is IngestMode.IF_MISSING:
                stored = self._stored_chunks(run.company)
                if stored:
                    record.skip("evidence already stored for this company")
                    record.detail["stored_chunks"] = stored
                    return
            try:
                result = self._gatherer.gather(run.request, lineage=run.lineage)
            except Exception as exc:  # ingestion or indexing outage: continue with stored evidence
                run.gathering_failed = True
                record.degrade(f"evidence gathering: live ingestion failed ({type(exc).__name__})")
                return
            run.gathering = result
            record.detail.update(result.summary())
            for warning in result.warnings:
                record.degrade(warning)
            injected = sum(1 for q in result.quarantined if "prompt_injection" in q.reasons)
            if injected:
                self._audit(
                    run, "guardrail.injection_blocked", {"stage": "evidence_gathering", "documents": injected}
                )
            if result.pii_redactions:
                self._audit(
                    run,
                    "guardrail.pii_redacted",
                    {"stage": "evidence_gathering", "redactions": result.pii_redactions},
                )
            if result.quarantined:
                self._audit(
                    run,
                    "evidence.quarantined",
                    {
                        "documents": [
                            {"doc_id": q.doc_id, "reasons": list(q.reasons)} for q in result.quarantined
                        ]
                    },
                )
            if not result.documents and not result.skipped.get("already_ingested"):
                run.gathering_failed = True
                record.degrade("evidence gathering: no public documents could be ingested")

    def _retrieval(self, run: _Run) -> None:
        rt = self._rt
        with self._step(run, StepName.RETRIEVAL, SpanType.RETRIEVER) as record:
            quarantined_docs = [q.doc_id for q in run.gathering.quarantined] if run.gathering else []
            run.blocklist = ChunkBlocklist(doc_ids=quarantined_docs)
            try:
                stored = rt.document_store.list_chunks(run.company)
            except Exception as exc:
                record.degrade(f"retrieval: document store unavailable ({type(exc).__name__})")
                stored = list(run.gathering.chunks) if run.gathering else []
            children = [c for c in stored if c.strategy is not ChunkStrategy.PARENT]
            record.detail.update({"stored_chunks": len(stored), "child_chunks": len(children)})
            if not children:
                run.retriever = GuardedRetriever(NoEvidenceRetriever(), run.blocklist)
                record.degrade(
                    f"retrieval: no public evidence is available for {run.company}; the brief reports "
                    "insufficient evidence"
                )
                return
            if run.gathering_failed:
                record.degrade(
                    f"retrieval: live ingestion unavailable; continuing with {len(children)} "
                    "previously stored chunks"
                )
            report = rt.poisoning_guard.evaluate(children, company_domain=run.request.domain)
            run.blocklist.add(chunk_ids=report.quarantined_ids)
            if report.quarantined_ids:
                self._audit(
                    run,
                    "guardrail.chunk_quarantined",
                    {
                        "stage": "retrieval",
                        "chunks": len(report.quarantined_ids),
                        "flagged": len(report.flagged_ids),
                    },
                )
                record.note(
                    f"retrieval: {len(report.quarantined_ids)} chunk(s) quarantined by the poisoning guard"
                )
            if report.flagged_ids:
                record.note(f"retrieval: {len(report.flagged_ids)} chunk(s) flagged as lower-trust evidence")
            corpus = [c for c in stored if not run.blocklist.blocks(c)]
            if rt.refresh_ingestion is not None and run.options.ingest is not IngestMode.NEVER:
                run.refresher = KnowledgeRefresher(
                    self._gatherer,
                    rt.refresh_ingestion,
                    run.request,
                    run.blocklist,
                    rt,
                    max_refreshes=rt.max_knowledge_refreshes,
                    lineage=run.lineage,
                )
            pipeline = rt.build_retriever(corpus, knowledge_refresh=run.refresher)
            run.retriever = GuardedRetriever(pipeline, run.blocklist)
            verdicts: dict[str, int] = {}
            relevance: list[float] = []
            probes = [q.query for q in (run.plan.queries if run.plan else ())][
                : run.options.plan_probe_queries
            ]
            for query in probes:
                try:
                    outcome = run.retriever.retrieve(query, company=run.company)
                except AgentError as exc:
                    record.degrade(f"retrieval: plan probe failed ({type(exc).__name__})")
                    continue
                verdicts[outcome.verdict.value] = verdicts.get(outcome.verdict.value, 0) + 1
                relevance.append(outcome.mean_relevance)
            record.detail.update(
                {
                    "corpus_chunks": len(corpus),
                    "quarantined_chunks": len(report.quarantined_ids),
                    "flagged_chunks": len(report.flagged_ids),
                    "probe_verdicts": verdicts,
                    "probe_mean_relevance": round(sum(relevance) / len(relevance), 4) if relevance else 0.0,
                    "knowledge_refresh_enabled": run.refresher is not None,
                }
            )

    def _qualification(self, run: _Run) -> None:
        rt = self._rt
        with self._step(run, StepName.QUALIFICATION, SpanType.AGENT) as record:
            retriever = run.retriever or GuardedRetriever(NoEvidenceRetriever(), run.blocklist)
            try:
                output = QualificationAgent(rt.llm, retriever, rt.prompts, rt.settings).qualify(run.company)
            except Exception as exc:
                record.degrade(f"qualification: agent error ({type(exc).__name__}); scored without evidence")
                output = QualificationAgent(None, NoEvidenceRetriever(), rt.prompts, rt.settings).qualify(
                    run.company
                )
            for warning in output.warnings:
                record.note(warning)
            run.qualification = output
            record.detail.update(
                {
                    "evidence_items": len(output.evidence),
                    "llm_criteria": sorted(c.value for c in output.llm_criteria),
                    "filtered_chunks": retriever.removed,
                    "knowledge_refreshes": run.refresher.refreshes_used if run.refresher else 0,
                    "refreshed_documents": run.refresher.documents_added if run.refresher else 0,
                }
            )

    def _scoring(self, run: _Run) -> None:
        with self._step(run, StepName.SCORING, SpanType.TOOL) as record:
            if run.qualification is None:
                raise AgentError("internal invariant violated: run.qualification is unset")
            breakdown = self._rt.scoring.breakdown(run.qualification.scores)
            run.breakdown = breakdown
            run.sensitivity = self._rt.scoring.sensitivity(breakdown.result)
            result = breakdown.result
            record.detail.update(
                {
                    "verdict": result.verdict.value,
                    "reason": breakdown.reason.value,
                    "weighted_score": result.weighted_score,
                    "overall_confidence": result.overall_confidence,
                    "without_evidence": [c.value for c in breakdown.without_evidence],
                    "verdict_robust": run.sensitivity.robust,
                }
            )

    def _brief_generation(self, run: _Run) -> None:
        rt = self._rt
        with self._step(run, StepName.BRIEF_GENERATION, SpanType.AGENT) as record:
            if run.qualification is None:
                raise AgentError("internal invariant violated: run.qualification is unset")
            if run.breakdown is None:
                raise AgentError("internal invariant violated: run.breakdown is unset")
            urls = set(run.qualification.evidence.urls())
            if run.retriever is not None:
                urls.update(run.retriever.retrieved_urls)
            versions = rt.model_versions()
            if run.plan is not None:
                versions["plan"] = run.plan.source
            kwargs: dict[str, Any] = {
                "run_id": run.state.run_id,
                "company": run.company,
                "qualification": run.breakdown.result,
                "evidence": run.qualification.evidence,
                "warnings": run.state.warnings,
                "model_versions": versions,
                "retrieved_urls": urls,
            }
            try:
                brief = BriefGenerationAgent(rt.llm, rt.prompts, rt.settings).generate(**kwargs)
            except Exception as exc:
                if rt.llm is None:
                    raise
                record.degrade(
                    f"brief generation: LLM path failed ({type(exc).__name__}); rebuilt deterministically"
                )
                kwargs["warnings"] = run.state.warnings
                brief = BriefGenerationAgent(None, rt.prompts, rt.settings).generate(**kwargs)
            run.brief = brief
            report = brief.citation_report
            record.detail.update(
                {
                    "statements": len(brief.all_statements()),
                    "verified_facts": len(brief.verified_facts),
                    "recommendations": len(brief.recommendations),
                    "citation_coverage": round(report.coverage, 4),
                    "citations_removed": len(report.removed_statements),
                }
            )

    def _validation(self, run: _Run) -> None:
        rt = self._rt
        with self._step(run, StepName.VALIDATION, SpanType.TOOL) as record:
            if run.brief is None:
                raise AgentError("internal invariant violated: run.brief is unset")
            brief = run.brief
            output_report = rt.output_guard.check_brief(brief)
            policy_report = rt.responsible_ai.evaluate(brief)
            if not output_report.ok:
                kinds: dict[str, int] = {}
                for violation in output_report.violations:
                    kinds[violation.kind.value] = kinds.get(violation.kind.value, 0) + 1
                self._audit(run, "guardrail.output_violation", {"stage": "validation", "violations": kinds})
            locations = [v.location for v in output_report.violations if v.kind in REMOVABLE_VIOLATIONS]
            locations.extend(f.location for f in policy_report.errors)
            cleaned, removed = strip_statements(brief, locations)
            if removed:
                note = (
                    f"validation: removed {removed} statement(s) flagged by the output guard "
                    "or responsible-AI policy"
                )
                cleaned = cleaned.model_copy(update={"warnings": (*cleaned.warnings, note)})
                record.note(note)
                output_report = rt.output_guard.check_brief(cleaned)
                policy_report = rt.responsible_ai.evaluate(cleaned)
            if not output_report.ok:
                unresolved = ", ".join(sorted(k.value for k in output_report.kinds))
                record.degrade(f"validation: unresolved output guard findings ({unresolved})")
            if policy_report.errors:
                record.degrade(f"validation: {len(policy_report.errors)} unresolved responsible-AI error(s)")
            run.brief, run.output_report, run.policy_report = cleaned, output_report, policy_report
            self._reflect(run, record)
            report = cleaned.citation_report
            record.detail.update(
                {
                    "citation_coverage": round(report.coverage, 4),
                    "fact_statements": report.total_statements,
                    "supported_statements": report.supported_statements,
                    "removed_by_citation_check": len(report.removed_statements),
                    "removed_by_guards": removed,
                    "output_violations": len(output_report.violations),
                    "policy_findings": len(policy_report.findings),
                }
            )

    def _output(self, run: _Run) -> RunResult:
        rt = self._rt
        if run.brief is None:
            raise AgentError("internal invariant violated: run.brief is unset")
        if run.qualification is None:
            raise AgentError("internal invariant violated: run.qualification is unset")
        brief = run.brief
        decision = rt.review_policy.decide(
            brief,
            sensitivity=run.sensitivity,
            output_report=run.output_report,
            policy_report=run.policy_report,
            degraded_steps=run.state.degraded_steps,
            reflection=run.reflection,
        )
        with self._step(run, StepName.OUTPUT, SpanType.TOOL) as record:
            try:
                rt.brief_repository.save(brief)
                record.detail["persisted"] = True
            except Exception as exc:
                record.detail["persisted"] = False
                record.degrade(f"output: the brief could not be persisted ({type(exc).__name__})")
            markdown = render_run_markdown(
                brief,
                review=decision,
                sensitivity=run.sensitivity,
                disclaimer=run.policy_report.disclaimer if run.policy_report else None,
            )
            rows: list[dict[str, Any]] = []
            try:
                run.lineage.record_brief(brief)
                rows = evidence_lineage_rows(brief, run.qualification.evidence)
            except Exception as exc:
                record.degrade(f"output: lineage could not be recorded ({type(exc).__name__})")
            if decision.needs_review:
                try:
                    rt.review_queue.enqueue(brief, decision, request=run.request)
                except Exception as exc:
                    record.degrade(f"output: review queue unavailable ({type(exc).__name__})")
            self._collect_metrics(run, brief)
            self._audit(
                run,
                "brief.generated",
                {
                    "verdict": brief.qualification.verdict.value,
                    "weighted_score": brief.qualification.weighted_score,
                    "citation_coverage": round(brief.citation_report.coverage, 4),
                    "needs_review": decision.needs_review,
                    "evidence": len(brief.evidence),
                },
            )
            try:
                self._track(run, brief, markdown, decision)
            except Exception as exc:  # experiment tracking is optional
                record.degrade(f"output: MLflow tracking failed ({type(exc).__name__})")
            record.detail.update(
                {
                    "needs_review": decision.needs_review,
                    "review_priority": decision.priority.value,
                    "lineage_rows": len(rows),
                    "mlflow_run_id": run.state.mlflow_run_id,
                    "markdown_chars": len(markdown),
                }
            )
        return RunResult(
            brief=brief, markdown=markdown, state=run.state, review=decision, lineage_rows=tuple(rows)
        )

    # ------------------------------------------------------------------ helpers
    def _reflect(self, run: _Run, record: StepRecord) -> None:
        """Self-RAG reflection: are the executive summary's factual claims supported by their evidence?

        Only verified-fact statements are critiqued; score narratives and recommendations are analysis
        of the scores, not claims about the company, and are labelled as such in the brief.
        """
        if run.brief is None:
            raise AgentError("internal invariant violated: run.brief is unset")
        if run.qualification is None:
            raise AgentError("internal invariant violated: run.qualification is unset")
        cited = [
            s
            for s in run.brief.executive_summary
            if s.evidence_ids and s.provenance is ProvenanceKind.VERIFIED_FACT
        ]
        chunks = [
            chunk
            for s in cited
            for evidence_id in s.evidence_ids
            if (chunk := run.qualification.evidence.chunk(evidence_id)) is not None
        ]
        if not cited or not chunks:
            record.detail["reflection"] = "not_applicable"
            return
        question = f"Is {run.company} a good fit for data and AI consulting services, and why?"
        answer = " ".join(s.text for s in cited)
        try:
            critique = SelfRagCritic(self._rt.llm).critique_answer(question, answer, chunks)
        except Exception as exc:  # reflection is advisory
            record.degrade(f"validation: self-reflection unavailable ({type(exc).__name__})")
            return
        run.reflection = critique
        record.detail["reflection"] = critique.is_supported.value
        record.detail["reflection_support_ratio"] = critique.support_ratio
        if critique.is_supported is SupportLevel.NO:
            note = "validation: self-reflection found the executive summary unsupported by its cited evidence"
            run.brief = run.brief.model_copy(update={"warnings": (*run.brief.warnings, note)})
            record.note(note)

    def _collect_metrics(self, run: _Run, brief: ClientBrief) -> None:
        q = brief.qualification
        metrics: dict[str, float] = {
            "weighted_score": float(q.weighted_score),
            "overall_confidence": float(q.overall_confidence),
            "citation_coverage": float(brief.citation_report.coverage),
            "evidence_items": float(len(brief.evidence)),
            "verified_facts": float(len(brief.verified_facts)),
            "recommendations": float(len(brief.recommendations)),
            "documents_ingested": float(len(run.gathering.documents)) if run.gathering else 0.0,
            "chunks_written": float(run.gathering.chunks_written) if run.gathering else 0.0,
        }
        for step in run.state.steps:
            metrics[f"step_ms.{step.name.value}"] = round(step.duration_ms, 3)
        if run.accounting is not None:
            metrics.update(run.accounting.costs.as_metrics())
            snapshot = run.accounting.budget.snapshot()
            metrics["budget_tokens_used"] = float(snapshot.tokens)
            metrics["budget_exhausted"] = 1.0 if run.accounting.exhausted else 0.0
        run.state.metrics.update(metrics)

    def _track(self, run: _Run, brief: ClientBrief, markdown: str, decision: ReviewDecision) -> None:
        rt = self._rt
        tags = {
            "cra.run_id": run.state.run_id,
            "cra.company": run.company,
            "cra.environment": rt.settings.environment.value,
            "cra.verdict": brief.qualification.verdict.value,
        }
        with RunTracker(
            enabled=rt.track_runs,
            experiment=rt.settings.observability.mlflow_experiment,
            run_name=f"brief-{run.state.run_id}"[:250],
            tags=tags,
        ) as tracker:
            if not tracker.active:
                return
            params: dict[str, Any] = {
                "company": run.company,
                "domain": run.request.domain,
                "ticker": run.request.ticker,
                "max_documents": run.request.max_documents,
                "requested_by": run.request.requested_by,
                **{f"version.{k}": v for k, v in brief.model_versions.items()},
            }
            tracker.log_params(params)
            tracker.log_metrics(run.state.metrics)
            tracker.log_dict(brief.model_dump(mode="json"), "brief.json")
            tracker.log_dict(run.lineage.to_dict(), "lineage.json")
            tracker.log_dict(run.state.to_dict(), "run_state.json")
            tracker.log_dict({"markdown": markdown, "review": decision.to_dict()}, "brief_markdown.json")
            if run.plan is not None:
                tracker.log_dict(run.plan.to_dict(), "research_plan.json")
            run.state.mlflow_run_id = tracker.run_id

    def _flush_audit(self) -> None:
        flush = getattr(self._rt.audit, "flush", None)
        if callable(flush):
            try:
                flush()
            except Exception as exc:
                _log.warning("audit.flush_failed", error=type(exc).__name__)

    def _audit(self, run: _Run, event_type: str, payload: Mapping[str, Any]) -> None:
        try:
            self._rt.audit.append(event_type, payload, principal=run.principal.id, run_id=run.state.run_id)
        except Exception as exc:  # the audit sink must not take a research run down
            get_metrics().increment("audit.write_failures")
            _log.warning("audit.write_failed", event_type=event_type, error=type(exc).__name__)
