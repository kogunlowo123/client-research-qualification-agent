"""Client Brief generation.

``BriefGenerationAgent.generate`` assembles every section of a ``ClientBrief``:

* Company Overview - verified facts only;
* Fit Assessment - the ``QualificationResult`` as scored;
* Evidence - every evidence item cited by the brief or the scores;
* Technology Priorities, Gartner-Relevant Insights, Opportunities, Risks -
  from ``OpportunityAnalysisAgent``;
* Executive Summary, Executive Talking Points, Recommended Next Actions - from
  the ``brief_writer`` prompt;
* exactly five Discovery Questions - from the ``discovery_questions`` prompt.

Every statement carries provenance and is then passed through
``CitationValidator``: unsupported facts are removed or re-labelled, and the
resulting ``CitationReport`` is attached. Each LLM step degrades to a
deterministic builder on ``OutputValidationError``/``TransientError``/
``CircuitOpenError``, and a section emptied by validation is refilled from the
deterministic builder (and validated again), so a complete brief is always
produced. Model and prompt versions are recorded for MLflow lineage.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.briefing.opportunity import OpportunityAnalysis, OpportunityAnalysisAgent
from client_research_agent.briefing.statements import (
    StatementDraft,
    drafts_to_statements,
    matching_sentence,
    qualification_summary,
    statements_summary,
    verbatim_fact,
)
from client_research_agent.citations.validator import CitationValidator
from client_research_agent.config.settings import AppSettings
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
)
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, span
from client_research_agent.prompts.registry import PromptRegistry
from client_research_agent.qualification.criteria import get_definition
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.text import truncate_words
from client_research_agent.services.ports import ChatMessage, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import CircuitOpenError, OutputValidationError, TransientError

DISCOVERY_QUESTION_COUNT = 5
MAX_QUESTION_WORDS = 60
_LLM_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)
_OVERVIEW_TYPES = (
    DocumentType.SEC_FILING,
    DocumentType.INVESTOR_RELATIONS,
    DocumentType.EARNINGS_RELEASE,
    DocumentType.CORPORATE_WEBPAGE,
)
VERDICT_LABELS: dict[FitVerdict, str] = {
    FitVerdict.GOOD_FIT: "a good fit",
    FitVerdict.POTENTIAL_FIT: "a potential fit",
    FitVerdict.NOT_ENOUGH_EVIDENCE: "not yet supported by enough evidence of fit",
}
NARRATIVE_SECTIONS = (
    "company_overview",
    "executive_summary",
    "executive_talking_points",
    "recommended_next_actions",
)
_log = get_logger(__name__)


class BriefDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    company_overview: list[StatementDraft] = Field(default_factory=list, max_length=12)
    executive_summary: list[StatementDraft] = Field(default_factory=list, max_length=12)
    executive_talking_points: list[StatementDraft] = Field(default_factory=list, max_length=12)
    recommended_next_actions: list[StatementDraft] = Field(default_factory=list, max_length=12)


class QuestionsDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    questions: list[str] = Field(min_length=1, max_length=12)


@dataclass(frozen=True, slots=True)
class _Narrative:
    sections: dict[str, list[BriefStatement]]
    warnings: tuple[str, ...]


def merge_reports(first: CitationReport, second: CitationReport) -> CitationReport:
    return CitationReport(
        checks=(*first.checks, *second.checks),
        total_statements=first.total_statements + second.total_statements,
        supported_statements=first.supported_statements + second.supported_statements,
        removed_statements=(*first.removed_statements, *second.removed_statements),
    )


class DeterministicBriefBuilder:
    """Builds every narrative section from scores and verbatim evidence, without an LLM."""

    def __init__(self, *, max_items: int = 5) -> None:
        self._max_items = max_items

    def company_overview(
        self, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        scale = _score(qualification, Criterion.COMPANY_SCALE)
        scale_ids = [i for i in (scale.evidence_ids if scale is not None else ()) if i in evidence]
        lexicon = get_definition(Criterion.COMPANY_SCALE).positive_signals
        facts: list[BriefStatement] = []
        seen: set[str] = set()
        for evidence_id in scale_ids:
            facts.append(verbatim_fact(evidence, evidence_id, lexicon))
            seen.add(facts[-1].text)
        # Authoritative corporate sources contribute only sentences that actually describe the company.
        for item in evidence.all():
            if len(facts) >= 4:
                break
            if item.document_type not in _OVERVIEW_TYPES or item.evidence_id in scale_ids:
                continue
            sentence = matching_sentence(evidence, item.evidence_id, lexicon)
            if sentence is not None and sentence not in seen:
                seen.add(sentence)
                facts.append(
                    BriefStatement(
                        text=sentence,
                        provenance=ProvenanceKind.VERIFIED_FACT,
                        evidence_ids=(item.evidence_id,),
                    )
                )
        return facts[:4]

    def executive_summary(self, company: str, qualification: QualificationResult) -> list[BriefStatement]:
        ranked = sorted(qualification.scores, key=lambda s: (-s.weighted, s.criterion.value))
        statements = [
            BriefStatement(
                text=(
                    f"{company} is assessed as {VERDICT_LABELS[qualification.verdict]} with a weighted "
                    "score of "
                    f"{qualification.weighted_score:.2f} out of 5 at {qualification.overall_confidence:.0%} "
                    "confidence."
                ),
                provenance=ProvenanceKind.AI_RECOMMENDATION,
            )
        ]
        if ranked:
            strongest, weakest = ranked[0], ranked[-1]
            statements.append(
                BriefStatement(
                    text=(
                        f"Strongest evidence is for {get_definition(strongest.criterion).title} "
                        f"({strongest.score}/5, rubric level: "
                        f"{get_definition(strongest.criterion).rubric[strongest.score]})"
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=strongest.evidence_ids[:2],
                )
            )
            if weakest.criterion is not strongest.criterion:
                statements.append(
                    BriefStatement(
                        text=(
                            f"The main gap is {get_definition(weakest.criterion).title} ({weakest.score}/5), "
                            "which should be tested in discovery."
                        ),
                        provenance=ProvenanceKind.AI_RECOMMENDATION,
                        evidence_ids=weakest.evidence_ids[:2],
                    )
                )
        return statements

    def talking_points(
        self, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        points: list[BriefStatement] = []
        used: set[str] = set()
        for score in sorted(qualification.scores, key=lambda s: (-s.weighted, s.criterion.value)):
            valid = [i for i in score.evidence_ids if i in evidence and i not in used]
            if score.score < 2 or not valid:
                continue
            used.add(valid[0])
            quote = truncate_words(" ".join(evidence.require(valid[0]).quote.split()), 220)
            points.append(
                BriefStatement(
                    text=(
                        f"On {get_definition(score.criterion).title.lower()}, acknowledge what {company} "
                        "has said "
                        f'publicly ("{quote}") and ask how it is progressing.'
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=(valid[0],),
                )
            )
            if len(points) >= 3:
                break
        if not points:
            points.append(
                BriefStatement(
                    text=(
                        f"Open by asking how {company} is prioritising data and AI investment this year, "
                        "since "
                        "public sources say little about it."
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                )
            )
        return points

    def next_actions(self, company: str, qualification: QualificationResult) -> list[BriefStatement]:
        ranked = sorted(qualification.scores, key=lambda s: (-s.weighted, s.criterion.value))
        top = get_definition(ranked[0].criterion).title if ranked else "the strongest criterion"
        weakest = get_definition(ranked[-1].criterion).title if ranked else "the weakest criterion"
        if qualification.verdict is FitVerdict.GOOD_FIT:
            actions = [
                f"Identify and engage the executive sponsor accountable for {top.lower()} at {company}.",
                f"Prepare a tailored point of view that links {top.lower()} to measurable business outcomes.",
                "Propose a scoped discovery workshop to confirm priorities, budget and timeline.",
                f"Use the discovery questions to close the remaining gap on {weakest.lower()}.",
            ]
        elif qualification.verdict is FitVerdict.POTENTIAL_FIT:
            actions = [
                f"Validate the evidence gap on {weakest.lower()} before investing in a full pursuit.",
                f"Arrange an exploratory conversation anchored on {top.lower()}.",
                "Monitor public sources for triggering events such as leadership changes or new programmes.",
                "Re-run qualification after discovery to confirm or downgrade the account.",
            ]
        else:
            actions = [
                f"Gather more public evidence on {company} (recent filings, investor materials, "
                "leadership news).",
                "Keep the account in a monitoring list rather than active pursuit.",
                f"If a warm introduction exists, test the {weakest.lower()} gap informally.",
                "Re-run qualification when new public information is available.",
            ]
        return [BriefStatement(text=a, provenance=ProvenanceKind.AI_RECOMMENDATION) for a in actions]

    def discovery_questions(self, company: str, qualification: QualificationResult) -> list[str]:
        scores = {s.criterion: s for s in qualification.scores}
        ordered = sorted(
            Criterion,
            key=lambda c: (
                scores[c].confidence if c in scores else 0.0,
                scores[c].score if c in scores else 0,
                c.value,
            ),
        )
        return [get_definition(c).render_discovery_question(company) for c in ordered][
            :DISCOVERY_QUESTION_COUNT
        ]

    def section(
        self, name: str, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        if name == "company_overview":
            return self.company_overview(qualification, evidence)
        if name == "executive_summary":
            return self.executive_summary(company, qualification)
        if name == "executive_talking_points":
            return self.talking_points(company, qualification, evidence)
        if name == "recommended_next_actions":
            return self.next_actions(company, qualification)
        return []


def _score(result: QualificationResult, criterion: Criterion) -> CriterionScore | None:
    return next((s for s in result.scores if s.criterion is criterion), None)


class BriefGenerationAgent:
    def __init__(
        self,
        llm: LLMClient | None,
        registry: PromptRegistry,
        settings: AppSettings,
        *,
        validator: CitationValidator | None = None,
        opportunity_agent: OpportunityAnalysisAgent | None = None,
        builder: DeterministicBriefBuilder | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = registry
        self._settings = settings
        self._validator = validator or CitationValidator(llm, settings.guardrails, prompts=registry)
        self._opportunity = opportunity_agent or OpportunityAnalysisAgent(llm, registry)
        self._builder = builder or DeterministicBriefBuilder()
        self._writer = registry.get("brief_writer")
        self._questions = registry.get("discovery_questions")

    def generate(
        self,
        *,
        run_id: str,
        company: str,
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
        warnings: Sequence[str] = (),
        model_versions: Mapping[str, str] | None = None,
        retrieved_urls: Collection[str] | None = None,
    ) -> ClientBrief:
        all_warnings: list[str] = list(warnings)
        with span("briefing.generate", SpanType.AGENT, company=company, run_id=run_id):
            analysis = self._opportunity.analyze(company, qualification, evidence)
            all_warnings.extend(analysis.warnings)
            narrative = self._narrative(company, qualification, evidence, analysis)
            all_warnings.extend(narrative.warnings)
            questions, question_warnings = self._discovery_questions(company, qualification, evidence)
            all_warnings.extend(question_warnings)

            candidate: dict[str, Sequence[BriefStatement]] = {
                "company_overview": narrative.sections["company_overview"],
                **analysis.sections(),
                "executive_summary": narrative.sections["executive_summary"],
                "executive_talking_points": narrative.sections["executive_talking_points"],
                "recommended_next_actions": narrative.sections["recommended_next_actions"],
            }
            outcome = self._validator.validate(candidate, evidence, retrieved_urls=retrieved_urls)
            sections = dict(outcome.sections)
            sections["company_overview"] = _facts_only(sections.get("company_overview", ()))
            report = outcome.report
            all_warnings.extend(outcome.warnings)

            if self._llm is not None:
                # LLM-written sections emptied by validation are rebuilt from verbatim evidence.
                refill = {
                    name: self._builder.section(name, company, qualification, evidence)
                    for name in NARRATIVE_SECTIONS
                    if not sections.get(name) and candidate.get(name)
                }
                refill = {name: items for name, items in refill.items() if items}
                if refill:
                    second = self._validator.validate(refill, evidence, retrieved_urls=retrieved_urls)
                    report = merge_reports(report, second.report)
                    all_warnings.extend(second.warnings)
                    for name, items in second.sections.items():
                        sections[name] = _facts_only(items) if name == "company_overview" else items
                        all_warnings.append(
                            f"{name}: all statements failed validation; rebuilt deterministically"
                        )

            brief = ClientBrief(
                run_id=run_id,
                company=company,
                company_overview=_facts_only(sections.get("company_overview", ())),
                qualification=qualification,
                evidence=self._cited_evidence(sections, qualification, evidence),
                technology_priorities=sections.get("technology_priorities", ()),
                gartner_relevant_insights=sections.get("gartner_relevant_insights", ()),
                opportunities=sections.get("opportunities", ()),
                risks=sections.get("risks", ()),
                executive_summary=sections.get("executive_summary", ()),
                discovery_questions=tuple(questions),
                executive_talking_points=sections.get("executive_talking_points", ()),
                recommended_next_actions=sections.get("recommended_next_actions", ()),
                citation_report=report,
                model_versions=self._model_versions(model_versions),
                warnings=tuple(dict.fromkeys(all_warnings)),
            )
        metrics = get_metrics()
        metrics.observe("brief_citation_coverage", report.coverage)
        metrics.increment("briefs_generated_total", verdict=qualification.verdict.value)
        return brief

    # ------------------------------------------------------------------ narrative
    def _narrative(
        self,
        company: str,
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
        analysis: OpportunityAnalysis,
    ) -> _Narrative:
        deterministic = {
            name: self._builder.section(name, company, qualification, evidence) for name in NARRATIVE_SECTIONS
        }
        if self._llm is None or len(evidence) == 0:
            return _Narrative(sections=deterministic, warnings=())
        try:
            draft = self._ask_writer(self._llm, company, qualification, evidence, analysis)
        except _LLM_FAILURES as exc:
            get_metrics().increment("brief_writer_fallback_total")
            _log.warning("briefing.writer_fallback", error=type(exc).__name__)
            return _Narrative(
                sections=deterministic,
                warnings=(
                    f"brief writer LLM failed ({type(exc).__name__}); used deterministic brief sections",
                ),
            )
        warnings: list[str] = []
        sections: dict[str, list[BriefStatement]] = {}
        for name in NARRATIVE_SECTIONS:
            items, section_warnings = drafts_to_statements(
                getattr(draft, name), section=name, facts_only=name == "company_overview"
            )
            warnings.extend(section_warnings)
            if not items and deterministic[name]:
                items = deterministic[name]
                warnings.append(f"{name}: LLM produced no usable statements; used deterministic statements")
            sections[name] = items
        return _Narrative(sections=sections, warnings=tuple(warnings))

    def _ask_writer(
        self,
        llm: LLMClient,
        company: str,
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
        analysis: OpportunityAnalysis,
    ) -> BriefDraft:
        analysis_text = "\n".join(
            statements_summary(label, items)
            for label, items in (
                ("Technology priorities", analysis.technology_priorities),
                ("Trend theme mappings", analysis.gartner_insights),
                ("Opportunities", analysis.opportunities),
                ("Risks", analysis.risks),
            )
        )
        prompt = self._writer.render(
            company=company,
            qualification_summary=qualification_summary(company, qualification),
            analysis_summary=analysis_text,
            evidence_ids=", ".join(evidence.ids()),
            evidence=evidence.render_block(),
        )
        messages = [
            ChatMessage(
                role="system",
                content="You write precise, evidence-grounded executive briefs and answer in JSON.",
            ),
            ChatMessage(role="user", content=prompt),
        ]
        with span("briefing.writer", SpanType.LLM):
            draft, _ = complete_structured(
                llm, messages, BriefDraft, max_tokens=self._settings.serving.max_output_tokens
            )
        return draft

    # ------------------------------------------------------------------ discovery questions
    def _discovery_questions(
        self, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> tuple[list[str], list[str]]:
        fallback = self._builder.discovery_questions(company, qualification)
        if self._llm is None:
            return fallback, []
        gaps = "\n".join(
            f"- {get_definition(s.criterion).title}: score {s.score}/5, confidence {s.confidence:.2f}, "
            f"{len(s.evidence_ids)} evidence item(s)"
            for s in sorted(qualification.scores, key=lambda s: (s.confidence, s.score, s.criterion.value))
        )
        prompt = self._questions.render(
            company=company,
            qualification_summary=qualification_summary(company, qualification),
            gaps=gaps,
            evidence=evidence.render_block(),
        )
        messages = [
            ChatMessage(role="system", content="You prepare discovery questions and answer in JSON."),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            draft, _ = complete_structured(self._llm, messages, QuestionsDraft, max_tokens=800)
        except _LLM_FAILURES as exc:
            get_metrics().increment("discovery_questions_fallback_total")
            return fallback, [
                f"discovery questions LLM failed ({type(exc).__name__}); used rubric-based questions"
            ]
        questions: list[str] = []
        rejected = 0
        for raw in draft.questions:
            question = " ".join(raw.split())
            if not question.endswith("?") or len(question.split()) > MAX_QUESTION_WORDS:
                rejected += 1
                continue
            if question.casefold() not in {q.casefold() for q in questions}:
                questions.append(question)
        warnings = [f"discovery questions: rejected {rejected} malformed question(s)"] if rejected else []
        for question in fallback:
            if len(questions) >= DISCOVERY_QUESTION_COUNT:
                break
            if question.casefold() not in {q.casefold() for q in questions}:
                questions.append(question)
                warnings.append("discovery questions: padded with rubric-based question")
        return questions[:DISCOVERY_QUESTION_COUNT], list(dict.fromkeys(warnings))

    # ------------------------------------------------------------------ assembly helpers
    @staticmethod
    def _cited_evidence(
        sections: Mapping[str, Sequence[BriefStatement]],
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
    ) -> tuple[Evidence, ...]:
        cited: set[str] = {i for items in sections.values() for s in items for i in s.evidence_ids}
        cited.update(i for score in qualification.scores for i in score.evidence_ids)
        selected = tuple(e for e in evidence.all() if e.evidence_id in cited)
        return selected or evidence.all()

    def _model_versions(self, extra: Mapping[str, str] | None) -> dict[str, str]:
        versions: dict[str, str] = {"llm": self._llm.model_name if self._llm is not None else "none"}
        versions.update(
            self._prompts.model_versions(
                [
                    "criterion_qualifier",
                    "opportunity_analysis",
                    "brief_writer",
                    "discovery_questions",
                    "citation_judge",
                ]
            )
        )
        versions.update(extra or {})
        return versions


def _facts_only(statements: Sequence[BriefStatement]) -> tuple[BriefStatement, ...]:
    """The overview is facts-only: facts downgraded by validation are removed from it."""
    return tuple(s for s in statements if s.provenance is ProvenanceKind.VERIFIED_FACT)
