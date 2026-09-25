"""Opportunity analysis: technology priorities, trend-theme mappings, opportunities and risks.

Trend themes are the publicly known *themes* of Gartner's strategic
technology trends (AI engineering, data fabric, platform engineering,
AI TRiSM, cloud-native / industry cloud platforms, ...). Mapping a company's
evidence to a theme is our own analysis, so every theme insight is an
``AI_RECOMMENDATION`` that says it is a mapping. Text that attributes a
statement to Gartner ("Gartner predicts ...") is only allowed when a cited
evidence item is of type ``ANALYST_PUBLIC`` and itself mentions Gartner;
otherwise the insight is discarded.

The LLM path uses the ``opportunity_analysis`` prompt; any LLM failure falls
back to deterministic rules driven by the qualification scores and the
evidence text, so the section is always populated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.briefing.statements import (
    StatementDraft,
    drafts_to_statements,
    qualification_summary,
    verbatim_fact,
)
from client_research_agent.models import (
    BriefStatement,
    Criterion,
    CriterionScore,
    DocumentType,
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
from client_research_agent.qualification.text import phrase_pattern, strip_citation_markers, truncate_words
from client_research_agent.services.ports import ChatMessage, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import CircuitOpenError, OutputValidationError, TransientError

_LLM_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)
_ATTRIBUTION = re.compile(
    r"\b(?:according to gartner|gartner(?:'s)?\s+(?:\w+\s+){0,3}?"
    r"(?:says|said|states|stated|predicts|predicted|forecasts|expects|recommends|recommended|ranks|ranked|"
    r"named|rated|identifies|identified|reports|reported|found|finds|estimates|positions|positioned|"
    r"research|analysis|survey|report|magic quadrant|hype cycle))",
    re.IGNORECASE,
)
MAPPING_NOTE = "This is an analytical mapping to a public trend theme, not a statement by Gartner."
_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrendTheme:
    name: str
    description: str
    keywords: tuple[str, ...]


TREND_THEMES: tuple[TrendTheme, ...] = (
    TrendTheme(
        "AI engineering",
        "Industrialising the delivery of AI and generative AI into production (MLOps, LLMOps, model "
        "lifecycle).",
        (
            "machine learning",
            "generative ai",
            "genai",
            "large language model",
            "llm",
            "mlops",
            "ai model",
            "ai-powered",
            "copilot",
            "artificial intelligence",
        ),
    ),
    TrendTheme(
        "Data fabric",
        "Integrated, metadata-driven data architecture spanning platforms (lakehouse, catalog, integration).",
        (
            "data platform",
            "lakehouse",
            "data lake",
            "data warehouse",
            "data integration",
            "data mesh",
            "data catalog",
            "unity catalog",
            "data governance",
            "single source of truth",
        ),
    ),
    TrendTheme(
        "Platform engineering",
        "Internal developer platforms that give teams self-service, governed delivery tooling.",
        ("platform engineering", "developer platform", "devops", "kubernetes", "ci/cd", "self-service"),
    ),
    TrendTheme(
        "AI TRiSM",
        "AI trust, risk and security management: governance, model risk, privacy and responsible AI.",
        (
            "responsible ai",
            "ai governance",
            "model risk",
            "ai ethics",
            "ai safety",
            "explainability",
            "ai regulation",
        ),
    ),
    TrendTheme(
        "Cloud-native and industry cloud platforms",
        "Cloud-native architectures and industry-specific cloud platforms replacing legacy estates.",
        (
            "cloud migration",
            "cloud-first",
            "cloud-native",
            "hybrid cloud",
            "microservices",
            "industry cloud",
            "saas",
        ),
    ),
    TrendTheme(
        "Intelligent applications and automation",
        "Applications and processes that embed AI-driven automation and decision intelligence.",
        (
            "automation",
            "intelligent automation",
            "decision intelligence",
            "personalization",
            "predictive",
            "rpa",
        ),
    ),
)
_THEMES_BY_NAME = {theme.name.casefold(): theme for theme in TREND_THEMES}

_OPPORTUNITY_TEMPLATES: dict[Criterion, str] = {
    Criterion.COMPANY_SCALE: (
        "Scale supports an enterprise-wide programme: propose a multi-business-unit data and AI operating "
        "model rather than a single-team pilot."
    ),
    Criterion.TECH_MODERNIZATION: (
        "Attach data platform modernisation to the stated technology modernisation agenda so migration "
        "spend also delivers governed, analytics-ready data."
    ),
    Criterion.AI_DATA_FOCUS: (
        "Offer an AI use-case acceleration engagement that moves prioritised use cases from pilot to "
        "production with MLOps and governance built in."
    ),
    Criterion.INDUSTRY_TRENDS: (
        "Lead with an industry-specific point of view on how data and AI address the pressures the company "
        "has acknowledged publicly."
    ),
    Criterion.NEAR_TERM_OPPORTUNITY: (
        "Engage now: recent triggering events suggest a buying window, so propose a short discovery "
        "workshop with the newly accountable executive."
    ),
}
_RISK_LEXICON: tuple[str, ...] = (
    "layoffs",
    "restructuring",
    "cost reduction",
    "cost-cutting",
    "hiring freeze",
    "budget freeze",
    "spending cuts",
    "impairment",
    "data breach",
    "regulatory fine",
    "litigation",
    "going concern",
    "profit warning",
    "headwinds",
    "decline in revenue",
    "outage",
)


class InsightDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    theme: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class OpportunityDraft(BaseModel):
    model_config = ConfigDict(extra="ignore")

    technology_priorities: list[StatementDraft] = Field(default_factory=list, max_length=12)
    gartner_insights: list[InsightDraft] = Field(default_factory=list, max_length=12)
    opportunities: list[StatementDraft] = Field(default_factory=list, max_length=12)
    risks: list[StatementDraft] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True, slots=True)
class OpportunityAnalysis:
    technology_priorities: tuple[BriefStatement, ...]
    gartner_insights: tuple[BriefStatement, ...]
    opportunities: tuple[BriefStatement, ...]
    risks: tuple[BriefStatement, ...]
    warnings: tuple[str, ...] = ()
    used_llm: bool = False

    def sections(self) -> dict[str, tuple[BriefStatement, ...]]:
        return {
            "technology_priorities": self.technology_priorities,
            "gartner_relevant_insights": self.gartner_insights,
            "opportunities": self.opportunities,
            "risks": self.risks,
        }


def themes_catalogue() -> str:
    return "\n".join(f"- {theme.name}: {theme.description}" for theme in TREND_THEMES)


def attributes_to_gartner(text: str) -> bool:
    return _ATTRIBUTION.search(text) is not None


def has_analyst_gartner_source(registry: EvidenceRegistry, evidence_ids: Sequence[str]) -> bool:
    for evidence_id in evidence_ids:
        evidence = registry.get(evidence_id)
        if (
            evidence is not None
            and evidence.document_type is DocumentType.ANALYST_PUBLIC
            and "gartner" in registry.source_text(evidence_id).lower()
        ):
            return True
    return False


class OpportunityAnalysisAgent:
    def __init__(self, llm: LLMClient | None, registry: PromptRegistry, *, max_items: int = 5) -> None:
        self._llm = llm
        self._prompts = registry
        self._template = registry.get("opportunity_analysis")
        self._max_items = max_items

    @property
    def prompt_versions(self) -> dict[str, str]:
        return self._prompts.model_versions(["opportunity_analysis"])

    def analyze(
        self, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> OpportunityAnalysis:
        with span("briefing.opportunity", SpanType.AGENT, company=company):
            if self._llm is None or len(evidence) == 0:
                return self.deterministic(company, qualification, evidence)
            try:
                draft = self._ask_llm(self._llm, company, qualification, evidence)
            except _LLM_FAILURES as exc:
                get_metrics().increment("opportunity_llm_fallback_total")
                _log.warning("opportunity.llm_fallback", error=type(exc).__name__)
                fallback = self.deterministic(company, qualification, evidence)
                return OpportunityAnalysis(
                    technology_priorities=fallback.technology_priorities,
                    gartner_insights=fallback.gartner_insights,
                    opportunities=fallback.opportunities,
                    risks=fallback.risks,
                    warnings=(
                        *fallback.warnings,
                        f"opportunity analysis LLM failed ({type(exc).__name__}); used rule-based analysis",
                    ),
                    used_llm=False,
                )
            return self._from_draft(company, qualification, evidence, draft)

    # ------------------------------------------------------------------ LLM path
    def _ask_llm(
        self,
        llm: LLMClient,
        company: str,
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
    ) -> OpportunityDraft:
        prompt = self._template.render(
            company=company,
            qualification_summary=qualification_summary(company, qualification),
            themes=themes_catalogue(),
            evidence_ids=", ".join(evidence.ids()),
            evidence=evidence.render_block(),
        )
        messages = [
            ChatMessage(
                role="system",
                content="You are a precise account strategist. Use only supplied evidence. Answer in JSON.",
            ),
            ChatMessage(role="user", content=prompt),
        ]
        draft, _ = complete_structured(llm, messages, OpportunityDraft)
        return draft

    def _from_draft(
        self,
        company: str,
        qualification: QualificationResult,
        evidence: EvidenceRegistry,
        draft: OpportunityDraft,
    ) -> OpportunityAnalysis:
        warnings: list[str] = []
        priorities, w1 = drafts_to_statements(
            draft.technology_priorities, section="technology_priorities", limit=self._max_items
        )
        opportunities, w2 = drafts_to_statements(
            draft.opportunities, section="opportunities", limit=self._max_items
        )
        risks, w3 = drafts_to_statements(draft.risks, section="risks", limit=self._max_items)
        warnings.extend((*w1, *w2, *w3))
        insights = self._insights_from_drafts(draft.gartner_insights, evidence, warnings)
        fallback: OpportunityAnalysis | None = None
        if not (priorities and insights and opportunities and risks):
            fallback = self.deterministic(company, qualification, evidence)
        if fallback is not None:
            for name, current, backup in (
                ("technology_priorities", priorities, fallback.technology_priorities),
                ("gartner_relevant_insights", insights, fallback.gartner_insights),
                ("opportunities", opportunities, fallback.opportunities),
                ("risks", risks, fallback.risks),
            ):
                if not current and backup:
                    current.extend(backup)
                    warnings.append(f"{name}: LLM returned no usable items; used rule-based items")
        return OpportunityAnalysis(
            technology_priorities=tuple(priorities),
            gartner_insights=tuple(insights),
            opportunities=tuple(opportunities),
            risks=tuple(risks),
            warnings=tuple(warnings),
            used_llm=True,
        )

    def _insights_from_drafts(
        self, drafts: Sequence[InsightDraft], evidence: EvidenceRegistry, warnings: list[str]
    ) -> list[BriefStatement]:
        insights: list[BriefStatement] = []
        for item in drafts:
            if len(insights) >= self._max_items:
                break
            theme = _THEMES_BY_NAME.get(item.theme.strip().casefold())
            if theme is None:
                warnings.append(
                    f"gartner_relevant_insights: dropped insight with unknown theme {item.theme!r}"
                )
                continue
            ids = tuple(dict.fromkeys(i.strip() for i in item.evidence_ids if i.strip()))
            text = truncate_words(strip_citation_markers(item.text), 500)
            if attributes_to_gartner(text) and not has_analyst_gartner_source(evidence, ids):
                warnings.append(
                    "gartner_relevant_insights: dropped insight attributing a claim to Gartner without an "
                    "analyst_public source"
                )
                continue
            insights.append(
                BriefStatement(
                    text=f"Mapped to the '{theme.name}' trend theme: {text} {MAPPING_NOTE}",
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=ids,
                )
            )
        return insights

    # ------------------------------------------------------------------ deterministic path
    def deterministic(
        self, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> OpportunityAnalysis:
        scores = {score.criterion: score for score in qualification.scores}
        return OpportunityAnalysis(
            technology_priorities=tuple(self._rule_priorities(company, scores, evidence)),
            gartner_insights=tuple(self._rule_insights(evidence)),
            opportunities=tuple(self._rule_opportunities(company, qualification, evidence)),
            risks=tuple(self._rule_risks(qualification, evidence)),
            used_llm=False,
        )

    def _rule_priorities(
        self, company: str, scores: dict[Criterion, CriterionScore], evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        statements: list[BriefStatement] = []
        seen: set[str] = set()
        for criterion in (Criterion.AI_DATA_FOCUS, Criterion.TECH_MODERNIZATION):
            score = scores.get(criterion)
            if score is None:
                continue
            lexicon = get_definition(criterion).positive_signals
            for evidence_id in score.evidence_ids[:2]:
                if evidence_id not in evidence:
                    continue
                fact = verbatim_fact(evidence, evidence_id, lexicon)
                if fact.text not in seen:
                    seen.add(fact.text)
                    statements.append(fact)
        ranked = sorted(
            (s for c, s in scores.items() if c in (Criterion.AI_DATA_FOCUS, Criterion.TECH_MODERNIZATION)),
            key=lambda s: (-s.score, s.criterion.value),
        )
        if ranked and ranked[0].score >= 3:
            top = ranked[0]
            statements.append(
                BriefStatement(
                    text=(
                        f"Inferred priority: {get_definition(top.criterion).title} appears to be "
                        f"{company}'s most "
                        "evidenced technology focus and should anchor the conversation."
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=top.evidence_ids[:2],
                )
            )
        return statements[: self._max_items]

    def _rule_insights(self, evidence: EvidenceRegistry) -> list[BriefStatement]:
        scored: list[tuple[float, TrendTheme, list[str], list[str]]] = []
        for theme in TREND_THEMES:
            patterns = [(keyword, phrase_pattern(keyword)) for keyword in theme.keywords]
            matched_keywords: list[str] = []
            weighted_ids: list[tuple[float, str]] = []
            total = 0.0
            for item in evidence.all():
                text = evidence.source_text(item.evidence_id)
                hits = [keyword for keyword, pattern in patterns if pattern.search(text)]
                if hits:
                    weight = len(hits) * (0.5 + item.relevance)
                    total += weight
                    weighted_ids.append((weight, item.evidence_id))
                    matched_keywords.extend(k for k in hits if k not in matched_keywords)
            if total > 0:
                top_ids = [eid for _, eid in sorted(weighted_ids, key=lambda pair: (-pair[0], pair[1]))[:2]]
                scored.append((total, theme, matched_keywords, top_ids))
        scored.sort(key=lambda entry: (-entry[0], entry[1].name))
        insights: list[BriefStatement] = []
        for _, theme, keywords, ids in scored[: min(4, self._max_items)]:
            insights.append(
                BriefStatement(
                    text=(
                        f"Mapped to the '{theme.name}' trend theme: public evidence references "
                        f"{', '.join(keywords[:3])}, which aligns with this theme ({theme.description}) "
                        f"{MAPPING_NOTE}"
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=tuple(ids),
                )
            )
        return insights

    def _rule_opportunities(
        self, company: str, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        strong = sorted(
            (s for s in qualification.scores if s.score >= 3),
            key=lambda s: (-s.weighted, s.criterion.value),
        )
        statements = [
            BriefStatement(
                text=_OPPORTUNITY_TEMPLATES[score.criterion],
                provenance=ProvenanceKind.AI_RECOMMENDATION,
                evidence_ids=tuple(i for i in score.evidence_ids[:2] if i in evidence),
            )
            for score in strong
        ]
        if not statements:
            statements.append(
                BriefStatement(
                    text=(
                        f"No criterion for {company} reaches a strong score yet; the opportunity is to run a "
                        "low-cost discovery conversation to test whether unpublished initiatives exist."
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                )
            )
        return statements[: self._max_items]

    def _rule_risks(
        self, qualification: QualificationResult, evidence: EvidenceRegistry
    ) -> list[BriefStatement]:
        statements: list[BriefStatement] = []
        seen: set[str] = set()
        for item in evidence.all():
            if len(statements) >= 2:
                break
            fact = verbatim_fact(evidence, item.evidence_id, _RISK_LEXICON)
            lowered = fact.text.lower()
            if any(phrase_pattern(term).search(lowered) for term in _RISK_LEXICON) and fact.text not in seen:
                seen.add(fact.text)
                statements.append(fact)
        for score in sorted(qualification.scores, key=lambda s: (s.score, s.confidence, s.criterion.value)):
            if score.score <= 2 or score.confidence < 0.4:
                statements.append(
                    BriefStatement(
                        text=(
                            f"Evidence gap: {get_definition(score.criterion).title} is weakly supported "
                            f"(score {score.score}/5, confidence {score.confidence:.0%}); validate it in "
                            "discovery before committing pursuit resources."
                        ),
                        provenance=ProvenanceKind.AI_RECOMMENDATION,
                        evidence_ids=tuple(i for i in score.evidence_ids[:2] if i in evidence),
                    )
                )
        if not statements and qualification.scores:
            weakest = min(qualification.scores, key=lambda s: (s.confidence, s.score, s.criterion.value))
            statements.append(
                BriefStatement(
                    text=(
                        f"Least certain criterion: {get_definition(weakest.criterion).title} (confidence "
                        f"{weakest.confidence:.0%}); a pursuit plan should not depend on it until confirmed."
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                    evidence_ids=tuple(i for i in weakest.evidence_ids[:2] if i in evidence),
                )
            )
        if qualification.verdict is FitVerdict.NOT_ENOUGH_EVIDENCE:
            statements.insert(
                0,
                BriefStatement(
                    text=(
                        "Qualification risk: the public evidence does not yet demonstrate fit, so any "
                        "pursuit "
                        "should be framed as exploratory."
                    ),
                    provenance=ProvenanceKind.AI_RECOMMENDATION,
                ),
            )
        return statements[: self._max_items]
