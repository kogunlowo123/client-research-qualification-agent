"""Citation validation for Client Brief statements.

Every statement tagged ``VERIFIED_FACT`` must be entailed by evidence it
cites, and that evidence must be a document actually retrieved in this run.

Per fact statement:

1. Cited ids are the union of ``evidence_ids`` and inline ``[E3]`` markers
   (markers are then stripped from the text; the renderer re-adds them).
2. Ids that are unknown to the run's ``EvidenceRegistry``, whose URL is not
   http(s), or whose URL does not belong to a retrieved document are rejected.
3. Each remaining id gets a support score: lexical entailment (content-word
   coverage with hard number and entity consistency rules, see
   ``citations.entailment``), optionally blended with an LLM judge using the
   ``citation_judge`` prompt. The judge is consulted only when every figure and
   named entity in the claim appears in the evidence (hard failures need no
   judge); it can lower support freely but raise it by at most
   ``JUDGE_MAX_UPLIFT`` over the lexical score, so text inside untrusted
   evidence cannot talk its way into "supported".
4. Support is the max over cited ids.
   * ``>= min_citation_support``: kept as a fact, citing only the ids that
     individually reach the threshold.
   * Figures that appear in none of the cited evidence: removed outright; a
     fabricated number is never re-labelled as a recommendation.
   * Partial support (``>= partial_support_ratio * min_citation_support``):
     re-tagged ``AI_RECOMMENDATION`` with an explicit note.
   * Otherwise removed.

``AI_RECOMMENDATION`` statements are kept, with invalid ids stripped.

``CitationReport.total_statements`` counts the fact statements evaluated and
``supported_statements`` those kept as facts, so ``coverage`` is the share of
claimed facts that survived grounding.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.citations.entailment import SupportAssessment, lexical_support
from client_research_agent.config.settings import GuardrailSettings
from client_research_agent.models import BriefStatement, CitationCheck, CitationReport, ProvenanceKind
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, span
from client_research_agent.prompts.registry import PromptRegistry, default_registry
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.text import citation_markers, strip_citation_markers
from client_research_agent.services.ports import ChatMessage, LLMClient
from client_research_agent.services.structured import complete_structured
from client_research_agent.utils.errors import CircuitOpenError, OutputValidationError, TransientError

JUDGE_MAX_UPLIFT = 0.2
PARTIAL_SUPPORT_NOTE = " (Inference: the cited evidence only partially supports this statement.)"
_LLM_FAILURES: tuple[type[Exception], ...] = (OutputValidationError, TransientError, CircuitOpenError)
_log = get_logger(__name__)


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    supported: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=1000)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    sections: dict[str, tuple[BriefStatement, ...]]
    report: CitationReport
    warnings: tuple[str, ...]
    downgraded: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _IdSupport:
    evidence_id: str
    score: float
    lexical: SupportAssessment
    reason: str


def is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


class CitationValidator:
    def __init__(
        self,
        llm: LLMClient | None,
        settings: GuardrailSettings,
        *,
        prompts: PromptRegistry | None = None,
        judge_weight: float = 0.5,
        partial_support_ratio: float = 0.5,
        max_judge_calls: int = 60,
    ) -> None:
        if not 0.0 <= judge_weight <= 1.0:
            raise ValueError("judge_weight must be within [0, 1]")
        if not 0.0 < partial_support_ratio <= 1.0:
            raise ValueError("partial_support_ratio must be within (0, 1]")
        self._llm = llm
        self._settings = settings
        self._prompts = prompts or default_registry()
        self._judge_weight = judge_weight
        self._partial_ratio = partial_support_ratio
        self._max_judge_calls = max_judge_calls
        self._template = self._prompts.get("citation_judge")

    @property
    def threshold(self) -> float:
        return self._settings.min_citation_support

    @property
    def prompt_versions(self) -> dict[str, str]:
        return self._prompts.model_versions(["citation_judge"])

    # ------------------------------------------------------------------ evidence checks
    def evidence_problem(
        self, evidence_id: str, registry: EvidenceRegistry, retrieved_urls: Collection[str] | None = None
    ) -> str | None:
        """Why ``evidence_id`` cannot be cited, or None when it is valid."""
        evidence = registry.get(evidence_id)
        chunk = registry.chunk(evidence_id)
        if evidence is None or chunk is None:
            return "unknown evidence id"
        if not is_http_url(evidence.url):
            return "evidence URL is not http(s)"
        if evidence.url != chunk.url:
            return "evidence URL does not match the retrieved document"
        if retrieved_urls is not None and evidence.url not in retrieved_urls:
            return "evidence URL is not among retrieved documents"
        return None

    # ------------------------------------------------------------------ main entry point
    def validate(
        self,
        sections: Mapping[str, Sequence[BriefStatement]],
        registry: EvidenceRegistry,
        *,
        retrieved_urls: Collection[str] | None = None,
    ) -> ValidationOutcome:
        checks: list[CitationCheck] = []
        warnings: list[str] = []
        removed: list[str] = []
        downgraded: list[str] = []
        total = 0
        supported = 0
        judge_budget = [self._max_judge_calls]
        validated: dict[str, tuple[BriefStatement, ...]] = {}
        with span("citations.validate", SpanType.PARSER, statements=sum(len(v) for v in sections.values())):
            for section, statements in sections.items():
                kept: list[BriefStatement] = []
                for statement in statements:
                    if statement.provenance is ProvenanceKind.VERIFIED_FACT:
                        total += 1
                        result = self._validate_fact(
                            statement,
                            registry,
                            retrieved_urls=retrieved_urls,
                            checks=checks,
                            warnings=warnings,
                            judge_budget=judge_budget,
                        )
                        if result is None:
                            removed.append(statement.text)
                        elif result.provenance is ProvenanceKind.VERIFIED_FACT:
                            supported += 1
                            kept.append(result)
                        else:
                            downgraded.append(statement.text)
                            kept.append(result)
                    else:
                        kept.append(
                            self._validate_recommendation(statement, registry, retrieved_urls, warnings)
                        )
                validated[section] = tuple(kept)
        metrics = get_metrics()
        metrics.increment("citations_removed_total", len(removed))
        metrics.increment("citations_downgraded_total", len(downgraded))
        if removed:
            warnings.append(f"removed {len(removed)} statement(s) not supported by cited evidence")
        if downgraded:
            warnings.append(
                f"re-labelled {len(downgraded)} partially supported fact(s) as AI recommendations"
            )
        report = CitationReport(
            checks=tuple(checks),
            total_statements=total,
            supported_statements=supported,
            removed_statements=tuple(removed),
        )
        return ValidationOutcome(
            sections=validated,
            report=report,
            warnings=tuple(dict.fromkeys(warnings)),
            downgraded=tuple(downgraded),
        )

    # ------------------------------------------------------------------ statements
    @staticmethod
    def _cited_ids(statement: BriefStatement) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*statement.evidence_ids, *citation_markers(statement.text))))

    def _split_ids(
        self,
        ids: Sequence[str],
        registry: EvidenceRegistry,
        retrieved_urls: Collection[str] | None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        valid: list[str] = []
        invalid: list[tuple[str, str]] = []
        for evidence_id in ids:
            problem = self.evidence_problem(evidence_id, registry, retrieved_urls)
            if problem is None:
                valid.append(evidence_id)
            else:
                invalid.append((evidence_id, problem))
        return valid, invalid

    def _validate_recommendation(
        self,
        statement: BriefStatement,
        registry: EvidenceRegistry,
        retrieved_urls: Collection[str] | None,
        warnings: list[str],
    ) -> BriefStatement:
        valid, invalid = self._split_ids(self._cited_ids(statement), registry, retrieved_urls)
        if invalid:
            warnings.append(
                f"stripped invalid evidence ids {[i for i, _ in invalid]} from recommendation: "
                f"{statement.text[:80]}"
            )
        text = strip_citation_markers(statement.text) or statement.text
        return BriefStatement(
            text=text, provenance=ProvenanceKind.AI_RECOMMENDATION, evidence_ids=tuple(valid)
        )

    def _validate_fact(
        self,
        statement: BriefStatement,
        registry: EvidenceRegistry,
        *,
        retrieved_urls: Collection[str] | None,
        checks: list[CitationCheck],
        warnings: list[str],
        judge_budget: list[int],
    ) -> BriefStatement | None:
        text = strip_citation_markers(statement.text)
        if not text:
            return None
        valid, invalid = self._split_ids(self._cited_ids(statement), registry, retrieved_urls)
        for evidence_id, problem in invalid:
            evidence = registry.get(evidence_id)
            checks.append(
                CitationCheck(
                    evidence_id=evidence_id,
                    url=evidence.url if evidence is not None else "",
                    supported=False,
                    support_score=0.0,
                    reason=problem,
                )
            )
        if invalid:
            warnings.append(f"fact cited invalid evidence {[i for i, _ in invalid]}: {text[:80]}")
        if not valid:
            return None

        supports = [
            self._support(text, evidence_id, registry, judge_budget, warnings) for evidence_id in valid
        ]
        for item in supports:
            checks.append(
                CitationCheck(
                    evidence_id=item.evidence_id,
                    url=registry.require(item.evidence_id).url,
                    supported=item.score >= self.threshold,
                    support_score=round(item.score, 4),
                    reason=item.reason[:500],
                )
            )
        best = max(item.score for item in supports)
        if best >= self.threshold:
            ids = tuple(item.evidence_id for item in supports if item.score >= self.threshold)
            return BriefStatement(text=text, provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=ids)
        if all(not item.lexical.numbers_consistent for item in supports):
            return None
        if best >= self.threshold * self._partial_ratio:
            ordered = sorted((i for i in supports if i.score > 0), key=lambda i: -i.score)
            return BriefStatement(
                text=f"{text.rstrip()}{PARTIAL_SUPPORT_NOTE}",
                provenance=ProvenanceKind.AI_RECOMMENDATION,
                evidence_ids=tuple(i.evidence_id for i in ordered),
            )
        return None

    def _support(
        self,
        text: str,
        evidence_id: str,
        registry: EvidenceRegistry,
        judge_budget: list[int],
        warnings: list[str],
    ) -> _IdSupport:
        source = registry.source_text(evidence_id)
        evidence = registry.require(evidence_id)
        lexical = lexical_support(text, f"{evidence.title}\n{source}")
        reason = lexical.reason()
        score = lexical.score
        if (
            self._llm is not None
            and lexical.numbers_consistent
            and not lexical.missing_entities
            and judge_budget[0] > 0
            and self._judge_weight > 0
        ):
            judge_budget[0] -= 1
            verdict = self._judge(self._llm, text, evidence_id, registry, warnings)
            if verdict is None:
                judge_budget[0] = 0  # the judge is failing; stop spending calls on it this run
            else:
                blended = (1 - self._judge_weight) * lexical.score + self._judge_weight * verdict.score
                if not verdict.supported:
                    blended = min(blended, verdict.score)
                score = max(0.0, min(blended, lexical.score + JUDGE_MAX_UPLIFT, 1.0))
                label = verdict.reason or ("supported" if verdict.supported else "unsupported")
                reason = f"{reason}; judge: {label}"
        return _IdSupport(evidence_id=evidence_id, score=score, lexical=lexical, reason=reason)

    def _judge(
        self,
        llm: LLMClient,
        text: str,
        evidence_id: str,
        registry: EvidenceRegistry,
        warnings: list[str],
    ) -> JudgeVerdict | None:
        prompt = self._template.render(
            statement=text,
            evidence=registry.render_block([evidence_id], full_text=True, max_chars_per_item=3000),
        )
        messages = [
            ChatMessage(
                role="system", content="You are a strict, sceptical fact-checking judge. Answer in JSON."
            ),
            ChatMessage(role="user", content=prompt),
        ]
        try:
            verdict, _ = complete_structured(llm, messages, JudgeVerdict, max_repairs=1, max_tokens=300)
        except _LLM_FAILURES as exc:
            get_metrics().increment("citation_judge_fallback_total")
            _log.warning("citations.judge_fallback", error=type(exc).__name__)
            warnings.append(
                f"citation judge unavailable ({type(exc).__name__}); used lexical entailment only"
            )
            return None
        return verdict
