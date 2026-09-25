"""Helpers shared by the briefing agents: LLM statement drafts and verbatim fact extraction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from client_research_agent.models import BriefStatement, CriterionScore, ProvenanceKind, QualificationResult
from client_research_agent.qualification.criteria import get_definition
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.text import (
    citation_markers,
    phrase_pattern,
    split_sentences,
    strip_citation_markers,
    truncate_words,
)

MAX_STATEMENT_CHARS = 600
MAX_FACT_SENTENCE_CHARS = 320


class StatementDraft(BaseModel):
    """One brief statement as proposed by an LLM (validated before use)."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(min_length=1, max_length=2000)
    provenance: Literal["verified_fact", "ai_recommendation"] = "ai_recommendation"
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


def drafts_to_statements(
    drafts: Iterable[StatementDraft],
    *,
    section: str,
    facts_only: bool = False,
    limit: int = 5,
) -> tuple[list[BriefStatement], list[str]]:
    """Convert drafts to ``BriefStatement``; citation validity is checked later by the validator.

    A draft labelled as a fact but citing nothing is dropped (it can never be
    verified). With ``facts_only`` recommendations are dropped too, which keeps
    sections such as the company overview strictly factual.
    """
    statements: list[BriefStatement] = []
    warnings: list[str] = []
    for draft in drafts:
        if len(statements) >= limit:
            warnings.append(f"{section}: truncated to {limit} statements")
            break
        ids = tuple(
            dict.fromkeys(
                (*(i.strip() for i in draft.evidence_ids if i.strip()), *citation_markers(draft.text))
            )
        )
        text = truncate_words(strip_citation_markers(draft.text), MAX_STATEMENT_CHARS)
        if not text:
            continue
        if draft.provenance == "verified_fact":
            if not ids:
                warnings.append(f"{section}: dropped uncited fact: {text[:80]}")
                continue
            statements.append(
                BriefStatement(text=text, provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=ids)
            )
        elif facts_only:
            warnings.append(
                f"{section}: dropped non-factual statement from a facts-only section: {text[:80]}"
            )
        else:
            statements.append(
                BriefStatement(text=text, provenance=ProvenanceKind.AI_RECOMMENDATION, evidence_ids=ids)
            )
    return statements, warnings


def matching_sentence(registry: EvidenceRegistry, evidence_id: str, lexicon: Sequence[str]) -> str | None:
    """Verbatim sentence from the evidence's source text with the most lexicon hits (None if no hit)."""
    patterns = [phrase_pattern(phrase) for phrase in lexicon]
    best: tuple[int, str] | None = None
    for sentence in split_sentences(registry.source_text(evidence_id)):
        hits = sum(1 for pattern in patterns if pattern.search(sentence))
        if hits and (best is None or hits > best[0]):
            best = (hits, sentence)
    if best is None:
        return None
    return truncate_words(" ".join(best[1].split()), MAX_FACT_SENTENCE_CHARS)


def verbatim_fact(
    registry: EvidenceRegistry, evidence_id: str, lexicon: Sequence[str] = ()
) -> BriefStatement:
    """A fact statement that restates evidence verbatim (lexicon-matching sentence, else the quote)."""
    sentence = matching_sentence(registry, evidence_id, lexicon) if lexicon else None
    text = sentence or truncate_words(
        " ".join(registry.require(evidence_id).quote.split()), MAX_FACT_SENTENCE_CHARS
    )
    return BriefStatement(text=text, provenance=ProvenanceKind.VERIFIED_FACT, evidence_ids=(evidence_id,))


def qualification_summary(company: str, result: QualificationResult) -> str:
    """Compact, prompt-friendly summary of a qualification result."""
    lines = [
        f"Company: {company}",
        f"Verdict: {result.verdict.value}",
        f"Weighted score: {result.weighted_score:.2f} / 5; overall confidence "
        f"{result.overall_confidence:.2f}",
        f"Rationale: {result.verdict_rationale}",
    ]
    for score in result.scores:
        lines.append(_score_line(score))
    return "\n".join(lines)


def _score_line(score: CriterionScore) -> str:
    ids = ", ".join(score.evidence_ids) or "none"
    return (
        f"- {get_definition(score.criterion).title}: {score.score}/5 (weight {score.weight:.2f}, "
        f"confidence {score.confidence:.2f}, evidence {ids}): {truncate_words(score.rationale, 300)}"
    )


def statements_summary(label: str, statements: Sequence[BriefStatement]) -> str:
    if not statements:
        return f"{label}: none"
    lines = [f"{label}:"]
    for statement in statements:
        ids = ", ".join(statement.evidence_ids)
        lines.append(f"- [{statement.provenance.value}] {statement.text}" + (f" ({ids})" if ids else ""))
    return "\n".join(lines)
