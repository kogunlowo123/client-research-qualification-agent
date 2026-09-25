"""Lexical entailment: a cheap, deterministic grounding check for one claim against one source.

The support score is the fraction of the claim's content words found in the
evidence, then adjusted by two hard consistency rules:

* **numbers** - every number in the claim (amounts with magnitude applied,
  percentages, years) must appear by value in the evidence; any unmatched
  number caps the score at ``NUMBER_MISMATCH_CAP``, far below any sensible
  support threshold, because a wrong figure is the most damaging kind of
  hallucination in a client brief;
* **entities** - capitalised names and acronyms in the claim must occur in
  the evidence; each missing entity scales the score down proportionally and
  any missing entity caps the score at ``ENTITY_MISMATCH_CAP`` (below the
  default support threshold), since a claim about the wrong product, firm or
  person is not supported even when the rest of the wording overlaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from client_research_agent.qualification.text import (
    content_words,
    extract_entities,
    strip_citation_markers,
    unmatched_numbers,
)

NUMBER_MISMATCH_CAP = 0.05
ENTITY_PENALTY = 0.5
ENTITY_MISMATCH_CAP = 0.25


@dataclass(frozen=True, slots=True)
class SupportAssessment:
    score: float
    coverage: float
    missing_numbers: tuple[str, ...]
    missing_entities: tuple[str, ...]

    @property
    def numbers_consistent(self) -> bool:
        return not self.missing_numbers

    def reason(self) -> str:
        parts = [f"{self.coverage:.0%} of claim terms found in evidence"]
        if self.missing_numbers:
            parts.append(f"figures not in evidence: {', '.join(self.missing_numbers[:5])}")
        if self.missing_entities:
            parts.append(f"names not in evidence: {', '.join(self.missing_entities[:5])}")
        return "; ".join(parts)


def _entity_present(entity: str, evidence_lower: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(entity) + r"(?![a-z0-9])", evidence_lower) is not None


def lexical_support(claim: str, evidence_text: str) -> SupportAssessment:
    cleaned = strip_citation_markers(claim)
    claim_terms = content_words(cleaned)
    evidence_terms = content_words(evidence_text)
    coverage = len(claim_terms & evidence_terms) / len(claim_terms) if claim_terms else 0.0
    missing_numbers = tuple(unmatched_numbers(cleaned, evidence_text))
    evidence_lower = evidence_text.lower()
    entities = sorted(extract_entities(cleaned))
    missing_entities = tuple(e for e in entities if not _entity_present(e, evidence_lower))
    score = coverage
    if entities:
        score *= 1.0 - ENTITY_PENALTY * len(missing_entities) / len(entities)
    if missing_entities:
        score = min(score, ENTITY_MISMATCH_CAP)
    if missing_numbers:
        score = min(score, NUMBER_MISMATCH_CAP)
    return SupportAssessment(
        score=round(max(0.0, min(1.0, score)), 4),
        coverage=round(coverage, 4),
        missing_numbers=missing_numbers,
        missing_entities=missing_entities,
    )
