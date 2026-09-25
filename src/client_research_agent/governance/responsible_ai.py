"""Responsible-AI policy checks on a finished :class:`ClientBrief`.

Rules
-----
* ``grounding``            every VERIFIED_FACT cites evidence present in the brief.
* ``recommendation_label`` advice ("we recommend", "should consider", next actions) is
  labelled AI_RECOMMENDATION, never presented as a verified fact.
* ``protected_attributes`` no reasoning about individuals based on protected
  characteristics (age, race, ethnicity, religion, gender, sexual orientation,
  disability, pregnancy, national origin, marital status, health, politics).
* ``personal_speculation`` no speculation about individuals' private lives.

Findings with severity ``error`` fail the brief; ``warning`` findings are
surfaced to the reader. :meth:`ResponsibleAIPolicy.disclaimer` renders the
standard AI-generated-content notice.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from client_research_agent.models import BriefStatement, ClientBrief, ProvenanceKind


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PolicyFinding:
    rule: str
    severity: Severity
    message: str
    location: str
    text: str = ""


@dataclass(frozen=True, slots=True)
class PolicyReport:
    findings: tuple[PolicyFinding, ...]
    disclaimer: str
    verified_facts: int
    recommendations: int

    @property
    def passed(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)

    @property
    def errors(self) -> tuple[PolicyFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[PolicyFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)


_INDIVIDUAL_RE = re.compile(
    r"\b(?:he|she|his|her|him|hers|ceo|cfo|cto|cio|coo|ciso|cdo|founder|co-founder|chair(?:man|woman|person)?|"
    r"executive|president|director|officer|vp|vice\s+president|head\s+of|leader|manager|board\s+member)\b",
    re.IGNORECASE,
)
_PERSON = (
    r"(?:he|she|they|ceo|cfo|cto|cio|coo|ciso|founder|co-founder|chair(?:man|woman|person)?|executive|president|"
    r"director|officer|leader|manager|(?-i:[A-Z][a-z]+))"
)
_PROTECTED_ATTRIBUTE_NOUN = (
    r"(?:age|race|ethnicity|ethnic\s+background|skin\s+colou?r|religion|religious\s+beliefs|faith|gender|sex|"
    r"sexual\s+orientation|disability|pregnancy|nationality|national\s+origin|immigration\s+status|accent|"
    r"marital\s+status|health|medical\s+condition|political\s+(?:views|affiliation|leanings))"
)
_PROTECTED_ADJECTIVE = (
    r"(?:\d{2}[- ]years?[- ]old|elderly|too\s+old|too\s+young|old|young|muslim|christian|jewish|hindu|"
    r"buddhist|sikh|atheist|catholic|gay|lesbian|bisexual|transgender|disabled|pregnant|an?\s+immigrant|"
    r"married|divorced|single|female|male|a\s+woman|a\s+man|black|white|asian|hispanic|latin[oa]|"
    r"foreign-born|sick|ill)"
)
# A company can have an "age" or "financial health"; these nouns only count after personal pronouns.
_NAME_SAFE_NOUN = (
    r"(?:race|ethnicity|ethnic\s+background|skin\s+colou?r|religion|religious\s+beliefs|gender|"
    r"sexual\s+orientation|disability|pregnancy|nationality|national\s+origin|immigration\s+status|"
    r"marital\s+status|medical\s+condition|political\s+(?:views|affiliation|leanings))"
)
# Possessive forms identify an individual on their own and are checked without the individual gate.
_POSSESSIVE_PATTERNS = (
    re.compile(rf"\b(?:his|her)\s+(?:\w+\s+)?{_PROTECTED_ATTRIBUTE_NOUN}\b", re.IGNORECASE),
    re.compile(rf"\b(?-i:[A-Z][a-z]+)'s\s+(?:\w+\s+)?{_NAME_SAFE_NOUN}\b", re.IGNORECASE),
)
_PROTECTED_PATTERNS = (
    re.compile(
        rf"\b{_PERSON}\s+(?:is|was|being|seems|appears)\s+(?:\w+\s+)?{_PROTECTED_ADJECTIVE}\b", re.IGNORECASE
    ),
    re.compile(
        rf"\b(?:is|was)\s+(?:the\s+|our\s+|their\s+)?{_PERSON}\s+(?:\w+\s+)?{_PROTECTED_ADJECTIVE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{2}[- ]years?[- ]old\s+(?:ceo|cfo|cto|cio|founder|executive|chair\w*|president|director)\b|"
        r"\b(?:nearing|approaching|close\s+to)\s+retirement\b",
        re.IGNORECASE,
    ),
)
_PERSONAL_LIFE_RE = re.compile(
    r"\b(?:personal\s+life|private\s+life|family\s+(?:life|problems|issues)|wife|husband|spouse|partner'?s|"
    r"children|kids|divorce|dating|affair|relationship|home\s+life|lifestyle|hobbies|vacation|health|illness|"
    r"medical|drinking|addiction|personal\s+finances|debts?)\b",
    re.IGNORECASE,
)
_SPECULATION_RE = re.compile(
    r"\b(?:likely|probably|possibly|may\s+be|might|could\s+be|rumou?r(?:ed|s)?|reportedly|allegedly|seems|"
    r"appears|suggests?|speculat\w*|presumably|we\s+suspect|perhaps|gossip)\b",
    re.IGNORECASE,
)
_ADVICE_RE = re.compile(
    r"\b(?:we\s+recommend|recommend(?:ed|s)?\s+(?:that|to)|should\s+(?:consider|prioriti[sz]e|engage|pursue|"
    r"position|target|lead|explore|propose)|we\s+suggest|consider\s+(?:positioning|proposing|offering|leading)|"
    r"propose\s+(?:a|an|that)|next\s+step\s+(?:is|should))\b",
    re.IGNORECASE,
)

DISCLAIMER_TEMPLATE = (
    "AI-generated research brief for {company} (run {run_id}, generated {generated}). "
    "It contains {facts} verified statement(s) citing {sources} public source(s) and {recs} AI "
    "recommendation(s), which are opinions of the model, not facts. Facts were extracted from public web "
    "sources that may be incomplete or out of date; verify material claims against the cited sources before "
    "acting. The brief does not assess individuals and must not be used for decisions about people."
)


class ResponsibleAIPolicy:
    RECOMMENDATION_SECTIONS = ("recommended_next_actions",)

    def evaluate(self, brief: ClientBrief) -> PolicyReport:
        findings: list[PolicyFinding] = []
        evidence_ids = {e.evidence_id for e in brief.evidence}
        for section, index, statement in self._statements(brief):
            location = f"{section}[{index}]"
            findings.extend(self._grounding(statement, evidence_ids, location))
            findings.extend(self._labelling(statement, section, location))
            findings.extend(self.check_text(statement.text, location))
        for index, question in enumerate(brief.discovery_questions):
            findings.extend(self.check_text(question, f"discovery_questions[{index}]"))
        return PolicyReport(
            findings=tuple(findings),
            disclaimer=self.disclaimer(brief),
            verified_facts=len(brief.verified_facts),
            recommendations=len(brief.recommendations),
        )

    def check_text(self, text: str, location: str = "text") -> list[PolicyFinding]:
        """Protected-attribute and personal-speculation rules for a single string."""
        findings: list[PolicyFinding] = []
        individual = bool(_INDIVIDUAL_RE.search(text))
        patterns = _POSSESSIVE_PATTERNS + (_PROTECTED_PATTERNS if individual else ())
        protected = next((m for m in (p.search(text) for p in patterns) if m), None)
        if protected:
            findings.append(
                PolicyFinding(
                    "protected_attributes",
                    Severity.ERROR,
                    f"references protected characteristic '{protected.group(0)}' of an individual",
                    location,
                    text,
                )
            )
        personal = _PERSONAL_LIFE_RE.search(text)
        if individual and personal and _SPECULATION_RE.search(text):
            findings.append(
                PolicyFinding(
                    "personal_speculation",
                    Severity.ERROR,
                    f"speculates about an individual's private life ('{personal.group(0)}')",
                    location,
                    text,
                )
            )
        return findings

    def disclaimer(self, brief: ClientBrief) -> str:
        return DISCLAIMER_TEMPLATE.format(
            company=brief.company,
            run_id=brief.run_id,
            generated=brief.generated_at.date().isoformat(),
            facts=len(brief.verified_facts),
            sources=len({e.url for e in brief.evidence}),
            recs=len(brief.recommendations),
        )

    @staticmethod
    def _statements(brief: ClientBrief) -> Iterable[tuple[str, int, BriefStatement]]:
        sections: dict[str, tuple[BriefStatement, ...]] = {
            "company_overview": brief.company_overview,
            "technology_priorities": brief.technology_priorities,
            "gartner_relevant_insights": brief.gartner_relevant_insights,
            "opportunities": brief.opportunities,
            "risks": brief.risks,
            "executive_summary": brief.executive_summary,
            "executive_talking_points": brief.executive_talking_points,
            "recommended_next_actions": brief.recommended_next_actions,
        }
        for name, statements in sections.items():
            for index, statement in enumerate(statements):
                yield name, index, statement

    @staticmethod
    def _grounding(statement: BriefStatement, evidence_ids: set[str], location: str) -> list[PolicyFinding]:
        if statement.provenance is not ProvenanceKind.VERIFIED_FACT:
            return []
        missing = [e for e in statement.evidence_ids if e not in evidence_ids]
        if not statement.evidence_ids or missing:
            return [
                PolicyFinding(
                    "grounding",
                    Severity.ERROR,
                    f"verified fact cites unknown evidence: {', '.join(missing) or 'none'}",
                    location,
                    statement.text,
                )
            ]
        return []

    def _labelling(self, statement: BriefStatement, section: str, location: str) -> list[PolicyFinding]:
        if statement.provenance is ProvenanceKind.AI_RECOMMENDATION:
            return []
        if section in self.RECOMMENDATION_SECTIONS:
            return [
                PolicyFinding(
                    "recommendation_label",
                    Severity.ERROR,
                    "next actions must be labelled as AI recommendations",
                    location,
                    statement.text,
                )
            ]
        if _ADVICE_RE.search(statement.text):
            return [
                PolicyFinding(
                    "recommendation_label",
                    Severity.WARNING,
                    "advisory language presented as a verified fact",
                    location,
                    statement.text,
                )
            ]
        return []
