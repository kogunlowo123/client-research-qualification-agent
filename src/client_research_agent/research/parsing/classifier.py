"""Rule-based document-type classification.

Each rule inspects one field (URL, title or the opening of the body text) and
votes for a :class:`DocumentType` with a weight. Title and URL evidence is
weighted higher than body text because body text of a press release routinely
mentions earnings, filings and executives in passing. Specific types beat the
generic press-release bucket on ties via ``_PRIORITY``. Confidence grows with
accumulated evidence and is capped below certainty; the fallback
``CORPORATE_WEBPAGE`` carries a deliberately low confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from client_research_agent.models import DocumentType
from client_research_agent.research.url_guard import host_matches

Field = Literal["url", "title", "text", "host"]


@dataclass(frozen=True, slots=True)
class Rule:
    document_type: DocumentType
    field: Field
    pattern: re.Pattern[str]
    weight: float
    label: str


@dataclass(frozen=True, slots=True)
class Classification:
    document_type: DocumentType
    confidence: float
    signals: tuple[str, ...] = ()


def _rule(document_type: DocumentType, field: Field, pattern: str, weight: float, label: str) -> Rule:
    return Rule(document_type, field, re.compile(pattern, re.IGNORECASE), weight, label)


_SEC = DocumentType.SEC_FILING
_EARN = DocumentType.EARNINGS_RELEASE
_LEAD = DocumentType.LEADERSHIP_ANNOUNCEMENT
_IR = DocumentType.INVESTOR_RELATIONS
_PR = DocumentType.PRESS_RELEASE
_AN = DocumentType.ANALYST_PUBLIC
_EXEC_TITLE = (
    r"(?:chief\s+\w+(?:\s+\w+)?\s+officer|\bC[EFITOD]O\b|\bCISO\b|\bCAIO\b|president|"
    r"chair(?:man|woman|person)?|general counsel|head of)"
)

RULES: tuple[Rule, ...] = (
    _rule(_SEC, "url", r"/Archives/edgar/|/cgi-bin/browse-edgar", 4.0, "url:edgar"),
    _rule(_SEC, "title", r"\b(?:form\s+)?(?:10-K|10-Q|8-K|20-F|6-K|S-1|DEF\s*14A)\b", 3.0, "title:form"),
    _rule(_SEC, "text", r"securities and exchange commission", 1.5, "text:sec"),
    _rule(
        _SEC, "text", r"pursuant to section 13 or 15\(d\)|commission file number", 2.0, "text:exchange-act"
    ),
    _rule(_SEC, "text", r"\bform\s+(?:10-K|10-Q|8-K)\b", 1.0, "text:form"),
    _rule(_EARN, "url", r"earnings|quarterly-results|financial-results|/results/", 2.0, "url:earnings"),
    _rule(
        _EARN,
        "title",
        r"(?:first|second|third|fourth|\bQ[1-4]\b)[\w\s,-]{0,40}(?:results|earnings)|"
        r"(?:fiscal|full)[\s-]year[\w\s,-]{0,30}results|quarterly results|reports?[\w\s,-]{0,60}results",
        3.0,
        "title:results",
    ),
    _rule(_EARN, "text", r"diluted (?:earnings per share|EPS)|\bnon-GAAP\b", 1.5, "text:eps"),
    _rule(
        _EARN, "text", r"(?:total )?revenue (?:of|was|grew|increased|rose)[^.]{0,40}\$", 1.0, "text:revenue"
    ),
    _rule(_EARN, "text", r"(?:conference call|webcast)[^.]{0,80}(?:results|earnings)", 1.0, "text:call"),
    _rule(_LEAD, "url", r"leadership|/management|executive-team|/board", 1.5, "url:leadership"),
    _rule(
        _LEAD,
        "title",
        rf"\b(?:appoints?|names?|hires?|promotes?|welcomes?)\b[^|]{{0,80}}{_EXEC_TITLE}|"
        rf"{_EXEC_TITLE}[^|]{{0,60}}\b(?:appointed|named|joins|to step down|retire)",
        3.0,
        "title:appointment",
    ),
    _rule(
        _LEAD,
        "text",
        rf"\b(?:appointed|named|promoted to)\b[^.]{{0,80}}{_EXEC_TITLE}|"
        rf"\bjoins\b[^.]{{0,40}}as {_EXEC_TITLE}",
        1.5,
        "text:appointment",
    ),
    _rule(_IR, "url", r"(?:^|[/.])(?:investors?|ir|shareholders?|stockholders?)(?:[/.-]|$)", 2.5, "url:ir"),
    _rule(
        _IR, "title", r"investor relations|shareholder|annual (?:general )?meeting|dividend", 2.0, "title:ir"
    ),
    _rule(_IR, "text", r"investor relations|stock information|analyst coverage|sec filings", 1.0, "text:ir"),
    _rule(_PR, "url", r"/news|/press|newsroom|/media|press-release", 1.5, "url:news"),
    _rule(
        _PR, "title", r"\bannounces?\b|\blaunch(?:es)?\b|\bunveils?\b|\bpartners? with\b", 1.0, "title:news"
    ),
    _rule(
        _PR,
        "text",
        r"PRNewswire|Business Wire|GLOBE NEWSWIRE|ACCESSWIRE|press release|\(NYSE:|\(NASDAQ:",
        1.5,
        "text:wire",
    ),
    _rule(_AN, "host", r"(?:^|\.)(?:gartner|forrester|idc)\.com$", 4.0, "host:analyst"),
)

_PRIORITY = (_SEC, _AN, _EARN, _LEAD, _IR, _PR)
_TEXT_WINDOW = 3000


class DocumentTypeClassifier:
    def __init__(self, rules: tuple[Rule, ...] = RULES, *, min_score: float = 1.0) -> None:
        self._rules = rules
        self._min_score = min_score

    def classify(
        self,
        url: str,
        title: str,
        text: str,
        *,
        hint: DocumentType | None = None,
    ) -> Classification:
        """Classify a document. ``hint`` (e.g. from the source that produced the URL) wins when given."""
        parts = urlsplit(url)
        fields: dict[Field, str] = {
            "url": f"{parts.netloc}{parts.path}",
            "title": title,
            "text": text[:_TEXT_WINDOW],
            "host": (parts.hostname or "").lower(),
        }
        scores: dict[DocumentType, float] = {}
        signals: dict[DocumentType, list[str]] = {}
        for rule in self._rules:
            if rule.pattern.search(fields[rule.field]):
                scores[rule.document_type] = scores.get(rule.document_type, 0.0) + rule.weight
                signals.setdefault(rule.document_type, []).append(rule.label)
        if host_matches(fields["host"], "sec.gov") and "url:edgar" not in signals.get(_SEC, []):
            scores[_SEC] = scores.get(_SEC, 0.0) + 2.0
            signals.setdefault(_SEC, []).append("host:sec.gov")

        if hint is not None:
            evidence = scores.get(hint, 0.0)
            return Classification(hint, _confidence(max(evidence, 3.0)), ("hint", *signals.get(hint, [])))
        if not scores or max(scores.values()) < self._min_score:
            return Classification(DocumentType.CORPORATE_WEBPAGE, 0.4, ())
        best = max(scores.values())
        winner = next(t for t in _PRIORITY if scores.get(t, 0.0) == best)
        return Classification(winner, _confidence(best), tuple(signals[winner]))


def _confidence(score: float) -> float:
    return round(min(0.95, 0.45 + 0.1 * score), 3)
