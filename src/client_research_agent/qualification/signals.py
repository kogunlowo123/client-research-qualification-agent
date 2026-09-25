"""Source trust, recency decay and numeric scale signals.

Trust reflects how authoritative a *public* document type is for factual
claims about a company: regulated filings outrank marketing pages. Recency
uses an exponential half-life so a two-year-old press release still counts,
just less than last quarter's earnings release.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

from client_research_agent.models import DocumentType

SOURCE_TRUST: dict[DocumentType, float] = {
    DocumentType.SEC_FILING: 1.0,
    DocumentType.EARNINGS_RELEASE: 0.95,
    DocumentType.INVESTOR_RELATIONS: 0.9,
    DocumentType.PRESS_RELEASE: 0.8,
    DocumentType.LEADERSHIP_ANNOUNCEMENT: 0.75,
    DocumentType.ANALYST_PUBLIC: 0.7,
    DocumentType.CORPORATE_WEBPAGE: 0.6,
}

UNDATED_RECENCY = 0.5
"""Recency weight for documents without a publication date: neither fresh nor stale."""


def source_trust(document_type: DocumentType) -> float:
    return SOURCE_TRUST.get(document_type, 0.5)


def recency_weight(publication_date: date | None, *, today: date, half_life_days: int = 365) -> float:
    """``0.5 ** (age / half_life)`` clamped to ``[0.05, 1.0]``; future dates count as fresh."""
    if publication_date is None:
        return UNDATED_RECENCY
    age_days = max(0, (today - publication_date).days)
    return max(0.05, min(1.0, math.pow(0.5, age_days / max(1, half_life_days))))


_SCALE = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "mn": 1e6, "m": 1e6}
_MONEY = r"[$€£]\s?(?P<amount>\d[\d,]*(?:\.\d+)?)\s?(?P<unit>trillion|billion|million|bn|mn|b|m)\b"
_REVENUE_BEFORE = re.compile(
    r"(?:revenue|revenues|net sales|sales|turnover)[^.;]{0,80}?" + _MONEY, re.IGNORECASE
)
_REVENUE_AFTER = re.compile(
    _MONEY + r"[^.;]{0,40}?(?:in\s+)?(?:annual\s+)?(?:revenue|revenues|net sales|sales|turnover)",
    re.IGNORECASE,
)
_EMPLOYEES = re.compile(
    r"(?P<count>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(?P<unit>thousand|k)?\s?\+?\s*"
    r"(?:full[- ]time\s+)?(?:employees|staff|workers|team members|associates|people worldwide)",
    re.IGNORECASE,
)
_WORKFORCE = re.compile(
    r"(?:workforce|headcount)\s+of\s+(?:approximately\s+|about\s+|over\s+|more than\s+)?"
    r"(?P<count>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(?P<unit>thousand|k)?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ScaleSignals:
    revenue_usd: float | None
    employees: int | None

    @property
    def found(self) -> bool:
        return self.revenue_usd is not None or self.employees is not None


def _money(amount: str, unit: str) -> float:
    return float(amount.replace(",", "")) * _SCALE[unit.lower()]


def extract_scale_signals(text: str) -> ScaleSignals:
    """Largest revenue and employee figures stated in ``text`` (None when absent)."""
    revenues = [
        _money(match.group("amount"), match.group("unit"))
        for pattern in (_REVENUE_BEFORE, _REVENUE_AFTER)
        for match in pattern.finditer(text)
    ]
    headcounts: list[int] = []
    for pattern in (_EMPLOYEES, _WORKFORCE):
        for match in pattern.finditer(text):
            count = float(match.group("count").replace(",", ""))
            if (match.group("unit") or "").lower() in {"thousand", "k"}:
                count *= 1000
            headcounts.append(int(count))
    return ScaleSignals(
        revenue_usd=max(revenues) if revenues else None,
        employees=max(headcounts) if headcounts else None,
    )


def revenue_band(revenue_usd: float) -> int:
    """Map annual revenue to the 0-5 scale rubric."""
    for threshold, band in ((10e9, 5), (1e9, 4), (250e6, 3), (50e6, 2)):
        if revenue_usd >= threshold:
            return band
    return 1 if revenue_usd > 0 else 0


def employee_band(employees: int) -> int:
    """Map headcount to the 0-5 scale rubric."""
    for threshold, band in ((50_000, 5), (10_000, 4), (1_000, 3), (200, 2)):
        if employees >= threshold:
            return band
    return 1 if employees > 0 else 0
