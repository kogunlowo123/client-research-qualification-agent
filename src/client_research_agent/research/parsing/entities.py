"""Deterministic entity extraction (regex + gazetteer, no ML dependency).

Extracts the facts qualification cares about: monetary amounts, percentages,
fiscal periods, headcount, executive appointments/titles and technology
terms. Being rule-based keeps it fast, explainable and identical across
environments; the output feeds chunk metadata (``Chunk.entities``) and
keyword retrieval rather than being treated as ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, TypeVar

T = TypeVar("T")

_SCALES: Final[dict[str, float]] = {
    "trillion": 1e12,
    "tn": 1e12,
    "t": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mm": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}
_CURRENCIES: Final[dict[str, str]] = {
    "us$": "USD",
    "$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
}
_MONEY = re.compile(
    r"(?P<cur>US\$|\$|€|£|\b(?:USD|EUR|GBP)\s?)"
    r"(?P<num>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?:\s?(?P<scale>trillion|billion|million|thousand|bn|tn|mm|mn|[TBMK])\b)?",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?<![\w.])(?P<num>-?\d{1,3}(?:\.\d+)?)\s?(?:%|percent\b|per cent\b)", re.IGNORECASE)
_ORDINAL_QUARTER = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_QUARTER = re.compile(
    r"\b(?:Q(?P<q>[1-4])|(?P<word>first|second|third|fourth)\s+quarter)"
    r"(?:\s+of)?(?:\s+(?:fiscal(?:\s+year)?|FY))?\s*'?(?P<year>(?:19|20)\d{2}|\d{2})\b",
    re.IGNORECASE,
)
_FISCAL_YEAR = re.compile(
    r"\b(?:FY\s?'?(?P<fy>(?:19|20)\d{2}|\d{2})|fiscal(?:\s+year)?\s+(?P<year>(?:19|20)\d{2}))\b"
)
_HEADCOUNT = re.compile(
    r"(?:(?:approximately|about|over|more than|nearly|roughly|some|around)\s+)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<scale>thousand|million|k)?\s+"
    r"(?:full-time\s+|global\s+|dedicated\s+)?(?:employees|team members|associates|staff members|workers)\b"
    r"|workforce of\s+(?:approximately\s+|about\s+|over\s+|more than\s+)?"
    r"(?P<num2>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<scale2>thousand|million|k)?",
    re.IGNORECASE,
)

_TITLE_ACRONYMS: Final[dict[str, str]] = {
    "CEO": "Chief Executive Officer",
    "CFO": "Chief Financial Officer",
    "CTO": "Chief Technology Officer",
    "CIO": "Chief Information Officer",
    "CDO": "Chief Data Officer",
    "COO": "Chief Operating Officer",
    "CISO": "Chief Information Security Officer",
    "CMO": "Chief Marketing Officer",
    "CAIO": "Chief AI Officer",
    "CDAO": "Chief Data and Analytics Officer",
}
_TITLE = (
    r"(?:Chief(?:\s+(?:[A-Z][a-z]+|AI|and|&)){1,4}\s+Officer|C(?:EO|FO|TO|IO|DO|OO|ISO|MO|AIO|DAO)\b|"
    r"(?:Executive\s+|Senior\s+)?Vice\s+President(?:\s+(?:of|and)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)?|"
    r"President|Chair(?:man|woman|person)?|General\s+Counsel|Head\s+of\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)"
)
_NAME_PART = r"(?:[A-Z]['\u2019]|Ma?c)?[A-Z][a-z]+(?:-[A-Z][a-z]+)?"
_NAME = rf"{_NAME_PART}(?:\s[A-Z]\.)?(?:\s(?:van|de|von|da)\b)?(?:\s{_NAME_PART}){{1,2}}"
_EXEC_PATTERNS = (
    re.compile(
        rf"\b(?:appoint(?:s|ed|ment\s+of)|nam(?:es|ed)|hir(?:es|ed)|promot(?:es|ed))\s+(?P<name>{_NAME})\s+"
        rf"(?:as|to\s+(?:the\s+(?:role|position)\s+of\s+)?)?\s*(?:its|the|our)?\s*(?:new\s+)?(?P<title>{_TITLE})"
    ),
    re.compile(
        rf"(?P<name>{_NAME})\s+(?:has\s+been|was|is|will\s+be)\s+(?:appointed|named|promoted)\s+"
        rf"(?:as\s+|to\s+)?(?:the\s+|its\s+)?(?:new\s+)?(?P<title>{_TITLE})"
    ),
    re.compile(
        rf"(?P<name>{_NAME}),\s+(?:the\s+company's\s+|our\s+|its\s+)?(?:new\s+|incoming\s+)?(?P<title>{_TITLE})"
    ),
    re.compile(rf"\b(?P<title>{_TITLE})\s+(?P<name>{_NAME})\b"),
)
_NAME_STOPWORDS = frozenset(
    {
        "The", "Our", "Its", "This", "Company", "Board", "Group", "Inc", "Corporation", "Today", "Global",
        "Senior", "Executive", "Vice", "Chief", "President", "New", "Former", "Interim", "Acting",
    }
)  # fmt: skip

_TECH_GAZETTEER: Final[tuple[tuple[str, str, bool], ...]] = (
    ("Databricks", r"\bDatabricks\b", False),
    ("Snowflake", r"\bSnowflake\b", False),
    ("AWS", r"\bAWS\b|\bAmazon Web Services\b", True),
    ("Azure", r"\b(?:Microsoft\s+)?Azure\b", False),
    ("GCP", r"\bGCP\b|\bGoogle Cloud(?: Platform)?\b", True),
    ("SAP", r"\bSAP\b|\bS/4\s?HANA\b", True),
    ("Salesforce", r"\bSalesforce\b", False),
    ("ServiceNow", r"\bServiceNow\b", False),
    ("Oracle", r"\bOracle\b", True),
    ("Workday", r"\bWorkday\b", True),
    ("Kubernetes", r"\bKubernetes\b", False),
    ("Generative AI", r"\bgenerative\s+AI\b|\bGenAI\b|\bgen\s?AI\b", False),
    ("Agentic AI", r"\bagentic\s+AI\b|\bAI\s+agents?\b", False),
    ("LLM", r"\bLLMs?\b|\blarge\s+language\s+models?\b", False),
    ("Artificial intelligence", r"\bartificial\s+intelligence\b|\bAI\b", False),
    ("Machine learning", r"\bmachine\s+learning\b|\bML\b", False),
    ("MLOps", r"\bMLOps\b|\bLLMOps\b", False),
    ("Data lakehouse", r"\b(?:data\s+)?lakehouse\b", False),
    ("Data warehouse", r"\bdata\s+warehous(?:e|ing)\b", False),
    ("Data platform", r"\bdata\s+(?:platform|foundation|mesh|fabric)\b", False),
    ("Data governance", r"\bdata\s+governance\b|\bUnity\s+Catalog\b", False),
    ("Analytics", r"\b(?:advanced\s+)?analytics\b", False),
    ("Cloud migration", r"\bcloud\s+migration\b|\bmigrat\w*\s+(?:\w+\s+){0,3}to\s+the\s+cloud\b", False),
    ("Cloud", r"\bcloud\b", False),
    ("ERP modernization", r"\bERP\s+(?:modernization|transformation|upgrade|migration)\b", False),
    ("ERP", r"\bERP\b", True),
    ("Digital transformation", r"\bdigital\s+transformation\b", False),
    ("Automation", r"\b(?:intelligent\s+)?automation\b|\bRPA\b", False),
    ("Cybersecurity", r"\bcyber\s?security\b|\bzero\s+trust\b", False),
    ("Modernization", r"\b(?:legacy|application|IT|technology)\s+modernization\b", False),
)
_TECH_PATTERNS: Final = tuple(
    (label, re.compile(pattern, 0 if case_sensitive else re.IGNORECASE))
    for label, pattern, case_sensitive in _TECH_GAZETTEER
)
_MAX_FLAT_ENTITIES = 60


@dataclass(frozen=True, slots=True)
class MonetaryAmount:
    text: str
    value: float
    currency: str


@dataclass(frozen=True, slots=True)
class Percentage:
    text: str
    value: float


@dataclass(frozen=True, slots=True)
class Headcount:
    text: str
    value: int


@dataclass(frozen=True, slots=True)
class ExecutiveMention:
    name: str
    title: str

    def __str__(self) -> str:
        return f"{self.name} ({self.title})"


@dataclass(frozen=True, slots=True)
class ExtractedEntities:
    monetary_amounts: tuple[MonetaryAmount, ...] = ()
    percentages: tuple[Percentage, ...] = ()
    fiscal_periods: tuple[str, ...] = ()
    headcounts: tuple[Headcount, ...] = ()
    executives: tuple[ExecutiveMention, ...] = ()
    technologies: tuple[str, ...] = ()
    technology_mentions: dict[str, int] = field(default_factory=dict)
    entities: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.entities


def _to_number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _two_digit_year(raw: str) -> int:
    year = int(raw)
    return 2000 + year if year < 100 else year


def _normalize_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip()
    return _TITLE_ACRONYMS.get(title.upper(), title) if len(title) <= 5 else title


def _dedupe(items: list[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(items))


class EntityExtractor:
    def __init__(self, *, max_text_chars: int = 200_000) -> None:
        self._max_chars = max_text_chars

    def extract(self, text: str) -> ExtractedEntities:
        text = text[: self._max_chars]
        money = self._money(text)
        percentages = self._percentages(text)
        periods = self._fiscal_periods(text)
        headcounts = self._headcounts(text)
        executives = self._executives(text)
        mentions = self._technologies(text)
        technologies = tuple(mentions)
        flat = _dedupe(
            [
                *technologies,
                *(str(e) for e in executives),
                *periods,
                *(h.text for h in headcounts),
                *(m.text for m in money),
                *(p.text for p in percentages),
            ]
        )[:_MAX_FLAT_ENTITIES]
        return ExtractedEntities(
            monetary_amounts=money,
            percentages=percentages,
            fiscal_periods=periods,
            headcounts=headcounts,
            executives=executives,
            technologies=technologies,
            technology_mentions=mentions,
            entities=flat,
        )

    @staticmethod
    def _money(text: str) -> tuple[MonetaryAmount, ...]:
        found: list[MonetaryAmount] = []
        for match in _MONEY.finditer(text):
            number = _to_number(match.group("num"))
            if match.group("dec"):
                number += float("0." + match.group("dec"))
            scale = (match.group("scale") or "").lower()
            value = number * _SCALES.get(scale, 1.0)
            currency = _CURRENCIES.get(match.group("cur").strip().lower(), "USD")
            found.append(MonetaryAmount(re.sub(r"\s+", " ", match.group(0)).strip(), value, currency))
        return _dedupe(found)

    @staticmethod
    def _percentages(text: str) -> tuple[Percentage, ...]:
        return _dedupe(
            [
                Percentage(re.sub(r"\s+", " ", m.group(0)), float(m.group("num")))
                for m in _PERCENT.finditer(text)
            ]
        )

    @staticmethod
    def _fiscal_periods(text: str) -> tuple[str, ...]:
        periods: list[str] = []
        for match in _QUARTER.finditer(text):
            quarter = match.group("q") or str(_ORDINAL_QUARTER[match.group("word").lower()])
            periods.append(f"Q{quarter} FY{_two_digit_year(match.group('year'))}")
        for match in _FISCAL_YEAR.finditer(text):
            year = match.group("fy") or match.group("year")
            periods.append(f"FY{_two_digit_year(year)}")
        return _dedupe(periods)

    @staticmethod
    def _headcounts(text: str) -> tuple[Headcount, ...]:
        found: list[Headcount] = []
        for match in _HEADCOUNT.finditer(text):
            raw = match.group("num") or match.group("num2")
            scale = (match.group("scale") or match.group("scale2") or "").lower()
            value = _to_number(raw) * _SCALES.get(scale, 1.0)
            if value >= 1:
                found.append(Headcount(re.sub(r"\s+", " ", match.group(0)).strip(), int(value)))
        return _dedupe(found)

    @staticmethod
    def _executives(text: str) -> tuple[ExecutiveMention, ...]:
        found: dict[str, ExecutiveMention] = {}
        for pattern in _EXEC_PATTERNS:
            for match in pattern.finditer(text):
                name = re.sub(r"\s+", " ", match.group("name")).strip()
                tokens = name.split()
                if tokens[0] in _NAME_STOPWORDS or any(t in _NAME_STOPWORDS for t in tokens[1:]):
                    continue
                if name not in found:
                    found[name] = ExecutiveMention(name, _normalize_title(match.group("title")))
        return tuple(found.values())

    @staticmethod
    def _technologies(text: str) -> dict[str, int]:
        hits: list[tuple[int, str, int]] = []
        for label, pattern in _TECH_PATTERNS:
            matches = list(pattern.finditer(text))
            if matches:
                hits.append((matches[0].start(), label, len(matches)))
        hits.sort()
        return {label: count for _, label, count in hits}
