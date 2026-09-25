"""Deterministic text analysis shared by qualification, ranking and citation validation.

Everything here is pure, dependency-free and fast enough to run on every chunk
and every brief statement. The functions intentionally favour precision over
recall: they are used to *verify* grounded claims, so a false "supported" is
far more costly than a false "unsupported".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had
    has have having he her here hers herself him himself his how i if in into is it its itself just
    me more most my myself no nor not now of off on once only or other our ours ourselves out over
    own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who
    whom why will with would you your yours yourself yourselves across per via within without
    including new company companies inc corp corporation ltd plc llc co group year years
    """.split()  # noqa: SIM905 - readable word list
)

_CITATION_MARKER = re.compile(r"\[\s*E\d+(?:\s*,\s*E\d+)*\s*\]")
_MARKER_WITH_SPACE = re.compile(r"\s*" + _CITATION_MARKER.pattern)
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[$])|\n\s*\n|\n(?=\s*[-*•])")
_ENTITY = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&]*[A-Z0-9][a-zA-Z0-9&]*|[A-Z][a-z0-9]{2,}(?:[A-Z][a-z0-9]+)*)\b")
_NUMBER = re.compile(
    r"(?P<currency>[$€£])?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<unit>%|percent\b|per\s?cent\b|trillion\b|billion\b|million\b|thousand\b|bn\b|mn\b|tn\b|[kKmMbB]\b))?"
)
_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("izations", "iz"),
    ("ization", "iz"),
    ("isations", "is"),
    ("isation", "is"),
    ("ations", "at"),
    ("ation", "at"),
    ("ments", ""),
    ("ment", ""),
    ("ings", ""),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ers", "er"),
    ("ed", ""),
    ("es", ""),
    ("s", ""),
)
_SCALE_WORDS: dict[str, float] = {
    "trillion": 1e12,
    "tn": 1e12,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "thousand": 1e3,
    "k": 1e3,
}


def strip_citation_markers(text: str) -> str:
    """Remove ``[E3]`` / ``[E1, E2]`` markers so they are not mistaken for claim content."""
    return re.sub(r"\s{2,}", " ", _MARKER_WITH_SPACE.sub("", text)).strip()


def citation_markers(text: str) -> tuple[str, ...]:
    """Evidence ids referenced inline in ``text`` in first-seen order."""
    ids: list[str] = []
    for match in _CITATION_MARKER.finditer(text):
        for token in re.findall(r"E\d+", match.group(0)):
            if token not in ids:
                ids.append(token)
    return tuple(ids)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def stem(word: str) -> str:
    """Very light suffix stripping: 'modernize', 'modernizing' and 'modernization' share a stem."""
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: len(word) - len(suffix)] + replacement
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def content_words(text: str) -> set[str]:
    """Stemmed, non-numeric, non-stopword tokens of at least three characters."""
    return {
        stem(token)
        for token in tokenize(strip_citation_markers(text))
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences in ``text`` (whitespace trimmed, empty sentences skipped)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    trimmed: list[tuple[int, int]] = []
    for raw_begin, raw_end in spans:
        begin, end = raw_begin, raw_end
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            trimmed.append((begin, end))
    return trimmed


def split_sentences(text: str) -> list[str]:
    return [text[begin:end] for begin, end in sentence_spans(text)]


class NumberKind(StrEnum):
    PERCENT = "percent"
    AMOUNT = "amount"


@dataclass(frozen=True, slots=True)
class NumberMention:
    value: float
    kind: NumberKind
    raw: str

    def matches(self, other: NumberMention, *, tolerance: float = 1e-6) -> bool:
        """Same kind and same value; the tolerance only absorbs float error from magnitude scaling."""
        if self.kind is not other.kind:
            return False
        if self.value == other.value:
            return True
        scale = max(abs(self.value), abs(other.value))
        return scale > 0 and abs(self.value - other.value) / scale <= tolerance


def extract_numbers(text: str) -> list[NumberMention]:
    """Numbers with their magnitude applied: ``$1.2 billion`` and ``1,200 million`` compare equal."""
    mentions: list[NumberMention] = []
    for match in _NUMBER.finditer(strip_citation_markers(text)):
        raw_number = match.group("num")
        try:
            value = float(raw_number.replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees a parseable number
            continue
        unit = (match.group("unit") or "").lower().replace(" ", "")
        kind = NumberKind.AMOUNT
        if unit == "%" or unit.startswith("per"):
            kind = NumberKind.PERCENT
        elif unit:
            value *= _SCALE_WORDS.get(unit, 1.0)
        mentions.append(NumberMention(value=value, kind=kind, raw=match.group(0).strip()))
    return mentions


def unmatched_numbers(claim: str, evidence_text: str) -> list[str]:
    """Numbers stated in ``claim`` that do not appear (by value) anywhere in ``evidence_text``."""
    available = extract_numbers(evidence_text)
    missing: list[str] = []
    for mention in extract_numbers(claim):
        if not any(mention.matches(candidate) for candidate in available):
            missing.append(mention.raw)
    return missing


def extract_entities(text: str) -> set[str]:
    """Capitalised names and acronyms that are not merely sentence-initial words (lower-cased)."""
    cleaned = strip_citation_markers(text)
    starts = {begin for begin, _ in sentence_spans(cleaned)}
    entities: set[str] = set()
    for match in _ENTITY.finditer(cleaned):
        token = match.group(0)
        if match.start() in starts and not any(ch.isupper() for ch in token[1:]):
            continue
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        entities.add(lowered)
    return entities


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Case-insensitive whole-word pattern for a lexicon phrase (spaces match any whitespace/hyphen)."""
    parts = [re.escape(part) for part in phrase.lower().split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]+".join(parts) + r"(?![a-z0-9])", re.IGNORECASE)


def truncate_words(text: str, limit: int) -> str:
    """Prefix of ``text`` no longer than ``limit`` characters, cut on a word boundary."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(" ")
    return (cut[:boundary] if boundary > limit // 2 else cut).rstrip(" ,;:")
