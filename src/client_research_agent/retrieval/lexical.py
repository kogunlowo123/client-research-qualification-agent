"""Shared lexical primitives: term normalisation, sentence segmentation, overlap.

A single analyzer is used by BM25, the hashing embedder, the rerankers and the
relevance fallbacks so that "a term" means the same thing everywhere in the
retrieval stack.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "s",
        "t",
        "don",
        "shall",
        "may",
        "might",
        "must",
        "via",
        "per",
        "upon",
        "within",
        "without",
        "across",
        "among",
        "amongst",
        "yet",
        "whether",
    }
)

# Longest suffix first; a rule applies only if the remaining stem keeps >= 3 characters.
_SUFFIX_RULES: tuple[tuple[str, str], ...] = (
    ("izations", "iz"),
    ("ization", "iz"),
    ("ations", "at"),
    ("ation", "at"),
    ("ators", "at"),
    ("ator", "at"),
    ("ments", ""),
    ("ment", ""),
    ("sses", "ss"),
    ("ings", ""),
    ("ing", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("ed", ""),
    ("es", ""),
    ("ly", ""),
    ("s", ""),
)
_NO_PLURAL_S = ("ss", "us", "is")
_TERM = re.compile(r"[a-z0-9]+")

_ABBREVIATIONS = frozenset(
    {
        "inc",
        "corp",
        "co",
        "ltd",
        "llc",
        "plc",
        "mr",
        "mrs",
        "ms",
        "dr",
        "st",
        "no",
        "vs",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "approx",
        "est",
        "fig",
    }
)
_CLOSING_QUOTES = chr(0x201D) + chr(0x2019)
_BOUNDARY = re.compile(r"([.!?][\"')\]" + _CLOSING_QUOTES + r"]*)(\s+)|(\s*\n\s*)")
_OPENERS = "\"'([-*" + chr(0x201C) + chr(0x2018) + chr(0x2022)


def stem(word: str) -> str:
    """Light, deterministic suffix stripper (Porter-flavoured, single pass)."""
    if len(word) <= 3 or word.isdigit():
        return word
    stemmed = word
    for suffix, replacement in _SUFFIX_RULES:
        if not word.endswith(suffix):
            continue
        if suffix == "s" and word.endswith(_NO_PLURAL_S):
            continue
        candidate = word[: -len(suffix)] + replacement
        if len(candidate) >= 3:
            stemmed = candidate
            break
    if stemmed.endswith("e") and len(stemmed) > 4:
        stemmed = stemmed[:-1]
    return stemmed


def tokenize(text: str, *, remove_stopwords: bool = True, stem_terms: bool = True) -> list[str]:
    """Lowercase alphanumeric terms, optionally stopword-filtered and stemmed."""
    terms: list[str] = []
    for raw in _TERM.findall(text.lower()):
        if len(raw) < 2 and not raw.isdigit():
            continue
        if remove_stopwords and raw in STOPWORDS:
            continue
        terms.append(stem(raw) if stem_terms else raw)
    return terms


def term_set(text: str) -> set[str]:
    return set(tokenize(text))


def coverage(query_terms: Iterable[str], text_terms: Iterable[str]) -> float:
    """Fraction of distinct query terms that occur in the text."""
    wanted = set(query_terms)
    if not wanted:
        return 0.0
    present = set(text_terms)
    return len(wanted & present) / len(wanted)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_abbreviation(text: str, start: int, punct_start: int) -> bool:
    segment = text[start:punct_start]
    match = re.search(r"(\S+)$", segment)
    if match is None:
        return False
    word = match.group(1).lower().lstrip("\"'(")
    if word in _ABBREVIATIONS:
        return True
    # Single-letter initials such as "J. Smith".
    return len(word) == 1 and word.isalpha()


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences; newlines always end a sentence."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if match.group(3) is not None:
            end, next_start = match.start(), match.end()
        else:
            follower = text[match.end() : match.end() + 1]
            if "\n" not in match.group(2):
                if not follower or not (follower.isupper() or follower.isdigit() or follower in _OPENERS):
                    continue
                if _is_abbreviation(text, start, match.start(1)):
                    continue
            end, next_start = match.end(1), match.end()
        _append_trimmed(text, start, end, spans)
        start = next_start
    _append_trimmed(text, start, len(text), spans)
    return spans


def _append_trimmed(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))


def split_sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]
