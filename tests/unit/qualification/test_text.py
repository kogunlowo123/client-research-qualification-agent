from __future__ import annotations

import pytest

from client_research_agent.qualification.text import (
    NumberKind,
    citation_markers,
    content_words,
    extract_entities,
    extract_numbers,
    jaccard,
    phrase_pattern,
    sentence_spans,
    split_sentences,
    stem,
    strip_citation_markers,
    tokenize,
    truncate_words,
    unmatched_numbers,
)


def test_citation_markers_are_extracted_and_stripped() -> None:
    text = "Revenue grew [E1] and margins improved [E2, E3] again [E1]."
    assert citation_markers(text) == ("E1", "E2", "E3")
    assert strip_citation_markers(text) == "Revenue grew and margins improved again."


def test_tokenize_and_stem() -> None:
    assert tokenize("Acme's AI-Platform 2026") == ["acme's", "ai", "platform", "2026"]
    stems = {stem(w) for w in ("modernize", "modernizing", "modernization", "modernized")}
    assert len(stems) == 1
    assert stem("companies") == "company"
    assert stem("data") == "data"


def test_content_words_drop_stopwords_numbers_and_markers() -> None:
    words = content_words("The company is migrating 300 platforms to the cloud [E4]")
    assert "cloud" in words
    assert "platform" in words
    assert "the" not in words
    assert "300" not in words
    assert "e4" not in words


def test_jaccard() -> None:
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_sentence_splitting_keeps_decimals_and_trims() -> None:
    text = "  Revenue was $1.2 billion. Headcount rose 5%!  New CEO named?\n\n- bullet item"
    assert split_sentences(text) == [
        "Revenue was $1.2 billion.",
        "Headcount rose 5%!",
        "New CEO named?",
        "- bullet item",
    ]
    for begin, end in sentence_spans(text):
        assert text[begin:end] == text[begin:end].strip()
    assert split_sentences("   ") == []


def test_numbers_apply_magnitude_and_kind() -> None:
    mentions = extract_numbers("Sales of $1.2 billion, up 12% (12 percent) with 45,000 staff and 3bn users")
    values = [(m.value, m.kind) for m in mentions]
    assert (1.2e9, NumberKind.AMOUNT) in values
    assert (12.0, NumberKind.PERCENT) in values
    assert (45000.0, NumberKind.AMOUNT) in values
    assert (3e9, NumberKind.AMOUNT) in values
    assert sum(1 for m in mentions if m.kind is NumberKind.PERCENT) == 2


def test_number_matching_across_formats() -> None:
    assert unmatched_numbers("Revenue was 1,200 million dollars", "revenue of $1.2 billion") == []
    assert unmatched_numbers("Revenue was $1.3 billion", "revenue of $1.2 billion") == ["$1.3 billion"]
    assert unmatched_numbers("Growth of 12%", "growth of 12 amount units") == ["12%"]
    assert unmatched_numbers("in 2026 [E12]", "announced in 2026") == []
    first, second = extract_numbers("0 and 0")
    assert first.matches(second)


def test_entities_ignore_sentence_initial_words() -> None:
    entities = extract_entities("Acme partnered with Databricks and AWS. The deal closed.")
    assert {"databricks", "aws"} <= entities
    assert "the" not in entities
    assert "acme" not in entities
    assert "acmecorp" in extract_entities("AcmeCorp leads.")


def test_phrase_pattern_is_whole_word_and_hyphen_tolerant() -> None:
    pattern = phrase_pattern("cloud migration")
    assert pattern.search("A Cloud-Migration effort")
    assert pattern.search("cloud   migration")
    assert not pattern.search("cloud migrations")
    assert not phrase_pattern("api").search("capital")


def test_truncate_words() -> None:
    assert truncate_words("short", 10) == "short"
    assert truncate_words("alpha beta gamma delta", 12) == "alpha beta"
    assert truncate_words("x" * 30, 10) == "x" * 10
