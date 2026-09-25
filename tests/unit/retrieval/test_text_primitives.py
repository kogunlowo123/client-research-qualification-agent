from __future__ import annotations

import pytest

from client_research_agent.retrieval import tokenization
from client_research_agent.retrieval.lexical import (
    coverage,
    jaccard,
    sentence_spans,
    split_sentences,
    stem,
    term_set,
    tokenize,
)
from client_research_agent.retrieval.tokenization import Tokenizer, count_tokens, get_tokenizer, load_encoding


class TestTokenizer:
    def test_exact_encoder_counts_bpe_tokens(self) -> None:
        tokenizer = get_tokenizer()
        assert tokenizer.exact
        assert tokenizer.count("hello world") == 2
        assert count_tokens("") == 0

    def test_special_tokens_are_counted_as_text(self) -> None:
        assert get_tokenizer().count("<|endoftext|>") > 1

    def test_whitespace_fallback(self) -> None:
        tokenizer = Tokenizer(None)
        assert not tokenizer.exact
        assert tokenizer.count("Acme  migrated\nto the cloud") == 5

    def test_load_failure_degrades_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(name: str) -> None:
            raise OSError(f"cannot download {name}")

        monkeypatch.setattr(tokenization.tiktoken, "get_encoding", boom)
        load_encoding.cache_clear()
        try:
            assert load_encoding("cl100k_base") is None
            assert Tokenizer.default().count("three word text") == 3
        finally:
            load_encoding.cache_clear()

    def test_word_spans_and_truncate(self) -> None:
        tokenizer = Tokenizer(None)
        text = "alpha beta gamma delta"
        assert tokenizer.word_spans(text, 6) == [(6, 10), (11, 16), (17, 22)]
        assert tokenizer.truncate(text, 2) == "alpha beta"
        assert tokenizer.truncate(text, 10) == text
        assert tokenizer.truncate(text, 0) == ""

    def test_truncate_with_bpe(self) -> None:
        tokenizer = get_tokenizer()
        text = "cloud migration " * 50
        truncated = tokenizer.truncate(text, 10)
        assert tokenizer.count(truncated) <= 10
        assert text.startswith(truncated)


class TestLexical:
    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("migration", "migrat"),
            ("migrating", "migrat"),
            ("migrated", "migrat"),
            ("migrate", "migrat"),
            ("companies", "company"),
            ("processes", "process"),
            ("process", "process"),
            ("platforms", "platform"),
            ("services", "servic"),
            ("service", "servic"),
            ("quarterly", "quarter"),
            ("ai", "ai"),
            ("2026", "2026"),
            ("analysis", "analysis"),
            ("uses", "use"),
        ],
    )
    def test_stem(self, word: str, expected: str) -> None:
        assert stem(word) == expected

    def test_tokenize_filters_stopwords_and_short_terms(self) -> None:
        assert tokenize("The AI platforms of Acme's 2 plants") == ["ai", "platform", "acme", "2", "plant"]
        assert tokenize("The cloud", remove_stopwords=False, stem_terms=False) == ["the", "cloud"]
        assert term_set("cloud clouds") == {"cloud"}

    def test_coverage_and_jaccard(self) -> None:
        assert coverage([], ["a"]) == 0.0
        assert coverage(["cloud", "ai"], ["cloud"]) == 0.5
        assert jaccard([], []) == 0.0
        assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_sentence_split_handles_abbreviations_and_decimals(self) -> None:
        text = (
            "Acme Corp. reported $4.2 billion in revenue. The U.S. business grew. "
            "J. Smith joined as CTO! Is growth durable? yes it is.\nNew line starts here"
        )
        assert split_sentences(text) == [
            "Acme Corp. reported $4.2 billion in revenue.",
            "The U.S. business grew.",
            "J. Smith joined as CTO!",
            "Is growth durable? yes it is.",
            "New line starts here",
        ]

    def test_sentence_split_quotes_and_newline_after_period(self) -> None:
        text = 'He said "we are done." Then left.\nlowercase continues. "Quoted start" here.'
        assert split_sentences(text) == [
            'He said "we are done."',
            "Then left.",
            "lowercase continues.",
            '"Quoted start" here.',
        ]

    def test_sentence_spans_are_exact_offsets(self) -> None:
        text = "  First one.  Second one.  "
        spans = sentence_spans(text)
        assert [text[s:e] for s, e in spans] == ["First one.", "Second one."]
        assert sentence_spans("   ") == []
