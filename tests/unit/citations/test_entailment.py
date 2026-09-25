from __future__ import annotations

import pytest

from client_research_agent.citations.entailment import NUMBER_MISMATCH_CAP, lexical_support

EVIDENCE = (
    "Acme Corp reported annual revenue of $12.4 billion for fiscal 2025. "
    "The company employs approximately 45,000 employees and partners with Databricks."
)


def test_paraphrase_with_matching_figures_is_supported() -> None:
    support = lexical_support(
        "Acme Corp reported annual revenue of $12.4 billion in fiscal 2025 [E1].", EVIDENCE
    )
    assert support.score == pytest.approx(1.0)
    assert support.numbers_consistent
    assert support.missing_entities == ()
    assert "100% of claim terms" in support.reason()


def test_fabricated_number_is_capped() -> None:
    support = lexical_support("Acme Corp reported annual revenue of $15 billion.", EVIDENCE)
    assert support.score <= NUMBER_MISMATCH_CAP
    assert support.missing_numbers == ("$15 billion",)
    assert "figures not in evidence" in support.reason()


def test_equivalent_magnitudes_match() -> None:
    assert lexical_support("Revenue was 12,400 million dollars.", EVIDENCE).numbers_consistent


def test_wrong_entity_is_penalised() -> None:
    support = lexical_support("Acme Corp partners with Snowflake.", EVIDENCE)
    assert "snowflake" in support.missing_entities
    assert support.score < support.coverage
    assert "names not in evidence" in support.reason()


def test_unrelated_claim_scores_low_and_empty_claim_scores_zero() -> None:
    assert lexical_support("The weather in Paris was sunny.", EVIDENCE).score < 0.2
    assert lexical_support("[E1]", EVIDENCE).score == 0.0
