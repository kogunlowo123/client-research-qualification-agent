from __future__ import annotations

import pytest

from client_research_agent.research.parsing.entities import (
    EntityExtractor,
    ExecutiveMention,
    ExtractedEntities,
)

extractor = EntityExtractor()


@pytest.mark.parametrize(
    ("text", "value", "currency"),
    [
        ("Revenue was $12.4 billion in 2025.", 12.4e9, "USD"),
        ("a US$3,500,000 contract", 3.5e6, "USD"),
        ("investing €250 million in plants", 250e6, "EUR"),
        ("£1.2bn programme", 1.2e9, "GBP"),
        ("USD 40 million facility", 40e6, "USD"),
        ("a $4.2B deal", 4.2e9, "USD"),
        ("cost $500 per seat", 500.0, "USD"),
    ],
)
def test_monetary_amounts(text: str, value: float, currency: str) -> None:
    amounts = extractor.extract(text).monetary_amounts
    assert len(amounts) == 1
    assert amounts[0].value == pytest.approx(value)
    assert amounts[0].currency == currency


def test_scale_letter_requires_word_boundary() -> None:
    amounts = extractor.extract("We spent $5 more on coffee.").monetary_amounts
    assert amounts[0].value == 5.0


def test_percentages() -> None:
    found = extractor.extract("Revenue grew 8% while margin reached 17.5 percent and churn fell -2 per cent.")
    assert [p.value for p in found.percentages] == [8.0, 17.5, -2.0]


def test_fiscal_periods() -> None:
    text = (
        "Results for the fourth quarter of fiscal 2025 and Q1 FY26 guidance. Full FY2025 outlook; "
        "fiscal year 2024 comparison; Q3'25 pipeline."
    )
    periods = extractor.extract(text).fiscal_periods
    assert periods == ("Q4 FY2025", "Q1 FY2026", "Q3 FY2025", "FY2025", "FY2026", "FY2024")


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("Northwind employs approximately 42,000 employees worldwide.", 42000),
        ("a team of 1.5 thousand full-time employees", 1500),
        ("more than 900 team members", 900),
        ("with a workforce of about 12,000 across plants", 12000),
    ],
)
def test_headcount(text: str, value: int) -> None:
    assert [h.value for h in extractor.extract(text).headcounts] == [value]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Northwind today announced it has appointed Rahul Mehta as Chief Information Officer.",
            ExecutiveMention("Rahul Mehta", "Chief Information Officer"),
        ),
        (
            "Acme names Laura van Dijk its new CDO effective May 1.",
            ExecutiveMention("Laura van Dijk", "Chief Data Officer"),
        ),
        (
            '"Data is our advantage," said Maria Chen, Chief Executive Officer.',
            ExecutiveMention("Maria Chen", "Chief Executive Officer"),
        ),
        (
            "Priya K. Raman has been appointed Chief Data and Analytics Officer.",
            ExecutiveMention("Priya K. Raman", "Chief Data and Analytics Officer"),
        ),
        (
            "CFO Tom O\u2019Brien said the quarter was strong.",
            ExecutiveMention("Tom O\u2019Brien", "Chief Financial Officer"),
        ),
        (
            "Sam Patel, Executive Vice President of Operations, will lead the program.",
            ExecutiveMention("Sam Patel", "Executive Vice President of Operations"),
        ),
    ],
)
def test_executives(text: str, expected: ExecutiveMention) -> None:
    assert expected in extractor.extract(text).executives


def test_executive_stopwords_filter_false_positives() -> None:
    executives = extractor.extract("The Board, President of the association, met today.").executives
    assert executives == ()


def test_technology_gazetteer_counts_and_orders_by_first_mention() -> None:
    text = (
        "We are migrating workloads to the cloud on AWS and Microsoft Azure, building a data lakehouse on "
        "Databricks, replacing our Snowflake warehouse, and piloting generative AI with LLMs. "
        "Our SAP S/4HANA ERP modernization continues; Salesforce remains the CRM. More AWS spend is planned."
    )
    found = extractor.extract(text)
    assert found.technologies[:3] == ("Cloud migration", "Cloud", "AWS")
    for term in ("Azure", "Data lakehouse", "Databricks", "Snowflake", "Generative AI", "LLM", "SAP",
                 "ERP modernization", "Salesforce"):  # fmt: skip
        assert term in found.technologies
    assert found.technology_mentions["AWS"] == 2
    assert "Oracle" not in found.technologies


def test_case_sensitive_terms_do_not_match_lowercase_words() -> None:
    found = extractor.extract("The sap flows in spring; aws is not an acronym here; said oracle.")
    assert found.technologies == ()


def test_flat_entities_and_empty_result() -> None:
    found = extractor.extract(
        "Northwind appointed Rahul Mehta as CIO. Revenue was $12.4 billion, up 8%, in fiscal 2025 "
        "with 42,000 employees and a Databricks lakehouse."
    )
    assert found.entities[:2] == ("Databricks", "Data lakehouse")
    assert "Rahul Mehta (Chief Information Officer)" in found.entities
    assert "FY2025" in found.entities
    assert "$12.4 billion" in found.entities
    assert "8%" in found.entities
    assert "42,000 employees" in found.entities
    assert not found.is_empty
    assert ExtractedEntities().is_empty
    assert extractor.extract("").is_empty


def test_text_is_truncated_to_budget() -> None:
    short = EntityExtractor(max_text_chars=10)
    assert short.extract("filler....Databricks").technologies == ()
