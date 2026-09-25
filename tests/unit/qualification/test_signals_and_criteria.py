from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.models import Criterion, DocumentType
from client_research_agent.qualification.criteria import (
    CRITERIA,
    CriterionDefinition,
    all_definitions,
    get_definition,
)
from client_research_agent.qualification.signals import (
    UNDATED_RECENCY,
    employee_band,
    extract_scale_signals,
    recency_weight,
    revenue_band,
    source_trust,
)

TODAY = date(2026, 9, 1)


def test_source_trust_orders_filings_above_marketing() -> None:
    assert source_trust(DocumentType.SEC_FILING) == 1.0
    assert source_trust(DocumentType.SEC_FILING) > source_trust(DocumentType.PRESS_RELEASE)
    assert source_trust(DocumentType.PRESS_RELEASE) > source_trust(DocumentType.CORPORATE_WEBPAGE)


def test_recency_weight_half_life() -> None:
    assert recency_weight(None, today=TODAY) == UNDATED_RECENCY
    assert recency_weight(TODAY, today=TODAY) == 1.0
    assert recency_weight(date(2027, 1, 1), today=TODAY) == 1.0
    assert recency_weight(date(2025, 9, 1), today=TODAY, half_life_days=365) == pytest.approx(0.5)
    assert recency_weight(date(1990, 1, 1), today=TODAY) == 0.05


@pytest.mark.parametrize(
    ("text", "revenue", "employees"),
    [
        ("Net sales were $4.5 billion in fiscal 2025.", 4.5e9, None),
        ("The group generated $820 million in annual revenue.", 820e6, None),
        ("Revenue reached €2,100 million while we employ 12,500 employees.", 2.1e9, 12500),
        ("A workforce of approximately 3.2 thousand people.", None, 3200),
        ("We have 45k employees worldwide.", None, 45000),
        ("No figures here.", None, None),
    ],
)
def test_extract_scale_signals(text: str, revenue: float | None, employees: int | None) -> None:
    signals = extract_scale_signals(text)
    assert signals.revenue_usd == (pytest.approx(revenue) if revenue else None)
    assert signals.employees == employees
    assert signals.found is (revenue is not None or employees is not None)


def test_scale_bands() -> None:
    assert [revenue_band(v) for v in (0, 1e6, 60e6, 300e6, 2e9, 20e9)] == [0, 1, 2, 3, 4, 5]
    assert [employee_band(v) for v in (0, 50, 500, 5000, 20000, 90000)] == [0, 1, 2, 3, 4, 5]


def test_all_five_criteria_are_defined() -> None:
    assert set(CRITERIA) == set(Criterion)
    titles = [d.title for d in all_definitions()]
    assert titles == [
        "Company Size & Scale",
        "Technology Modernization",
        "AI and Data Focus",
        "Industry Trends",
        "Near-Term Opportunity",
    ]
    for definition in all_definitions():
        assert sorted(definition.rubric) == [0, 1, 2, 3, 4, 5]
        assert definition.positive_signals
        assert definition.negative_signals
        assert all("Acme" in q for q in definition.render_queries("Acme"))
        assert definition.rubric_text().splitlines()[0].startswith("0 - ")
        assert "Acme" in definition.render_discovery_question("Acme")
        assert definition.render_discovery_question("Acme").endswith("?")


def test_definition_validation() -> None:
    base = get_definition(Criterion.COMPANY_SCALE)
    with pytest.raises(ValueError, match="levels"):
        CriterionDefinition(
            criterion=base.criterion,
            title=base.title,
            description=base.description,
            rubric={0: "x"},
            queries=base.queries,
            positive_signals=base.positive_signals,
            negative_signals=base.negative_signals,
            discovery_question=base.discovery_question,
        )
    with pytest.raises(ValueError, match="company"):
        CriterionDefinition(
            criterion=base.criterion,
            title=base.title,
            description=base.description,
            rubric=base.rubric,
            queries=("revenue without placeholder",),
            positive_signals=base.positive_signals,
            negative_signals=base.negative_signals,
            discovery_question=base.discovery_question,
        )
