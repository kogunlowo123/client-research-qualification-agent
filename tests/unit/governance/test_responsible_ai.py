from __future__ import annotations

import pytest

from client_research_agent.governance.responsible_ai import ResponsibleAIPolicy, Severity
from tests.unit.governance.briefs import build_brief, fact, recommendation


@pytest.fixture
def policy() -> ResponsibleAIPolicy:
    return ResponsibleAIPolicy()


def test_clean_brief_passes_with_disclaimer(policy: ResponsibleAIPolicy) -> None:
    report = policy.evaluate(build_brief())
    assert report.passed
    assert report.findings == ()
    assert report.verified_facts == 3
    assert report.recommendations == 5
    assert "Acme Corp" in report.disclaimer
    assert "run-123" in report.disclaimer
    assert "2026-09-01" in report.disclaimer
    assert "1 public source" in report.disclaimer


def test_grounding_unknown_evidence(policy: ResponsibleAIPolicy) -> None:
    report = policy.evaluate(build_brief(overview=[fact("Acme has 10,000 staff.", "ev-404")]))
    assert not report.passed
    assert report.errors[0].rule == "grounding"
    assert report.errors[0].location == "company_overview[0]"


def test_next_actions_must_be_recommendations(policy: ResponsibleAIPolicy) -> None:
    report = policy.evaluate(build_brief(next_actions=[fact("Book a workshop.")]))
    assert [f.rule for f in report.errors] == ["recommendation_label"]


def test_advice_presented_as_fact_warns(policy: ResponsibleAIPolicy) -> None:
    report = policy.evaluate(build_brief(opportunities=[fact("We recommend that Acme adopt a lakehouse.")]))
    assert report.passed
    assert report.warnings[0].rule == "recommendation_label"
    assert report.warnings[0].severity is Severity.WARNING


@pytest.mark.parametrize(
    "text",
    [
        "The CEO is 67-year-old and may resist change.",
        "Given her religion, the CFO may oppose the deal.",
        "The founder is Muslim, so outreach should differ.",
        "Smith's ethnicity suggests a different pitch.",
        "The chairman is nearing retirement, so wait.",
        "A 62-year-old CEO is unlikely to fund AI.",
    ],
)
def test_protected_attribute_reasoning_blocked(policy: ResponsibleAIPolicy, text: str) -> None:
    report = policy.evaluate(build_brief(risks=[recommendation(text)]))
    assert "protected_attributes" in {f.rule for f in report.errors}


@pytest.mark.parametrize(
    "text",
    [
        "The CFO is probably distracted by a divorce.",
        "Rumors suggest the CEO has health problems.",
    ],
)
def test_personal_speculation_blocked(policy: ResponsibleAIPolicy, text: str) -> None:
    report = policy.evaluate(build_brief(risks=[recommendation(text)]))
    assert "personal_speculation" in {f.rule for f in report.errors}


@pytest.mark.parametrize(
    "text",
    [
        "The CEO launched a women-in-technology scholarship program.",
        "Jane Doe, CFO, announced record revenue.",
        "The company operates in the healthcare industry.",
        "The CIO may prioritize data governance next year.",
        "Acme is a young company founded in 2019.",
        "Acme's financial health improved after the CFO restructured debt.",
    ],
)
def test_business_text_not_flagged(policy: ResponsibleAIPolicy, text: str) -> None:
    assert policy.check_text(text) == []


def test_discovery_questions_checked(policy: ResponsibleAIPolicy) -> None:
    questions = [
        "Is the CEO married?",
        "Is his health a factor for the CFO?",
        "What is the budget?",
        "Who decides?",
        "When?",
    ]
    report = policy.evaluate(build_brief(questions=questions))
    locations = {f.location for f in report.errors}
    assert {"discovery_questions[0]", "discovery_questions[1]"} <= locations
