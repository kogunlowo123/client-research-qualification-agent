from __future__ import annotations

import pytest

from client_research_agent.security.pii import (
    PiiDetector,
    PiiFinding,
    PiiPolicy,
    PiiRedactor,
    PiiType,
    iban_valid,
    luhn_valid,
)


@pytest.fixture
def redactor() -> PiiRedactor:
    return PiiRedactor()


def _types(text: str) -> list[PiiType]:
    return [f.type for f in PiiDetector().detect(text)]


def test_luhn_and_iban_validation() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("12345")
    assert iban_valid("GB82 WEST 1234 5698 7654 32")
    assert iban_valid("DE89370400440532013000")
    assert not iban_valid("GB82 WEST 1234 5698 7654 33")
    assert not iban_valid("DE8937040044053201300")
    assert not iban_valid("not an iban")
    assert iban_valid("XK051212012345678906")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Card 4111 1111 1111 1111 on file", PiiType.CREDIT_CARD),
        ("Card 4111-1111-1111-1111 on file", PiiType.CREDIT_CARD),
        ("SSN 123-45-6789", PiiType.SSN),
        ("social security number: 123456789", PiiType.SSN),
        ("IBAN GB82 WEST 1234 5698 7654 32 please", PiiType.IBAN),
        ("server 192.168.10.25 responded", PiiType.IPV4),
        ("host 10.20.30.40.", PiiType.IPV4),
        ("call +1 (415) 555-0134 today", PiiType.PHONE),
        ("call +44 20 7946 0958 today", PiiType.PHONE),
        ("call 415-555-0134 today", PiiType.PHONE),
        ("call +14155550134 today", PiiType.PHONE),
        ("write to jane.doe@gmail.com", PiiType.EMAIL),
    ],
)
def test_detects_pii(text: str, expected: PiiType) -> None:
    assert _types(text) == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Card 4111 1111 1111 1112 invalid",
        "Released on 2026-05-01 in Boston",
        "Revenue was 1 250 000 units",
        "Version 999.1.1.1 shipped",
        "Order 12345",
    ],
)
def test_no_false_positives(text: str) -> None:
    assert _types(text) == []


def test_invalid_ssn_area_not_classified_as_ssn() -> None:
    assert PiiType.SSN not in _types("SSN 000-12-3456 invalid area")
    assert PiiType.SSN not in _types("id 666-12-3456")


def test_policy_keeps_names_titles_and_role_mailboxes(redactor: PiiRedactor) -> None:
    text = (
        "Jane Doe, Chief Financial Officer, said results were strong. Media: press@acme.com. "
        "Jane can be reached at jane.doe@acme.com or +1 415 555 0134."
    )
    result = redactor.redact(text)
    assert "Jane Doe, Chief Financial Officer" in result.text
    assert "press@acme.com" in result.text
    assert "jane.doe@acme.com" not in result.text
    assert "[REDACTED_EMAIL]" in result.text
    assert "[REDACTED_PHONE]" in result.text
    assert result.redacted_count == 2
    assert [k.type for k in result.kept] == [PiiType.EMAIL]
    assert result.counts() == {"email": 1, "phone": 1}


def test_addresses_residential_vs_business(redactor: PiiRedactor) -> None:
    text = (
        "The company is headquartered at 500 Main Street, Suite 200. "
        "The founder lives at 12 Oak Tree Lane, Apt 4B with family."
    )
    result = redactor.redact(text)
    assert "500 Main Street, Suite 200" in result.text
    assert "12 Oak Tree Lane" not in result.text
    assert "[REDACTED_ADDRESS]" in result.text
    home = redactor.redact("His home is 44 Elm Road in Springfield.")
    assert "[REDACTED_ADDRESS]" in home.text


def test_strict_policy_redacts_role_mailboxes_and_business_addresses() -> None:
    strict = PiiRedactor(policy=PiiPolicy(keep_role_mailboxes=False, keep_business_addresses=False))
    result = strict.redact("press@acme.com, 500 Main Street")
    assert result.text == "[REDACTED_EMAIL], [REDACTED_ADDRESS]"


def test_policy_default_branch_redacts_unknown_types() -> None:
    policy = PiiPolicy(always_redact=frozenset())
    finding = PiiFinding(type=PiiType.SSN, start=0, end=1, fingerprint="x")
    assert policy.should_redact(finding)
    assert finding.span == (0, 1)


def test_overlap_resolution_prefers_higher_priority() -> None:
    findings = PiiDetector().detect("IBAN DE89 3704 0044 0532 0130 00")
    assert [f.type for f in findings] == [PiiType.IBAN]


def test_findings_do_not_store_raw_values() -> None:
    finding = PiiDetector().detect("jane@gmail.com")[0]
    assert "jane" not in repr(finding)
    assert len(finding.fingerprint) == 12


def test_redact_value_recurses(redactor: PiiRedactor) -> None:
    payload = {"a": "ssn 123-45-6789", "b": ["jane@gmail.com", ("x",)], "c": 3}
    cleaned = redactor.redact_value(payload)
    assert cleaned == {"a": "ssn [REDACTED_US_SSN]", "b": ["[REDACTED_EMAIL]", ("x",)], "c": 3}
    assert redactor.detector is not None


def test_contains_pii(redactor: PiiRedactor) -> None:
    assert redactor.contains_pii("jane@gmail.com")
    assert not redactor.contains_pii("investors@acme.com")


def test_custom_role_mailboxes() -> None:
    detector = PiiDetector(role_mailboxes=["Deals"])
    assert not detector.detect("deals@acme.com")[0].personal
