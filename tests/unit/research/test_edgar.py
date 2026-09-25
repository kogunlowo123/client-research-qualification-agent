from __future__ import annotations

import json
from datetime import date

import pytest

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.research.sources.edgar import (
    COMPANY_CONCEPT_URL,
    COMPANY_FACTS_URL,
    SUBMISSIONS_URL,
    TICKERS_URL,
    EdgarClient,
    latest_fact,
    normalize_company_name,
    pad_cik,
)
from client_research_agent.utils.errors import CrawlPolicyViolationError, UpstreamServiceError
from tests.unit.research.helpers import ScriptedFetcher, fixture_text

CIK = "0001234567"


@pytest.fixture
def client(scripted: ScriptedFetcher) -> EdgarClient:
    scripted.add(TICKERS_URL, fixture_text("company_tickers.json"), content_type="application/json")
    scripted.add(
        SUBMISSIONS_URL.format(cik=CIK), fixture_text("submissions.json"), content_type="application/json"
    )
    scripted.add(
        COMPANY_FACTS_URL.format(cik=CIK), fixture_text("companyfacts.json"), content_type="application/json"
    )
    return EdgarClient.from_settings(
        scripted, CrawlerSettings(user_agent="TestAgent/1.0", contact_email="ops@agent.example")
    )


def test_sec_user_agent_declares_contact(client: EdgarClient, scripted: ScriptedFetcher) -> None:
    client.resolve_cik(ticker="NWND")
    _url, headers = scripted.calls[0]
    assert headers is not None
    assert headers["User-Agent"] == "TestAgent/1.0 ops@agent.example"


def test_contact_email_is_mandatory(scripted: ScriptedFetcher) -> None:
    with pytest.raises(ValueError, match="contact e-mail"):
        EdgarClient(scripted, user_agent="x", contact_email="nobody")


def test_resolve_cik_by_ticker_and_name(client: EdgarClient, scripted: ScriptedFetcher) -> None:
    assert client.resolve_cik(ticker="nwnd") == CIK
    assert client.resolve_cik(company_name="Northwind Industries") == CIK
    assert client.resolve_cik(company_name="Microsoft Corporation") == "0000789019"
    assert client.resolve_cik(ticker="ZZZZ", company_name="Unknown Co") is None
    assert client.resolve_cik() is None
    assert scripted.urls.count(TICKERS_URL) == 1


def test_recent_filings_filters_orders_and_builds_urls(client: EdgarClient) -> None:
    filings = client.recent_filings(CIK, max_filings=10)
    assert [f.form for f in filings] == ["10-K", "8-K", "10-Q", "DEF 14A"]
    assert [f.filing_date for f in filings] == [
        date(2026, 2, 13),
        date(2026, 2, 12),
        date(2025, 11, 4),
        date(2025, 4, 2),
    ]
    ten_k = filings[0]
    assert ten_k.url == (
        "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000012/nwnd-20251231.htm"
    )
    assert ten_k.report_date == date(2025, 12, 31)
    assert ten_k.accession_number == "0001234567-26-000012"
    assert filings[3].report_date is None
    assert all("passwd" not in f.url for f in filings)


def test_recent_filings_respects_limit_and_form_override(client: EdgarClient) -> None:
    assert len(client.recent_filings(CIK, max_filings=2)) == 2
    assert [f.form for f in client.recent_filings("1234567", forms=["10-Q"])] == ["10-Q"]


@pytest.mark.parametrize(
    "payload",
    [
        {"filings": {"recent": {"accessionNumber": "oops"}}},
        {"filings": {"recent": []}},
        [],
        {
            "filings": {
                "recent": {
                    "accessionNumber": ["bad-acc-x"],
                    "form": ["10-K"],
                    "filingDate": ["2026-01-01"],
                    "primaryDocument": ["a.htm"],
                }
            }
        },
    ],
)
def test_recent_filings_tolerates_malformed_payloads(scripted: ScriptedFetcher, payload: object) -> None:
    scripted.add(SUBMISSIONS_URL.format(cik=CIK), json.dumps(payload))
    client = EdgarClient(scripted, user_agent="a", contact_email="a@b.example")
    assert client.recent_filings(CIK) == []


def test_company_facts_picks_latest_annual_values(client: EdgarClient) -> None:
    facts = client.company_facts(CIK)
    assert facts is not None
    assert facts.entity_name == "Northwind Industries, Inc."
    assert facts.revenue is not None
    assert facts.revenue.value == 12_400_000_000
    assert facts.revenue.concept == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert facts.revenue.period_end == date(2025, 12, 31)
    assert facts.revenue.fiscal_year == 2025
    assert facts.shares_outstanding is not None
    assert facts.shares_outstanding.value == 405_500_000
    assert facts.employees is not None
    assert facts.employees.value == 42_000
    assert facts.has_any
    assert facts.source_url == COMPANY_FACTS_URL.format(cik=CIK)


def test_company_facts_falls_back_to_concept_api(scripted: ScriptedFetcher) -> None:
    scripted.fail(COMPANY_FACTS_URL.format(cik=CIK), CrawlPolicyViolationError("response exceeds cap"))
    concept = COMPANY_CONCEPT_URL.format(cik=CIK, taxonomy="us-gaap", concept="Revenues")
    scripted.add(
        concept,
        json.dumps(
            {
                "entityName": "Northwind Industries, Inc.",
                "units": {"USD": [{"start": "2025-01-01", "end": "2025-12-31", "val": 12.4e9, "fp": "FY"}]},
            }
        ),
    )
    scripted.add(
        COMPANY_CONCEPT_URL.format(cik=CIK, taxonomy="dei", concept="EntityCommonStockSharesOutstanding"),
        json.dumps({"units": {"shares": [{"end": "2026-02-06", "val": 405500000}]}}),
    )
    scripted.add(COMPANY_CONCEPT_URL.format(cik=CIK, taxonomy="us-gaap", concept="SalesRevenueNet"), "[]")
    client = EdgarClient(scripted, user_agent="a", contact_email="a@b.example")
    facts = client.company_facts(CIK)
    assert facts is not None
    assert facts.revenue is not None
    assert facts.revenue.value == 12.4e9
    assert facts.shares_outstanding is not None
    assert facts.source_url == concept
    assert facts.entity_name == "Northwind Industries, Inc."


def test_company_facts_none_when_nothing_available(scripted: ScriptedFetcher) -> None:
    client = EdgarClient(scripted, user_agent="a", contact_email="a@b.example")
    assert client.company_facts(CIK) is None
    scripted.add(COMPANY_FACTS_URL.format(cik=CIK), "[1, 2]")
    assert client.company_facts(CIK) is None


def test_company_facts_with_sparse_payload(scripted: ScriptedFetcher) -> None:
    scripted.add(COMPANY_FACTS_URL.format(cik=CIK), json.dumps({"entityName": "X", "facts": {"dei": []}}))
    facts = EdgarClient(scripted, user_agent="a", contact_email="a@b.example").company_facts(CIK)
    assert facts is not None
    assert not facts.has_any


def test_http_errors_and_invalid_json_raise(scripted: ScriptedFetcher) -> None:
    client = EdgarClient(scripted, user_agent="a", contact_email="a@b.example")
    with pytest.raises(UpstreamServiceError, match="HTTP 404"):
        client.recent_filings(CIK)
    scripted.add(SUBMISSIONS_URL.format(cik=CIK), "<html>maintenance</html>")
    with pytest.raises(UpstreamServiceError, match="invalid JSON"):
        client.recent_filings(CIK)


def test_helpers() -> None:
    assert pad_cik(320193) == "0000320193"
    assert pad_cik("0000320193") == "0000320193"
    assert pad_cik("0") == "0000000000"
    for bad in ("abc", "12345678901"):
        with pytest.raises(ValueError, match="invalid CIK"):
            pad_cik(bad)
    assert normalize_company_name("AT&T Inc.") == "at and t"
    assert normalize_company_name("Northwind Industries, Inc.") == "northwind industries"
    assert latest_fact("X", {"USD": "nope"}, unit_names=("USD",)) is None
    quarterly = latest_fact(
        "X", {"USD": [{"val": 1, "fp": "Q1"}, {"val": 2, "fp": "FY"}, {"val": True}]},
        unit_names=("USD",), annual_only=True,
    )  # fmt: skip
    assert quarterly is not None
    assert quarterly.value == 2
