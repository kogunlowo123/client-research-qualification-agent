from __future__ import annotations

import pytest

from client_research_agent.models import DocumentType
from client_research_agent.research.parsing.classifier import DocumentTypeClassifier

classifier = DocumentTypeClassifier()


@pytest.mark.parametrize(
    ("url", "title", "text", "expected"),
    [
        (
            "https://www.sec.gov/Archives/edgar/data/1234567/000123456726000012/nwnd-20251231.htm",
            "nwnd-20251231",
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION Washington, D.C. FORM 10-K ANNUAL REPORT "
            "PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934",
            DocumentType.SEC_FILING,
        ),
        (
            "https://northwind.example/newsroom/press-releases/2026/q4-results",
            "Northwind Reports Fourth Quarter and Fiscal Year 2025 Results",
            "CHICAGO /PRNewswire/ -- Revenue was $12.4 billion. Diluted EPS was $2.14. Non-GAAP margin 17%.",
            DocumentType.EARNINGS_RELEASE,
        ),
        (
            "https://northwind.example/newsroom/press-releases/2026/cio",
            "Northwind Appoints Rahul Mehta as Chief Information Officer",
            "CHICAGO (BUSINESS WIRE) -- Northwind today announced it has appointed Rahul Mehta as CIO.",
            DocumentType.LEADERSHIP_ANNOUNCEMENT,
        ),
        (
            "https://investors.northwind.example/stock-information",
            "Investor Relations | Northwind",
            "Stock information, analyst coverage and SEC filings for shareholders.",
            DocumentType.INVESTOR_RELATIONS,
        ),
        (
            "https://northwind.example/newsroom/press-releases/2026/platform",
            "Northwind Launches Industrial AI Platform",
            "CHICAGO, March 2, 2026 /PRNewswire/ -- Northwind today announced the launch of a platform.",
            DocumentType.PRESS_RELEASE,
        ),
        (
            "https://www.gartner.com/en/newsroom/press-releases/2026-01-10-gartner-forecast",
            "Gartner Forecasts Worldwide IT Spending",
            "Gartner forecasts worldwide IT spending to grow in 2026.",
            DocumentType.ANALYST_PUBLIC,
        ),
    ],
)
def test_rules(url: str, title: str, text: str, expected: DocumentType) -> None:
    result = classifier.classify(url, title, text)
    assert result.document_type is expected
    assert 0.5 <= result.confidence <= 0.95
    assert result.signals


def test_fallback_is_low_confidence_corporate_webpage() -> None:
    result = classifier.classify("https://northwind.example/about", "About Northwind", "We build pumps.")
    assert result.document_type is DocumentType.CORPORATE_WEBPAGE
    assert result.confidence == 0.4
    assert result.signals == ()


def test_sec_host_without_archive_path_still_counts() -> None:
    result = classifier.classify("https://www.sec.gov/cgi-bin/viewer", "Filing viewer", "")
    assert result.document_type is DocumentType.SEC_FILING
    assert "host:sec.gov" in result.signals


def test_hint_overrides_rules_with_floor_confidence() -> None:
    result = classifier.classify(
        "https://www.gartner.com/en/articles/x", "Some article", "text", hint=DocumentType.ANALYST_PUBLIC
    )
    assert result.document_type is DocumentType.ANALYST_PUBLIC
    assert result.signals[0] == "hint"
    assert result.confidence >= 0.75


def test_min_score_threshold() -> None:
    strict = DocumentTypeClassifier(min_score=10.0)
    result = strict.classify("https://northwind.example/news/a", "Northwind announces", "")
    assert result.document_type is DocumentType.CORPORATE_WEBPAGE
