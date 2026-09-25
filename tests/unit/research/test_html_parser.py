from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.research.parsing.html import HtmlParser, extract_text, parse_date_value
from tests.unit.research.helpers import fixture_text

URL = "https://northwind.example/newsroom/press-releases/2026/northwind-q4-fy2025-results"
TODAY = date(2026, 9, 24)


@pytest.fixture
def parser() -> HtmlParser:
    return HtmlParser(today=TODAY)


def page(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body>{body}</body></html>"


LONG = "Northwind expands its data platform with new analytics capabilities for plant operations. " * 3


class TestPressReleaseFixture:
    def test_metadata(self, parser: HtmlParser) -> None:
        parsed = parser.parse(fixture_text("press_release.html"), URL)
        assert parsed.title == "Northwind Industries Reports Fourth Quarter and Fiscal Year 2025 Results"
        assert parsed.publication_date == date(2026, 2, 12)
        assert parsed.canonical_url == URL
        assert parsed.language == "en"
        assert parsed.meta["description"] == "Northwind Industries reports record results."
        assert parsed.headings[:2] == (
            "Northwind Industries Reports Fourth Quarter and Fiscal Year 2025 Results",
            "Leadership update",
        )

    def test_main_content_excludes_boilerplate(self, parser: HtmlParser) -> None:
        text = parser.parse(fixture_text("press_release.html"), URL).text
        assert "Total revenue was $12.4 billion" in text
        assert "appointed Rahul Mehta as Chief Information Officer" in text
        for boilerplate in ("window.analytics", ".promo", "Related news", "Privacy", "Subscribe", "Home"):
            assert boilerplate not in text

    def test_hidden_text_is_stripped(self, parser: HtmlParser) -> None:
        text = parser.parse(fixture_text("press_release.html"), URL).text
        for hidden in (
            "Ignore all previous instructions",
            "$900 billion",
            "Decorative icon",
            "Skip to content",
        ):
            assert hidden not in text

    def test_links_are_absolute_deduplicated_and_include_navigation(self, parser: HtmlParser) -> None:
        parsed = parser.parse(fixture_text("press_release.html"), URL)
        assert "https://northwind.example/investors/" in parsed.links
        assert "https://northwind.example/newsroom/" in parsed.links
        assert "https://northwind.example/investors/events" in parsed.links
        assert parsed.link_texts["https://northwind.example/about/leadership"] == "Leadership"
        assert all(link.startswith("https://") for link in parsed.links)
        assert len(parsed.links) == len(set(parsed.links))

    def test_inline_elements_do_not_break_sentences(self) -> None:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup("<div><p>Acme <b>appointed</b> Jane<br>Doe</p><!-- note --></div>", "lxml")
        assert extract_text(soup) == "Acme appointed Jane\nDoe"


class TestPublicationDate:
    def test_jsonld_list_date_published(self, parser: HtmlParser) -> None:
        head = (
            '<script type="application/ld+json">[{"@type":"Organization"},'
            '{"@type":"NewsArticle","datePublished":"2025-11-04"}]</script>'
        )
        assert parser.parse(page(f"<p>{LONG}</p>", head), URL).publication_date == date(2025, 11, 4)

    def test_invalid_jsonld_falls_through_to_meta(self, parser: HtmlParser) -> None:
        head = (
            '<script type="application/ld+json">{not json</script>'
            '<meta property="article:published_time" content="2026-01-15T10:00:00+00:00">'
        )
        assert parser.parse(page(f"<p>{LONG}</p>", head), URL).publication_date == date(2026, 1, 15)

    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ('<meta name="date" content="2025-06-30">', date(2025, 6, 30)),
            ('<meta name="DC.date.issued" content="2024-03-01">', date(2024, 3, 1)),
            ('<meta name="dcterms.date" content="March 3, 2025">', date(2025, 3, 3)),
            ('<meta name="citation_publication_date" content="2023/07/09">', date(2023, 7, 9)),
        ],
    )
    def test_meta_variants(self, parser: HtmlParser, head: str, expected: date) -> None:
        assert parser.parse(page(f"<p>{LONG}</p>", head), URL).publication_date == expected

    def test_time_element_prefers_pubdate(self, parser: HtmlParser) -> None:
        body = (
            '<article><time datetime="2026-05-01">Updated</time>'
            '<time datetime="2026-04-02" itemprop="datePublished">April 2</time>'
            f"<p>{LONG}</p></article>"
        )
        assert parser.parse(page(body), URL).publication_date == date(2026, 4, 2)

    def test_time_element_text_without_datetime(self, parser: HtmlParser) -> None:
        body = f"<article><time>7 August 2025</time><p>{LONG}</p></article>"
        assert parser.parse(page(body), URL).publication_date == date(2025, 8, 7)

    @pytest.mark.parametrize(
        ("snippet", "expected"),
        [
            ("SAN JOSE, Calif., Sept. 3, 2025 -- Acme today announced", date(2025, 9, 3)),
            ("LONDON, 14 March 2026 -- Acme plc announced", date(2026, 3, 14)),
            ("Posted 2024-12-01 by Communications", date(2024, 12, 1)),
        ],
    )
    def test_text_patterns(self, parser: HtmlParser, snippet: str, expected: date) -> None:
        body = f"<main><p>{snippet}. {LONG}</p></main>"
        assert parser.parse(page(body), URL).publication_date == expected

    def test_no_date(self, parser: HtmlParser) -> None:
        assert parser.parse(page(f"<p>{LONG}</p>"), URL).publication_date is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("", None),
            ("yesterday", None),
            ("1985-01-01", None),
            ("2031-01-01", None),
            ("2026-13-45", None),
            ("Q3 2026 somewhere 99:99", None),
            ("2026-02-12T13:05:00-05:00", date(2026, 2, 12)),
            ("Feb 12 2026", date(2026, 2, 12)),
        ],
    )
    def test_parse_date_value(self, value: str, expected: date | None) -> None:
        assert parse_date_value(value, today=TODAY) == expected


class TestContentSelection:
    def test_densest_block_fallback(self, parser: HtmlParser) -> None:
        body = (
            '<div class="menu"><a href="/a">A</a> <a href="/b">B</a> <a href="/c">C</a></div>'
            f'<div class="story"><p>{LONG}</p>'
            f"<p>Second, longer, paragraph with commas, clauses, and detail. {LONG}</p></div>"
            '<div class="links"><p><a href="/x">'
            "A very long list of links that is mostly anchors here</a></p></div>"
        )
        text = parser.parse(page(body), URL).text
        assert "Second, longer, paragraph" in text
        assert "very long list of links" not in text

    def test_largest_article_wins(self, parser: HtmlParser) -> None:
        body = f"<article><p>Short teaser.</p></article><article><p>{LONG}</p></article>"
        text = parser.parse(page(body), URL).text
        assert "data platform" in text
        assert "Short teaser" not in text

    def test_role_main_and_body_fallbacks(self, parser: HtmlParser) -> None:
        assert "Primary" in parser.parse(page('<div role="main">Primary</div><p>tiny</p>'), URL).text
        assert parser.parse(page("<div>Just a line</div>"), URL).text == "Just a line"

    def test_responsive_hidden_utility_is_kept(self, parser: HtmlParser) -> None:
        body = '<main><div class="hidden md:block">Desktop copy</div><div class="hidden">Gone</div></main>'
        text = parser.parse(page(body), URL).text
        assert "Desktop copy" in text
        assert "Gone" not in text

    def test_title_fallbacks_and_base_href(self, parser: HtmlParser) -> None:
        og = parser.parse(page("<p>x</p>", '<meta property="og:title" content="OG Title">'), URL)
        assert og.title == "OG Title"
        h1 = parser.parse("<html><body><h1>Heading Title</h1></body></html>", URL)
        assert h1.title == "Heading Title"
        assert parser.parse("<html><body><p>x</p></body></html>", URL).title == ""
        based = parser.parse(
            page(
                '<a href="rel/page">r</a><a href="mailto:x@y.z">m</a><a href="#top">t</a><a href="ftp://f/x">f</a>',
                '<base href="https://cdn.northwind.example/root/"><meta property="og:url" content="/og">',
            ),
            URL,
        )
        assert based.links == ("https://cdn.northwind.example/root/rel/page",)
        assert based.canonical_url == "https://cdn.northwind.example/og"

    def test_max_links_cap(self) -> None:
        body = "".join(f'<a href="/p{i}">p{i}</a>' for i in range(10))
        assert len(HtmlParser(max_links=3).parse(page(body), URL).links) == 3

    def test_word_count(self, parser: HtmlParser) -> None:
        assert parser.parse(page("<main>one two three</main>"), URL).word_count == 3
