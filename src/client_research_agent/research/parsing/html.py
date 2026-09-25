"""HTML -> clean text with metadata.

Boilerplate (scripts, styles, navigation, footers, forms ...) is removed, and so
is every element a human reader would not see: ``hidden`` attributes,
``aria-hidden="true"``, inline ``display:none`` / ``visibility:hidden`` /
zero-size text, and screen-reader-only utility classes. Invisible text is the
classic carrier for retrieval poisoning and indirect prompt injection ("ignore
previous instructions ..." in a white-on-white div), so it never reaches the
index.

Publication date resolution order: JSON-LD ``datePublished`` -> meta tags
(``article:published_time``, ``date``, ``dc.date`` ...) -> ``<time datetime>``
-> dated text patterns near the top of the main content.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urldefrag, urljoin, urlsplit

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag
from dateutil import parser as date_parser

from client_research_agent.observability.tracing import SpanType, traced

_REMOVE_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "iframe",
    "object",
    "embed",
    "svg",
    "canvas",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "nav",
    "footer",
    "aside",
    "dialog",
)
_BOILERPLATE_ROLES = frozenset({"navigation", "banner", "contentinfo", "complementary", "search", "dialog"})
_HIDDEN_CLASSES = frozenset(
    {"sr-only", "visually-hidden", "screen-reader-text", "screenreader-only", "hidden"}
)
_CONTENT_CONTAINERS = frozenset({"article", "main"})
_RESPONSIVE_SHOW = (":block", ":flex", ":grid", ":inline", ":inline-block", ":table")
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|em|rem|%)?\s*(?:;|$)|opacity\s*:\s*0(?:\.0+)?\s*(?:;|$)",
    re.IGNORECASE,
)
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption", "figure",
        "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p", "pre",
        "section", "table", "tbody", "td", "tfoot", "th", "thead", "tr", "ul",
    }
)  # fmt: skip
_SKIP_STRING_TYPES = (Comment, Declaration, Doctype, ProcessingInstruction)
_WS = re.compile(r"[ \t\r\f\v\u00a0\u200b]+")

_DATE_META_KEYS = (
    "article:published_time",
    "og:published_time",
    "datepublished",
    "publish-date",
    "publish_date",
    "publishdate",
    "pubdate",
    "date",
    "dc.date",
    "dc.date.issued",
    "dcterms.date",
    "dcterms.created",
    "citation_publication_date",
    "parsely-pub-date",
    "sailthru.date",
)
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_TEXT_DATE_PATTERNS = (
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}\b"),
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\.?,?\s+(?:19|20)\d{{2}}\b"),
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
)
_JSONLD_DATE_KEYS = ("datePublished", "dateCreated", "uploadDate")
_JSONLD_MAX_CHARS = 200_000


@dataclass(frozen=True, slots=True)
class ParsedPage:
    url: str
    title: str
    text: str
    publication_date: date | None = None
    canonical_url: str | None = None
    links: tuple[str, ...] = ()
    link_texts: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, str] = field(default_factory=dict)
    headings: tuple[str, ...] = ()
    language: str | None = None

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def _attr(tag: Tag, name: str) -> str:
    value = tag.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return " ".join(str(v) for v in value).strip()


def _clean_line(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    if _attr(tag, "aria-hidden").lower() == "true":
        return True
    if tag.name == "input" and _attr(tag, "type").lower() == "hidden":
        return True
    style = _attr(tag, "style")
    if style and _HIDDEN_STYLE.search(style):
        return True
    classes = {c.lower() for c in _attr(tag, "class").split()}
    if not classes & _HIDDEN_CLASSES:
        return False
    # Responsive utilities ("hidden md:block") are visible on some viewports: keep them.
    return not any(c.endswith(_RESPONSIVE_SHOW) for c in classes)


def _within(tag: Tag, names: frozenset[str]) -> bool:
    return any(parent.name in names for parent in tag.parents if isinstance(parent, Tag))


def extract_text(root: Tag) -> str:
    """Visible text with block elements on their own lines; iterative so deep DOMs cannot recurse out."""
    pieces: list[str] = []
    stack: list[Tag | NavigableString | None] = [root]
    while stack:
        node = stack.pop()
        if node is None:
            pieces.append("\n")
            continue
        if isinstance(node, NavigableString):
            if not isinstance(node, _SKIP_STRING_TYPES):
                pieces.append(str(node))
            continue
        is_block = node.name in _BLOCK_TAGS
        if is_block:
            pieces.append("\n")
            stack.append(None)
        stack.extend(child for child in reversed(node.contents) if isinstance(child, Tag | NavigableString))
    lines = (_clean_line(line) for line in "".join(pieces).split("\n"))
    return "\n".join(line for line in lines if line)


def _iter_jsonld(value: Any) -> Iterator[Mapping[str, Any]]:
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        elif isinstance(item, dict):
            yield item
            graph = item.get("@graph")
            if isinstance(graph, list | dict):
                stack.append(graph)


def parse_date_value(value: str, *, today: date | None = None) -> date | None:
    """Parse an ISO/free-form date string; rejects values without a 4-digit year or out of range."""
    value = value.strip()
    if not value or not re.search(r"(?:19|20)\d{2}", value):
        return None
    parsed: datetime | None
    try:
        parsed = date_parser.isoparse(value)
    except (ValueError, OverflowError):
        try:
            parsed = date_parser.parse(value, default=datetime(2000, 1, 1))
        except (ValueError, OverflowError):
            parsed = None
    if parsed is None:
        return None
    result = parsed.date()
    limit = today or datetime.now(UTC).date()
    if result.year < 1990 or result > date(limit.year + 1, limit.month, min(limit.day, 28)):
        return None
    return result


class HtmlParser:
    """Deterministic HTML extraction (BeautifulSoup + lxml)."""

    def __init__(
        self, *, min_paragraph_chars: int = 25, max_links: int = 2000, today: date | None = None
    ) -> None:
        self._min_paragraph_chars = min_paragraph_chars
        self._max_links = max_links
        self._today = today

    @traced(span_type=SpanType.PARSER)
    def parse(self, html: str, url: str) -> ParsedPage:
        soup = BeautifulSoup(html, "lxml")
        jsonld = self._jsonld_objects(soup)
        meta = self._meta(soup)
        base = self._base_url(soup, url)
        html_tag = soup.find("html")
        language = _attr(html_tag, "lang")[:2].lower() or None if isinstance(html_tag, Tag) else None

        self._remove_hidden(soup)
        links, link_texts = self._links(soup, base)
        title = self._title(soup, meta, jsonld)
        canonical = self._canonical(soup, base, meta)
        self._remove_boilerplate(soup)
        content = self._main_content(soup)
        text = extract_text(content) if content is not None else ""
        headings = self._headings(content if content is not None else soup)
        published = self._publication_date(soup, meta, jsonld, text)
        return ParsedPage(
            url=url,
            title=title,
            text=text,
            publication_date=published,
            canonical_url=canonical,
            links=links,
            link_texts=link_texts,
            meta=meta,
            headings=headings,
            language=language,
        )

    @staticmethod
    def _jsonld_objects(soup: BeautifulSoup) -> list[Mapping[str, Any]]:
        objects: list[Mapping[str, Any]] = []
        for script in soup.find_all("script"):
            if not isinstance(script, Tag) or "ld+json" not in _attr(script, "type").lower():
                continue
            raw = script.get_text()[:_JSONLD_MAX_CHARS]
            try:
                payload = json.loads(raw)
            except ValueError:
                continue
            objects.extend(_iter_jsonld(payload))
        return objects

    @staticmethod
    def _meta(soup: BeautifulSoup) -> dict[str, str]:
        meta: dict[str, str] = {}
        for tag in soup.find_all("meta"):
            if not isinstance(tag, Tag):
                continue
            key = (_attr(tag, "property") or _attr(tag, "name") or _attr(tag, "itemprop")).lower()
            content = _attr(tag, "content")
            if key and content and key not in meta:
                meta[key] = content[:2000]
        return meta

    @staticmethod
    def _base_url(soup: BeautifulSoup, url: str) -> str:
        base = soup.find("base")
        if isinstance(base, Tag):
            href = _attr(base, "href")
            candidate = urljoin(url, href) if href else ""
            if urlsplit(candidate).scheme in ("http", "https"):
                return candidate
        return url

    @staticmethod
    def _remove_hidden(soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(True)):
            if isinstance(tag, Tag) and not tag.decomposed and tag.name != "html" and _is_hidden(tag):
                tag.decompose()

    def _links(self, soup: BeautifulSoup, base: str) -> tuple[tuple[str, ...], dict[str, str]]:
        ordered: dict[str, str] = {}
        for anchor in soup.find_all("a"):
            if not isinstance(anchor, Tag):
                continue
            href = _attr(anchor, "href")
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            absolute, _fragment = urldefrag(urljoin(base, href))
            if urlsplit(absolute).scheme not in ("http", "https"):
                continue
            label = _clean_line(anchor.get_text(" "))[:200]
            if not ordered.get(absolute):
                ordered[absolute] = label
            if len(ordered) >= self._max_links:
                break
        return tuple(ordered), ordered

    @staticmethod
    def _title(soup: BeautifulSoup, meta: Mapping[str, str], jsonld: list[Mapping[str, Any]]) -> str:
        for obj in jsonld:
            headline = obj.get("headline")
            if isinstance(headline, str) and headline.strip():
                return _clean_line(headline)[:500]
        if meta.get("og:title"):
            return _clean_line(meta["og:title"])[:500]
        title = soup.find("title")
        if isinstance(title, Tag) and title.get_text(strip=True):
            return _clean_line(title.get_text(" "))[:500]
        heading = soup.find("h1")
        if isinstance(heading, Tag):
            return _clean_line(heading.get_text(" "))[:500]
        return ""

    @staticmethod
    def _canonical(soup: BeautifulSoup, base: str, meta: Mapping[str, str]) -> str | None:
        for link in soup.find_all("link"):
            if isinstance(link, Tag) and "canonical" in _attr(link, "rel").lower().split():
                href = _attr(link, "href")
                if href:
                    return urljoin(base, href)
        og_url = meta.get("og:url")
        return urljoin(base, og_url) if og_url else None

    @staticmethod
    def _remove_boilerplate(soup: BeautifulSoup) -> None:
        for tag in list(soup.find_all(True)):
            if not isinstance(tag, Tag) or tag.decomposed:
                continue
            boilerplate = tag.name in _REMOVE_TAGS or _attr(tag, "role").lower() in _BOILERPLATE_ROLES
            stray_header = tag.name == "header" and not _within(tag, _CONTENT_CONTAINERS)
            if boilerplate or stray_header:
                tag.decompose()

    def _main_content(self, soup: BeautifulSoup) -> Tag | None:
        articles = [a for a in soup.find_all("article") if isinstance(a, Tag)]
        if articles:
            return max(articles, key=lambda a: len(a.get_text(" ", strip=True)))
        for selector in ("main", '[role="main"]'):
            found = soup.select_one(selector)
            if found is not None and found.get_text(strip=True):
                return found
        dense = self._densest_block(soup)
        if dense is not None:
            return dense
        body = soup.find("body")
        return body if isinstance(body, Tag) else soup

    def _densest_block(self, soup: BeautifulSoup) -> Tag | None:
        """Readability-style scoring: paragraphs vote for their parent (full) and grandparent (half)."""
        scores: dict[int, float] = {}
        nodes: dict[int, Tag] = {}
        for paragraph in soup.find_all(["p", "pre", "blockquote"]):
            if not isinstance(paragraph, Tag):
                continue
            text = paragraph.get_text(" ", strip=True)
            if len(text) < self._min_paragraph_chars:
                continue
            points = 1.0 + text.count(",") + min(len(text) / 100.0, 3.0)
            parent = paragraph.parent
            for weight, node in ((1.0, parent), (0.5, parent.parent if parent is not None else None)):
                if isinstance(node, Tag) and node.name not in ("html", "[document]"):
                    nodes[id(node)] = node
                    scores[id(node)] = scores.get(id(node), 0.0) + points * weight
        if not scores:
            return None

        def adjusted(key: int) -> float:
            node = nodes[key]
            total = len(node.get_text(" ", strip=True)) or 1
            linked = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
            return scores[key] * (1.0 - min(linked / total, 1.0))

        return nodes[max(scores, key=adjusted)]

    @staticmethod
    def _headings(root: Tag) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for heading in root.find_all(["h1", "h2", "h3"]):
            if isinstance(heading, Tag):
                text = _clean_line(heading.get_text(" "))
                if text:
                    seen.setdefault(text[:300], None)
        return tuple(seen)[:50]

    def _publication_date(
        self,
        soup: BeautifulSoup,
        meta: Mapping[str, str],
        jsonld: list[Mapping[str, Any]],
        text: str,
    ) -> date | None:
        for obj in jsonld:
            for key in _JSONLD_DATE_KEYS:
                value = obj.get(key)
                if isinstance(value, str) and (parsed := parse_date_value(value, today=self._today)):
                    return parsed
        for key in _DATE_META_KEYS:
            value = meta.get(key)
            if value and (parsed := parse_date_value(value, today=self._today)):
                return parsed
        times = [t for t in soup.find_all("time") if isinstance(t, Tag)]
        times.sort(
            key=lambda t: 0 if (t.has_attr("pubdate") or _attr(t, "itemprop") == "datePublished") else 1
        )
        for time_tag in times:
            value = _attr(time_tag, "datetime") or time_tag.get_text(" ", strip=True)
            if value and (parsed := parse_date_value(value, today=self._today)):
                return parsed
        head = text[:2000]
        for pattern in _TEXT_DATE_PATTERNS:
            match = pattern.search(head)
            if match and (parsed := parse_date_value(match.group(0), today=self._today)):
                return parsed
        return None
