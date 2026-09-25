"""SEC EDGAR client (public, free, no key).

Uses the documented JSON endpoints:

* ``https://www.sec.gov/files/company_tickers.json`` - ticker/name -> CIK
* ``https://data.sec.gov/submissions/CIK##########.json`` - recent filings
* ``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`` - XBRL facts
  (falls back to the much smaller per-concept ``companyconcept`` endpoint
  when the full facts document is unavailable or exceeds the size cap)

SEC fair-access policy requires a declared User-Agent that includes a contact
e-mail and at most 10 requests/second; the User-Agent is sent on every call
and pacing is enforced by the shared fetcher's per-host rate limiter.

Facts taken from XBRL are *reported values*, so downstream scoring may treat
them as verified facts (for example revenue for company-scale scoring).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import quote

from client_research_agent.config.settings import CrawlerSettings
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.services.ports import HttpFetcher
from client_research_agent.utils.errors import AgentError, UpstreamServiceError

_log = get_logger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{document}"

DEFAULT_FORMS: tuple[str, ...] = ("10-K", "10-Q", "8-K", "DEF 14A")
REVENUE_CONCEPTS: tuple[str, ...] = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
_CORPORATE_SUFFIXES = re.compile(
    r"\b(?:inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|holdings?|group|sa|ag|nv|se)\b\.?",
    re.IGNORECASE,
)
_SAFE_DOCUMENT = re.compile(r"^[A-Za-z0-9._-]{1,200}$")
_ANNUAL_MIN_DAYS = 330


@dataclass(frozen=True, slots=True)
class EdgarFiling:
    cik: str
    accession_number: str
    form: str
    filing_date: date
    primary_document: str
    url: str
    report_date: date | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class FactValue:
    concept: str
    value: float
    unit: str
    period_end: date | None = None
    period_start: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    form: str | None = None
    filed: date | None = None


@dataclass(frozen=True, slots=True)
class CompanyFacts:
    cik: str
    entity_name: str
    source_url: str
    revenue: FactValue | None = None
    shares_outstanding: FactValue | None = None
    employees: FactValue | None = None

    @property
    def has_any(self) -> bool:
        return any(v is not None for v in (self.revenue, self.shares_outstanding, self.employees))


def pad_cik(cik: str | int) -> str:
    digits = str(cik).strip().lstrip("0") or "0"
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"invalid CIK: {cik!r}")
    return digits.zfill(10)


def normalize_company_name(name: str) -> str:
    cleaned = _CORPORATE_SUFFIXES.sub(" ", name.lower().replace("&", " and "))
    return re.sub(r"[^a-z0-9]+", " ", cleaned).strip()


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _column(recent: Mapping[str, Any], key: str, length: int) -> list[Any]:
    values = recent.get(key)
    if not isinstance(values, list):
        return [None] * length
    return values + [None] * (length - len(values))


def _fact_from_entry(concept: str, unit: str, entry: Mapping[str, Any]) -> FactValue | None:
    value = entry.get("val")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    fy = entry.get("fy")
    return FactValue(
        concept=concept,
        value=float(value),
        unit=unit,
        period_end=_parse_date(entry.get("end")),
        period_start=_parse_date(entry.get("start")),
        fiscal_year=fy if isinstance(fy, int) else None,
        fiscal_period=str(entry["fp"]) if entry.get("fp") else None,
        form=str(entry["form"]) if entry.get("form") else None,
        filed=_parse_date(entry.get("filed")),
    )


def _is_annual(fact: FactValue) -> bool:
    if fact.period_start and fact.period_end:
        return (fact.period_end - fact.period_start).days >= _ANNUAL_MIN_DAYS
    return fact.fiscal_period == "FY"


def latest_fact(
    concept: str,
    units: Mapping[str, Any],
    *,
    unit_names: Sequence[str],
    annual_only: bool = False,
) -> FactValue | None:
    """Most recent value (by period end, then filing date) for ``concept`` in the given units."""
    best: FactValue | None = None
    for unit in unit_names:
        entries = units.get(unit)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            fact = _fact_from_entry(concept, unit, entry)
            if fact is None or (annual_only and not _is_annual(fact)):
                continue
            key = (fact.period_end or date.min, fact.filed or date.min)
            if best is None or key > (best.period_end or date.min, best.filed or date.min):
                best = fact
    return best


def _latest_of(facts: Iterable[FactValue | None]) -> FactValue | None:
    present = [f for f in facts if f is not None]
    if not present:
        return None
    return max(present, key=lambda f: (f.period_end or date.min, f.filed or date.min))


class EdgarClient:
    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        user_agent: str,
        contact_email: str,
        forms: Sequence[str] = DEFAULT_FORMS,
    ) -> None:
        if "@" not in contact_email:
            raise ValueError("SEC fair-access policy requires a contact e-mail in the User-Agent")
        self._fetcher = fetcher
        self._headers = {"User-Agent": f"{user_agent} {contact_email}", "Accept": "application/json"}
        self._forms = tuple(forms)
        self._tickers: list[tuple[str, str, str]] | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, fetcher: HttpFetcher, crawler: CrawlerSettings) -> EdgarClient:
        return cls(fetcher, user_agent=crawler.user_agent, contact_email=crawler.contact_email)

    def _get_json(self, url: str) -> Any:
        result = self._fetcher.fetch(url, headers=self._headers)
        if not result.ok:
            raise UpstreamServiceError(
                f"SEC returned HTTP {result.status_code} for {url}", result.status_code
            )
        try:
            return json.loads(result.text)
        except ValueError as exc:
            raise UpstreamServiceError(f"SEC returned invalid JSON for {url}") from exc

    def _ticker_table(self) -> list[tuple[str, str, str]]:
        with self._lock:
            if self._tickers is None:
                payload = self._get_json(TICKERS_URL)
                rows = payload.values() if isinstance(payload, dict) else payload
                table: list[tuple[str, str, str]] = []
                for row in rows if isinstance(rows, Iterable) else ():
                    if isinstance(row, Mapping) and row.get("cik_str") is not None and row.get("ticker"):
                        table.append(
                            (str(row["ticker"]).upper(), str(row.get("title", "")), pad_cik(row["cik_str"]))
                        )
                self._tickers = table
            return self._tickers

    @traced(span_type=SpanType.TOOL)
    def resolve_cik(self, *, ticker: str | None = None, company_name: str | None = None) -> str | None:
        """10-digit CIK for a ticker (exact) or company name (normalized exact match)."""
        table = self._ticker_table()
        if ticker:
            wanted = ticker.strip().upper()
            for symbol, _title, cik in table:
                if symbol == wanted:
                    return cik
        if company_name:
            wanted_name = normalize_company_name(company_name)
            for _symbol, title, cik in table:
                if wanted_name and normalize_company_name(title) == wanted_name:
                    return cik
        return None

    @traced(span_type=SpanType.TOOL)
    def recent_filings(
        self,
        cik: str,
        *,
        max_filings: int = 10,
        forms: Sequence[str] | None = None,
    ) -> list[EdgarFiling]:
        padded = pad_cik(cik)
        payload = self._get_json(SUBMISSIONS_URL.format(cik=padded))
        recent = payload.get("filings", {}).get("recent", {}) if isinstance(payload, Mapping) else {}
        if not isinstance(recent, Mapping):
            return []
        accessions = recent.get("accessionNumber")
        if not isinstance(accessions, list):
            return []
        length = len(accessions)
        wanted = {f.upper() for f in (forms or self._forms)}
        filings: list[EdgarFiling] = []
        rows = zip(
            accessions,
            _column(recent, "form", length),
            _column(recent, "filingDate", length),
            _column(recent, "reportDate", length),
            _column(recent, "primaryDocument", length),
            _column(recent, "primaryDocDescription", length),
            strict=True,
        )
        cik_int = int(padded)
        for accession, form, filed, reported, document, description in rows:
            filing_date = _parse_date(filed)
            if not isinstance(accession, str) or not isinstance(form, str) or filing_date is None:
                continue
            if (
                form.upper() not in wanted
                or not isinstance(document, str)
                or not _SAFE_DOCUMENT.match(document)
            ):
                continue
            accession_path = accession.replace("-", "")
            if not accession_path.isdigit():
                continue
            filings.append(
                EdgarFiling(
                    cik=padded,
                    accession_number=accession,
                    form=form,
                    filing_date=filing_date,
                    primary_document=document,
                    url=ARCHIVE_URL.format(
                        cik_int=cik_int, accession=accession_path, document=quote(document)
                    ),
                    report_date=_parse_date(reported),
                    description=str(description or ""),
                )
            )
        filings.sort(key=lambda f: (f.filing_date, f.accession_number), reverse=True)
        return filings[: max(0, max_filings)]

    @traced(span_type=SpanType.TOOL)
    def company_facts(self, cik: str) -> CompanyFacts | None:
        """Latest annual revenue, shares outstanding and (if reported) employee count."""
        padded = pad_cik(cik)
        url = COMPANY_FACTS_URL.format(cik=padded)
        try:
            payload = self._get_json(url)
        except AgentError as exc:
            _log.info("edgar.companyfacts_unavailable", cik=padded, error=type(exc).__name__)
            return self._facts_from_concepts(padded)
        if not isinstance(payload, Mapping):
            return None
        facts = payload.get("facts")
        facts = facts if isinstance(facts, Mapping) else {}
        gaap = _mapping(facts.get("us-gaap"))
        dei = _mapping(facts.get("dei"))

        def units(namespace: Mapping[str, Any], concept: str) -> Mapping[str, Any]:
            node = namespace.get(concept)
            found = node.get("units") if isinstance(node, Mapping) else None
            return found if isinstance(found, Mapping) else {}

        revenue = _latest_of(
            latest_fact(c, units(gaap, c), unit_names=("USD",), annual_only=True) for c in REVENUE_CONCEPTS
        )
        shares = latest_fact(
            "EntityCommonStockSharesOutstanding",
            units(dei, "EntityCommonStockSharesOutstanding"),
            unit_names=("shares",),
        )
        employees = latest_fact(
            "EntityNumberOfEmployees",
            units(dei, "EntityNumberOfEmployees"),
            unit_names=("employee", "employees", "pure", "Employee"),
        )
        return CompanyFacts(
            cik=padded,
            entity_name=str(payload.get("entityName", "")),
            source_url=url,
            revenue=revenue,
            shares_outstanding=shares,
            employees=employees,
        )

    def _concept(self, cik: str, taxonomy: str, concept: str) -> tuple[str, Mapping[str, Any]]:
        try:
            payload = self._get_json(COMPANY_CONCEPT_URL.format(cik=cik, taxonomy=taxonomy, concept=concept))
        except AgentError:
            return "", {}
        if not isinstance(payload, Mapping):
            return "", {}
        found = payload.get("units")
        return str(payload.get("entityName", "")), found if isinstance(found, Mapping) else {}

    def _facts_from_concepts(self, cik: str) -> CompanyFacts | None:
        entity_name = ""
        revenue_candidates: list[FactValue | None] = []
        for concept in REVENUE_CONCEPTS:
            name, found = self._concept(cik, "us-gaap", concept)
            entity_name = entity_name or name
            revenue_candidates.append(latest_fact(concept, found, unit_names=("USD",), annual_only=True))
        name, share_units = self._concept(cik, "dei", "EntityCommonStockSharesOutstanding")
        entity_name = entity_name or name
        revenue = _latest_of(revenue_candidates)
        cited = ("us-gaap", revenue.concept) if revenue else ("dei", "EntityCommonStockSharesOutstanding")
        facts = CompanyFacts(
            cik=cik,
            entity_name=entity_name,
            source_url=COMPANY_CONCEPT_URL.format(cik=cik, taxonomy=cited[0], concept=cited[1]),
            revenue=revenue,
            shares_outstanding=latest_fact(
                "EntityCommonStockSharesOutstanding", share_units, unit_names=("shares",)
            ),
        )
        return facts if facts.has_any else None
