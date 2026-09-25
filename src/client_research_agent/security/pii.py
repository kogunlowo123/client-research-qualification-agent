"""PII detection and policy-driven redaction (OWASP LLM02 sensitive information disclosure).

Policy for client research
--------------------------
Executives named in press releases, filings and leadership announcements are
*public business information*: person names and job titles are kept so that a
brief can say "Jane Doe, CFO, announced ...". Everything that identifies or
reaches a private individual is removed:

* personal e-mail addresses are always redacted; generic role mailboxes such
  as ``press@`` or ``investors@`` are public corporate contact points and kept
  (configurable);
* phone numbers, US SSNs, payment card numbers (Luhn-validated), IBANs
  (ISO 13616 mod-97 validated) and IPv4 addresses are always redacted;
* street addresses are redacted when they look residential (apartment/unit
  markers or residence context such as "lives at"); corporate headquarters
  addresses are public and kept.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PiiType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "us_ssn"
    CREDIT_CARD = "credit_card"
    IBAN = "iban"
    IPV4 = "ipv4"
    ADDRESS = "address"


@dataclass(frozen=True, slots=True)
class PiiFinding:
    type: PiiType
    start: int
    end: int
    fingerprint: str
    personal: bool = True

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    findings: tuple[PiiFinding, ...]
    kept: tuple[PiiFinding, ...] = ()

    @property
    def redacted_count(self) -> int:
        return len(self.findings)

    def counts(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        for finding in self.findings:
            totals[finding.type.value] = totals.get(finding.type.value, 0) + 1
        return totals


ROLE_MAILBOXES = frozenset(
    {
        "press", "media", "pr", "news", "newsroom", "communications", "comms", "investors", "investor",
        "ir", "investor.relations", "investorrelations", "info", "contact", "hello", "support", "sales",
        "enquiries", "inquiries", "careers", "jobs", "partners", "marketing", "office", "legal", "privacy",
        "security", "webmaster", "noreply", "no-reply",
    }
)  # fmt: skip

_EMAIL_RE = re.compile(r"(?<![\w.+-])([A-Za-z0-9._%+-]{1,64})@((?:[A-Za-z0-9-]{1,63}\.)+[A-Za-z]{2,24})\b")
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_SSN_CONTEXT_RE = re.compile(
    r"\b(?:ssn|social\s+security(?:\s+(?:number|no\.?))?)\s*[:#]?\s*((?!000|666|9\d\d)\d{3}\s?(?!00)\d{2}\s?(?!0000)\d{4})\b",
    re.IGNORECASE,
)
_CARD_RE = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_IBAN_RE = re.compile(r"\b([A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30})\b")
_IPV4_RE = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d|\.\d)"
)
_PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}(?:[\s.-]\d{2,4}){1,3}(?![\w])"
    r"|(?<![\w+])\+\d{8,15}(?!\w)"
)
_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|"
    r"Circle|Cir|Parkway|Pkwy|Highway|Hwy|Square|Sq)"
)
_ADDRESS_RE = re.compile(
    rf"\b\d{{1,6}}\s+(?:[A-Z][A-Za-z'-]+\s){{1,4}}{_STREET_SUFFIX}\b\.?"
    r"(?:,?\s*(?:Apt\.?|Apartment|Unit|Suite|Ste\.?|#)\s*[A-Za-z0-9-]+)?"
)
_RESIDENTIAL_UNIT_RE = re.compile(r"\b(?:Apt\.?|Apartment|Unit|#)\s*[A-Za-z0-9-]+", re.IGNORECASE)
_RESIDENCE_CONTEXT_RE = re.compile(
    r"\b(?:home|resides?|residing|residence|lives|living|personal|domicile|his\s+house|her\s+house|mailing)\b",
    re.IGNORECASE,
)

_IBAN_LENGTHS: Mapping[str, int] = {
    "AD": 24, "AE": 23, "AT": 20, "BE": 16, "BG": 22, "BH": 22, "BR": 29, "CH": 21, "CY": 28, "CZ": 24,
    "DE": 22, "DK": 18, "EE": 20, "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GI": 23, "GR": 27, "HR": 21,
    "HU": 28, "IE": 22, "IL": 23, "IS": 26, "IT": 27, "KW": 30, "KZ": 20, "LB": 28, "LI": 21, "LT": 20,
    "LU": 20, "LV": 21, "MC": 27, "MT": 31, "NL": 18, "NO": 15, "PK": 24, "PL": 28, "PT": 25, "QA": 29,
    "RO": 24, "SA": 24, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "TR": 26, "UA": 29,
}  # fmt: skip

_PRIORITY: Mapping[PiiType, int] = {
    PiiType.SSN: 0,
    PiiType.CREDIT_CARD: 1,
    PiiType.IBAN: 2,
    PiiType.EMAIL: 3,
    PiiType.IPV4: 4,
    PiiType.PHONE: 5,
    PiiType.ADDRESS: 6,
}


def luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 12:
        return False
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        doubled = digit * 2 if index % 2 == 1 else digit
        checksum += doubled - 9 if doubled > 9 else doubled
    return checksum % 10 == 0


def iban_valid(candidate: str) -> bool:
    iban = candidate.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", iban):
        return False
    expected = _IBAN_LENGTHS.get(iban[:2])
    if expected is not None and len(iban) != expected:
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _phone_digit_count_ok(value: str) -> bool:
    digits = sum(ch.isdigit() for ch in value)
    return 9 <= digits <= 15


def _looks_like_date_or_number(value: str) -> bool:
    stripped = value.strip()
    if re.fullmatch(r"\d{4}[-./]\d{2}[-./]\d{2}", stripped) or re.fullmatch(
        r"\d{2}[-./]\d{2}[-./]\d{4}", stripped
    ):
        return True
    # Grouped numbers written with single spaces and 3-digit groups are usually quantities ("1 250 000").
    return bool(re.fullmatch(r"\d{1,3}(?: \d{3})+", stripped))


class PiiDetector:
    """Finds PII spans; validation (Luhn, mod-97, octet ranges) keeps false positives low."""

    def __init__(self, *, role_mailboxes: Iterable[str] = ROLE_MAILBOXES) -> None:
        self._role_mailboxes = frozenset(m.lower() for m in role_mailboxes)

    def detect(self, text: str) -> list[PiiFinding]:
        candidates: list[PiiFinding] = []
        candidates.extend(self._emails(text))
        for match in _SSN_RE.finditer(text):
            candidates.append(self._finding(PiiType.SSN, match.start(), match.end(), match.group(0)))
        for match in _SSN_CONTEXT_RE.finditer(text):
            candidates.append(self._finding(PiiType.SSN, match.start(1), match.end(1), match.group(1)))
        for match in _CARD_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19 and luhn_valid(digits):
                candidates.append(self._finding(PiiType.CREDIT_CARD, match.start(), match.end(), digits))
        for match in _IBAN_RE.finditer(text):
            if iban_valid(match.group(1)):
                candidates.append(self._finding(PiiType.IBAN, match.start(1), match.end(1), match.group(1)))
        for match in _IPV4_RE.finditer(text):
            candidates.append(self._finding(PiiType.IPV4, match.start(), match.end(), match.group(0)))
        for match in _PHONE_RE.finditer(text):
            value = match.group(0)
            if _phone_digit_count_ok(value) and not _looks_like_date_or_number(value):
                candidates.append(self._finding(PiiType.PHONE, match.start(), match.end(), value))
        for match in _ADDRESS_RE.finditer(text):
            window = text[max(0, match.start() - 60) : match.start()]
            residential = bool(_RESIDENTIAL_UNIT_RE.search(match.group(0))) and not re.search(
                r"\b(?:Suite|Ste)\b", match.group(0)
            )
            personal = residential or bool(_RESIDENCE_CONTEXT_RE.search(window))
            candidates.append(
                self._finding(PiiType.ADDRESS, match.start(), match.end(), match.group(0), personal=personal)
            )
        return self._resolve_overlaps(candidates)

    def _emails(self, text: str) -> Iterable[PiiFinding]:
        for match in _EMAIL_RE.finditer(text):
            local = match.group(1).lower()
            personal = local not in self._role_mailboxes
            yield self._finding(PiiType.EMAIL, match.start(), match.end(), match.group(0), personal=personal)

    @staticmethod
    def _finding(kind: PiiType, start: int, end: int, value: str, *, personal: bool = True) -> PiiFinding:
        return PiiFinding(type=kind, start=start, end=end, fingerprint=_fingerprint(value), personal=personal)

    @staticmethod
    def _resolve_overlaps(candidates: list[PiiFinding]) -> list[PiiFinding]:
        ordered = sorted(candidates, key=lambda f: (_PRIORITY[f.type], f.start, -(f.end - f.start)))
        chosen: list[PiiFinding] = []
        for finding in ordered:
            if all(finding.end <= other.start or finding.start >= other.end for other in chosen):
                chosen.append(finding)
        return sorted(chosen, key=lambda f: f.start)


@dataclass(frozen=True, slots=True)
class PiiPolicy:
    """Which finding types are redacted, and whether public business contacts are preserved."""

    always_redact: frozenset[PiiType] = field(
        default_factory=lambda: frozenset(
            {PiiType.PHONE, PiiType.SSN, PiiType.CREDIT_CARD, PiiType.IBAN, PiiType.IPV4}
        )
    )
    keep_role_mailboxes: bool = True
    keep_business_addresses: bool = True

    def should_redact(self, finding: PiiFinding) -> bool:
        if finding.type in self.always_redact:
            return True
        if finding.type is PiiType.EMAIL:
            return finding.personal or not self.keep_role_mailboxes
        if finding.type is PiiType.ADDRESS:
            return finding.personal or not self.keep_business_addresses
        return True


class PiiRedactor:
    def __init__(self, detector: PiiDetector | None = None, policy: PiiPolicy | None = None) -> None:
        self._detector = detector or PiiDetector()
        self._policy = policy or PiiPolicy()

    @property
    def detector(self) -> PiiDetector:
        return self._detector

    def redact(self, text: str) -> RedactionResult:
        findings = self._detector.detect(text)
        redacted: list[PiiFinding] = []
        kept: list[PiiFinding] = []
        pieces: list[str] = []
        cursor = 0
        for finding in findings:
            if not self._policy.should_redact(finding):
                kept.append(finding)
                continue
            pieces.append(text[cursor : finding.start])
            pieces.append(f"[REDACTED_{finding.type.value.upper()}]")
            cursor = finding.end
            redacted.append(finding)
        pieces.append(text[cursor:])
        return RedactionResult(text="".join(pieces), findings=tuple(redacted), kept=tuple(kept))

    def redact_value(self, value: Any) -> Any:
        """Recursively redact strings inside mappings / sequences (used for audit payloads)."""
        if isinstance(value, str):
            return self.redact(value).text
        if isinstance(value, Mapping):
            return {key: self.redact_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self.redact_value(item) for item in value)
        return value

    def contains_pii(self, text: str) -> bool:
        return any(self._policy.should_redact(f) for f in self._detector.detect(text))
