"""Validation of generated brief text before it leaves the system (OWASP LLM05 improper output handling).

A brief is rendered into CRM notes, e-mails and dashboards, so model output is
treated as untrusted too. :class:`OutputGuard` rejects text that:

* leaks prompt scaffolding (chat-template tokens, spotlight delimiters,
  datamark characters or configured system-prompt canaries);
* contains credential-shaped secrets (same patterns as log scrubbing);
* contains PII that the redaction policy would remove;
* carries unsafe links (anything but http/https, credentials in URLs,
  markdown image beacons, or links not backed by cited evidence);
* instructs the reader to run code or shell commands.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from client_research_agent.models import ClientBrief
from client_research_agent.observability.logging import scrub
from client_research_agent.security.pii import PiiRedactor
from client_research_agent.security.prompt_injection import DATAMARK
from client_research_agent.utils.errors import SecurityViolationError


class ViolationKind(StrEnum):
    PROMPT_LEAK = "prompt_leak"
    SECRET = "secret"  # noqa: S105  # nosec B105 - violation label, not a credential
    PII = "pii"
    UNSAFE_URL = "unsafe_url"
    UNCITED_URL = "uncited_url"
    CODE_EXECUTION = "code_execution_instruction"


@dataclass(frozen=True, slots=True)
class OutputViolation:
    kind: ViolationKind
    detail: str
    location: str = "text"


@dataclass(frozen=True, slots=True)
class OutputReport:
    violations: tuple[OutputViolation, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def kinds(self) -> frozenset[ViolationKind]:
        return frozenset(v.kind for v in self.violations)


_PROMPT_MARKERS = re.compile(
    r"<\|\s*(?:im_start|im_end|system|endoftext|begin_of_text|start_header_id|eot_id)\s*\|>|\[/?INST\]|<</?SYS>>|"
    r"<</?\s*UNTRUSTED_CONTENT\b|\bBEGIN\s+SYSTEM\s+PROMPT\b|\bEND\s+SYSTEM\s+PROMPT\b",
    re.IGNORECASE,
)
_EXTRA_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    re.compile(
        r"\b(?:password|passwd|pwd|secret|api[_-]?key|client[_-]?secret)\s*[:=]\s*\S{6,}", re.IGNORECASE
    ),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
)
_SCHEME_RE = re.compile(r"(?<![\w.+-])([a-z][a-z0-9+.-]{1,20}):(?://)?[^\s<>\"')\]]+", re.IGNORECASE)
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_HTML_ACTIVE_RE = re.compile(
    r"<\s*(?:script|iframe|object|embed|img|svg|form)\b|\bon[a-z]+\s*=\s*[\"']", re.IGNORECASE
)
_DANGEROUS_SCHEMES = frozenset(
    {"javascript", "data", "vbscript", "file", "blob", "about", "ftp", "smb", "ms-msdt"}
)
_CODE_EXECUTION_RE = (
    re.compile(
        r"\b(?:run|execute|paste|type|enter)\s+(?:the\s+)?(?:following|this|these|below)\s+"
        r"(?:\w+\s+)?(?:commands?|scripts?|code|snippet|one-liner|in\s+(?:your\s+)?(?:terminal|console|shell))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:paste|run|type)\s+(?:\w+\s+){0,3}(?:into|in)\s+(?:your\s+|a\s+|the\s+)?(?:terminal|console|shell|powershell|command\s+prompt)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:curl|wget|iwr|invoke-webrequest)\b[^\n|]{0,200}\|\s*(?:ba|z)?sh\b", re.IGNORECASE),
    re.compile(r"\bpowershell(?:\.exe)?\s+(?:-\w+\s+)*-(?:enc|encodedcommand|e)\b", re.IGNORECASE),
    re.compile(r"\b(?:iex|invoke-expression)\s*\(", re.IGNORECASE),
    re.compile(r"\brm\s+-rf\s+[/~]|\bsudo\s+\w+|\bchmod\s+\+x\b|\bpip\s+install\s+\S+|\bnpm\s+install\s+\S+"),
    re.compile(r"```\s*(?:bash|sh|shell|zsh|powershell|ps1|cmd|bat|console)\b", re.IGNORECASE),
)


class OutputGuard:
    def __init__(
        self,
        *,
        redactor: PiiRedactor | None = None,
        canaries: Iterable[str] = (),
        allowed_url_hosts: Iterable[str] = (),
    ) -> None:
        self._redactor = redactor or PiiRedactor()
        self._canaries = tuple(c for c in canaries if len(c) >= 8)
        self._allowed_hosts = frozenset(h.lower() for h in allowed_url_hosts)

    def check(
        self, text: str, *, allowed_urls: Iterable[str] | None = None, location: str = "text"
    ) -> OutputReport:
        violations: list[OutputViolation] = []
        violations.extend(self._prompt_leaks(text, location))
        violations.extend(self._secrets(text, location))
        violations.extend(self._pii(text, location))
        violations.extend(self._urls(text, location, allowed_urls))
        violations.extend(self._code_execution(text, location))
        return OutputReport(tuple(violations))

    def check_brief(self, brief: ClientBrief) -> OutputReport:
        """Validate every rendered string of a brief; only URLs of cited evidence are permitted."""
        allowed = {e.url for e in brief.evidence}
        sections: dict[str, tuple[str, ...]] = {
            "company_overview": tuple(s.text for s in brief.company_overview),
            "technology_priorities": tuple(s.text for s in brief.technology_priorities),
            "gartner_relevant_insights": tuple(s.text for s in brief.gartner_relevant_insights),
            "opportunities": tuple(s.text for s in brief.opportunities),
            "risks": tuple(s.text for s in brief.risks),
            "executive_summary": tuple(s.text for s in brief.executive_summary),
            "executive_talking_points": tuple(s.text for s in brief.executive_talking_points),
            "recommended_next_actions": tuple(s.text for s in brief.recommended_next_actions),
            "discovery_questions": brief.discovery_questions,
            "qualification": (
                brief.qualification.verdict_rationale,
                *(s.rationale for s in brief.qualification.scores),
            ),
            "warnings": brief.warnings,
        }
        violations: list[OutputViolation] = []
        for section, texts in sections.items():
            for index, value in enumerate(texts):
                report = self.check(value, allowed_urls=allowed, location=f"{section}[{index}]")
                violations.extend(report.violations)
        return OutputReport(tuple(violations))

    def enforce(self, text: str, *, allowed_urls: Iterable[str] | None = None) -> None:
        report = self.check(text, allowed_urls=allowed_urls)
        if not report.ok:
            kinds = ", ".join(sorted(k.value for k in report.kinds))
            raise SecurityViolationError(f"generated output rejected: {kinds}")

    # -- checks ---------------------------------------------------------------------------------------
    def _prompt_leaks(self, text: str, location: str) -> Iterable[OutputViolation]:
        match = _PROMPT_MARKERS.search(text)
        if match:
            yield OutputViolation(ViolationKind.PROMPT_LEAK, f"prompt marker {match.group(0)!r}", location)
        if DATAMARK in text:
            yield OutputViolation(
                ViolationKind.PROMPT_LEAK, "spotlight datamark character in output", location
            )
        lowered = text.lower()
        for canary in self._canaries:
            if canary.lower() in lowered:
                yield OutputViolation(ViolationKind.PROMPT_LEAK, "system prompt canary present", location)

    @staticmethod
    def _secrets(text: str, location: str) -> Iterable[OutputViolation]:
        if scrub(text) != text:
            yield OutputViolation(ViolationKind.SECRET, "credential-shaped token", location)
            return
        for pattern in _EXTRA_SECRET_PATTERNS:
            if pattern.search(text):
                yield OutputViolation(ViolationKind.SECRET, "credential-shaped token", location)
                return

    def _pii(self, text: str, location: str) -> Iterable[OutputViolation]:
        result = self._redactor.redact(text)
        for finding in result.findings:
            yield OutputViolation(
                ViolationKind.PII, f"{finding.type.value} at {finding.start}-{finding.end}", location
            )

    def _urls(
        self, text: str, location: str, allowed_urls: Iterable[str] | None
    ) -> Iterable[OutputViolation]:
        if _MARKDOWN_IMAGE_RE.search(text):
            yield OutputViolation(ViolationKind.UNSAFE_URL, "markdown image (exfiltration beacon)", location)
        if _HTML_ACTIVE_RE.search(text):
            yield OutputViolation(ViolationKind.UNSAFE_URL, "active HTML content", location)
        for match in _SCHEME_RE.finditer(text):
            scheme = match.group(1).lower()
            if scheme in _DANGEROUS_SCHEMES:
                yield OutputViolation(ViolationKind.UNSAFE_URL, f"disallowed scheme {scheme}:", location)
        allowed = None if allowed_urls is None else {u.rstrip("/.,") for u in allowed_urls}
        for match in _HTTP_URL_RE.finditer(text):
            url = match.group(0).rstrip(".,;:")
            authority = url.split("://", 1)[1].split("/", 1)[0]
            if "@" in authority:
                yield OutputViolation(ViolationKind.UNSAFE_URL, "credentials embedded in URL", location)
                continue
            host = authority.split(":", 1)[0].lower()
            if allowed is not None and url.rstrip("/") not in allowed and host not in self._allowed_hosts:
                yield OutputViolation(
                    ViolationKind.UNCITED_URL, f"URL not backed by evidence: {host}", location
                )

    @staticmethod
    def _code_execution(text: str, location: str) -> Iterable[OutputViolation]:
        for pattern in _CODE_EXECUTION_RE:
            match = pattern.search(text)
            if match:
                yield OutputViolation(ViolationKind.CODE_EXECUTION, match.group(0)[:80], location)
                return
