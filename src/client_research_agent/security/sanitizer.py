"""Normalisation and neutralisation of untrusted web content (OWASP LLM01 / LLM05).

Everything scraped from the public web is attacker-controllable. Before any of
it is chunked, embedded or placed in a prompt it passes through
:class:`ContentSanitizer`, which:

* applies NFKC normalisation so full-width / compatibility forms collapse to
  their canonical characters (defeats many homoglyph and spacing tricks);
* removes zero-width, bidirectional-override and Unicode *tag* characters that
  are used to hide instructions from human reviewers ("ASCII smuggling");
* strips other C0/C1 control characters except newline and tab;
* removes markdown image beacons and HTML ``<img>`` tags (a classic data
  exfiltration channel once rendered) and unwraps markdown links to their text;
* collapses whitespace and caps the length to ``max_input_chars``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from client_research_agent.config.settings import GuardrailSettings

ZERO_WIDTH_CHARS = "\u200b\u200c\u200d\u2060\u2061\u2062\u2063\u2064\ufeff\u180e\u00ad\u034f"
BIDI_CONTROL_CHARS = "\u200e\u200f\u061c\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"

ZERO_WIDTH_RE = re.compile(f"[{re.escape(ZERO_WIDTH_CHARS)}]")
BIDI_RE = re.compile(f"[{re.escape(BIDI_CONTROL_CHARS)}]")
TAG_CHARS_RE = re.compile("[\U000e0000-\U000e007f]")
_CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_VARIATION_SELECTOR_RE = re.compile("[\ufe00-\ufe0f\U000e0100-\U000e01ef]")

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]{0,500}\]\(\s*[^)\s]{1,2048}(?:\s+\"[^\"]*\")?\s*\)")
HTML_IMG_RE = re.compile(r"<img\b[^>]{0,2048}>", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]{1,500})\]\(\s*[^)\s]{1,2048}(?:\s+\"[^\"]*\")?\s*\)")
MARKDOWN_REF_DEF_RE = re.compile(r"(?m)^\s{0,3}\[[^\]\n]{1,200}\]:\s*\S+.*$")
_INLINE_WS_RE = re.compile(r"[ \t\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")
_MANY_NEWLINES_RE = re.compile(r"\n\s*\n\s*\n+")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_LEADING_WS_RE = re.compile(r"\n[ \t]+")


def decode_tag_characters(text: str) -> str:
    """Map Unicode tag characters (U+E0000 block) back to the ASCII they smuggle."""
    return "".join(
        chr(ord(ch) - 0xE0000) for ch in TAG_CHARS_RE.findall(text) if 0x20 <= ord(ch) - 0xE0000 < 0x7F
    )


@dataclass(frozen=True, slots=True)
class SanitizedText:
    text: str
    removed_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    truncated: bool = False
    original_length: int = 0

    @property
    def modified(self) -> bool:
        return self.truncated or any(self.removed_counts.values())

    @property
    def total_removed(self) -> int:
        return sum(self.removed_counts.values())


class ContentSanitizer:
    """Deterministic, idempotent sanitiser for untrusted text."""

    def __init__(self, max_input_chars: int = 20_000, *, unwrap_links: bool = True) -> None:
        if max_input_chars < 1:
            raise ValueError("max_input_chars must be positive")
        self._max_chars = max_input_chars
        self._unwrap_links = unwrap_links

    @classmethod
    def from_settings(cls, settings: GuardrailSettings) -> ContentSanitizer:
        return cls(max_input_chars=settings.max_input_chars)

    @property
    def max_input_chars(self) -> int:
        return self._max_chars

    def sanitize_untrusted(self, text: str) -> SanitizedText:
        counts: dict[str, int] = {}
        original_length = len(text)

        normalized = unicodedata.normalize("NFKC", text)
        if normalized != text:
            counts["nfkc_changed"] = 1
        value = normalized

        value = self._strip(value, TAG_CHARS_RE, "tag_chars", counts)
        value = self._strip(value, ZERO_WIDTH_RE, "zero_width", counts)
        value = self._strip(value, BIDI_RE, "bidi_controls", counts)
        value = self._strip(value, _VARIATION_SELECTOR_RE, "variation_selectors", counts)
        value = value.replace("\r\n", "\n").replace("\r", "\n")
        value = self._strip(value, _CONTROL_RE, "control_chars", counts)

        value = self._strip(value, MARKDOWN_IMAGE_RE, "markdown_images", counts)
        value = self._strip(value, HTML_IMG_RE, "html_images", counts)
        value = self._strip(value, MARKDOWN_REF_DEF_RE, "markdown_reference_links", counts)
        if self._unwrap_links:
            value, n_links = MARKDOWN_LINK_RE.subn(r"\1", value)
            if n_links:
                counts["markdown_links"] = n_links

        value = _INLINE_WS_RE.sub(" ", value)
        value = _TRAILING_WS_RE.sub("\n", value)
        value = _LEADING_WS_RE.sub("\n", value)
        value = _MANY_NEWLINES_RE.sub("\n\n", value)
        value = value.strip()

        truncated = len(value) > self._max_chars
        if truncated:
            counts["truncated_chars"] = len(value) - self._max_chars
            value = value[: self._max_chars].rstrip()

        return SanitizedText(
            text=value,
            removed_counts=MappingProxyType({k: v for k, v in counts.items() if v}),
            truncated=truncated,
            original_length=original_length,
        )

    @staticmethod
    def _strip(value: str, pattern: re.Pattern[str], name: str, counts: dict[str, int]) -> str:
        cleaned, n = pattern.subn("", value)
        if n:
            counts[name] = counts.get(name, 0) + n
        return cleaned
