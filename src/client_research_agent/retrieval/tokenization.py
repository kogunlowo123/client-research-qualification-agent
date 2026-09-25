"""Token counting for chunk budgets.

Uses the ``cl100k_base`` BPE (the tokenizer family used by the chat and
embedding endpoints we budget against). The encoder is loaded once and cached;
when the BPE file cannot be loaded (air-gapped cluster, no cache) counting
degrades to whitespace-delimited words so chunking keeps working offline.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import tiktoken

from client_research_agent.observability.logging import get_logger

ENCODING_NAME = "cl100k_base"

_WORD = re.compile(r"\S+")
_logger = get_logger(__name__)


@lru_cache(maxsize=4)
def load_encoding(name: str = ENCODING_NAME) -> Any | None:
    """Return the tiktoken encoding, or ``None`` when it cannot be loaded."""
    try:
        return tiktoken.get_encoding(name)
    except Exception as exc:  # offline / missing BPE cache -> whitespace fallback
        _logger.warning("tiktoken_encoding_unavailable", encoding=name, error=str(exc)[:200])
        return None


class Tokenizer:
    """Counts tokens with a BPE encoder when available, whitespace words otherwise."""

    def __init__(self, encoding: Any | None = None) -> None:
        self._encoding = encoding

    @classmethod
    def default(cls) -> Tokenizer:
        return cls(load_encoding())

    @property
    def exact(self) -> bool:
        """True when counts come from the real BPE rather than the whitespace fallback."""
        return self._encoding is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text, disallowed_special=()))
        return len(_WORD.findall(text))

    def word_spans(self, text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
        """Character spans of whitespace-delimited words inside ``text[start:end]``."""
        stop = len(text) if end is None else end
        return [(m.start(), m.end()) for m in _WORD.finditer(text, start, stop)]

    def truncate(self, text: str, max_tokens: int) -> str:
        """Longest word-aligned prefix of ``text`` within ``max_tokens``."""
        if max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text
        used = 0
        end = 0
        for start, stop in self.word_spans(text):
            cost = self.count(" " + text[start:stop]) if self.exact else 1
            if used + cost > max_tokens:
                break
            used += cost
            end = stop
        return text[:end]


@lru_cache(maxsize=1)
def get_tokenizer() -> Tokenizer:
    return Tokenizer.default()


def count_tokens(text: str) -> int:
    return get_tokenizer().count(text)
