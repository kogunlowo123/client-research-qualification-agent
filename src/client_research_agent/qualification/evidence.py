"""Evidence registry: the per-run catalogue of citable evidence.

Retrieved chunks become ``Evidence`` items with stable, human-friendly ids
(``E1``..``En``) assigned in registration order and deduplicated by
``chunk_id``. Every LLM prompt, every score and every brief statement refers
to evidence exclusively through these ids, which is what makes citation
validation possible: an id either resolves to a chunk that was actually
retrieved for this run or it is rejected.

Quotes are verbatim, contiguous substrings of the chunk text (the best
supporting sentence window for the query) so a reader can find them in the
source document.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Iterable, Iterator, Sequence

from client_research_agent.models import Chunk, Evidence, RetrievedChunk
from client_research_agent.qualification.text import content_words, extract_numbers, sentence_spans

MAX_QUOTE_CHARS = 500
_EVIDENCE_TAG = re.compile(r"<\s*/?\s*evidence\b[^>]*>", re.IGNORECASE)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_relevance(scores: Sequence[float]) -> list[float]:
    """Map heterogeneous retriever scores (cosine, RRF, cross-encoder logits) onto ``[0, 1]``.

    Scores already in ``[0, 1]`` are kept as-is so calibrated similarities are
    not distorted; otherwise the batch is min-max scaled into ``[0.1, 1.0]``
    (a single out-of-range score goes through a logistic squash).
    """
    if not scores:
        return []
    if all(0.0 <= score <= 1.0 for score in scores):
        return [float(score) for score in scores]
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, score)))) for score in scores]
    return [0.1 + 0.9 * (score - low) / (high - low) for score in scores]


def sanitize_untrusted(text: str) -> str:
    """Neutralise delimiter spoofing and control characters in untrusted source text."""
    return _CONTROL.sub(" ", _EVIDENCE_TAG.sub("[tag removed]", text))


def best_quote(text: str, query: str | None = None, *, max_chars: int = MAX_QUOTE_CHARS) -> str:
    """Best contiguous window of whole sentences (<= ``max_chars``) supporting ``query``.

    Sentences are scored by overlap with the query's content words plus a small
    bonus for concrete numbers. The returned string is a verbatim slice of
    ``text``; if the best single sentence is longer than ``max_chars`` it is
    cut on a word boundary (still a verbatim prefix).
    """
    stripped = text.strip()
    spans = sentence_spans(stripped)
    if not spans:
        return stripped[:max_chars]
    query_terms = content_words(query or "")

    def sentence_score(span: tuple[int, int]) -> float:
        sentence = stripped[span[0] : span[1]]
        overlap = len(content_words(sentence) & query_terms) if query_terms else 0
        return overlap + (0.25 if extract_numbers(sentence) else 0.0)

    scores = [sentence_score(span) for span in spans]
    best_start, best_end, best_value = 0, 0, -1.0
    for first in range(len(spans)):
        total = 0.0
        for last in range(first, len(spans)):
            if spans[last][1] - spans[first][0] > max_chars and last > first:
                break
            total += scores[last]
            # Prefer higher score; on ties prefer the earlier, shorter window.
            if total > best_value:
                best_start, best_end, best_value = first, last, total
    begin, end = spans[best_start][0], spans[best_end][1]
    window = stripped[begin:end]
    if len(window) > max_chars:
        cut = window[:max_chars]
        boundary = cut.rfind(" ")
        window = cut[:boundary] if boundary > max_chars // 2 else cut
    return window.strip()


class EvidenceRegistry:
    """Thread-safe, append-only catalogue of evidence for one research run."""

    def __init__(self, *, id_prefix: str = "E", max_quote_chars: int = MAX_QUOTE_CHARS) -> None:
        if not id_prefix.isalpha():
            raise ValueError("id_prefix must be alphabetic")
        self._prefix = id_prefix
        self._max_quote_chars = max_quote_chars
        self._lock = threading.RLock()
        self._evidence: dict[str, Evidence] = {}
        self._chunks: dict[str, Chunk] = {}
        self._by_chunk: dict[str, str] = {}

    # ------------------------------------------------------------------ registration
    def register(
        self, retrieved: RetrievedChunk, *, query: str | None = None, relevance: float | None = None
    ) -> Evidence:
        """Register one retrieved chunk; re-registering a chunk returns the existing id.

        A duplicate registration keeps the original id and quote but raises the
        stored relevance if the new retrieval scored it higher.
        """
        score = relevance if relevance is not None else normalize_relevance([retrieved.score])[0]
        score = max(0.0, min(1.0, score))
        chunk = retrieved.chunk
        with self._lock:
            existing_id = self._by_chunk.get(chunk.chunk_id)
            if existing_id is not None:
                existing = self._evidence[existing_id]
                if score > existing.relevance:
                    existing = existing.model_copy(update={"relevance": score})
                    self._evidence[existing_id] = existing
                return existing
            evidence_id = f"{self._prefix}{len(self._evidence) + 1}"
            quote = best_quote(chunk.text, query, max_chars=self._max_quote_chars) or chunk.title
            evidence = Evidence(
                evidence_id=evidence_id,
                chunk_id=chunk.chunk_id,
                url=chunk.url,
                title=chunk.title,
                quote=quote,
                document_type=chunk.document_type,
                publication_date=chunk.publication_date,
                relevance=score,
            )
            self._evidence[evidence_id] = evidence
            self._chunks[evidence_id] = chunk
            self._by_chunk[chunk.chunk_id] = evidence_id
            return evidence

    def register_many(
        self, retrieved: Sequence[RetrievedChunk], *, query: str | None = None
    ) -> list[Evidence]:
        """Register a batch, normalising relevance across the batch; returns evidence in input order."""
        relevances = normalize_relevance([item.score for item in retrieved])
        with self._lock:
            return [
                self.register(item, query=query, relevance=relevance)
                for item, relevance in zip(retrieved, relevances, strict=True)
            ]

    # ------------------------------------------------------------------ lookup
    def get(self, evidence_id: str) -> Evidence | None:
        with self._lock:
            return self._evidence.get(evidence_id)

    def require(self, evidence_id: str) -> Evidence:
        evidence = self.get(evidence_id)
        if evidence is None:
            raise KeyError(f"unknown evidence id {evidence_id!r}")
        return evidence

    def chunk(self, evidence_id: str) -> Chunk | None:
        with self._lock:
            return self._chunks.get(evidence_id)

    def source_text(self, evidence_id: str) -> str:
        """Full text of the chunk behind ``evidence_id`` ('' when unknown)."""
        chunk = self.chunk(evidence_id)
        return chunk.text if chunk is not None else ""

    def id_for_chunk(self, chunk_id: str) -> str | None:
        with self._lock:
            return self._by_chunk.get(chunk_id)

    def __contains__(self, evidence_id: object) -> bool:
        with self._lock:
            return evidence_id in self._evidence

    def __len__(self) -> int:
        with self._lock:
            return len(self._evidence)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self.all())

    def all(self) -> tuple[Evidence, ...]:
        with self._lock:
            return tuple(self._evidence.values())

    def ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._evidence)

    def urls(self) -> frozenset[str]:
        with self._lock:
            return frozenset(chunk.url for chunk in self._chunks.values())

    def partition_ids(self, evidence_ids: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split ids into (known, unknown), de-duplicated, preserving order."""
        known: list[str] = []
        unknown: list[str] = []
        for evidence_id in evidence_ids:
            target = known if evidence_id in self else unknown
            if evidence_id not in target:
                target.append(evidence_id)
        return tuple(known), tuple(unknown)

    def subset(self, evidence_ids: Iterable[str]) -> tuple[Evidence, ...]:
        known, _ = self.partition_ids(evidence_ids)
        return tuple(self.require(evidence_id) for evidence_id in known)

    # ------------------------------------------------------------------ prompting
    def render_block(
        self,
        evidence_ids: Iterable[str] | None = None,
        *,
        full_text: bool = False,
        max_chars_per_item: int = 1200,
    ) -> str:
        """Render evidence for a prompt inside ``<evidence>`` delimiters.

        Source text is sanitised so it cannot close the delimiter or smuggle
        control characters; the prompt instructs the model to treat everything
        inside the block as untrusted data.
        """
        selected = self.all() if evidence_ids is None else self.subset(evidence_ids)
        lines = ["<evidence>"]
        for evidence in selected:
            date_text = evidence.publication_date.isoformat() if evidence.publication_date else "undated"
            body = self.source_text(evidence.evidence_id) if full_text else evidence.quote
            body = sanitize_untrusted(" ".join(body.split()))[:max_chars_per_item]
            title = sanitize_untrusted(" ".join(evidence.title.split()))[:200]
            lines.append(
                f"[{evidence.evidence_id}] type={evidence.document_type.value} | date={date_text} | "
                f"title={title} | url={evidence.url}"
            )
            lines.append(f"text: {body}")
        if len(lines) == 1:
            lines.append("(no evidence available)")
        lines.append("</evidence>")
        return "\n".join(lines)
