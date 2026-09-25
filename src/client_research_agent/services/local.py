"""In-process adapters for the hexagonal ports.

Local development, CI and the end-to-end test suite bind these adapters instead
of the Databricks ones. They honour exactly the same behavioural contract (the
suite in ``tests/contract`` runs against both), so a pipeline that works here
works against Unity Catalog and Vector Search.

Metadata filters
----------------
``VectorIndex.search`` accepts the Databricks Vector Search filter dialect so
callers never need to know which adapter is bound:

* ``{"company": "Acme"}`` - equality
* ``{"document_type": ["sec_filing", "press_release"]}`` - membership (``IN``)
* ``{"publication_date >=": date(2026, 1, 1)}`` - comparison (``>``, ``>=``, ``<``, ``<=``)
* ``{"company NOT": "Acme"}`` - negation (``NOT IN`` for a list); ``"company !="`` and
  ``"NOT company"`` are accepted as aliases

Rows whose field is ``NULL`` never match a comparison, mirroring SQL semantics.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from client_research_agent.models import Chunk, ChunkStrategy, ClientBrief, RetrievedChunk, SourceDocument
from client_research_agent.models.domain import utc_now

FILTERABLE_FIELDS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "doc_id",
        "company",
        "url",
        "document_type",
        "source_domain",
        "strategy",
        "parent_id",
        "publication_date",
        "industry",
        "chunk_index",
        "confidence",
    }
)
_OPERATORS: frozenset[str] = frozenset({"=", ">", ">=", "<", "<=", "NOT"})
DENSE_RETRIEVER = "dense"


@dataclass(frozen=True, slots=True)
class FilterClause:
    """One parsed metadata filter: ``field <op> value``."""

    field: str
    operator: str
    value: Any


def normalize_filter_value(value: Any) -> Any:
    """Convert enums, dates and collections to plain comparable scalars."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list | tuple | set | frozenset):
        return [normalize_filter_value(item) for item in value]
    return value


def parse_filters(filters: Mapping[str, Any] | None) -> list[FilterClause]:
    """Parse the Vector Search filter dialect, rejecting unknown fields and operators."""
    clauses: list[FilterClause] = []
    for raw_key, raw_value in (filters or {}).items():
        parts = raw_key.strip().split()
        if len(parts) == 1:
            field, operator = parts[0], "="
        elif len(parts) == 2 and parts[0].upper() == "NOT":
            field, operator = parts[1], "NOT"
        elif len(parts) == 2:
            field, operator = parts[0], parts[1].upper()
        else:
            raise ValueError(f"malformed filter key {raw_key!r}")
        operator = "NOT" if operator == "!=" else operator
        if operator not in _OPERATORS:
            raise ValueError(f"unsupported filter operator {operator!r} in {raw_key!r}")
        if field not in FILTERABLE_FIELDS:
            raise ValueError(f"field {field!r} is not filterable; allowed: {sorted(FILTERABLE_FIELDS)}")
        value = normalize_filter_value(raw_value)
        if isinstance(value, list) and operator not in ("=", "NOT"):
            raise ValueError(f"operator {operator!r} does not accept a list value")
        clauses.append(FilterClause(field=field, operator=operator, value=value))
    return clauses


def _compare(actual: Any, operator: str, expected: Any) -> bool:
    if isinstance(expected, list):
        found = actual in expected
        return found if operator == "=" else not found
    if operator == "=":
        return bool(actual == expected)
    if operator == "NOT":
        return bool(actual != expected)
    if actual is None or expected is None:
        return False
    try:
        if operator == ">":
            return bool(actual > expected)
        if operator == ">=":
            return bool(actual >= expected)
        if operator == "<":
            return bool(actual < expected)
        return bool(actual <= expected)
    except TypeError:
        return False


def chunk_matches(chunk: Chunk, clauses: Iterable[FilterClause]) -> bool:
    """Evaluate parsed filter clauses against a chunk's metadata."""
    for clause in clauses:
        actual = normalize_filter_value(getattr(chunk, clause.field))
        if not _compare(actual, clause.operator, clause.value):
            return False
    return True


class InMemoryVectorIndex:
    """Exact cosine-similarity index over numpy arrays; thread-safe."""

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._lock = threading.RLock()
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, npt.NDArray[np.float64]] = {}

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def __len__(self) -> int:
        with self._lock:
            return len(self._chunks)

    @staticmethod
    def _as_unit_vector(vector: Sequence[float], dimension: int | None) -> npt.NDArray[np.float64]:
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1 or array.size == 0:
            raise ValueError("embedding must be a non-empty 1-D vector")
        if dimension is not None and array.size != dimension:
            raise ValueError(f"embedding dimension {array.size} != index dimension {dimension}")
        if not np.all(np.isfinite(array)):
            raise ValueError("embedding contains non-finite values")
        norm = float(np.linalg.norm(array))
        return array / norm if norm > 0.0 else array

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0
        with self._lock:
            dimension = self._dimension if self._dimension is not None else len(embeddings[0])
            prepared = [
                (chunk, self._as_unit_vector(vector, dimension))
                for chunk, vector in zip(chunks, embeddings, strict=True)
            ]
            self._dimension = dimension
            for chunk, unit in prepared:
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = unit
        return len({chunk.chunk_id for chunk in chunks})

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int,
        filters: Mapping[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k must be >= 1")
        clauses = parse_filters(filters)
        with self._lock:
            if not self._chunks:
                return []
            query = self._as_unit_vector(query_vector, self._dimension)
            candidates = [cid for cid, chunk in self._chunks.items() if chunk_matches(chunk, clauses)]
            if not candidates:
                return []
            matrix = np.vstack([self._vectors[cid] for cid in candidates])
            scores = matrix @ query
            order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))[:k]
            return [
                RetrievedChunk(
                    chunk=self._chunks[candidates[i]],
                    score=float(scores[i]),
                    retriever=DENSE_RETRIEVER,
                    rank=rank,
                )
                for rank, i in enumerate(order)
            ]

    def delete_company(self, company: str) -> int:
        with self._lock:
            doomed = [cid for cid, chunk in self._chunks.items() if chunk.company == company]
            for cid in doomed:
                del self._chunks[cid]
                del self._vectors[cid]
            return len(doomed)


def _chunk_sort_key(chunk: Chunk) -> tuple[str, int, str]:
    return (chunk.doc_id, chunk.chunk_index, chunk.chunk_id)


class InMemoryDocumentStore:
    """Dict-backed system of record for documents and chunks; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._documents: dict[str, SourceDocument] = {}
        self._chunks: dict[str, Chunk] = {}

    def save_documents(self, documents: Sequence[SourceDocument]) -> int:
        with self._lock:
            for document in documents:
                self._documents[document.doc_id] = document
        return len({document.doc_id for document in documents})

    def save_chunks(self, chunks: Sequence[Chunk]) -> int:
        with self._lock:
            for chunk in chunks:
                self._chunks[chunk.chunk_id] = chunk
        return len({chunk.chunk_id for chunk in chunks})

    def get_document(self, doc_id: str) -> SourceDocument | None:
        with self._lock:
            return self._documents.get(doc_id)

    def list_documents(self, company: str) -> list[SourceDocument]:
        with self._lock:
            found = [doc for doc in self._documents.values() if doc.company == company]
        return sorted(found, key=lambda doc: doc.doc_id)

    def list_chunks(self, company: str) -> list[Chunk]:
        with self._lock:
            found = [chunk for chunk in self._chunks.values() if chunk.company == company]
        return sorted(found, key=_chunk_sort_key)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        with self._lock:
            seen: set[str] = set()
            ordered: list[Chunk] = []
            for chunk_id in chunk_ids:
                chunk = self._chunks.get(chunk_id)
                if chunk is not None and chunk_id not in seen:
                    seen.add(chunk_id)
                    ordered.append(chunk)
            return ordered

    def get_parents(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Resolve the distinct parent chunks of ``chunks`` (small-to-big retrieval)."""
        parent_ids = list(dict.fromkeys(c.parent_id for c in chunks if c.parent_id is not None))
        return [c for c in self.get_chunks(parent_ids) if c.strategy is ChunkStrategy.PARENT]

    def known_hashes(self, company: str) -> set[str]:
        with self._lock:
            return {doc.content_hash for doc in self._documents.values() if doc.company == company}

    def delete_company(self, company: str) -> int:
        with self._lock:
            doomed_chunks = [cid for cid, chunk in self._chunks.items() if chunk.company == company]
            for cid in doomed_chunks:
                del self._chunks[cid]
            doomed_docs = [did for did, doc in self._documents.items() if doc.company == company]
            for did in doomed_docs:
                del self._documents[did]
            return len(doomed_chunks)


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")


def validate_run_id(run_id: str) -> str:
    """Reject run ids that could escape the storage directory or break keys."""
    if not _RUN_ID.fullmatch(run_id) or ".." in run_id:
        raise ValueError(f"invalid run_id {run_id!r}")
    return run_id


class JsonlBriefRepository:
    """Directory of ``<run_id>.json`` briefs written atomically (temp file + ``os.replace``)."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._directory

    def _path(self, run_id: str) -> Path:
        return self._directory / f"{validate_run_id(run_id)}.json"

    def save(self, brief: ClientBrief) -> str:
        target = self._path(brief.run_id)
        payload = brief.model_dump_json(indent=2)
        with self._lock:
            handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed explicitly before replace
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=f".{brief.run_id}.",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(handle.name, target)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        return brief.run_id

    def get(self, run_id: str) -> ClientBrief | None:
        path = self._path(run_id)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return ClientBrief.model_validate_json(raw)

    def list_run_ids(self) -> list[str]:
        return sorted(p.stem for p in self._directory.glob("*.json") if not p.name.startswith("."))


class JsonlAuditSink:
    """Append-only JSON Lines audit log (one event per line, UTF-8)."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not event_type:
            raise ValueError("event_type must be non-empty")
        line = json.dumps(
            {"ts": utc_now().isoformat(), "event_type": event_type, "payload": dict(payload)},
            default=str,
            sort_keys=True,
            ensure_ascii=False,
        )
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        with self._lock:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
