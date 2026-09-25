"""Replays captured public documents as an ingestion source.

:class:`SnapshotIngestion` implements the ingestion-runner contract
(``run(request) -> IngestionResult``) over documents captured ahead of time, so
an evaluation example produces the same evidence on every run and needs no
network access. Documents get the same identifiers, content hashes and trust
scores the live pipeline would assign.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from client_research_agent.evaluation.dataset import EvalExample, SnapshotDocument
from client_research_agent.models import DocumentType, ResearchRequest, SourceDocument
from client_research_agent.research.ingestion import (
    IngestionResult,
    Origin,
    SkippedSource,
    SkipReason,
    content_hash,
    document_id,
    trust_score,
)
from client_research_agent.research.url_guard import host_of
from client_research_agent.services.ports import DocumentStore

_ORIGINS: Mapping[DocumentType, Origin] = {
    DocumentType.SEC_FILING: Origin.EDGAR,
    DocumentType.ANALYST_PUBLIC: Origin.ANALYST,
}


def to_source_document(snapshot: SnapshotDocument, request: ResearchRequest) -> SourceDocument:
    host = host_of(snapshot.url)
    origin = _ORIGINS.get(snapshot.document_type, Origin.CORPORATE)
    return SourceDocument(
        doc_id=document_id(snapshot.url),
        company=request.company_name,
        url=snapshot.url,
        title=snapshot.title,
        text=snapshot.text,
        document_type=snapshot.document_type,
        source_domain=host,
        content_hash=content_hash(snapshot.text),
        publication_date=snapshot.publication_date,
        industry=request.industry,
        trust_score=trust_score(host, origin, request.domain),
        metadata={"origin": "snapshot", "snapshot_origin": origin.value},
    )


class SnapshotIngestion:
    """Ingestion runner over captured documents, keyed by company name (case-insensitive)."""

    def __init__(
        self,
        documents: Mapping[str, Sequence[SnapshotDocument]],
        *,
        document_store: DocumentStore | None = None,
    ) -> None:
        self._documents = {company.casefold(): tuple(docs) for company, docs in documents.items()}
        self._store = document_store

    @classmethod
    def from_examples(
        cls, examples: Iterable[EvalExample], *, document_store: DocumentStore | None = None
    ) -> SnapshotIngestion:
        grouped: dict[str, list[SnapshotDocument]] = {}
        for example in examples:
            grouped.setdefault(example.inputs.company_name, []).extend(example.snapshot)
        return cls(grouped, document_store=document_store)

    def run(self, request: ResearchRequest) -> IngestionResult:
        known = self._store.known_hashes(request.company_name) if self._store is not None else set()
        accepted: list[SourceDocument] = []
        skipped: list[SkippedSource] = []
        seen: set[str] = set()
        for snapshot in self._documents.get(request.company_name.casefold(), ()):
            document = to_source_document(snapshot, request)
            if document.content_hash in known:
                skipped.append(SkippedSource(document.url, SkipReason.ALREADY_INGESTED))
            elif document.content_hash in seen:
                skipped.append(SkippedSource(document.url, SkipReason.DUPLICATE))
            elif len(accepted) >= request.max_documents:
                skipped.append(SkippedSource(document.url, SkipReason.LIMIT_REACHED))
            else:
                seen.add(document.content_hash)
                accepted.append(document)
        return IngestionResult(documents=accepted, skipped=skipped)
