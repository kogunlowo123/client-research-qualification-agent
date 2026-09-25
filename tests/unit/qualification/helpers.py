"""Test-only helpers shared by the qualification, scoring, citation and briefing tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

from client_research_agent.models import (
    Chunk,
    Criterion,
    CriterionScore,
    DocumentType,
    RetrievedChunk,
)
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.qualification.text import content_words
from client_research_agent.utils.errors import TransientError
from tests.support.doubles import make_chunk

TODAY = date(2026, 9, 1)
COMPANY = "Acme Corp"

DEFAULT_WEIGHTS = {
    Criterion.COMPANY_SCALE: 0.20,
    Criterion.TECH_MODERNIZATION: 0.20,
    Criterion.AI_DATA_FOCUS: 0.25,
    Criterion.INDUSTRY_TRENDS: 0.10,
    Criterion.NEAR_TERM_OPPORTUNITY: 0.25,
}


@dataclass
class ListOutcome:
    chunks: list[RetrievedChunk]


@dataclass
class KeywordRetriever:
    """In-memory retriever: ranks chunks of the requested company by content-word overlap."""

    corpus: Sequence[Chunk]
    fail_on: tuple[str, ...] = ()
    ignore_company: bool = False
    queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str, *, company: str, top_k: int | None = None) -> ListOutcome:
        self.queries.append(query)
        if any(marker in query for marker in self.fail_on):
            raise TransientError(f"search backend timeout for {query!r}")
        terms = content_words(query)
        scored = [
            (len(terms & content_words(chunk.text)), chunk)
            for chunk in self.corpus
            if self.ignore_company or chunk.company == company
        ]
        ranked = sorted(
            (item for item in scored if item[0] > 0), key=lambda item: (-item[0], item[1].chunk_id)
        )
        return ListOutcome(
            [
                RetrievedChunk(chunk=chunk, score=float(score), retriever="keyword", rank=rank)
                for rank, (score, chunk) in enumerate(ranked[: top_k or 8])
            ]
        )


def acme_corpus() -> list[Chunk]:
    return [
        make_chunk(
            "c-10k",
            "Acme Corp reported annual revenue of $12.4 billion for fiscal 2025. "
            "The company employs approximately 45,000 employees in 30 countries.",
            doc_id="10k",
            document_type=DocumentType.SEC_FILING,
            url="https://www.sec.gov/Archives/acme-10k",
        ),
        make_chunk(
            "c-pr1",
            "Acme announced a multi-year digital transformation program to migrate legacy systems to the "
            "cloud. "
            "The cloud migration will complete by the end of 2027.",
            doc_id="pr1",
        ),
        make_chunk(
            "c-pr2",
            "Acme is investing in generative AI and machine learning on a new lakehouse data platform. "
            "A chief data officer was appointed in March 2026 to lead data governance.",
            doc_id="pr2",
        ),
        make_chunk(
            "c-ir1",
            "Industry competition and new regulation are driving retailers to invest in personalization "
            "and supply chain analytics.",
            doc_id="ir1",
            document_type=DocumentType.INVESTOR_RELATIONS,
        ),
        make_chunk(
            "c-pr3",
            "Acme appoints new chief information officer and announces a $200 million technology investment "
            "roadmap launching next fiscal year.",
            doc_id="pr3",
            document_type=DocumentType.LEADERSHIP_ANNOUNCEMENT,
        ),
    ]


def retrieved(chunk: Chunk, score: float = 0.8, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, retriever="test", rank=rank)


def registry_for(chunks: Sequence[Chunk], *, query: str | None = None) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    for rank, chunk in enumerate(chunks):
        registry.register(retrieved(chunk, 0.9 - rank * 0.05, rank), query=query)
    return registry


def make_score(
    criterion: Criterion,
    score: int,
    confidence: float = 0.8,
    evidence_ids: tuple[str, ...] = ("E1",),
    rationale: str = "Evidence reviewed.",
) -> CriterionScore:
    return CriterionScore(
        criterion=criterion,
        score=score,
        weight=DEFAULT_WEIGHTS[criterion],
        confidence=confidence,
        rationale=rationale,
        evidence_ids=evidence_ids,
    )


def uniform_scores(score: int, confidence: float = 0.8) -> list[CriterionScore]:
    return [make_score(c, score, confidence, (f"E{i + 1}",)) for i, c in enumerate(Criterion)]
