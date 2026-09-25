from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.models import Chunk, ChunkStrategy, DocumentType, RetrievedChunk
from client_research_agent.ranking.evidence_ranker import EvidenceRanker
from tests.support.doubles import make_chunk

TODAY = date(2026, 9, 1)


def rc(chunk: Chunk, score: float, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, retriever="test", rank=rank)


def with_domain(chunk_id: str, text: str, domain: str, doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        company="Acme Corp",
        url=f"https://{domain}/{doc_id}",
        title=doc_id,
        document_type=DocumentType.PRESS_RELEASE,
        source_domain=domain,
        chunk_index=0,
        strategy=ChunkStrategy.CHILD,
        publication_date=date(2026, 6, 1),
    )


def test_base_score_combines_relevance_trust_and_recency() -> None:
    ranker = EvidenceRanker(today=TODAY)
    filing = make_chunk("a", "Filing text.", doc_id="a", document_type=DocumentType.SEC_FILING)
    webpage = make_chunk("b", "Web text.", doc_id="b", document_type=DocumentType.CORPORATE_WEBPAGE)
    old = make_chunk(
        "c", "Old text.", doc_id="c", document_type=DocumentType.SEC_FILING, publication_date=date(2018, 1, 1)
    )
    ranked = ranker.base_scores([rc(webpage, 0.9), rc(filing, 0.9), rc(old, 0.9)])
    assert [r.chunk_id for r in ranked] == ["a", "b", "c"]
    top = ranked[0]
    assert top.score == pytest.approx(top.relevance * top.trust * top.recency)
    assert top.trust == 1.0


def test_duplicates_keep_best_score() -> None:
    chunk = make_chunk("a", "Text.")
    ranked = EvidenceRanker(today=TODAY).base_scores([rc(chunk, 0.2), rc(chunk, 0.7), rc(chunk, 0.5)])
    assert len(ranked) == 1
    assert ranked[0].relevance == pytest.approx(0.7)


def test_mmr_prefers_diverse_documents() -> None:
    text = "Acme is migrating its legacy ERP estate to the cloud with a multi-year programme."
    a1 = make_chunk("a1", text, doc_id="a", index=0)
    a2 = make_chunk("a2", text + " Phase two follows.", doc_id="a", index=1)
    b1 = make_chunk("b1", "Acme appointed a chief data officer to lead AI governance.", doc_id="b")
    ranked = EvidenceRanker(today=TODAY, mmr_lambda=0.5).rank([rc(a1, 0.95), rc(a2, 0.94), rc(b1, 0.8)], k=2)
    assert [r.chunk_id for r in ranked] == ["a1", "b1"]
    no_diversity = EvidenceRanker(today=TODAY, mmr_lambda=1.0).rank(
        [rc(a1, 0.95), rc(a2, 0.94), rc(b1, 0.8)], 2
    )
    assert [r.chunk_id for r in no_diversity] == ["a1", "a2"]


def test_domain_cap_is_soft() -> None:
    chunks = [
        with_domain(f"x{i}", f"Distinct topic {i} about cloud {i}.", "one.example", f"d{i}") for i in range(4)
    ]
    other = with_domain("y", "Separate source about data platforms.", "two.example", "o1")
    ranker = EvidenceRanker(today=TODAY, max_per_domain=2, mmr_lambda=1.0)
    ranked = ranker.rank([rc(c, 0.9 - i * 0.01) for i, c in enumerate(chunks)] + [rc(other, 0.1)], k=3)
    assert [r.chunk_id for r in ranked] == ["x0", "x1", "y"]
    relaxed = ranker.rank([rc(c, 0.9) for c in chunks], k=4)
    assert len(relaxed) == 4


def test_rank_edge_cases_and_validation() -> None:
    ranker = EvidenceRanker(today=TODAY)
    assert ranker.rank([], 3) == []
    assert ranker.rank([rc(make_chunk("a", "x"), 0.5)], 0) == []
    zero = ranker.rank([rc(make_chunk("a", "x"), 0.0)], 1)
    assert zero[0].score == 0.0
    with pytest.raises(ValueError, match="mmr_lambda"):
        EvidenceRanker(mmr_lambda=1.5)
    with pytest.raises(ValueError, match="max_per_domain"):
        EvidenceRanker(max_per_domain=0)
    assert EvidenceRanker().today == date.today()


def test_similarity_bounds() -> None:
    a = make_chunk("a", "cloud migration programme", doc_id="d")
    b = make_chunk("b", "cloud migration programme", doc_id="d")
    assert EvidenceRanker.similarity(rc(a, 1), rc(b, 1)) == 1.0
