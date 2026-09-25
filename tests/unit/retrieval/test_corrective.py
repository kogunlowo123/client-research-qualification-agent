from __future__ import annotations

from collections.abc import Sequence

import pytest

from client_research_agent.config.settings import RetrievalSettings
from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.crag import (
    CorrectiveRetriever,
    CragAction,
    RelevanceGrader,
    RetrievalVerdict,
)
from client_research_agent.retrieval.query_rewriting import QueryRewriter
from client_research_agent.retrieval.self_rag import SelfRagCritic, SupportLevel
from client_research_agent.utils.errors import UpstreamServiceError
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.retrieval.conftest import FailingLLM

RELEVANT = make_chunk("rel", "Acme completed its cloud migration to Azure and closed two data centers.")
PARTIAL = make_chunk("part", "Acme cloud budget increased this year.")
NOISE = make_chunk("noise", "Acme sponsored a local marathon for charity.")


def _hit(chunk: Chunk, rank: int = 0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=1.0 / (rank + 1), retriever="hybrid", rank=rank)


class QueryRoutedRetriever:
    """Returns different results depending on which query it receives."""

    def __init__(self, by_query: dict[str, list[Chunk]], default: list[Chunk]) -> None:
        self.by_query = by_query
        self.default = default
        self.queries: list[str] = []

    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        for needle, chunks in self.by_query.items():
            if needle in query:
                return [_hit(c, i) for i, c in enumerate(chunks)][:k]
        return [_hit(c, i) for i, c in enumerate(self.default)][:k]


class TestRelevanceGrader:
    def test_lexical_grade_ignores_company_terms(self) -> None:
        grader = RelevanceGrader()
        relevant = grader.lexical_grade("Acme cloud migration", RELEVANT)
        noise = grader.lexical_grade("Acme cloud migration", NOISE)
        assert relevant.relevant
        assert relevant.source == "lexical"
        assert relevant.score == pytest.approx(1.0)
        assert not noise.relevant
        assert noise.score == 0.0
        only_company = grader.lexical_grade("Acme Corp", NOISE)
        assert only_company.score > 0

    def test_llm_grade_and_fallback(self, failing_llm: FailingLLM) -> None:
        llm = ScriptedLLM(routes={"grade-relevance": {"relevant": True, "score": 0.9, "reason": "on topic"}})
        grade = RelevanceGrader(llm).grade("cloud", NOISE)
        assert (grade.relevant, grade.score, grade.reason, grade.source) == (True, 0.9, "on topic", "llm")
        assert "untrusted data" in llm.calls[0][0].content
        fallback = RelevanceGrader(failing_llm).grade_many("cloud migration", [_hit(RELEVANT)])
        assert fallback[0].source == "lexical"
        assert get_metrics().counter("retrieval.crag.grader_fallback") == 1


class TestCorrectiveRetriever:
    def _crag(self, retriever: QueryRoutedRetriever, **kwargs: object) -> CorrectiveRetriever:
        return CorrectiveRetriever(
            retriever,
            grader=RelevanceGrader(),
            rewriter=QueryRewriter(),
            min_relevance=0.5,
            max_corrections=2,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_accepts_good_retrieval(self) -> None:
        retriever = QueryRoutedRetriever({}, [RELEVANT, NOISE])
        result = self._crag(retriever).retrieve("Acme cloud migration", k=5, company="Acme Corp")
        assert result.verdict is RetrievalVerdict.CORRECT
        assert result.actions == [CragAction.ACCEPT]
        assert [r.chunk.chunk_id for r in result.chunks] == ["rel"]
        assert retriever.queries == ["Acme cloud migration"]
        assert result.final_relevance >= 0.5
        assert result.grades["noise"].relevant is False
        assert result.action_trace[-1] == "verdict:correct"

    def test_corrects_by_rewriting_then_accepts(self) -> None:
        retriever = QueryRoutedRetriever({"azure": [RELEVANT]}, [NOISE])
        result = self._crag(retriever).retrieve("Acme cloud migration", k=5, company="Acme Corp")
        assert CragAction.CORRECT in result.actions
        assert result.verdict is RetrievalVerdict.CORRECT
        assert [r.chunk.chunk_id for r in result.chunks] == ["rel"]
        assert len(retriever.queries) == 2
        assert result.queries[1] == retriever.queries[1]
        assert any(line.startswith("correct:1") for line in result.action_trace)

    def test_refresh_hook_invoked_when_still_incorrect(self) -> None:
        state = {"ingested": False}

        class RefreshingRetriever(QueryRoutedRetriever):
            def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
                self.queries.append(query)
                return [_hit(RELEVANT)] if state["ingested"] else [_hit(NOISE)]

        def refresh(query: str) -> int:
            state["ingested"] = True
            return 3

        retriever = RefreshingRetriever({}, [])
        result = self._crag(retriever, knowledge_refresh=refresh).retrieve("cloud migration", k=5)
        assert result.actions == [
            CragAction.CORRECT,
            CragAction.CORRECT,
            CragAction.REFRESH,
            CragAction.ACCEPT,
        ]
        assert "refresh:ingested=3" in result.action_trace
        assert result.verdict is RetrievalVerdict.CORRECT
        assert [r.chunk.chunk_id for r in result.chunks] == ["rel"]

    def test_refresh_failure_and_zero_ingest(self) -> None:
        retriever = QueryRoutedRetriever({}, [NOISE])

        def broken(query: str) -> int:
            raise UpstreamServiceError("crawler down", status_code=503)

        result = self._crag(retriever, knowledge_refresh=broken).retrieve("cloud migration", k=5)
        assert "refresh:failed UpstreamServiceError" in result.action_trace
        assert result.verdict is RetrievalVerdict.INCORRECT
        assert result.chunks == []
        nothing = self._crag(QueryRoutedRetriever({}, [NOISE]), knowledge_refresh=lambda q: 0).retrieve(
            "cloud", k=5
        )
        assert "refresh:ingested=0" in nothing.action_trace

    def test_empty_results_give_up(self) -> None:
        result = self._crag(QueryRoutedRetriever({}, [])).retrieve("cloud migration", k=5)
        assert result.verdict is RetrievalVerdict.INCORRECT
        assert result.actions[-1] is CragAction.GIVE_UP
        assert result.final_relevance == 0.0

    def test_ambiguous_verdict_and_ungraded_pass_through(self) -> None:
        pool = [PARTIAL, *[make_chunk(f"n{i}", f"Acme event {i}") for i in range(3)]]
        crag = CorrectiveRetriever(
            QueryRoutedRetriever({}, pool),
            grader=RelevanceGrader(),
            rewriter=QueryRewriter(),
            min_relevance=0.6,
            max_corrections=0,
            max_graded=2,
            relevance_window=2,
        )
        result = crag.retrieve("cloud migration budget", k=10)
        assert result.verdict is RetrievalVerdict.AMBIGUOUS
        assert [r.chunk.chunk_id for r in result.chunks] == ["part", "n1", "n2"]  # n0 graded out
        assert "n2" not in result.grades

    def test_from_settings_and_validation(self) -> None:
        crag = CorrectiveRetriever.from_settings(
            QueryRoutedRetriever({}, [RELEVANT]),
            RetrievalSettings(crag_min_relevance=0.2, crag_max_corrections=0),
            grader=RelevanceGrader(),
            rewriter=QueryRewriter(),
        )
        assert crag.verdict_for(0.25) is RetrievalVerdict.CORRECT
        assert crag.verdict_for(0.15) is RetrievalVerdict.AMBIGUOUS
        assert crag.verdict_for(0.05) is RetrievalVerdict.INCORRECT
        with pytest.raises(ValueError, match="max_graded"):
            CorrectiveRetriever(
                QueryRoutedRetriever({}, []), grader=RelevanceGrader(), rewriter=QueryRewriter(), max_graded=0
            )
        with pytest.raises(ValueError, match="relevance_window"):
            CorrectiveRetriever(
                QueryRoutedRetriever({}, []),
                grader=RelevanceGrader(),
                rewriter=QueryRewriter(),
                relevance_window=0,
            )

    def test_stops_when_corrections_exhausted(self) -> None:
        class OneQueryRewriter(QueryRewriter):
            def correction_queries(self, query: str, *, company: str | None = None) -> list[str]:
                return ["only one"]

        crag = CorrectiveRetriever(
            QueryRoutedRetriever({}, [NOISE]),
            grader=RelevanceGrader(),
            rewriter=OneQueryRewriter(),
            min_relevance=0.9,
            max_corrections=3,
        )
        result = crag.retrieve("cloud migration", k=3)
        assert result.actions == [CragAction.CORRECT]
        assert result.queries == ["cloud migration", "only one"]


class TestSelfRag:
    CHUNKS: Sequence[Chunk] = (RELEVANT, NOISE)

    def test_heuristic_reflection(self) -> None:
        critic = SelfRagCritic()
        assert critic.needs_retrieval("What did Acme announce about cloud migration?")
        assert not critic.needs_retrieval("hello there")
        assert not critic.needs_retrieval("what is a lakehouse?")
        assert critic.needs_retrieval("What is Acme's lakehouse?")
        relevance = critic.judge_relevance("Acme cloud migration", [_hit(RELEVANT), NOISE])
        assert relevance == {"rel": True, "noise": False}
        assert critic.judge_relevance("x", []) == {}

    def test_heuristic_support_levels(self) -> None:
        critic = SelfRagCritic()
        full = critic.critique_answer(
            "Did Acme migrate to the cloud?", "Acme completed its cloud migration to Azure.", self.CHUNKS
        )
        assert full.is_supported is SupportLevel.FULLY
        assert full.source == "heuristic"
        assert full.is_useful >= 3
        partial = critic.critique_answer(
            "Did Acme migrate?",
            "Acme completed its cloud migration to Azure. Acme closed two data centers. "
            "Acme bought a satellite.",
            self.CHUNKS,
        )
        assert partial.is_supported is SupportLevel.PARTIALLY
        assert partial.unsupported_claims == ("Acme bought a satellite.",)
        assert partial.support_ratio == pytest.approx(2 / 3, abs=1e-3)
        none = critic.critique_answer("Q?", "Globex acquired a bank. Initech hired pilots.", self.CHUNKS)
        assert none.is_supported is SupportLevel.NO
        empty = critic.critique_answer("Q?", "  ", self.CHUNKS)
        assert (empty.is_supported, empty.is_useful) == (SupportLevel.NO, 1)

    def test_llm_reflection_tokens(self) -> None:
        llm = ScriptedLLM(
            routes={
                "self-rag-retrieve": {"is_retrieval_needed": False, "reason": "chit-chat"},
                "self-rag-isrel": {"judgements": [{"id": "noise", "is_relevant": True}]},
                "self-rag-issup": {
                    "is_supported": "partially_supported",
                    "is_useful": 4,
                    "unsupported_claims": ["Acme bought a satellite.", " "],
                    "reason": "one claim unsupported",
                },
            }
        )
        report = SelfRagCritic(llm).reflect(
            "Did Acme migrate?", "Acme migrated. Acme bought a satellite.", [RELEVANT, NOISE]
        )
        assert report.is_retrieval_needed is False
        assert report.is_relevant == {"rel": True, "noise": True}
        assert report.is_supported is SupportLevel.PARTIALLY
        assert report.is_useful == 4
        assert report.critique.source == "llm"
        assert report.critique.unsupported_claims == ("Acme bought a satellite.",)
        assert report.notes == ["1 unsupported claim(s)"]

    def test_llm_failure_falls_back_everywhere(self, failing_llm: FailingLLM) -> None:
        critic = SelfRagCritic(failing_llm)
        report = critic.reflect("What did Acme say about cloud?", "Zebras migrate annually.", [NOISE])
        assert report.is_retrieval_needed is True
        assert report.is_relevant == {"noise": False}
        assert report.critique.source == "heuristic"
        assert report.notes == ["retrieval needed but no relevant passages", "1 unsupported claim(s)"]
        assert get_metrics().counter("retrieval.self_rag.fallback", token="issup") == 1
        assert failing_llm.calls >= 3
