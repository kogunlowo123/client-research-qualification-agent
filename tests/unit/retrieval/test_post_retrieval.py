from __future__ import annotations

from datetime import date

import pytest

from client_research_agent.models import Chunk, ChunkStrategy, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.retrieval.compression import ContextCompressor
from client_research_agent.retrieval.graph import EntityGraph, GraphRetriever, normalise_entity
from client_research_agent.retrieval.parent_child import ParentExpander
from client_research_agent.retrieval.reranking import (
    LexicalReranker,
    LLMReranker,
    Reranker,
    RerankWeights,
    maximal_marginal_relevance,
    proximity_score,
    recency_score,
)
from tests.support.doubles import ScriptedLLM, make_chunk
from tests.unit.retrieval._stores import MemoryDocumentStore
from tests.unit.retrieval.conftest import FailingLLM

TODAY = date(2026, 9, 1)


def _hit(chunk: Chunk, rank: int = 0, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk, score=score, retriever="hybrid", rank=rank)


def _candidates() -> list[RetrievedChunk]:
    chunks = [
        make_chunk("off", "Acme declared a quarterly dividend.", publication_date=TODAY),
        make_chunk("new", "Acme cloud migration to Azure is complete.", publication_date=TODAY),
        make_chunk("old", "Acme cloud migration to Azure is finished.", publication_date=date(2020, 1, 1)),
    ]
    return [_hit(c, rank=i) for i, c in enumerate(chunks)]


class TestFeatureScores:
    def test_recency(self) -> None:
        assert recency_score(None, TODAY, 365) == 0.5
        assert recency_score(TODAY, TODAY, 365) == 1.0
        assert recency_score(date(2025, 9, 1), TODAY, 365) == pytest.approx(0.5)
        assert recency_score(date(2027, 1, 1), TODAY, 365) == 1.0

    def test_proximity(self) -> None:
        assert proximity_score({"a", "b"}, ["x", "y"]) == 0.0
        assert proximity_score({"a", "b"}, ["a", "x"]) == 0.5
        assert proximity_score({"a", "b"}, ["a", "b", "x"]) == 1.0
        assert proximity_score({"a", "b"}, ["a", "x", "x", "b", "a", "b"]) == 1.0
        assert proximity_score({"a", "b"}, ["a", "x", "x", "b"]) == pytest.approx(0.5)


class TestLexicalReranker:
    def test_prefers_relevant_and_recent(self) -> None:
        reranker = LexicalReranker(recency_half_life_days=365, today=lambda: TODAY)
        assert isinstance(reranker, Reranker)
        ranked = reranker.rerank("Acme cloud migration Azure", _candidates())
        assert [r.chunk.chunk_id for r in ranked] == ["new", "old", "off"]
        assert [r.rank for r in ranked] == [0, 1, 2]
        assert ranked[0].score > ranked[1].score > ranked[2].score
        assert len(reranker.rerank("cloud", _candidates(), top_n=1)) == 1
        assert reranker.rerank("cloud", []) == []

    def test_weights_and_validation(self) -> None:
        only_trust = RerankWeights(0, 0, 0, 0, 1.0, 0)
        scores = LexicalReranker(weights=only_trust).score_all("", _candidates())
        assert scores == [pytest.approx(0.8)] * 3
        with pytest.raises(ValueError, match="half_life"):
            LexicalReranker(recency_half_life_days=0)


class TestLLMReranker:
    def test_blends_llm_scores(self) -> None:
        llm = ScriptedLLM(
            routes={
                "listwise-rerank": {
                    "scores": [{"id": "[1]", "score": 10}, {"id": "0", "score": 2}, {"id": "x", "score": 5}]
                }
            }
        )
        reranker = LLMReranker(llm, fallback=LexicalReranker(today=lambda: TODAY), max_candidates=2)
        ranked = reranker.rerank("Acme cloud migration", _candidates())
        assert [r.chunk.chunk_id for r in ranked] == ["new", "off", "old"]  # judged first, unjudged last
        assert "untrusted" in llm.calls[0][0].content

    def test_falls_back_on_failure_or_useless_output(self, failing_llm: FailingLLM) -> None:
        lexical = LexicalReranker(today=lambda: TODAY)
        expected = [r.chunk.chunk_id for r in lexical.rerank("Acme cloud migration Azure", _candidates())]
        failed = LLMReranker(failing_llm, fallback=lexical).rerank(
            "Acme cloud migration Azure", _candidates()
        )
        assert [r.chunk.chunk_id for r in failed] == expected
        useless = LLMReranker(ScriptedLLM(routes={"listwise-rerank": {"scores": [{"id": "zz", "score": 3}]}}))
        assert len(useless.rerank("Acme cloud migration Azure", _candidates(), top_n=2)) == 2
        assert useless.rerank("q", []) == []
        assert get_metrics().counter("retrieval.rerank.llm_fallback") == 2

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="llm_weight"):
            LLMReranker(ScriptedLLM(), llm_weight=2.0)


class TestMMR:
    def test_diversifies_near_duplicates(self) -> None:
        hits = [
            _hit(make_chunk("a", "Acme cloud migration Azure data centers"), score=1.0),
            _hit(make_chunk("a2", "Acme cloud migration Azure data centers closed"), score=0.95),
            _hit(make_chunk("b", "Acme hired a chief data officer"), score=0.6),
        ]
        selected = maximal_marginal_relevance(hits, k=2, lambda_mult=0.3)
        assert [r.chunk.chunk_id for r in selected] == ["a", "b"]
        assert [r.chunk.chunk_id for r in maximal_marginal_relevance(hits, k=2, lambda_mult=1.0)] == [
            "a",
            "a2",
        ]

    def test_embedding_similarity(self) -> None:
        hits = [_hit(make_chunk(cid, "same text"), score=1.0) for cid in ("x", "y", "z")]
        embeddings = {"x": [1.0, 0.0], "y": [1.0, 0.0], "z": [0.0, 1.0]}
        selected = maximal_marginal_relevance(hits, k=2, lambda_mult=0.5, embeddings=embeddings)
        assert [r.chunk.chunk_id for r in selected] == ["x", "z"]

    def test_edge_cases(self) -> None:
        assert maximal_marginal_relevance([], k=3) == []
        assert maximal_marginal_relevance(_candidates(), k=0) == []
        with pytest.raises(ValueError, match="lambda_mult"):
            maximal_marginal_relevance(_candidates(), k=1, lambda_mult=1.5)


class TestCompression:
    TEXT = (
        "Acme opened a new office. The cafeteria was renovated. Acme moved 70 percent of workloads to Azure. "
        "Parking was expanded. The cloud migration will finish in 2027. Staff enjoyed a picnic."
    )

    def test_keeps_relevant_sentences_verbatim_in_order(self) -> None:
        chunk = make_chunk("c", self.TEXT)
        compressed = ContextCompressor(max_sentences=2).compress(
            "Acme cloud migration Azure workloads", [_hit(chunk)]
        )
        text = compressed[0].chunk.text
        assert text == "Acme moved 70 percent of workloads to Azure. The cloud migration will finish in 2027."
        for sentence in text.split(". "):
            assert sentence.rstrip(".") in self.TEXT
        assert compressed[0].chunk.metadata["compressed"] is True
        assert compressed[0].chunk.metadata["kept_sentences"] == 2
        assert compressed[0].chunk.chunk_id == "c"
        assert compressed[0].chunk.token_count < chunk.token_count
        assert get_metrics().counter("retrieval.compression.applied") == 1

    def test_short_chunks_untouched_and_lead_fallback(self) -> None:
        short = _hit(make_chunk("s", "One sentence only."))
        compressor = ContextCompressor(max_sentences=2)
        assert compressor.compress("anything", [short])[0] is short
        lead = compressor.compress_chunk("zzz unrelated", make_chunk("c", self.TEXT))
        assert lead.text == "Acme opened a new office. The cafeteria was renovated."

    def test_parent_text_is_compressed_too(self) -> None:
        chunk = make_chunk("c", "Short child.").model_copy(update={"metadata": {"parent_text": self.TEXT}})
        compressed = ContextCompressor(max_sentences=1, parent_max_sentences=1).compress_chunk("Azure", chunk)
        assert compressed.text == "Short child."
        assert compressed.metadata["parent_text"] == "Acme moved 70 percent of workloads to Azure."
        assert compressed.metadata["parent_text_compressed"] is True

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="max_sentences"):
            ContextCompressor(max_sentences=0)


class TestParentExpander:
    def _store(self) -> tuple[MemoryDocumentStore, Chunk, list[RetrievedChunk]]:
        store = MemoryDocumentStore()
        parent = make_chunk("p1", "Parent context. " * 20, strategy=ChunkStrategy.PARENT)
        store.save_chunks([parent])
        children = [
            _hit(make_chunk("c1", "Parent context.", parent_id="p1")),
            _hit(make_chunk("c2", "Parent context.", parent_id="p1"), rank=1),
            _hit(make_chunk("c3", "orphan", parent_id="missing"), rank=2),
            _hit(make_chunk("c4", "no parent"), rank=3),
        ]
        return store, parent, children

    def test_attaches_parent_text_once_and_keeps_child_anchor(self) -> None:
        store, parent, children = self._store()
        expanded = ParentExpander(store).expand(children)
        assert [r.chunk.chunk_id for r in expanded] == ["c1", "c2", "c3", "c4"]
        assert expanded[0].chunk.metadata["parent_text"] == parent.text
        assert expanded[0].chunk.metadata["parent_id"] == "p1"
        assert expanded[1].chunk.metadata["parent_text_ref"] == "c1"
        assert "parent_text" not in expanded[2].chunk.metadata
        assert get_metrics().counter("retrieval.parent_expansion.attached") == 1

    def test_no_dedupe_and_clipping(self) -> None:
        store, parent, children = self._store()
        expanded = ParentExpander(store, max_parent_chars=50, dedupe_parents=False).expand(children[:2])
        assert all(len(r.chunk.metadata["parent_text"]) == 50 for r in expanded)
        assert expanded[0].chunk.metadata["parent_text"] in parent.text
        unrelated = make_chunk("u", "text that is not inside the parent", parent_id="p1")
        clipped = ParentExpander(store, max_parent_chars=30).expand([_hit(unrelated)])
        assert clipped[0].chunk.metadata["parent_text"] == parent.text[:30]

    def test_store_outage_and_no_parents(self) -> None:
        store, _, children = self._store()
        store.fail_get = True
        assert ParentExpander(store).expand(children) == children
        assert ParentExpander(store).expand(children[3:]) == children[3:]
        with pytest.raises(ValueError, match="max_parent_chars"):
            ParentExpander(store, max_parent_chars=0)


def _entity_chunk(chunk_id: str, entities: tuple[str, ...], company: str = "Acme Corp") -> Chunk:
    return make_chunk(chunk_id, f"text {chunk_id}", company=company).model_copy(update={"entities": entities})


class TestGraph:
    def _graph(self) -> EntityGraph:
        return EntityGraph.build(
            [
                _entity_chunk("c1", ("Acme Corp", "Maria Chen", "Microsoft Azure")),
                _entity_chunk("c2", ("Acme Corp", "Maria Chen", "Lakehouse Program")),
                _entity_chunk("c3", ("Acme Corp", "Robert Klein", "Board")),
                _entity_chunk("c4", ("Globex", "Priya Natarajan", "AWS"), company="Globex"),
                _entity_chunk("c5", ("Globex", "AWS"), company="Globex"),
                _entity_chunk("c5", ("ignored duplicate",)),
                _entity_chunk("c6", ("Solo", "solo", "")),
            ]
        )

    def test_structure(self) -> None:
        graph = self._graph()
        assert normalise_entity("  Maria-Chen ") == "maria chen"
        assert graph.chunk_count == 6
        assert len(graph) == len(graph.entities) == 10
        assert graph.neighbours("Maria Chen") == {
            "acme corp": 2.0,
            "microsoft azure": 1.0,
            "lakehouse program": 1.0,
        }
        assert graph.chunks_for("maria chen") == {"c1", "c2"}
        assert graph.display_name("maria chen") == "Maria Chen"
        assert graph.display_name("unknown") == "unknown"
        assert graph.degree("acme corp") == 6.0
        assert graph.chunk("missing") is None
        assert graph.seed_entities("What is Maria Chen doing with the lakehouse program?") == [
            "lakehouse program",
            "maria chen",
        ]

    def test_label_propagation_communities(self) -> None:
        communities = self._graph().communities(top_n=2)
        by_member = {e: c.community_id for c in communities for e in c.entities}
        assert by_member["Maria Chen"] == by_member["Microsoft Azure"] == by_member["Robert Klein"]
        assert by_member["Priya Natarajan"] == by_member["AWS"]
        assert by_member["Maria Chen"] != by_member["AWS"]
        acme = communities[0]
        assert acme.top_entities[0] == "Acme Corp"
        assert len(acme.top_entities) == 2
        assert acme.representative_chunk_ids[0] in {"c1", "c2", "c3"}
        assert acme.summary.startswith("Community 0: Acme Corp")
        assert "chunks" in acme.summary
        solo = next(c for c in communities if "Solo" in c.entities)
        assert solo.chunk_count == 1

    def test_graph_retrieval_expands_neighbours(self) -> None:
        retriever = GraphRetriever(self._graph(), ignore_entities=("Board",))
        hits = retriever.retrieve("Maria Chen priorities", k=5, company="Acme Corp")
        ids = [h.chunk.chunk_id for h in hits]
        assert ids[:2] == ["c1", "c2"]
        assert all(h.retriever == "graph" for h in hits)
        weights = retriever.expansion("Maria Chen", company="Acme Corp")
        assert weights["maria chen"] == 1.0
        assert "acme corp" not in weights
        assert weights["microsoft azure"] == pytest.approx(0.25)  # 0.5 * edge 1 / strongest edge 2
        assert retriever.retrieve("Maria Chen", k=5, company="Globex") == []
        assert retriever.retrieve("nothing matches", k=5) == []
        assert retriever.retrieve("Maria Chen", k=0) == []
        assert len(retriever.retrieve("Maria Chen", k=1)) == 1
        assert retriever.graph.chunk_count == 6

    def test_isolated_seed(self) -> None:
        retriever = GraphRetriever(EntityGraph.build([_entity_chunk("z", ("Zeta",))]))
        assert [h.chunk.chunk_id for h in retriever.retrieve("zeta", k=3)] == ["z"]
