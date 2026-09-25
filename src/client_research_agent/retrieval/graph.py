"""GraphRAG over an entity co-occurrence graph.

Entities attached to chunks (people, products, partners, programmes) become
nodes; co-occurrence within a chunk adds weighted edges. Retrieval seeds on
entities named in the query and expands to their graph neighbours, surfacing
evidence that shares no vocabulary with the query ("Acme" -> "Jane Doe, CTO"
-> the chunk announcing her lakehouse programme). Communities come from
deterministic weighted label propagation and are summarised extractively.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations

from client_research_agent.models import Chunk, RetrievedChunk
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.observability.tracing import SpanType, traced
from client_research_agent.retrieval.base import company_matches

RETRIEVER_NAME = "graph"
_NORMALISE = re.compile(r"[^a-z0-9]+")


def normalise_entity(name: str) -> str:
    return _NORMALISE.sub(" ", name.casefold()).strip()


@dataclass(frozen=True, slots=True)
class Community:
    community_id: int
    entities: tuple[str, ...]
    top_entities: tuple[str, ...]
    representative_chunk_ids: tuple[str, ...]
    chunk_count: int
    summary: str


class EntityGraph:
    """Undirected weighted co-occurrence graph stored as adjacency dictionaries."""

    def __init__(self) -> None:
        self._adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        self._display: dict[str, str] = {}
        self._entity_chunks: dict[str, set[str]] = defaultdict(set)
        self._chunks: dict[str, Chunk] = {}

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> EntityGraph:
        graph = cls()
        for chunk in chunks:
            graph.add_chunk(chunk)
        return graph

    def add_chunk(self, chunk: Chunk) -> None:
        if chunk.chunk_id in self._chunks:
            return
        self._chunks[chunk.chunk_id] = chunk
        keys: list[str] = []
        for entity in chunk.entities:
            key = normalise_entity(entity)
            if not key or key in keys:
                continue
            keys.append(key)
            self._display.setdefault(key, entity.strip())
            self._entity_chunks[key].add(chunk.chunk_id)
            self._adjacency.setdefault(key, {})
        for a, b in combinations(sorted(keys), 2):
            self._adjacency[a][b] = self._adjacency[a].get(b, 0.0) + 1.0
            self._adjacency[b][a] = self._adjacency[b].get(a, 0.0) + 1.0

    @property
    def entities(self) -> list[str]:
        return sorted(self._adjacency)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def __len__(self) -> int:
        return len(self._adjacency)

    def display_name(self, entity: str) -> str:
        return self._display.get(entity, entity)

    def neighbours(self, entity: str) -> dict[str, float]:
        return dict(self._adjacency.get(normalise_entity(entity), {}))

    def chunks_for(self, entity: str) -> set[str]:
        return set(self._entity_chunks.get(normalise_entity(entity), set()))

    def chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def degree(self, entity: str) -> float:
        return sum(self._adjacency.get(entity, {}).values())

    def seed_entities(self, query: str) -> list[str]:
        """Entities whose normalised name appears as a whole phrase in the query."""
        padded = f" {normalise_entity(query)} "
        seeds = [e for e in self._adjacency if f" {e} " in padded]
        return sorted(seeds, key=lambda e: (-len(e), e))

    def label_propagation(self, *, max_iterations: int = 20) -> dict[str, str]:
        """Deterministic asynchronous weighted label propagation; returns entity -> label."""
        labels = {node: node for node in self._adjacency}
        order = sorted(self._adjacency)
        for _ in range(max_iterations):
            changed = False
            for node in order:
                neighbours = self._adjacency[node]
                if not neighbours:
                    continue
                votes: dict[str, float] = defaultdict(float)
                for neighbour, weight in neighbours.items():
                    votes[labels[neighbour]] += weight
                best = max(votes.values())
                winner = min(label for label, vote in votes.items() if vote == best)
                if labels[node] in votes and votes[labels[node]] == best:
                    winner = labels[node]
                if winner != labels[node]:
                    labels[node] = winner
                    changed = True
            if not changed:
                break
        return labels

    def communities(self, *, top_n: int = 5, max_iterations: int = 20) -> list[Community]:
        labels = self.label_propagation(max_iterations=max_iterations)
        groups: dict[str, list[str]] = defaultdict(list)
        for node, label in labels.items():
            groups[label].append(node)
        ordered = sorted(groups.values(), key=lambda members: (-len(members), sorted(members)[0]))
        return [self._summarise(index, sorted(members), top_n) for index, members in enumerate(ordered)]

    def _summarise(self, community_id: int, members: Sequence[str], top_n: int) -> Community:
        member_set = set(members)
        internal_degree = {
            m: sum(w for n, w in self._adjacency[m].items() if n in member_set)
            + len(self._entity_chunks[m]) * 0.01
            for m in members
        }
        top = sorted(members, key=lambda m: (-internal_degree[m], m))[:top_n]
        coverage: Counter[str] = Counter()
        for member in members:
            coverage.update(self._entity_chunks[member])
        representative = tuple(cid for cid, _ in sorted(coverage.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
        top_names = tuple(self.display_name(m) for m in top)
        summary = (
            f"Community {community_id}: {', '.join(top_names)} "
            f"({len(members)} entities across {len(coverage)} chunks)"
        )
        return Community(
            community_id=community_id,
            entities=tuple(self.display_name(m) for m in members),
            top_entities=top_names,
            representative_chunk_ids=representative,
            chunk_count=len(coverage),
            summary=summary,
        )


class GraphRetriever:
    """Seed entities from the query, expand one hop, score chunks by entity evidence."""

    def __init__(
        self,
        graph: EntityGraph,
        *,
        max_neighbours: int = 8,
        neighbour_weight: float = 0.5,
        ignore_entities: Iterable[str] = (),
    ) -> None:
        self._graph = graph
        self._max_neighbours = max_neighbours
        self._neighbour_weight = neighbour_weight
        self._ignore = {normalise_entity(e) for e in ignore_entities}

    @property
    def graph(self) -> EntityGraph:
        return self._graph

    def expansion(self, query: str, *, company: str | None = None) -> dict[str, float]:
        """Entity -> weight: seeds weigh 1.0, neighbours ``neighbour_weight * relative edge weight``."""
        ignore = set(self._ignore)
        if company:
            ignore.add(normalise_entity(company))
        seeds = [s for s in self._graph.seed_entities(query) if s not in ignore]
        weights: dict[str, float] = dict.fromkeys(seeds, 1.0)
        for seed in seeds:
            neighbours = self._graph.neighbours(seed)
            if not neighbours:
                continue
            strongest = max(neighbours.values())
            ranked = sorted(neighbours.items(), key=lambda kv: (-kv[1], kv[0]))[: self._max_neighbours]
            for neighbour, weight in ranked:
                if neighbour in ignore:
                    continue
                score = self._neighbour_weight * weight / strongest
                weights[neighbour] = max(weights.get(neighbour, 0.0), score)
        return weights

    @traced("graph.retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, *, k: int, company: str | None = None) -> list[RetrievedChunk]:
        weights = self.expansion(query, company=company)
        if not weights or k <= 0:
            return []
        scores: dict[str, float] = defaultdict(float)
        for entity, weight in weights.items():
            for chunk_id in self._graph.chunks_for(entity):
                scores[chunk_id] += weight
        ranked: list[RetrievedChunk] = []
        for chunk_id, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            chunk = self._graph.chunk(chunk_id)
            if chunk is None or not company_matches(chunk, company):
                continue
            ranked.append(
                RetrievedChunk(chunk=chunk, score=score, retriever=RETRIEVER_NAME, rank=len(ranked))
            )
            if len(ranked) >= k:
                break
        get_metrics().increment("retrieval.graph.queries")
        return ranked
