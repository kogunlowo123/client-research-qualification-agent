"""Run-level data lineage: run -> source URLs -> documents -> chunks -> evidence -> brief statements.

The graph is exported as a dict/JSON document (logged as an MLflow artifact)
and as flat edge rows for a Unity Catalog lineage table, so any sentence in a
brief can be traced back to the exact public pages it rests on.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from client_research_agent.models import BriefStatement, Chunk, ClientBrief, Evidence, SourceDocument


class NodeType(StrEnum):
    RUN = "run"
    SOURCE_URL = "source_url"
    DOCUMENT = "document"
    CHUNK = "chunk"
    EVIDENCE = "evidence"
    STATEMENT = "statement"


@dataclass(frozen=True, slots=True)
class LineageNode:
    type: NodeType
    id: str
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LineageEdge:
    upstream_type: NodeType
    upstream_id: str
    downstream_type: NodeType
    downstream_id: str


def statement_id(section: str, text: str) -> str:
    return hashlib.sha256(f"{section}\x1f{text}".encode()).hexdigest()[:16]


@dataclass
class _Graph:
    nodes: dict[tuple[NodeType, str], LineageNode] = field(default_factory=dict)
    edges: dict[LineageEdge, None] = field(default_factory=dict)


class LineageRecorder:
    def __init__(self, run_id: str, *, created_at: datetime | None = None) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        self._run_id = run_id
        self._created_at = created_at or datetime.now(UTC)
        self._graph = _Graph()
        self._lock = threading.Lock()
        self._add_node(NodeType.RUN, run_id, {})

    @property
    def run_id(self) -> str:
        return self._run_id

    def _add_node(self, kind: NodeType, node_id: str, attributes: dict[str, Any]) -> None:
        key = (kind, node_id)
        attrs = tuple(sorted((k, str(v)) for k, v in attributes.items() if v is not None))
        existing = self._graph.nodes.get(key)
        if existing is None or (attrs and not existing.attributes):
            self._graph.nodes[key] = LineageNode(kind, node_id, attrs)

    def _link(self, up: tuple[NodeType, str], down: tuple[NodeType, str]) -> None:
        self._graph.edges[LineageEdge(up[0], up[1], down[0], down[1])] = None

    def record_source(self, url: str) -> None:
        with self._lock:
            self._add_node(NodeType.SOURCE_URL, url, {})
            self._link((NodeType.RUN, self._run_id), (NodeType.SOURCE_URL, url))

    def record_document(self, document: SourceDocument) -> None:
        self.record_source(document.url)
        with self._lock:
            self._add_node(
                NodeType.DOCUMENT,
                document.doc_id,
                {
                    "content_hash": document.content_hash,
                    "document_type": document.document_type.value,
                    "source_domain": document.source_domain,
                    "trust_score": document.trust_score,
                },
            )
            self._link((NodeType.SOURCE_URL, document.url), (NodeType.DOCUMENT, document.doc_id))

    def record_chunk(self, chunk: Chunk) -> None:
        with self._lock:
            if (NodeType.DOCUMENT, chunk.doc_id) not in self._graph.nodes:
                self._add_node(NodeType.SOURCE_URL, chunk.url, {})
                self._link((NodeType.RUN, self._run_id), (NodeType.SOURCE_URL, chunk.url))
                self._add_node(NodeType.DOCUMENT, chunk.doc_id, {"source_domain": chunk.source_domain})
                self._link((NodeType.SOURCE_URL, chunk.url), (NodeType.DOCUMENT, chunk.doc_id))
            self._add_node(NodeType.CHUNK, chunk.chunk_id, {"strategy": chunk.strategy.value})
            self._link((NodeType.DOCUMENT, chunk.doc_id), (NodeType.CHUNK, chunk.chunk_id))

    def record_evidence(self, evidence: Evidence) -> None:
        with self._lock:
            self._add_node(NodeType.EVIDENCE, evidence.evidence_id, {"url": evidence.url})
            if (NodeType.CHUNK, evidence.chunk_id) not in self._graph.nodes:
                self._add_node(NodeType.CHUNK, evidence.chunk_id, {})
            self._link((NodeType.CHUNK, evidence.chunk_id), (NodeType.EVIDENCE, evidence.evidence_id))

    def record_statement(self, statement: BriefStatement, section: str) -> str:
        sid = statement_id(section, statement.text)
        with self._lock:
            self._add_node(
                NodeType.STATEMENT, sid, {"section": section, "provenance": statement.provenance.value}
            )
            for evidence_id in statement.evidence_ids:
                if (NodeType.EVIDENCE, evidence_id) not in self._graph.nodes:
                    self._add_node(NodeType.EVIDENCE, evidence_id, {})
                self._link((NodeType.EVIDENCE, evidence_id), (NodeType.STATEMENT, sid))
        return sid

    def record_brief(self, brief: ClientBrief) -> list[str]:
        for evidence in brief.evidence:
            self.record_evidence(evidence)
        sections: dict[str, Iterable[BriefStatement]] = {
            "company_overview": brief.company_overview,
            "technology_priorities": brief.technology_priorities,
            "gartner_relevant_insights": brief.gartner_relevant_insights,
            "opportunities": brief.opportunities,
            "risks": brief.risks,
            "executive_summary": brief.executive_summary,
            "executive_talking_points": brief.executive_talking_points,
            "recommended_next_actions": brief.recommended_next_actions,
        }
        return [self.record_statement(s, name) for name, items in sections.items() for s in items]

    # -- queries ---------------------------------------------------------------------------------------
    def upstream_sources(self, statement: str) -> frozenset[str]:
        """Source URLs a statement node (by id) ultimately rests on."""
        with self._lock:
            parents: dict[tuple[NodeType, str], list[tuple[NodeType, str]]] = {}
            for edge in self._graph.edges:
                down = (edge.downstream_type, edge.downstream_id)
                parents.setdefault(down, []).append((edge.upstream_type, edge.upstream_id))
        found: set[str] = set()
        stack = [(NodeType.STATEMENT, statement)]
        seen: set[tuple[NodeType, str]] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            if node[0] is NodeType.SOURCE_URL:
                found.add(node[1])
                continue
            stack.extend(parents.get(node, []))
        return frozenset(found)

    def orphans(self) -> list[tuple[str, str]]:
        """Evidence/chunk nodes with no upstream (cited but never ingested in this run)."""
        with self._lock:
            has_parent = {(e.downstream_type, e.downstream_id) for e in self._graph.edges}
            return sorted(
                (kind.value, node_id)
                for (kind, node_id) in self._graph.nodes
                if kind in (NodeType.CHUNK, NodeType.EVIDENCE) and (kind, node_id) not in has_parent
            )

    # -- export ----------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            nodes = [
                {"type": n.type.value, "id": n.id, "attributes": dict(n.attributes)}
                for n in sorted(self._graph.nodes.values(), key=lambda n: (n.type.value, n.id))
            ]
            edges = [
                {
                    "upstream_type": e.upstream_type.value,
                    "upstream_id": e.upstream_id,
                    "downstream_type": e.downstream_type.value,
                    "downstream_id": e.downstream_id,
                }
                for e in self._graph.edges
            ]
        return {
            "run_id": self._run_id,
            "created_at": self._created_at.isoformat(),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_table_rows(self) -> list[dict[str, Any]]:
        with self._lock:
            edges = list(self._graph.edges)
        return [
            {
                "run_id": self._run_id,
                "upstream_type": e.upstream_type.value,
                "upstream_id": e.upstream_id,
                "downstream_type": e.downstream_type.value,
                "downstream_id": e.downstream_id,
                "recorded_at": self._created_at,
            }
            for e in edges
        ]
