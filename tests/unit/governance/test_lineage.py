from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from client_research_agent.governance.lineage import LineageRecorder, NodeType, statement_id
from client_research_agent.models import DocumentType, SourceDocument
from tests.support.doubles import make_chunk
from tests.unit.governance.briefs import EVIDENCE_URL, build_brief, evidence, fact


def _document() -> SourceDocument:
    return SourceDocument(
        doc_id="doc-1",
        company="Acme Corp",
        url=EVIDENCE_URL,
        title="Acme completes cloud migration",
        text="Acme migrated its ERP platform to the cloud in 2025.",
        document_type=DocumentType.PRESS_RELEASE,
        source_domain="acme.example.com",
        content_hash="abc123",
        trust_score=0.9,
    )


def test_full_chain_traces_statement_to_source() -> None:
    recorder = LineageRecorder("run-1", created_at=datetime(2026, 9, 1, tzinfo=UTC))
    recorder.record_document(_document())
    recorder.record_chunk(make_chunk("chunk-1", "Acme migrated its ERP.", doc_id="doc-1", url=EVIDENCE_URL))
    brief = build_brief()
    ids = recorder.record_brief(brief)
    assert len(ids) == len(brief.all_statements())
    overview_id = statement_id("company_overview", brief.company_overview[0].text)
    assert overview_id in ids
    assert recorder.upstream_sources(overview_id) == {EVIDENCE_URL}
    assert recorder.orphans() == []
    assert recorder.run_id == "run-1"

    graph = recorder.to_dict()
    assert graph["run_id"] == "run-1"
    types = {n["type"] for n in graph["nodes"]}
    assert types == {t.value for t in NodeType}
    doc_node = next(n for n in graph["nodes"] if n["type"] == "document")
    assert doc_node["attributes"]["content_hash"] == "abc123"
    assert graph["edge_count"] == len(graph["edges"])
    assert json.loads(recorder.to_json(indent=2))["node_count"] == graph["node_count"]


def test_chunk_without_document_creates_source_edges() -> None:
    recorder = LineageRecorder("r")
    recorder.record_chunk(make_chunk("c9", "text", doc_id="doc-9", url="https://x.example/9"))
    recorder.record_evidence(evidence("ev-9", chunk_id="c9"))
    sid = recorder.record_statement(fact("A fact.", "ev-9"), "risks")
    assert recorder.upstream_sources(sid) == {"https://x.example/9"}
    rows = recorder.to_table_rows()
    assert all(r["run_id"] == "r" for r in rows)
    assert {"upstream_type", "downstream_id", "recorded_at"} <= set(rows[0])


def test_orphans_for_uningested_citations() -> None:
    recorder = LineageRecorder("r")
    recorder.record_evidence(evidence("ev-x", chunk_id="chunk-x"))
    recorder.record_statement(fact("Unbacked.", "ev-missing"), "risks")
    assert recorder.orphans() == [("chunk", "chunk-x"), ("evidence", "ev-missing")]


def test_duplicate_records_idempotent() -> None:
    recorder = LineageRecorder("r")
    recorder.record_source("https://a.example")
    recorder.record_source("https://a.example")
    graph = recorder.to_dict()
    assert graph["edge_count"] == 1
    assert graph["node_count"] == 2


def test_upstream_handles_cycles_and_unknown() -> None:
    recorder = LineageRecorder("r")
    assert recorder.upstream_sources("nope") == frozenset()


def test_run_id_required() -> None:
    with pytest.raises(ValueError, match="run_id"):
        LineageRecorder("")
