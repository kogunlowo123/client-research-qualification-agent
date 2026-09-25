from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.databricks.audit_sink import (
    AUDIT_COLUMNS,
    DeltaAuditSink,
    FanOutAuditLogger,
    audit_table_row,
)
from client_research_agent.databricks.unity_catalog import DDL, StatementExecutor
from client_research_agent.governance.audit import AuditRecord, compute_record_hash
from client_research_agent.observability.metrics import get_metrics
from tests.support.databricks_fakes import FakeExecutor, PlatformStatementExecution


def ddl_columns() -> list[str]:
    body = DDL["audit_log"]
    columns = []
    for line in body.splitlines():
        token = line.strip().split(" ", 1)[0]
        if (
            token.isidentifier()
            and token.upper() not in {"CREATE", "CONSTRAINT", "USING", "COMMENT", "TBLPROPERTIES"}
            and token == token.lower()
        ):
            columns.append(token)
    return columns


def record(sequence: int = 1) -> AuditRecord:
    data: dict[str, Any] = {
        "sequence": sequence,
        "timestamp": datetime(2026, 9, 24, 12, 0, tzinfo=UTC).isoformat(),
        "event_type": "run.started",
        "principal": "analyst",
        "run_id": "run-1",
        "payload": {"company": "Acme"},
        "prev_hash": "0" * 64,
    }
    data["record_hash"] = compute_record_hash(data)
    return AuditRecord.from_dict(data)


def test_row_matches_ddl_columns() -> None:
    row = audit_table_row(record())
    assert tuple(row) == AUDIT_COLUMNS
    assert list(AUDIT_COLUMNS) == ddl_columns()
    envelope = json.loads(row["payload_json"])
    assert envelope == {"principal": "analyst", "run_id": "run-1", "payload": {"company": "Acme"}}
    assert row["hash"] == record().record_hash
    assert row["event_id"] == row["hash"][:32]


def test_sink_batches_and_inserts_through_statement_executor() -> None:
    sql = PlatformStatementExecution()
    sink = DeltaAuditSink(StatementExecutor(sql, "wh"), catalog="cat", schema="sch", batch_size=2)
    sink.write(record(1))
    assert sink.pending == 1
    assert not sql.executed
    sink.write(record(2))
    assert sink.pending == 0
    statement = sql.executed[0].statement
    assert statement.startswith("INSERT INTO `cat`.`sch`.`audit_log` (event_id, sequence")
    assert ":event_id_1" in statement
    assert sql.executed[0].params["sequence_1"].type == "BIGINT"
    assert sql.executed[0].params["recorded_at_0"].type == "TIMESTAMP"


def test_sink_failures_are_retried_and_bounded() -> None:
    executor = FakeExecutor(fail=True)
    sink = DeltaAuditSink(executor, catalog="cat", schema="sch", batch_size=1, max_buffer=2)  # type: ignore[arg-type]
    for i in range(1, 4):
        sink.write(record(i))
    assert sink.pending == 2
    assert sink.dropped == 1
    assert get_metrics().counter("audit.delta_write_failures") >= 1
    executor.fail = False
    assert sink.flush() == 2
    assert sink.pending == 0
    sink.close()


def test_sink_record_port_chains_events() -> None:
    executor = FakeExecutor()
    sink = DeltaAuditSink(executor, catalog="cat", schema="sch", batch_size=10)  # type: ignore[arg-type]
    sink.record(
        "guardrail.pii_redacted",
        {"redactions": 2, "principal": "svc", "run_id": "r1", "note": "call 555-201-4477"},
    )
    sink.record("run.completed", {"status": "ok"})
    sink.close()
    params = executor.calls[0][1]
    assert params["sequence_0"] == 1
    assert params["sequence_1"] == 2
    assert params["prev_hash_1"] == params["hash_0"]
    first = json.loads(params["payload_json_0"])
    assert first["principal"] == "svc"
    assert first["run_id"] == "r1"
    assert "555-201-4477" not in params["payload_json_0"]


def test_sink_validates_batching() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DeltaAuditSink(FakeExecutor(), catalog="c", schema="s", batch_size=0)  # type: ignore[arg-type]


class BrokenReplica:
    def write(self, record: AuditRecord) -> None:
        raise RuntimeError("replica down")

    def flush(self) -> int:
        raise RuntimeError("replica down")


def test_fan_out_logger_keeps_local_chain_when_replicas_fail(tmp_path: Path) -> None:
    good = DeltaAuditSink(FakeExecutor(), catalog="c", schema="s", batch_size=100)  # type: ignore[arg-type]
    logger = FanOutAuditLogger(tmp_path / "audit.jsonl", [BrokenReplica(), good])
    logger.append("run.started", {"company": "Acme"}, run_id="r1")
    logger.record("run.completed", {"status": "ok"})
    assert logger.verify().valid
    assert good.pending == 2
    assert logger.flush() == 2
    assert get_metrics().counter("audit.replica_failures") == 2
