from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from client_research_agent.governance.audit import (
    GENESIS_HASH,
    AuditLogger,
    DeltaAuditRecordBuilder,
    compute_record_hash,
    iter_records,
    verify_chain,
)
from client_research_agent.observability.logging import log_context
from client_research_agent.services.ports import AuditSink


def _clock() -> datetime:
    return datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@pytest.fixture
def audit_path(tmp_path: Path) -> Path:
    return tmp_path / "audit" / "audit.jsonl"


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rewrite(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_implements_audit_sink_and_chains(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, clock=_clock, fsync=False)
    assert isinstance(logger, AuditSink)
    logger.record("run_started", {"company": "Acme", "run_id": "r1", "principal": "alice"})
    second = logger.append("run_finished", {"status": "ok"}, principal="bob", run_id="r1")
    records = list(logger.records())
    assert [r.sequence for r in records] == [1, 2]
    assert records[0].prev_hash == GENESIS_HASH
    assert records[1].prev_hash == records[0].record_hash
    assert records[0].principal == "alice"
    assert records[0].run_id == "r1"
    assert "principal" not in records[0].payload
    assert second.principal == "bob"
    assert logger.last_hash == second.record_hash
    assert logger.path == audit_path
    verification = logger.verify()
    assert verification.valid
    assert verification.records_checked == 2


def test_payload_scrubbed_and_redacted(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, clock=_clock, fsync=True)
    logger.record(
        "llm_call",
        {
            "token": "abc",
            "note": "used " + "dapi" + "0123456789abcdef" * 2,
            "contact": "jane.doe@gmail.com",
        },
    )
    payload = next(iter_records(audit_path)).payload
    assert payload["token"] == "[REDACTED]"
    assert "dapi" not in payload["note"]
    assert payload["contact"] == "[REDACTED_EMAIL]"


def test_context_and_default_principal(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, default_principal="svc", clock=_clock, fsync=False)
    with log_context(run_id="ctx-run", principal="ctx-user"):
        logger.record("step", {"n": 1})
    logger.record("step", {"n": 2})
    records = list(iter_records(audit_path))
    assert (records[0].principal, records[0].run_id) == ("ctx-user", "ctx-run")
    assert (records[1].principal, records[1].run_id) == ("svc", None)


def test_resume_existing_chain(audit_path: Path) -> None:
    AuditLogger(audit_path, clock=_clock, fsync=False).record("a", {})
    resumed = AuditLogger(audit_path, clock=_clock, fsync=False)
    resumed.record("b", {})
    assert [r.sequence for r in iter_records(audit_path)] == [1, 2]
    assert verify_chain(audit_path).valid


def test_invalid_event_type(audit_path: Path) -> None:
    with pytest.raises(ValueError, match="event_type"):
        AuditLogger(audit_path, fsync=False).record("bad event\n", {})


def test_tamper_payload_detected(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, clock=_clock, fsync=False)
    for i in range(3):
        logger.record("e", {"i": i})
    records = _lines(audit_path)
    records[1]["payload"] = {"i": 99}
    _rewrite(audit_path, records)
    result = verify_chain(audit_path)
    assert not result.valid
    assert result.first_invalid_sequence == 2
    assert result.records_checked == 1
    assert "hash" in (result.reason or "")


def test_tamper_rehash_breaks_link(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, clock=_clock, fsync=False)
    for i in range(3):
        logger.record("e", {"i": i})
    records = _lines(audit_path)
    records[1]["payload"] = {"i": 99}
    records[1]["record_hash"] = compute_record_hash(records[1])
    _rewrite(audit_path, records)
    result = verify_chain(audit_path)
    assert result.first_invalid_sequence == 3
    assert result.reason == "prev_hash does not match preceding record"


def test_deletion_and_reorder_detected(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, clock=_clock, fsync=False)
    for i in range(3):
        logger.record("e", {"i": i})
    records = _lines(audit_path)
    _rewrite(audit_path, [records[0], records[2]])
    assert "sequence gap" in (verify_chain(audit_path).reason or "")
    _rewrite(audit_path, [records[1], records[0]])
    assert not verify_chain(audit_path).valid


def test_unparseable_and_missing(tmp_path: Path) -> None:
    assert verify_chain(tmp_path / "missing.jsonl").valid
    assert list(iter_records(tmp_path / "missing.jsonl")) == []
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{not json}\n", encoding="utf-8")
    result = verify_chain(broken)
    assert not result.valid
    assert "unparseable" in (result.reason or "")


def test_concurrent_appends_keep_chain_valid(audit_path: Path) -> None:
    logger = AuditLogger(audit_path, fsync=False)

    def work(worker: int) -> None:
        for i in range(25):
            logger.record("e", {"worker": worker, "i": i})

    threads = [threading.Thread(target=work, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    result = verify_chain(audit_path)
    assert result.valid
    assert result.records_checked == 100


def test_delta_builder(audit_path: Path) -> None:
    record = AuditLogger(audit_path, clock=_clock, fsync=False).append("e", {"b": 1, "a": 2}, run_id="r")
    builder = DeltaAuditRecordBuilder()
    row = builder.to_row(record)
    assert set(row) == set(builder.SCHEMA)
    assert row["payload_json"] == '{"a":2,"b":1}'
    assert row["event_date"] == date(2026, 9, 1)
    assert builder.to_rows([record])[0] == row
    assert builder.spark_ddl_schema().startswith("sequence BIGINT, event_time TIMESTAMP")
    ddl = builder.create_table_sql("main.governance.audit_log")
    assert "'delta.appendOnly' = 'true'" in ddl
    assert "PARTITIONED BY (event_date)" in ddl
    with pytest.raises(ValueError, match="table name"):
        builder.create_table_sql("x; DROP TABLE y")
    assert record.to_dict()["record_hash"] == record.record_hash
