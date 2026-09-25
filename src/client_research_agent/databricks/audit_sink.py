"""Unity Catalog ``audit_log`` sink and the fan-out audit logger.

:class:`DeltaAuditSink` writes hash-chained audit records to the append-only
``<catalog>.<schema>.audit_log`` table (columns exactly as
``unity_catalog.DDL["audit_log"]``: ``event_id, sequence, event_type,
payload_json, recorded_at, prev_hash, hash``) with parameterised multi-row
``INSERT`` statements on the SQL warehouse. Rows are produced from
:class:`~client_research_agent.governance.audit.DeltaAuditRecordBuilder` rows;
``principal`` and ``run_id``, which the table has no columns for, travel inside
``payload_json`` together with the payload so the chain hash can be re-verified
from the table. Writes are buffered and flushed in batches; a failed flush is
logged and counted, never raised, and the batch is retried on the next flush
(bounded by ``max_buffer``).

:class:`FanOutAuditLogger` is an :class:`AuditLogger` (the local JSONL file stays
the tamper-evident primary) that replicates every record to further sinks such
as :class:`DeltaAuditSink`.

Used directly through the :class:`~client_research_agent.services.ports.AuditSink`
``record`` method, the sink keeps its own in-process hash chain.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from client_research_agent.databricks.unity_catalog import StatementExecutor, qualified_name
from client_research_agent.governance.audit import (
    GENESIS_HASH,
    AuditLogger,
    AuditRecord,
    DeltaAuditRecordBuilder,
    compute_record_hash,
)
from client_research_agent.observability.logging import get_logger
from client_research_agent.observability.metrics import get_metrics
from client_research_agent.security.pii import PiiRedactor

AUDIT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "sequence",
    "event_type",
    "payload_json",
    "recorded_at",
    "prev_hash",
    "hash",
)
_log = get_logger(__name__)


class AuditReplica(Protocol):
    def write(self, record: AuditRecord) -> None: ...

    def flush(self) -> int: ...


def audit_table_row(record: AuditRecord, builder: DeltaAuditRecordBuilder | None = None) -> dict[str, Any]:
    """Map an audit record onto the ``audit_log`` table columns."""
    row = (builder or DeltaAuditRecordBuilder()).to_row(record)
    envelope = {
        "principal": row["principal"],
        "run_id": row["run_id"],
        "payload": json.loads(row["payload_json"]),
    }
    return {
        "event_id": row["record_hash"][:32],
        "sequence": int(row["sequence"]),
        "event_type": row["event_type"],
        "payload_json": json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "recorded_at": row["event_time"],
        "prev_hash": row["prev_hash"],
        "hash": row["record_hash"],
    }


class DeltaAuditSink:
    """Buffered, fail-safe writer of audit records to the Unity Catalog ``audit_log`` table."""

    def __init__(
        self,
        executor: StatementExecutor,
        *,
        catalog: str,
        schema: str,
        batch_size: int = 50,
        max_buffer: int = 5000,
        default_principal: str = "system",
        redactor: PiiRedactor | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if batch_size < 1 or max_buffer < batch_size:
            raise ValueError("batch_size must be >= 1 and max_buffer >= batch_size")
        self._executor = executor
        self._table = qualified_name(catalog, schema, "audit_log")
        self._batch_size = batch_size
        self._max_buffer = max_buffer
        self._builder = DeltaAuditRecordBuilder()
        self._default_principal = default_principal
        self._redactor = redactor or PiiRedactor()
        self._clock = clock
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._sequence = 0
        self._last_hash = GENESIS_HASH
        self.dropped = 0

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def write(self, record: AuditRecord) -> None:
        row = audit_table_row(record, self._builder)
        with self._lock:
            self._buffer.append(row)
            overflow = len(self._buffer) - self._max_buffer
            if overflow > 0:
                del self._buffer[:overflow]
                self.dropped += overflow
                get_metrics().increment("audit.delta_dropped", overflow)
            ready = len(self._buffer) >= self._batch_size
        if ready:
            self.flush()

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """``AuditSink`` port: chain the event in-process and buffer it."""
        body = {k: v for k, v in payload.items() if k not in ("principal", "run_id")}
        with self._lock:
            data: dict[str, Any] = {
                "sequence": self._sequence + 1,
                "timestamp": self._clock().astimezone(UTC).isoformat(),
                "event_type": event_type,
                "principal": str(payload.get("principal") or self._default_principal),
                "run_id": None if payload.get("run_id") is None else str(payload["run_id"]),
                "payload": self._redactor.redact_value(json.loads(json.dumps(body, default=str))),
                "prev_hash": self._last_hash,
            }
            data["record_hash"] = compute_record_hash(data)
            self._sequence = data["sequence"]
            self._last_hash = data["record_hash"]
        self.write(AuditRecord.from_dict(data))

    def flush(self) -> int:
        """Insert buffered rows in batches; returns the number written. Never raises."""
        written = 0
        while True:
            with self._lock:
                batch = self._buffer[: self._batch_size]
            if not batch:
                return written
            try:
                self._insert(batch)
            except Exception as exc:  # audit replication must never fail a research run
                get_metrics().increment("audit.delta_write_failures")
                _log.warning("audit.delta_write_failed", rows=len(batch), error=type(exc).__name__)
                return written
            with self._lock:
                del self._buffer[: len(batch)]
            written += len(batch)
            get_metrics().increment("audit.delta_rows_written", len(batch))

    def close(self) -> None:
        """Flush what can be written; anything still buffered is reported, not raised."""
        self.flush()
        remaining = self.pending
        if remaining:
            _log.warning("audit.delta_unflushed", rows=remaining)

    def _insert(self, rows: Sequence[Mapping[str, Any]]) -> None:
        params: dict[str, Any] = {}
        tuples: list[str] = []
        for index, row in enumerate(rows):
            names = []
            for column in AUDIT_COLUMNS:
                name = f"{column}_{index}"
                params[name] = row[column]
                names.append(f":{name}")
            tuples.append(f"({', '.join(names)})")
        statement = (
            f"INSERT INTO {self._table} ({', '.join(AUDIT_COLUMNS)}) VALUES "  # noqa: S608 - validated identifiers  # nosec B608
            + ", ".join(tuples)
        )
        self._executor.execute(statement, params, op="insert_audit")


class FanOutAuditLogger(AuditLogger):
    """Local hash-chained JSONL audit log that also replicates each record to other sinks."""

    def __init__(self, path: str | Path, replicas: Sequence[AuditReplica], **kwargs: Any) -> None:
        super().__init__(path, **kwargs)
        self._replicas = tuple(replicas)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        principal: str | None = None,
        run_id: str | None = None,
    ) -> AuditRecord:
        record = super().append(event_type, payload, principal=principal, run_id=run_id)
        for replica in self._replicas:
            try:
                replica.write(record)
            except Exception as exc:
                get_metrics().increment("audit.replica_failures")
                _log.warning("audit.replica_failed", replica=type(replica).__name__, error=type(exc).__name__)
        return record

    def flush(self) -> int:
        written = 0
        for replica in self._replicas:
            try:
                written += replica.flush()
            except Exception as exc:
                _log.warning(
                    "audit.replica_flush_failed", replica=type(replica).__name__, error=type(exc).__name__
                )
        return written
