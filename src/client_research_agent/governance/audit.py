"""Tamper-evident audit trail.

:class:`AuditLogger` implements :class:`~client_research_agent.services.ports.AuditSink`
as an append-only JSON Lines file in which every record carries the SHA-256 of
its predecessor (a hash chain). Editing, deleting or reordering any line breaks
the chain, which :func:`verify_chain` detects. Payloads are secret-scrubbed and
PII-redacted *before* hashing so the trail itself never becomes a leak.

:class:`DeltaAuditRecordBuilder` turns the same records into rows for an
append-only Unity Catalog Delta table (``delta.appendOnly = true``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from client_research_agent.observability.logging import current_context, scrub
from client_research_agent.security.pii import PiiRedactor

GENESIS_HASH = "0" * 64
_RESERVED_KEYS = ("principal", "run_id")


def _canonical(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_record_hash(record: Mapping[str, Any]) -> str:
    body = {k: v for k, v in record.items() if k != "record_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    timestamp: str
    event_type: str
    principal: str
    run_id: str | None
    payload: Mapping[str, Any]
    prev_hash: str
    record_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "principal": self.principal,
            "run_id": self.run_id,
            "payload": dict(self.payload),
            "prev_hash": self.prev_hash,
            "record_hash": self.record_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AuditRecord:
        return cls(
            sequence=int(data["sequence"]),
            timestamp=str(data["timestamp"]),
            event_type=str(data["event_type"]),
            principal=str(data["principal"]),
            run_id=None if data.get("run_id") is None else str(data["run_id"]),
            payload=dict(data.get("payload") or {}),
            prev_hash=str(data["prev_hash"]),
            record_hash=str(data["record_hash"]),
        )


@dataclass(frozen=True, slots=True)
class ChainVerification:
    valid: bool
    records_checked: int
    first_invalid_sequence: int | None = None
    reason: str | None = None


class AuditLogger:
    """Hash-chained append-only JSONL audit sink (thread-safe within a process)."""

    def __init__(
        self,
        path: str | Path,
        *,
        default_principal: str = "system",
        redactor: PiiRedactor | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        fsync: bool = True,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._default_principal = default_principal
        self._redactor = redactor or PiiRedactor()
        self._clock = clock
        self._fsync = fsync
        self._lock = threading.Lock()
        self._sequence, self._last_hash = self._recover_tail()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def last_hash(self) -> str:
        return self._last_hash

    def _recover_tail(self) -> tuple[int, str]:
        last: Mapping[str, Any] | None = None
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last = json.loads(line)
        if last is None:
            return 0, GENESIS_HASH
        return int(last["sequence"]), str(last["record_hash"])

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.append(event_type, payload)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        principal: str | None = None,
        run_id: str | None = None,
    ) -> AuditRecord:
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", event_type):
            raise ValueError(f"invalid audit event_type {event_type!r}")
        context = current_context()
        body = {k: v for k, v in payload.items() if k not in _RESERVED_KEYS}
        resolved_principal = str(
            principal or payload.get("principal") or context.get("principal") or self._default_principal
        )
        run_value = run_id or payload.get("run_id") or context.get("run_id")
        clean_payload = self._redactor.redact_value(scrub(json.loads(_canonical(body))))

        with self._lock:
            data: dict[str, Any] = {
                "sequence": self._sequence + 1,
                "timestamp": self._clock().astimezone(UTC).isoformat(),
                "event_type": event_type,
                "principal": resolved_principal,
                "run_id": None if run_value is None else str(run_value),
                "payload": clean_payload,
                "prev_hash": self._last_hash,
            }
            data["record_hash"] = compute_record_hash(data)
            line = _canonical(data) + "\n"
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            self._sequence = data["sequence"]
            self._last_hash = data["record_hash"]
        return AuditRecord.from_dict(data)

    def records(self) -> Iterator[AuditRecord]:
        return iter_records(self._path)

    def verify(self) -> ChainVerification:
        return verify_chain(self._path)


def iter_records(path: str | Path) -> Iterator[AuditRecord]:
    file_path = Path(path)
    if not file_path.exists():
        return
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield AuditRecord.from_dict(json.loads(line))


def verify_chain(path: str | Path) -> ChainVerification:
    """Recompute every hash and link; report the first broken record."""
    file_path = Path(path)
    if not file_path.exists():
        return ChainVerification(valid=True, records_checked=0)
    expected_prev = GENESIS_HASH
    expected_sequence = 1
    checked = 0
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                sequence = int(data["sequence"])
                prev_hash = data["prev_hash"]
                record_hash = data["record_hash"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return ChainVerification(
                    False, checked, expected_sequence, f"unparseable record on line {line_number}"
                )
            if sequence != expected_sequence:
                return ChainVerification(
                    False, checked, sequence, f"sequence gap: expected {expected_sequence}, found {sequence}"
                )
            if prev_hash != expected_prev:
                return ChainVerification(
                    False, checked, sequence, "prev_hash does not match preceding record"
                )
            if compute_record_hash(data) != record_hash:
                return ChainVerification(False, checked, sequence, "record content does not match its hash")
            expected_prev = record_hash
            expected_sequence += 1
            checked += 1
    return ChainVerification(valid=True, records_checked=checked)


class DeltaAuditRecordBuilder:
    """Builds rows and DDL for the Unity Catalog audit table written by the Databricks adapter."""

    SCHEMA: Mapping[str, str] = {
        "sequence": "BIGINT",
        "event_time": "TIMESTAMP",
        "event_type": "STRING",
        "principal": "STRING",
        "run_id": "STRING",
        "payload_json": "STRING",
        "prev_hash": "STRING",
        "record_hash": "STRING",
        "event_date": "DATE",
    }
    _IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")

    def to_row(self, record: AuditRecord) -> dict[str, Any]:
        event_time = datetime.fromisoformat(record.timestamp)
        return {
            "sequence": record.sequence,
            "event_time": event_time,
            "event_type": record.event_type,
            "principal": record.principal,
            "run_id": record.run_id,
            "payload_json": _canonical(record.payload),
            "prev_hash": record.prev_hash,
            "record_hash": record.record_hash,
            "event_date": event_time.date(),
        }

    def to_rows(self, records: Iterator[AuditRecord] | list[AuditRecord]) -> list[dict[str, Any]]:
        return [self.to_row(r) for r in records]

    def spark_ddl_schema(self) -> str:
        return ", ".join(f"{name} {kind}" for name, kind in self.SCHEMA.items())

    def create_table_sql(self, table_name: str) -> str:
        if not self._IDENT.fullmatch(table_name):
            raise ValueError(f"invalid table name {table_name!r}")
        columns = ",\n  ".join(f"{name} {kind}" for name, kind in self.SCHEMA.items())
        return (
            f"CREATE TABLE IF NOT EXISTS {table_name} (\n  {columns}\n)\n"
            "USING DELTA\nPARTITIONED BY (event_date)\n"
            "COMMENT 'Hash-chained audit trail of the client research agent'\n"
            "TBLPROPERTIES ('delta.appendOnly' = 'true', 'delta.enableChangeDataFeed' = 'true')"
        )
