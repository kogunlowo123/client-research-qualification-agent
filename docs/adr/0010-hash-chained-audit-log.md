# ADR-0010: Tamper-evident, hash-chained audit log

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

Briefs influence commercial decisions about named companies and executives.
Reviewers must be able to answer: who requested a brief, which sources were
fetched, which guardrails fired, and whether the record has been altered.
An ordinary log table can be edited by anyone with `MODIFY`, and logs are a
common PII and credential leak path.

## Decision

- `governance/audit.py::AuditLogger` implements the `AuditSink` port as an
  append-only JSON Lines file. Every record carries a monotonic `sequence`,
  `timestamp`, `event_type`, `principal`, `run_id`, the scrubbed `payload`,
  `prev_hash` and `record_hash = SHA-256(canonical JSON of the record including
  prev_hash)`; the first record chains from `GENESIS_HASH` (64 zeros). Writes are
  serialised by a lock and fsync'd by default; on start-up the logger recovers the
  tail to continue the chain.
- Payloads are secret-scrubbed and PII-redacted (`security/pii.py::PiiRedactor`)
  **before** hashing, so the trail cannot become a leak and redaction cannot be
  reversed without breaking the chain.
- `verify_chain(path)` walks the file and reports the first break (edit, delete,
  reorder or insertion).
- In Databricks (dev, staging, prod), `agent/factory.py::build_runtime` binds
  `databricks/audit_sink.py::FanOutAuditLogger`, which appends to the local chain
  and replicates each record through `DeltaAuditSink` to the Unity Catalog table
  `audit_log` (`event_id`, `sequence`, `event_type`, `payload_json`,
  `recorded_at`, `prev_hash`, `hash`), created with `delta.appendOnly = true` and
  Change Data Feed. If the Delta sink cannot be created, the runtime logs
  `runtime.delta_audit_unavailable` and keeps the local chain only.
- Access: `audit_log` is tagged `data_classification = confidential`; a row filter
  (`audit_log_row_filter` in `infrastructure/unity_catalog/03_governance.sql`)
  limits reads to engineers and the agent service principal; `payload_json`
  carries the column tag `pii_scrubbed = true`.

## Consequences

- Positive: tampering is detectable offline by anyone with read access, without
  trusting the storage layer.
- Positive: `delta.appendOnly` blocks `UPDATE` and `DELETE` at the table level,
  so the chain is protected by two independent mechanisms in Databricks.
- Negative: the chain detects tampering but does not prevent truncation of the
  tail by a principal who can rewrite the file or drop the table. Anchoring the
  latest hash externally (for example in the MLflow run or a separate
  write-once location) is a documented follow-up.
- Negative: a single chain serialises writers. It is per process for the JSONL
  sink; concurrent serving replicas need either one chain per replica or a
  sequencer. See the residual risks in `docs/architecture/threat_model.md`.
- Negative: append-only tables complicate retention. Deleting expired audit rows
  requires temporarily lifting `delta.appendOnly` under change control.
