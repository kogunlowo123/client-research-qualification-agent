# Runbook: Prompt-injection alert

Controls: `security/prompt_injection.py::PromptInjectionDetector`,
`security/sanitizer.py::ContentSanitizer`,
`security/poisoning.py::RetrievalPoisoningGuard`,
`security/output_guard.py::OutputGuard`, AI Gateway safety guardrails on FMAPI
endpoints. Threat context: [threat_model.md](../architecture/threat_model.md) (LLM01, LLM04, LLM08).

## Symptoms

- Alert `guardrail_blocks_hourly`: more than 20 audit events in one hour with
  `event_type LIKE 'guardrail%'` or an `authorization` event with
  `decision = deny` (`infrastructure/monitoring/alerts.sql`).
- Audit events `evidence.quarantined` (payload `documents: [{doc_id, reasons}]`,
  reasons such as `prompt_injection` plus detector signal names, or poisoning
  reasons such as `seo_spam_cluster`).
- Metrics `evidence.documents_quarantined`; log events `evidence.quarantined`
  with `doc_id`, `domain`, `reasons`.
- Briefs with warnings such as "evidence gathering: N document(s) quarantined by
  injection/poisoning guards" or "retrieval: N chunk(s) quarantined by the
  poisoning guard".
- `OutputGuard` violations (`prompt_leak`, `secret`, `pii`, `unsafe_url`,
  `uncited_url`, `code_execution_instruction`) on generated briefs; the
  `validation` step removes the affected statements and the review policy
  queues the brief at `high` priority.
- AI Gateway safety filter rejections in FMAPI inference tables.

Guardrail audit events emitted by `ClientResearchOrchestrator` (payload always
carries `stage`):

| Event type | Stage | Payload |
|---|---|---|
| `guardrail.injection_blocked` | `company_input` | `field` (request field rejected) |
| `guardrail.injection_blocked` | `evidence_gathering` | `documents` (documents blocked by windowed screening) |
| `guardrail.pii_redacted` | `evidence_gathering` | `redactions` |
| `guardrail.chunk_quarantined` | `retrieval` | `chunks` (count quarantined by the poisoning guard) |
| `guardrail.output_violation` | `validation` | `violations` (`OutputGuard` kinds) |
| `evidence.quarantined` | `evidence_gathering` | `documents: [{doc_id, reasons}]` |

In dev, staging and prod the runtime's `FanOutAuditLogger`
(`databricks/audit_sink.py`) writes the local hash-chained JSON Lines file and
replicates every record to `audit_log` through `DeltaAuditSink`, so the queries
below run directly against the table. `guardrail_blocks_hourly` counts
`guardrail%` events, which includes `guardrail.pii_redacted`; a spike made up
only of redactions indicates PII-heavy sources, not an attack.

## Dashboards and queries

- `audit_events_daily` (events by type and decision).

```sql
SELECT recorded_at, event_type, payload_json
FROM `${catalog}`.`${schema}`.`audit_log`
WHERE recorded_at >= current_timestamp() - INTERVAL 2 HOURS
  AND (event_type LIKE 'guardrail%' OR event_type IN ('evidence.quarantined', 'authorization'))
ORDER BY recorded_at DESC
LIMIT 200;
```

Quarantine reasons over the last day (payloads are PII-scrubbed before storage):

```sql
SELECT recorded_at,
       get_json_object(payload_json, '$.documents') AS quarantined_documents
FROM `${catalog}`.`${schema}`.`audit_log`
WHERE recorded_at >= current_timestamp() - INTERVAL 24 HOURS
  AND event_type = 'evidence.quarantined'
ORDER BY recorded_at DESC;
```

Map `doc_id` to domain through `documents` only for documents that were
accepted; quarantined documents are not persisted, so use the log events
(`domain` field) or the MLflow `run_state.json` for their source.

Verify the audit chain has not been tampered with (JSONL sink):

```python
from client_research_agent.governance.audit import verify_chain

print(verify_chain("var/audit/audit.jsonl"))
```

## Diagnosis

| Pattern | Interpretation |
|---|---|
| Many blocks from one domain | A site (or a compromised page) is serving injection payloads; content is being quarantined as designed |
| Blocks spread across many unrelated domains, similar text | Coordinated SEO or poisoning campaign; `RetrievalPoisoningGuard` should also report `seo_spam_cluster` |
| Blocks on the caller side (`authorization` deny, rate-limit) | Abuse of the endpoint by a principal |
| Blocks on benign corporate text | False positive; capture the text for tuning |
| `OutputGuard` violations without input blocks | Injection evaded input detection but was caught at output; highest priority to analyse |

## Mitigation

1. Confirm that blocked content did not reach any released brief: search
   `briefs` in the window for the offending domain in `brief_json`.
2. For a malicious domain, stop fetching it: remove it from any watchlist entry
   (`companies_watchlist`) and seed URLs. The allow-list is request-scoped, so no
   code change is needed to stop future fetches.
3. Purge quarantined evidence already stored for the affected company:
   `DeltaDocumentStore.delete_company_chunks(company)` (or a targeted `DELETE` on
   `chunks` by `source_domain`), then sync the index; re-ingest.
4. For caller abuse: revoke the principal's `CAN_QUERY` on the endpoint or remove
   it from `cra-analysts`; AI Gateway per-user limits cap the rate meanwhile.
5. For false positives: add the sample to the benign corpus
   (`tests/security/corpus.py::BENIGN_PARAGRAPHS`) in a pull request; do not
   raise `guardrails.injection_block_threshold` in production as a hot fix.
6. For evasions caught only at output: add the payload to
   `INJECTION_PAYLOADS` and treat as a security bug under [SECURITY.md](../../SECURITY.md).

## Escalation

- SEV1 if an injected instruction altered a released brief, exfiltrated data, or
  an `OutputGuard` `secret` violation occurred. Engage security; preserve the
  audit log, MLflow traces and inference-table rows for the window.
- SEV2 for an active campaign that is being blocked.
