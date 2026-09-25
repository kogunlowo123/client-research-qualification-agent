# Incident Runbooks

Each runbook has the same structure: Symptoms, Dashboards and queries,
Diagnosis, Mitigation, Escalation. Queries reference the datasets in
`infrastructure/monitoring/dashboard_queries.sql` and the alert queries in
`infrastructure/monitoring/alerts.sql` (thresholds in
`infrastructure/monitoring/alerts.json`). Replace `${catalog}`, `${schema}` and
`${environment}` (for example `client_research`, `agent_prod`, `prod`) when
pasting into the SQL editor.

| Runbook | Primary alert or trigger |
|---|---|
| [Serving endpoint 5xx or latency](serving-endpoint-errors-and-latency.md) | `agent_error_rate`, `agent_latency_p95`, CD smoke test failure |
| [FM API rate limiting and circuit breaker open](fm-api-rate-limiting-and-circuit-breaker.md) | Rising `llm.fallback`, `CircuitOpenError` in logs, 429s on FMAPI endpoints |
| [Vector index sync failure](vector-index-sync-failure.md) | `cra_ingestion_refresh.sync_index` failure, stale dense results |
| [Ingestion blocked by robots or SSRF policy](ingestion-blocked-by-policy.md) | `ingestion.sources_skipped` spike, `not_enough_evidence_share` |
| [Citation coverage regression or quality gate failure](citation-coverage-and-quality-gate.md) | `citation_coverage_daily`, `eval_gate_failed`, failed `evaluate` task |
| [Prompt-injection alert](prompt-injection-alert.md) | `guardrail_blocks_hourly` |
| [Rollback procedure](rollback.md) | Any of the above where the latest model version is the cause |

## Severity and paging

| Severity | Definition | Response |
|---|---|---|
| SEV1 | Prod endpoint unavailable or returning unsupported "verified facts" | Page on-call (prod jobs page through `oncall_notification_destination_id`); roll back first, diagnose second |
| SEV2 | Prod degraded (latency SLO breach, fallback model serving most traffic, coverage below gate) | On-call during business hours; mitigation within the SLO window |
| SEV3 | Staging or dev failures, single job failures that retried successfully | Ticket |

General rules:

1. Stabilise before diagnosing. For prod quality incidents the default
   mitigation is the [rollback procedure](rollback.md).
2. Never widen `crawler.allowed_domain_suffixes`, `DEFAULT_ANALYST_POLICIES` or
   guardrail thresholds as an incident mitigation without a reviewed change.
3. Record every action in the incident ticket with timestamps; the audit log
   records agent actions, not operator actions.
