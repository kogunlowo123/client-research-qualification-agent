# Operational Guide

How the agent is run in production: service level objectives, monitoring and
alerting, capacity, cost controls, data retention, access reviews and on-call.
Incident procedures are in [docs/runbooks](../runbooks/README.md).

## 1. Service level objectives

SLOs apply to `prod`. Targets are aligned with the alert thresholds in
`infrastructure/monitoring/alerts.json` so that an alert fires before an SLO is
exhausted.

| SLI | Definition (source) | SLO target | Alert |
|---|---|---|---|
| Availability | Share of requests to `cra-agent-prod` with `status_code < 500` (`cra_agent_payload`, query `error_rate_hourly`) | 99.0% over 30 days | `agent_error_rate` > 2% over 1 hour |
| Latency | p95 `execution_duration_ms` of `cra-agent-prod` (`latency_percentiles_hourly`) | p95 <= 60 s over 7 days | `agent_latency_p95` > 60,000 ms over 1 hour |
| Grounding | Mean `citation_report.supported_statements / total_statements` of briefs generated per day (`citation_coverage_daily`) | >= 0.90 daily | `citation_coverage_daily` < 0.90 |
| Evaluation quality | `gate_passed` of nightly `cra_evaluation` runs (`eval_results`) | Every nightly run passes | `eval_gate_failed` > 0 in 24 h |
| Evidence sufficiency | Share of briefs with verdict `not_enough_evidence` over 7 days | <= 40% | `not_enough_evidence_share` > 0.4 |
| Cost per brief | Estimated LLM cost per run from `observability/cost.py::TokenCostTracker` (logged to MLflow), cross-checked with list-price spend in `system.billing.usage` divided by briefs generated | Every run completes within `RunBudget` (defaults: 400,000 tokens, 200 LLM calls, 25.0 estimated cost units); p95 cost per brief reviewed monthly against the contract rate | `serving_cost_usd_daily` > 150 USD/day list price |

Notes:

- The synchronous endpoint is designed for interactive single-company research.
  Batch research runs in `cra_brief_generation` (job timeout 5400 s, health rule
  at 2700 s) and is measured by job success rate, not by the endpoint SLOs.
- `NOT_ENOUGH_EVIDENCE` covers both missing evidence and evidence of low fit
  ([ADR-0008](../adr/0008-verdict-semantics.md)); a rise in this share is a
  signal to investigate ingestion, not necessarily a defect.

## 2. Monitoring

| Layer | Source | Where |
|---|---|---|
| Request logs | AI Gateway inference table `<catalog>.<schema>.cra_agent_payload` and FMAPI tables `fm_*_payload` | Dashboard datasets in `infrastructure/monitoring/dashboard_queries.sql` |
| Usage and cost | `system.serving.endpoint_usage`, `system.serving.served_entities`, `system.billing.usage`, `system.billing.list_prices` | `token_usage_by_endpoint_daily`, `serving_cost_usd_daily` |
| Brief quality | `briefs.brief_json`, `eval_results` | `citation_coverage_daily`, `verdict_distribution`, `eval_quality_trend` |
| Governance | `audit_log` | `audit_events_daily` |
| Drift | Lakehouse Monitoring on `cra_agent_payload` (time series on `request_time`, sliced by status and served entity) and `briefs` (sliced by `verdict`, tracks `weighted_score`) | Metric tables in `<schema>_monitoring` (`infrastructure/monitoring/lakehouse_monitor.py`) |
| Traces | MLflow Tracing (`ENABLE_MLFLOW_TRACING=true` on the endpoint; experiment `/Shared/client-research-agent-<env>`) | MLflow Trace UI |
| Metrics and traces (self-hosted / local) | OpenTelemetry via OTLP/HTTP (`observability.otlp_endpoint`); collector config `infrastructure/monitoring/otel-collector.yaml` | Jaeger, Prometheus scrape on the collector |
| Logs | structlog JSON with `run_id`, `company`, `step` context and secret scrubbing | Job and endpoint logs |

Key application metrics (emitted through `observability/metrics.py`):

| Metric | Meaning |
|---|---|
| `retrieval.latency_ms`, `retrieval.mean_relevance` | Retrieval pipeline latency and CRAG relevance |
| `retrieval.crag.corrections`, `retrieval.crag.verdict` | Corrective retrieval activity |
| `retrieval.hybrid.dense_fallback` | Dense leg failures (embedding endpoint or index) |
| `llm.fallback`, `llm.retries`, `llm.errors`, `llm.tokens.prompt`, `llm.tokens.completion`, `llm.cost.usd` | FMAPI health and consumption |
| `qualification_llm_fallback_total`, `qualification_invalid_citations_total` | Criterion scoring degradations and ungrounded citations |
| `citations_removed_total`, `citations_downgraded_total`, `brief_citation_coverage` | Grounding outcomes |
| `ingestion.sources_skipped`, `crawler.fetch_errors`, `crawler.fetch_latency_ms` | Crawl health |
| `briefs_generated_total` | Throughput |

## 3. Alerts

Provisioned by `infrastructure/monitoring/provision_alerts.py` from
`alerts.sql` and `alerts.json` (matched by display name, idempotent). Each alert
query returns one `value` row.

| Alert | Schedule | Threshold | Runbook |
|---|---|---|---|
| `agent_latency_p95` | every 10 min | > 60,000 ms | [serving](../runbooks/serving-endpoint-errors-and-latency.md) |
| `agent_error_rate` | every 10 min | > 0.02 | [serving](../runbooks/serving-endpoint-errors-and-latency.md) |
| `citation_coverage_daily` | 07:00 daily | < 0.90 | [coverage](../runbooks/citation-coverage-and-quality-gate.md) |
| `not_enough_evidence_share` | 07:00 daily | > 0.40 | [ingestion](../runbooks/ingestion-blocked-by-policy.md) |
| `guardrail_blocks_hourly` | every 15 min | > 20 | [prompt injection](../runbooks/prompt-injection-alert.md) |
| `serving_cost_usd_daily` | 08:00 daily | > 150 USD | Section 5 below |
| `eval_gate_failed` | 07:30 daily | > 0 | [coverage](../runbooks/citation-coverage-and-quality-gate.md) |

Job-level alerting (bundle): every job e-mails `alert_email` on failure and on
duration-threshold breach (`health.rules` on `RUN_DURATION_SECONDS`); in `prod`
every job also pages `oncall_notification_destination_id` on failure.

## 4. Capacity

| Component | Capacity lever | Current setting |
|---|---|---|
| Agent endpoint | `workload_size`, scale-to-zero, AI Gateway limits | prod Medium, no scale-to-zero, 1200/min endpoint, 60/min per user (`agent_endpoint.prod.json`) |
| FMAPI chat | Pay-per-token shared capacity; AI Gateway limits 3000/min endpoint, 300/min per user (prod) | Move to provisioned throughput on sustained 429s |
| FMAPI embeddings | 12000/min endpoint, 1200/min per user (prod); 150 inputs per request | Batch size 64 in indexing |
| Vector Search | `vector_search_endpoint_type` `STANDARD`; `TRIGGERED` sync | `STORAGE_OPTIMIZED` is available via the Terraform variable for large corpora |
| Batch research | `cra_brief_generation.max_concurrent_runs` 10 with queueing | Lower during FMAPI incidents |
| Ingestion | `cra_ingestion_refresh.max_concurrent_runs` 1, daily 03:00 UTC, 4 h timeout | Per-host rate `crawler.requests_per_second_per_host` 1.0 bounds crawl speed by design |
| SQL warehouse | `warehouse_size`, `warehouse_max_clusters`, auto-stop | prod X-Small, max 2 clusters, 15 min |

Rules of thumb:

- The crawl is intentionally slow per host; throughput comes from parallelism
  across hosts (`IngestionPipeline` thread pool) and from pre-ingesting the
  watchlist overnight, not from raising per-host rates.
- Per-request BM25 and entity graphs are built in memory from one company's
  chunks; memory scales with chunks per company, not with the total corpus.

## 5. Cost controls

| Control | Where |
|---|---|
| Per-run hard caps on tokens, calls and estimated cost (`BudgetExceededError`) | `security/rate_limiter.py::RunBudget` |
| Per-requester request rate | AI Gateway per-user limits; `PrincipalRateLimiter` in `ClientResearchResponsesAgent` keyed on the asserted requester |
| Cost estimation per step and endpoint (DBUs and USD at `usd_per_dbu`, default 0.07) | `observability/cost.py::TokenCostTracker`, `PricingTable`; override the default table with contract rates |
| Embeddings computed once and persisted | ADR-0003 |
| Embedding cache | `retrieval/embeddings.py::CachingEmbeddingClient` |
| Scale-to-zero outside prod | `agent_endpoint.dev.json`, `agent_endpoint.staging.json`, tfvars |
| Serverless jobs and warehouse auto-stop | Bundle `environments`, `warehouse_auto_stop_minutes` |
| Monthly budget alert on `app = client-research-agent` | `deployment/terraform/budget.tf` (dev 300, staging 500, prod 2000 USD) |
| Daily spend alert | `serving_cost_usd_daily` |
| Chargeback tags | `app`, `environment`, `cost_center` on endpoints, warehouse, cluster policy |

## 6. Data retention

| Data | Classification tag | Retention guidance |
|---|---|---|
| `documents`, `chunks`, `parent_chunks` | `public` | Refreshed daily for watchlist companies. Delete a company's data with `DeltaDocumentStore.delete_company` (documents and chunks) and let CDF propagate the deletion to the index. |
| `briefs`, `lineage`, `brief_exports` volume | `internal` | Keep as long as the brief is in commercial use; `lineage` is needed to withdraw briefs after a grounding incident. |
| `audit_log` | `confidential`, append-only | `governance/data_classification.py::uc_tags` proposes `data_retention` 3y (7y for RESTRICTED data). Deleting expired rows requires lifting `delta.appendOnly` under change control ([ADR-0010](../adr/0010-hash-chained-audit-log.md)). |
| AI Gateway inference tables | Contain request and response payloads | Apply the same retention as `briefs`; they can contain requester identity. |
| MLflow traces and evaluation runs | Internal | Retain evaluation runs for promoted versions at least as long as the version is `champion` or `previous_champion`. |
| CI artifacts | n/a | Coverage and JUnit 14 days; `dist` 30 days (`ci.yml`) |

No retention job exists in the repository today; Delta `VACUUM` and table-level
retention properties are platform defaults. See the production readiness
checklist in the [README](../../README.md#production-readiness-checklist).

## 7. Access reviews

Quarterly, per environment:

1. Group membership of `cra-engineers`, `cra-analysts` (and, where used,
   `cra-viewers`, `cra-operators`, `cra-admins`, `cra-service-principals`, which
   map to roles in `security/rbac.py::DEFAULT_GROUP_ROLE_MAPPING`).
2. UC grants: compare `SHOW GRANTS ON SCHEMA <catalog>.agent_<env>` with
   `deployment/terraform/grants.tf` and `infrastructure/unity_catalog/05_grants.sql`.
   Analysts must hold `SELECT` on `briefs` only.
3. Endpoint permissions on `cra-agent-<env>`: service principal `CAN_MANAGE`,
   engineers `CAN_VIEW`, analysts `CAN_QUERY`.
4. Secret scope ACLs on `client-research-agent`.
5. Federation policy subjects: only `repo:kogunlowo123/client-research-qualification-agent:environment:<env>`.
6. GitHub `prod` environment reviewers and CODEOWNERS.
7. Confirm no PAT is configured for staging or prod principals.

## 8. On-call

- Rotation owns prod SEV1 and SEV2 (see [runbooks/README.md](../runbooks/README.md)).
- Pages arrive through the workspace notification destination configured as
  `oncall_notification_destination_id` (prod job failures) and the SQL alert
  destinations provisioned by `provision_alerts.py`.
- First actions: check the dashboard datasets for the last two hours, identify
  whether a deployment happened (`cra_agent_deploy` run history, `champion`
  alias), and roll back if so.
- Handover notes must include open incidents, pending promotions (candidate
  versions awaiting `evaluate_candidate`), and any temporary job concurrency
  changes.

## 9. Routine operations

| Task | Command |
|---|---|
| Research one company in a workspace | `databricks bundle run -t <env> cra_brief_generation --params company="Example Corp",domain=example.com,ticker=EXMP` |
| Refresh the watchlist now | `databricks bundle run -t <env> cra_ingestion_refresh` |
| Run the evaluation gate | `databricks bundle run -t <env> cra_evaluation` |
| Promote a new version | `databricks bundle run -t <env> cra_agent_deploy --params action=log_and_deploy` |
| Redeploy the current champion | `databricks bundle run -t <env> cra_agent_deploy --params action=deploy_champion` |
| Roll back | `deployment/workflows/rollback.sh <env>` |
| Verify audit chain (JSONL) | `python -c "from client_research_agent.governance.audit import verify_chain; print(verify_chain('<path>'))"` |
