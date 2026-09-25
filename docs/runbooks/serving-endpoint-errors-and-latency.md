# Runbook: Serving endpoint 5xx or latency

Endpoint: `cra-agent-<env>` (bundle variable `agent_endpoint_name`; config in
`deployment/serving/agent_endpoint.<env>.json`).

## Symptoms

- Alert `agent_error_rate`: more than 2% of requests returned 5xx in the last hour.
- Alert `agent_latency_p95`: p95 `execution_duration_ms` above 60,000 ms in the last hour.
- `deployment/workflows/smoke_test.sh <env>` fails ("endpoint not ready",
  "unexpected HTTP status", "latency ... exceeded"). In prod CD this triggers an
  automatic `rollback.sh prod --skip-smoke`.
- Callers report timeouts from the Review App, Playground or integrations.

## Dashboards and queries

- `latency_percentiles_hourly` and `error_rate_hourly`
  (`infrastructure/monitoring/dashboard_queries.sql`) over
  `${catalog}.${schema}.cra_agent_payload`.
- `token_usage_by_endpoint_daily` to see whether FMAPI traffic changed at the same time.
- Endpoint state:

```bash
databricks serving-endpoints get cra-agent-prod --output json \
  | jq '{ready: .state.ready, config_update: .state.config_update,
         served: [.config.served_entities[] | {entity_name, entity_version, workload_size}]}'
databricks serving-endpoints logs cra-agent-prod client_research_agent
```

- Break down errors by status and served version:

```sql
SELECT date_trunc('MINUTE', request_time) AS minute, status_code,
       served_entity_id, count(*) AS n,
       percentile_approx(execution_duration_ms, 0.95) AS p95_ms
FROM `${catalog}`.`${schema}`.`cra_agent_payload`
WHERE request_time >= current_timestamp() - INTERVAL 2 HOURS
GROUP BY 1, 2, 3
ORDER BY 1 DESC;
```

- MLflow traces for slow requests in the experiment `/Shared/client-research-agent-<env>`:
  sort by duration; compare span times of `retrieval.pipeline`,
  `qualification.qualify`, `briefing.generate` and the `LLM` spans.

## Diagnosis

| Observation | Likely cause | Next step |
|---|---|---|
| `config_update = IN_PROGRESS` or `UPDATE_FAILED` | A deploy is rolling out or failed | Check the latest `cra_agent_deploy` run; wait or roll back |
| 5xx started exactly at a new `entity_version` | Regression in the new model version | [Rollback](rollback.md) |
| 429 from the agent endpoint | AI Gateway rate limit (prod 1200/min endpoint, 60/min per user) | Identify the caller in `cra_agent_payload`; throttle the integration or raise the limit through a reviewed change to `agent_endpoint.<env>.json` / Terraform vars |
| Latency concentrated in `LLM` spans, `llm.retries` rising | FMAPI slowness or throttling | [FM API rate limiting runbook](fm-api-rate-limiting-and-circuit-breaker.md) |
| Latency concentrated in crawler spans | Slow or rate-limited public sites, EDGAR pacing | Expected for cold companies; prefer pre-ingested companies via `cra_ingestion_refresh`; lower `max_documents` in the request |
| Container logs show import or auth errors at start | Missing dependency or broken environment variables | Roll back; fix in the model's logged requirements |
| Errors only after scale from zero (dev/staging `scale_to_zero_enabled = true`) | Cold start | Expected outside prod; prod has scale-to-zero disabled |

## Mitigation

1. If a new version is implicated: [roll back](rollback.md)
   (`deployment/workflows/rollback.sh <env>`).
2. If capacity-bound: increase `workload_size` in
   `deployment/serving/agent_endpoint.<env>.json` (or `serving_workload_size` in
   the Terraform tfvars when Terraform owns the endpoint) through a reviewed
   change, then redeploy with `databricks bundle run -t <env> cra_agent_deploy --params action=deploy_champion`.
3. If a single caller is abusive: reduce that caller's traffic at the source;
   AI Gateway per-user limits already cap it.
4. For bulk research traffic hitting the synchronous endpoint: move it to the
   `cra_brief_generation` job.

## Escalation

- SEV1 if prod error rate stays above 2% for 30 minutes after rollback, or the
  endpoint is not `READY`: escalate to the platform owner (CODEOWNERS
  `@kogunlowo123`) and open a Databricks support case with the endpoint name,
  served entity version and request ids from `cra_agent_payload`.
