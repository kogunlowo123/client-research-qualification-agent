# Runbook: Foundation Model API rate limiting and circuit breaker open

Endpoints: `serving.chat_endpoint` (`databricks-claude-sonnet-4`),
`serving.fallback_chat_endpoint` (`databricks-meta-llama-3-3-70b-instruct`),
`serving.judge_endpoint`, `serving.embedding_endpoint` (`databricks-gte-large-en`).
AI Gateway configuration: `deployment/serving/foundation_model_ai_gateway.<env>.json`.

## Background

Every FMAPI call runs under a per-endpoint `CircuitBreaker`
(`databricks/model_serving.py::endpoint_breaker`) and a retry policy built from
`ResilienceSettings`:

| Key | Default | local | prod |
|---|---|---|---|
| `resilience.max_attempts` | 4 | 2 | 5 |
| `resilience.initial_backoff_seconds` | 0.5 | 0.1 | 0.5 |
| `resilience.max_backoff_seconds` | 20.0 | 20.0 | 20.0 |
| `resilience.breaker_failure_threshold` | 5 | 5 | 5 |
| `resilience.breaker_reset_seconds` | 30.0 | 30.0 | 60 |

`429` maps to `RateLimitedError` (honouring `Retry-After`), `5xx` to
`UpstreamServiceError`, timeouts to `UpstreamTimeoutError`; all are
`TransientError`, are retried, and count against the breaker. When the breaker is
open, calls fail fast with `CircuitOpenError` until `breaker_reset_seconds`
elapse, then one half-open probe is allowed.

`FallbackLLMClient` routes to the fallback endpoint on `TransientError` or
`CircuitOpenError`. If both endpoints fail, each LLM step uses its deterministic
path ([ADR-0005](../adr/0005-deterministic-fallback-for-every-llm-step.md)), and
the brief records a warning per degraded step.

## Symptoms

- Metrics: `llm.fallback` (attributes `primary`, `fallback`, `reason`),
  `llm.retries`, `llm.errors`, and step fallbacks
  `qualification_llm_fallback_total`, `opportunity_llm_fallback_total`,
  `brief_writer_fallback_total`, `discovery_questions_fallback_total`,
  `citation_judge_fallback_total`, `retrieval.rerank.llm_fallback`,
  `retrieval.crag.grader_fallback`.
- Log events `llm.fallback` and `qualification.llm_fallback` with
  `error=CircuitOpenError` or `RateLimitedError`.
- Briefs with warnings such as "LLM assessment failed (CircuitOpenError); used
  deterministic heuristic score".
- `briefs.brief_json` `model_versions` naming the fallback model.
- Latency increase (retries with backoff) before the breaker opens.

## Dashboards and queries

- `token_usage_by_endpoint_daily`: sudden token growth indicates a loop or a
  traffic change; flat tokens with errors indicate provider-side throttling.
- FMAPI inference tables written by AI Gateway (`fm_claude_sonnet_4_*`,
  `fm_llama_3_3_70b_*`, `fm_gte_large_en_*` in the environment schema):

```sql
SELECT date_trunc('MINUTE', request_time) AS minute, status_code, count(*) AS n
FROM `${catalog}`.`${schema}`.`fm_claude_sonnet_4_payload`
WHERE request_time >= current_timestamp() - INTERVAL 2 HOURS
GROUP BY 1, 2 ORDER BY 1 DESC;
```

- Share of briefs produced with degraded steps:

```sql
SELECT CAST(generated_at AS DATE) AS day,
       count_if(brief_json LIKE '%used deterministic%') / count(*) AS degraded_share
FROM `${catalog}`.`${schema}`.`briefs`
WHERE generated_at >= current_timestamp() - INTERVAL 7 DAYS
GROUP BY 1 ORDER BY 1;
```

## Diagnosis

| Observation | Cause |
|---|---|
| 429 with `key = user` exhaustion in AI Gateway | Per-principal limit on the FMAPI endpoint (prod 300/min per user) reached by the agent service principal |
| 429 with endpoint-level exhaustion | Endpoint limit (prod 3000/min) or shared pay-per-token capacity |
| 5xx or timeouts without 429 | Provider incident |
| Token volume multiplied for one run | Reasoning loop or unusually large evidence; `RunBudget` should cap it |
| Only embeddings failing | Dense retrieval degrades to BM25 (`retrieval.hybrid.dense_fallback`); ingestion embedding fails and the run cannot index new children |

## Mitigation

1. Confirm the fallback endpoint is healthy; if it is, service continues at
   reduced quality and no immediate action is required beyond monitoring.
2. Reduce concurrency: lower `max_concurrent_runs` on `cra_brief_generation`
   (default 10) for the duration, or pause the `cra_ingestion_refresh` schedule
   if embedding capacity is the constraint.
3. If AI Gateway limits are the constraint and the traffic is legitimate, raise
   them in `foundation_model_ai_gateway.<env>.json` through a reviewed change and
   re-apply with `deployment/workflows/deploy.sh <env> --apply-fm-gateway`.
4. For sustained pay-per-token throttling, move the chat endpoint to provisioned
   throughput and point `CRA_SERVING__CHAT_ENDPOINT` at it (configuration only;
   no code change).
5. Do not lower `resilience.breaker_failure_threshold` or raise `max_attempts` in
   an incident: more retries amplify load on a throttled endpoint.

## Escalation

- SEV2 when both chat endpoints are unavailable for more than 30 minutes in prod
  (briefs are deterministic-only). Open a Databricks support case with endpoint
  names, time window and request ids from the inference tables.
- SEV1 if citation coverage also drops below the gate (see
  [citation coverage runbook](citation-coverage-and-quality-gate.md)).
