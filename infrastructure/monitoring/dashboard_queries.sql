-- Databricks SQL dashboard datasets for the Client Research Agent.
-- ${catalog} and ${schema} are rendered by provision_alerts.py (or replace them
-- when pasting into the SQL editor). Sources:
--   ${catalog}.${schema}.cra_agent_payload  AI Gateway inference table of cra-agent-<env>
--   ${catalog}.${schema}.briefs / audit_log / eval_results
--   system.serving.endpoint_usage, system.serving.served_entities,
--   system.billing.usage, system.billing.list_prices

-- query: latency_percentiles_hourly
SELECT
  date_trunc('HOUR', request_time)                         AS hour,
  count(*)                                                 AS requests,
  percentile_approx(execution_duration_ms, 0.50)           AS p50_ms,
  percentile_approx(execution_duration_ms, 0.95)           AS p95_ms,
  percentile_approx(execution_duration_ms, 0.99)           AS p99_ms
FROM `${catalog}`.`${schema}`.`cra_agent_payload`
WHERE request_time >= current_timestamp() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 1;

-- query: error_rate_hourly
SELECT
  date_trunc('HOUR', request_time)                                   AS hour,
  count(*)                                                           AS requests,
  count_if(status_code >= 500)                                       AS server_errors,
  count_if(status_code = 429)                                        AS rate_limited,
  count_if(status_code BETWEEN 400 AND 499 AND status_code <> 429)   AS client_errors,
  round(count_if(status_code >= 500) / nullif(count(*), 0), 4)       AS server_error_rate
FROM `${catalog}`.`${schema}`.`cra_agent_payload`
WHERE request_time >= current_timestamp() - INTERVAL 7 DAYS
GROUP BY 1
ORDER BY 1;

-- query: citation_coverage_daily
WITH b AS (
  SELECT
    generated_at,
    CAST(get_json_object(brief_json, '$.citation_report.supported_statements') AS DOUBLE)
      / nullif(CAST(get_json_object(brief_json, '$.citation_report.total_statements') AS DOUBLE), 0)
      AS citation_coverage
  FROM `${catalog}`.`${schema}`.`briefs`
)
SELECT
  CAST(generated_at AS DATE)                       AS day,
  count(*)                                         AS briefs,
  round(avg(citation_coverage), 4)                 AS mean_citation_coverage,
  round(percentile_approx(citation_coverage, 0.1), 4) AS p10_citation_coverage,
  count_if(citation_coverage < 0.9)                AS briefs_below_gate
FROM b
WHERE generated_at >= current_timestamp() - INTERVAL 30 DAYS
GROUP BY 1
ORDER BY 1;

-- query: verdict_distribution
SELECT
  CAST(generated_at AS DATE)             AS day,
  verdict,
  count(*)                               AS briefs,
  round(avg(weighted_score), 3)          AS mean_weighted_score,
  round(avg(CAST(get_json_object(brief_json, '$.qualification.overall_confidence') AS DOUBLE)), 3) AS mean_confidence
FROM `${catalog}`.`${schema}`.`briefs`
WHERE generated_at >= current_timestamp() - INTERVAL 30 DAYS
GROUP BY 1, 2
ORDER BY 1, 2;

-- query: token_usage_by_endpoint_daily
SELECT
  CAST(u.request_time AS DATE)          AS day,
  e.endpoint_name,
  count(*)                              AS requests,
  sum(u.input_token_count)              AS input_tokens,
  sum(u.output_token_count)             AS output_tokens
FROM system.serving.endpoint_usage AS u
JOIN system.serving.served_entities AS e
  ON u.served_entity_id = e.served_entity_id
WHERE u.request_time >= current_timestamp() - INTERVAL 30 DAYS
  AND e.endpoint_name IN (
    'cra-agent-${environment}',
    'databricks-claude-sonnet-4',
    'databricks-meta-llama-3-3-70b-instruct',
    'databricks-gte-large-en')
GROUP BY 1, 2
ORDER BY 1, 2;

-- query: serving_cost_usd_daily
SELECT
  u.usage_date                                        AS day,
  u.usage_metadata.endpoint_name                      AS endpoint_name,
  sum(u.usage_quantity)                               AS dbus,
  round(sum(u.usage_quantity * p.pricing.effective_list.default), 2) AS list_cost_usd
FROM system.billing.usage AS u
JOIN system.billing.list_prices AS p
  ON u.sku_name = p.sku_name
 AND u.cloud = p.cloud
 AND u.usage_unit = p.usage_unit
 AND u.usage_end_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_end_time < p.price_end_time)
WHERE u.usage_date >= current_date() - INTERVAL 30 DAYS
  AND u.billing_origin_product IN ('MODEL_SERVING', 'VECTOR_SEARCH', 'JOBS', 'SQL')
  AND (
    u.custom_tags['app'] = 'client-research-agent'
    OR u.usage_metadata.endpoint_name IN (
      'cra-agent-${environment}',
      'databricks-claude-sonnet-4',
      'databricks-meta-llama-3-3-70b-instruct',
      'databricks-gte-large-en')
  )
GROUP BY 1, 2
ORDER BY 1, 2;

-- query: audit_events_daily
SELECT
  CAST(recorded_at AS DATE)                              AS day,
  event_type,
  coalesce(get_json_object(payload_json, '$.decision'), 'n/a') AS decision,
  count(*)                                               AS events
FROM `${catalog}`.`${schema}`.`audit_log`
WHERE recorded_at >= current_timestamp() - INTERVAL 30 DAYS
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;

-- query: eval_quality_trend
SELECT
  evaluated_at,
  model_version,
  metric_name,
  metric_value,
  threshold,
  passed,
  gate_passed
FROM `${catalog}`.`${schema}`.`eval_results`
WHERE evaluated_at >= current_timestamp() - INTERVAL 90 DAYS
ORDER BY evaluated_at, metric_name;
