-- Databricks SQL alert queries. Each query returns exactly one row with a
-- `value` column evaluated by the thresholds in alerts.json.
-- ${catalog}, ${schema} and ${environment} are rendered by provision_alerts.py.
-- briefs/audit_log columns follow the adapter schema; brief metrics not stored
-- as columns are read from brief_json (serialised ClientBrief).

-- alert: agent_latency_p95
SELECT coalesce(percentile_approx(execution_duration_ms, 0.95), 0) AS value
FROM `${catalog}`.`${schema}`.`cra_agent_payload`
WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR;

-- alert: agent_error_rate
SELECT coalesce(count_if(status_code >= 500) / nullif(count(*), 0), 0) AS value
FROM `${catalog}`.`${schema}`.`cra_agent_payload`
WHERE request_time >= current_timestamp() - INTERVAL 1 HOUR;

-- alert: citation_coverage_daily
SELECT coalesce(avg(CAST(get_json_object(brief_json, '$.citation_report.supported_statements') AS DOUBLE)
  / nullif(CAST(get_json_object(brief_json, '$.citation_report.total_statements') AS DOUBLE), 0)), 1.0) AS value
FROM `${catalog}`.`${schema}`.`briefs`
WHERE generated_at >= current_timestamp() - INTERVAL 24 HOURS;

-- alert: not_enough_evidence_share
SELECT coalesce(count_if(verdict = 'not_enough_evidence') / nullif(count(*), 0), 0) AS value
FROM `${catalog}`.`${schema}`.`briefs`
WHERE generated_at >= current_timestamp() - INTERVAL 7 DAYS;

-- alert: guardrail_blocks_hourly
SELECT count(*) AS value
FROM `${catalog}`.`${schema}`.`audit_log`
WHERE recorded_at >= current_timestamp() - INTERVAL 1 HOUR
  AND (
    (event_type = 'authorization' AND get_json_object(payload_json, '$.decision') = 'deny')
    OR event_type LIKE 'guardrail%'
  );

-- alert: serving_cost_usd_daily
SELECT coalesce(sum(u.usage_quantity * p.pricing.effective_list.default), 0) AS value
FROM system.billing.usage AS u
JOIN system.billing.list_prices AS p
  ON u.sku_name = p.sku_name
 AND u.cloud = p.cloud
 AND u.usage_unit = p.usage_unit
 AND u.usage_end_time >= p.price_start_time
 AND (p.price_end_time IS NULL OR u.usage_end_time < p.price_end_time)
WHERE u.usage_date = current_date() - INTERVAL 1 DAY
  AND (
    u.custom_tags['app'] = 'client-research-agent'
    OR u.usage_metadata.endpoint_name = 'cra-agent-${environment}'
  );

-- alert: eval_gate_failed
SELECT count(*) AS value
FROM `${catalog}`.`${schema}`.`eval_results`
WHERE evaluated_at >= current_timestamp() - INTERVAL 24 HOURS
  AND gate_passed = false;
