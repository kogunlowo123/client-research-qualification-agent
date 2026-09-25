# Runbook: Citation coverage regression or quality gate failure

Metric: `CitationReport.coverage = supported_statements / total_statements`
(share of claimed facts that survived citation validation). Gates:
`eval_min_citation_coverage` (0.90) and `eval_min_pass_rate` (0.85) in
`databricks.yml`.

## Symptoms

- Alert `citation_coverage_daily`: mean coverage of briefs in the last 24 h below 0.90.
- Alert `eval_gate_failed`: an evaluation run in the last 24 h has `gate_passed = false`.
- `cra_brief_generation` task `evaluate` fails (`--fail-on-gate`).
- `cra_evaluation` fails (nightly, 06:00 UTC).
- `cra_agent_deploy` stops at `evaluate_candidate`; the candidate is not promoted
  (this is the gate working, not an outage).
- Metrics: `citations_removed_total`, `citations_downgraded_total` rising;
  `brief_citation_coverage` histogram shifting down.

## Dashboards and queries

- `citation_coverage_daily` (mean, p10, briefs below 0.9 per day).
- `eval_quality_trend` (per-metric values, thresholds and gate outcome by model version).

```sql
SELECT model_version, metric_name, avg(metric_value) AS value, min(threshold) AS threshold,
       count_if(NOT passed) AS failures
FROM `${catalog}`.`${schema}`.`eval_results`
WHERE evaluated_at >= current_timestamp() - INTERVAL 14 DAYS
GROUP BY 1, 2 ORDER BY 1, 2;
```

Removed statements of recent low-coverage briefs:

```sql
SELECT run_id, company,
       get_json_object(brief_json, '$.citation_report.removed_statements') AS removed,
       get_json_object(brief_json, '$.model_versions') AS versions
FROM `${catalog}`.`${schema}`.`briefs`
WHERE generated_at >= current_timestamp() - INTERVAL 1 DAY
ORDER BY CAST(get_json_object(brief_json, '$.citation_report.supported_statements') AS DOUBLE)
       / nullif(CAST(get_json_object(brief_json, '$.citation_report.total_statements') AS DOUBLE), 0)
LIMIT 20;
```

## Diagnosis

| Observation | Likely cause |
|---|---|
| Drop coincides with a new `model_versions.llm` or prompt fingerprint | Model or prompt change produces facts that paraphrase too far from the evidence, or invent figures |
| Drop coincides with the fallback model in `model_versions` | Primary endpoint degraded; see [FM API runbook](fm-api-rate-limiting-and-circuit-breaker.md) |
| Removed statements contain numbers | Fabricated or transformed figures (number rule caps support at 0.05); this is the most damaging hallucination and must be treated as a quality incident |
| Removed statements are correct paraphrases | Validator too strict for the new style; do not lower `guardrails.min_citation_support` without an evaluation run |
| Coverage fine, pass rate low | A judge-based metric regressed (for example correctness against `expected_facts`, or guideline adherence); inspect per-metric rows |
| Only some companies affected | Evidence quality (thin or stale sources); check ingestion |
| `eval_set` changed recently | Dataset change, not model change: compare `dataset_version` in `eval_results` |

## Mitigation

1. If a new champion caused it: [roll back](rollback.md).
2. If a candidate failed the gate: no action on prod. Investigate in the MLflow
   evaluation run (traces per row) before re-running `cra_agent_deploy`.
3. If the fallback model is the cause: restore the primary endpoint; briefs
   produced in the window can be regenerated through `cra_brief_generation`.
4. Never lower `eval_min_citation_coverage`, `eval_min_pass_rate` or
   `guardrails.min_citation_support` to make a gate pass. Threshold changes go
   through a pull request with an evaluation comparison.

## Escalation

- SEV1 if briefs with unsupported verified facts reached users (coverage below
  the gate in prod and briefs exported from `brief_exports`). Notify the business
  owner so affected briefs can be withdrawn; `lineage` identifies the runs and
  statements.
- SEV2 for a failing nightly evaluation with prod coverage still above the gate.
