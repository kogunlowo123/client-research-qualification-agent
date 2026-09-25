# ADR-0011: Serve the agent as an MLflow ResponsesAgent deployed with `databricks.agents.deploy`

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

The agent must be callable by humans (Databricks Playground, Review App) and by
systems (CRM integrations) with authentication, rate limiting, request logging,
tracing and a rollback path. Options considered:

1. A custom container on an external platform (FastAPI behind a gateway).
2. A generic MLflow `pyfunc` on Model Serving.
3. An MLflow `ResponsesAgent` (OpenAI Responses-compatible schema) registered in
   the Unity Catalog model registry and deployed with Mosaic AI Agent Framework
   `databricks.agents.deploy()`.

## Decision

Option 3.

- The agent is logged as an MLflow 3 `ResponsesAgent`, registered in UC as
  `<catalog>.<schema>.client_research_agent` (bundle resource
  `registered_models.cra_agent`), and deployed to the endpoint named by the bundle
  variable `agent_endpoint_name` (`cra-agent-dev`, `cra-agent-staging`,
  `cra-agent-prod`).
- Versions move through UC aliases, not stages (`databricks/mlflow_registry.py`):
  `challenger` (candidate under evaluation), `champion` (served), and
  `previous_champion` / `rolled_back` maintained by the deploy job and
  `deployment/workflows/rollback.sh`.
- The `cra_agent_deploy` job runs `log_and_register` -> `evaluate_candidate`
  (quality gate, `--fail-on-gate`) -> `promote_and_deploy`; with
  `action=deploy_champion` it redeploys whatever `champion` points to.
- Endpoint configuration per environment (workload size, scale-to-zero,
  environment variables, AI Gateway usage tracking, inference table
  `cra_agent_*`, endpoint and per-user rate limits) is versioned in
  `deployment/serving/agent_endpoint.<env>.json`.
- Request contract: Responses-style `input` messages plus `custom_inputs`
  carrying `company_name`, `ticker`, `domain`, `max_documents`, `requested_by`
  (see `deployment/workflows/smoke_test.sh`). The brief is returned as
  `output_text` in a `message` output item.
- Tracing: `ENABLE_MLFLOW_TRACING=true` on the endpoint; spans are emitted by
  `observability/tracing.py` with MLflow span types (`AGENT`, `RETRIEVER`, `LLM`,
  `PARSER`, ...).

## Consequences

- Positive: authentication, per-user rate limits, usage tracking, inference
  tables, Review App and Playground come from the platform; no custom gateway.
- Positive: rollback is an alias move plus an endpoint config update, scripted
  and exercised automatically by CD when the prod smoke test fails.
- Positive: the served model is immutable and traceable to a UC model version,
  its MLflow run, and prompt fingerprints recorded in `model_versions`.
- Negative: `databricks.agents.deploy()` owns the endpoint; Terraform manages it
  only when `manage_serving_endpoint = true`, and the two must not both be used
  for one environment.
- Negative: a research run is long (tens of seconds to minutes, bounded by the
  smoke test's `SMOKE_MAX_LATENCY_SECONDS`, default 180). Synchronous serving
  suits interactive use; bulk research belongs in the `cra_brief_generation`
  job, not the endpoint.
