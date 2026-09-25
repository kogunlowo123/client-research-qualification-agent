# Deployment Diagram

Where each artefact runs and how it gets there. Details:
[deployment_guide.md](../operations/deployment_guide.md).

```text
 Developer workstation                     GitHub
 +---------------------------+             +--------------------------------------------------+
 | uv / make / pre-commit    |  PR, tag    | ci.yml: lint, mypy, unit (3.11, 3.12, cov>=90),  |
 | cra CLI (local adapters)  |-----------> | marked suites, bandit, pip-audit, Trivy,         |
 | docker compose: MLflow,   |             | gitleaks, build wheel/sdist/image                |
 | OTel Collector, Jaeger    |             | codeql.yml: python, actions                      |
 | databricks auth login     |             | cd.yml: main -> staging; vX.Y.Z tag -> prod      |
 +---------------------------+             |   (environment approval), OIDC id-token          |
             |                             | Release: wheel/sdist, GHCR image (SBOM, provenance)
             | terraform apply             +------------------------+-------------------------+
             | (per env, prod first)                                 | OIDC token exchange
             v                                                       v (federation policy, no secrets)
 +------------------------------------------------------------------------------------------------+
 | Databricks account                                                                             |
 |   service principal cra-agent-<env> + federation policy github-<env>; budget                  |
 |                                                                                                |
 |  Databricks workspace (one per environment or shared; schema agent_<env> isolates data)       |
 |  +------------------------------------------------------------------------------------------+ |
 |  | Asset Bundle target <env>  (databricks bundle deploy; deploy.sh)                          | |
 |  |   wheel dist/*.whl  ->  serverless jobs (environment_version 3)                           | |
 |  |     cra_ingestion_refresh   cra_brief_generation   cra_evaluation   cra_agent_deploy      | |
 |  |   MLflow experiment, UC registered model, monitoring schema                               | |
 |  |                                                                                           | |
 |  | Model Serving                                                                             | |
 |  |   cra-agent-<env>  <- databricks.agents.deploy() from cra_agent_deploy (champion alias)   | |
 |  |   FMAPI: databricks-claude-sonnet-4 | databricks-meta-llama-3-3-70b-instruct |            | |
 |  |          databricks-gte-large-en            (AI Gateway via deploy.sh --apply-fm-gateway) | |
 |  |                                                                                           | |
 |  | Vector Search endpoint cra-vs-endpoint-<env> / index chunks_index      (Terraform)        | |
 |  | SQL warehouse cra-sql-<env>                                            (Terraform)        | |
 |  | Unity Catalog client_research.agent_<env>.*   tables via apply_ddl.py  (Terraform + DDL)  | |
 |  | Secret scope client-research-agent                                     (Terraform)        | |
 |  +------------------------------------------------------------------------------------------+ |
 +------------------------------------------------------------------------------------------------+
             ^                                      |
             | HTTPS (OAuth, CAN_QUERY)             | HTTPS egress (UrlGuard, robots.txt, rate limit)
             |                                      v
 +---------------------------+          +-----------------------------------------------+
 | Callers                   |          | Public internet                               |
 | Review App / Playground   |          | www.sec.gov, data.sec.gov                     |
 | CRM integrations          |          | corporate domains in the request scope        |
 | analysts (SQL on briefs)  |          | analyst public pages supplied as seed URLs    |
 +---------------------------+          +-----------------------------------------------+
```

Container image (Dockerfile): multi-stage build on `python:3.12-slim`, runs as
UID 10001, entrypoint `cra`, health check `cra serve-check`. It is published to
GHCR on release and used for local runs and self-hosted execution; the
Databricks deployment uses the wheel, not the image.
