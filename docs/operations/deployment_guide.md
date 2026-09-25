# Deployment Guide

This guide takes an empty Databricks workspace to a running, gated deployment of
the agent in `dev`, `staging` and `prod`, and describes promotion and rollback.

Ownership split (see `databricks.yml` header):

| Owner | Resources |
|---|---|
| Terraform (`deployment/terraform`) | Catalog (prod state only), per-environment schema `agent_<env>`, volumes `raw_documents` and `brief_exports`, UC grants, service principal and GitHub OIDC federation policy, secret scope and ACLs, Vector Search endpoint and `chunks_index`, serverless SQL warehouse `cra-sql-<env>`, cluster policy `cra-jobs-<env>`, budget, and (optionally) the agent serving endpoint |
| UC DDL (`infrastructure/unity_catalog`) | Tables, CHECK constraints, governance functions (`mask_email`, `audit_log_row_filter`), row filters, column masks, tags, table-level grants |
| Asset Bundle (`databricks.yml`, `deployment/databricks/resources`) | Wheel artifact `cra_wheel`, jobs `cra_ingestion_refresh`, `cra_brief_generation`, `cra_evaluation`, `cra_agent_deploy`, experiment `cra_experiment`, registered model `cra_agent`, monitoring schema `<schema>_monitoring` |
| `databricks.agents.deploy()` in `cra_agent_deploy` | Agent serving endpoint `cra-agent-<env>` (unless `manage_serving_endpoint = true` in Terraform) |
| `deploy.sh --apply-fm-gateway` | AI Gateway configuration of the FMAPI endpoints |

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Databricks workspace with Unity Catalog, serverless jobs, Model Serving, Vector Search, Mosaic AI Agent Framework | One workspace per environment or one shared workspace; the schema per environment isolates data |
| Account admin (first apply only) | Needed for `databricks_service_principal_federation_policy` and `databricks_budget` (account-level provider `databricks.account`) |
| Groups `cra-engineers`, `cra-analysts` (SCIM-synced) | Terraform reads them with `data "databricks_group"`; names are variables (`engineers_group`, `analysts_group`) |
| Terraform `>= 1.6.0, < 2.0.0`, provider `databricks/databricks ~> 1.134` | `deployment/terraform/versions.tf` |
| Remote state backend | `backend "azurerm" {}` with partial configuration (`deployment/terraform/backend.tf`); on AWS replace with `backend "s3" {}` |
| Databricks CLI `>= 0.250.0` | `bundle.databricks_cli_version`; CD pins `DATABRICKS_CLI_VERSION` 1.18.0 |
| `uv`, `jq`, `curl` | Used by `deploy.sh`, `rollback.sh`, `smoke_test.sh` |
| Python 3.11 or 3.12 with the `databricks` extra | For `infrastructure/**/*.py` bootstrap scripts |

## 2. Terraform apply order

One state per environment. **Prod is applied first** because the prod state owns
the shared catalog (`manage_catalog = true` in `envs/prod.tfvars`); dev and
staging read it with `data "databricks_catalog"`.

Some resources are gated behind variables because they depend on tables created
by the DDL step:

| Variable | Default | Gates |
|---|---|---|
| `create_vector_index` | `false` | `databricks_vector_search_index.chunks` (needs the `chunks` table with CDF) |
| `tables_provisioned` | `false` | Table-level grants such as `databricks_grant.briefs_analysts` |
| `manage_serving_endpoint` | `false` | `databricks_model_serving.agent` and its permissions (needs a registered model version) |
| `databricks_account_id` | `""` | Federation policy (`federation_enabled`) and, with `workspace_id` and `budget_alert_emails`, the budget |

For each environment, in the order `prod`, `staging`, `dev`:

```bash
cd deployment/terraform
terraform init \
  -backend-config="resource_group_name=$TFSTATE_RESOURCE_GROUP" \
  -backend-config="storage_account_name=$TFSTATE_STORAGE_ACCOUNT" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=client-research-agent/${ENVIRONMENT}.tfstate" \
  -backend-config="use_azuread_auth=true"

# 1. platform: catalog (prod), schema, volumes, SP, OIDC federation, secret scope,
#    VS endpoint, warehouse, cluster policy, budget
terraform apply -var-file=envs/${ENVIRONMENT}.tfvars \
  -var=databricks_host=... -var=databricks_account_host=... \
  -var=databricks_account_id=... -var=workspace_id=... \
  -var=github_repository=kogunlowo123/client-research-qualification-agent

# 2. tables, constraints, governance functions, tags, grants
python ../../infrastructure/unity_catalog/apply_ddl.py --environment ${ENVIRONMENT} \
  --agent-sp "$(terraform output -raw service_principal_application_id)"

# 3. index and table-level grants
terraform apply -var-file=envs/${ENVIRONMENT}.tfvars \
  -var=create_vector_index=true -var=tables_provisioned=true   # plus the -var flags above
```

Outputs used later: `service_principal_application_id`, `vector_search_endpoint_name`,
`vector_search_index_name`, `sql_warehouse_id`, `secret_scope`, `agent_endpoint_name`.

Per-environment differences (`envs/*.tfvars`):

| Setting | dev | staging | prod |
|---|---|---|---|
| `engineer_schema_privileges` | full build rights | `USE_SCHEMA`, `SELECT`, `EXECUTE`, `READ_VOLUME` | `USE_SCHEMA`, `SELECT` |
| `engineer_secret_permission` | `MANAGE` | `WRITE` | `READ` |
| `serving_workload_size` / scale-to-zero | Small / true | Small / true | Medium / false |
| Endpoint rate limit per minute (endpoint / user) | 120 / 30 | 300 / 60 | 1200 / 60 |
| `warehouse_size` | 2X-Small | 2X-Small | X-Small |
| `monthly_budget_usd` | 300 | 500 | 2000 |

Workspaces bootstrapped without Terraform can run the SQL in
`infrastructure/unity_catalog/00_catalog_schema.sql` and `05_grants.sql` through
`apply_ddl.py`; every statement is idempotent.

## 3. MLflow, Vector Search and monitoring bootstrap

These are idempotent and normally handled by Terraform and the bundle; use them
for bootstrap or break-glass:

```bash
python infrastructure/vector_search/provision_index.py --environment ${ENVIRONMENT} --sync
python infrastructure/mlflow/setup_mlflow.py --environment ${ENVIRONMENT}
python infrastructure/monitoring/provision_alerts.py --environment ${ENVIRONMENT} \
  --notify-email "$ALERT_EMAIL" --destination-id "$ONCALL_DESTINATION_ID"
python infrastructure/monitoring/lakehouse_monitor.py --environment ${ENVIRONMENT} \
  --notify-email "$ALERT_EMAIL"
```

`lakehouse_monitor.py` needs the AI Gateway inference table `cra_agent_payload`,
which exists only after the agent endpoint has served traffic; run it after the
first deployment.

## 4. GitHub OIDC setup

1. Terraform step 1 creates `databricks_service_principal_federation_policy.github`
   (policy id `github-<env>`) with subject
   `repo:kogunlowo123/client-research-qualification-agent:environment:<env>`.
2. In GitHub, create environments `staging` and `prod` (and `dev` if used).
   Configure required reviewers on `prod`.
3. Per environment, set **variables** (not secrets):

| Variable | Value |
|---|---|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_CLIENT_ID` | `terraform output -raw service_principal_application_id` |
| `DATABRICKS_ACCOUNT_ID` | Account id (OIDC token audience) |
| `ALERT_EMAIL` | Job failure mailbox |
| `ONCALL_NOTIFICATION_DESTINATION_ID` | Prod only: workspace notification destination for paging |
| `APPLY_FM_GATEWAY` | `"true"` to re-apply FMAPI AI Gateway configuration during deploy |
| `DATABRICKS_DEPLOY_ENABLED` | Repository-level variable. `"true"` enables the `cd.yml` deploy jobs; until then they are skipped so forks and fresh clones stay green |
| `SEC_CONTACT_EMAIL` | Contact mailbox for the SEC EDGAR User-Agent. `cd.yml` exports it as `BUNDLE_VAR_sec_contact_email` (bundle variable, no default, passed to every job as `--contact-email`); deploys fail validation without it |

`cd.yml` exports `DATABRICKS_AUTH_TYPE=github-oidc` and
`DATABRICKS_TOKEN_AUDIENCE=${{ vars.DATABRICKS_ACCOUNT_ID }}`, and requests
`id-token: write`. No Databricks secret is stored in GitHub. The only repository
secret referenced is the optional `SAFETY_API_KEY` for the safety scan in CI.

## 5. Secrets

- The default configuration needs no application secrets: Databricks access is
  OAuth (M2M in jobs and serving, OIDC in CI, U2M for developers).
- Terraform creates the secret scope `client-research-agent`
  (`secret_scope_name`), grants the service principal `READ` and engineers
  `engineer_secret_permission`.
- If an integration needs a secret, store it in the scope and reference it from
  job or endpoint environment variables as `{{secrets/client-research-agent/<key>}}`;
  read it in code through `security/secrets.py` (`EnvSecretProvider` /
  `DatabricksSecretProvider` / `ChainedSecretProvider`), which returns `SecretStr`.
- Never put secrets in `config/environments/*.yaml` or `.env`; YAML is shipped in
  the wheel.

## 6. Bundle deployment per target

```bash
make bundle-validate                 # validate dev, staging, prod
deployment/workflows/deploy.sh dev   # validate + deploy (make deploy-dev)
deployment/workflows/deploy.sh staging --evaluate          # also runs cra_evaluation (make deploy-staging)
deployment/workflows/deploy.sh prod --deploy-agent --apply-fm-gateway
```

`deploy.sh <dev|staging|prod>` options:

| Option | Effect |
|---|---|
| `--deploy-agent` | `databricks bundle run cra_agent_deploy --params action=log_and_deploy` |
| `--evaluate` | `databricks bundle run cra_evaluation`; fails if the quality gate fails |
| `--run <job_key>` | Run any bundle job by resource key (repeatable) |
| `--apply-fm-gateway` | `databricks serving-endpoints put-ai-gateway` for every endpoint in `deployment/serving/foundation_model_ai_gateway.<env>.json` |

The script deletes `dist/` first because jobs install `../../../dist/*.whl` and a
stale wheel must not be picked up.

Targets:

| Target | Mode | Schema | VS endpoint | Agent endpoint | Runs as |
|---|---|---|---|---|---|
| `dev` (default) | `development` (schedules paused, resources prefixed) | `agent_dev` | `cra-vs-endpoint-dev` | `cra-agent-dev` | deploying user |
| `staging` | `production` | `agent_staging` | `cra-vs-endpoint-staging` | `cra-agent-staging` | service principal |
| `prod` | `production` | `agent_prod` | `cra-vs-endpoint-prod` | `cra-agent-prod` | service principal; webhook paging on job failure |

## 7. Agent deployment and promotion

`cra_agent_deploy` (parameter `action`, default `log_and_deploy`):

```text
route_action (condition: action == log_and_deploy)
  true  -> log_and_register     cra-deploy-agent --stage log          -> task value model_version
        -> evaluate_candidate   cra-evaluate --mode dataset --model-uri models:/<uc_model>/<version>
                                --min-pass-rate --min-citation-coverage --fail-on-gate
        -> promote_and_deploy   cra-deploy-agent --stage promote --model-version <version>
                                --endpoint-name <agent_endpoint_name>
                                --endpoint-config deployment/serving/agent_endpoint.<env>.json
  false -> redeploy_champion    cra-deploy-agent --stage deploy-champion
```

Promotion moves `champion` to the evaluated version and keeps the previous one
as `previous_champion` for rollback.

CD flow (`.github/workflows/cd.yml`):

| Trigger | Environment | Steps |
|---|---|---|
| Push to `main` | `staging` | verify identity -> `deploy.sh staging` -> `cra_agent_deploy` -> `cra_evaluation` (quality gate) -> `smoke_test.sh staging` |
| Tag `vX.Y.Z` (must be on `main`) | `prod` (required reviewers) | verify identity -> `deploy.sh prod` -> `cra_agent_deploy` -> `smoke_test.sh prod` -> on failure `rollback.sh prod --skip-smoke` |
| `workflow_dispatch` | chosen | as above |

## 8. Smoke test

```bash
deployment/workflows/smoke_test.sh <env>
```

Checks, against `cra-agent-<env>` (override `CRA_AGENT_ENDPOINT`):

1. endpoint `state.ready == READY`;
2. a canned Responses request (Microsoft Corporation, `MSFT`, `microsoft.com`,
   `max_documents: 10`) returns HTTP 200 JSON with at least one output item;
3. the `output_text` contains an `http(s)://` citation and a verdict string;
4. latency within `SMOKE_MAX_LATENCY_SECONDS` (default 180).

## 9. Rollback

See the [rollback runbook](../runbooks/rollback.md). Summary:

```bash
deployment/workflows/rollback.sh prod                 # to previous_champion
deployment/workflows/rollback.sh prod --to-version 12 # explicit version
deployment/workflows/rollback.sh prod --via-job       # redeploy through cra_agent_deploy
```

Infrastructure changes are rolled back by reverting the Terraform or bundle
change and re-applying; data is not rolled back by any of these procedures
(Delta time travel is available on all tables for manual recovery).

## 10. Post-deployment checklist

- [ ] `terraform output` values recorded; `create_vector_index` and `tables_provisioned` true.
- [ ] `databricks bundle validate -t <env>` clean.
- [ ] `companies_watchlist` seeded with at least one active company; `cra_ingestion_refresh` succeeded once.
- [ ] `eval_set` populated (schema: `infrastructure/mlflow/eval_dataset_schema.json`).
- [ ] `cra_agent_deploy` passed `evaluate_candidate`; `champion` alias set.
- [ ] Smoke test passes.
- [ ] Alerts provisioned (`provision_alerts.py`) with the on-call destination; Lakehouse monitors created.
- [ ] SEC contact mailbox supplied: `BUNDLE_VAR_sec_contact_email` for jobs (no default; set GitHub environment variable `SEC_CONTACT_EMAIL`, which `cd.yml` exports) and secret `client-research-agent/sec-contact-email` for the endpoint (`databricks secrets put-secret client-research-agent sec-contact-email`).
