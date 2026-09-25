# ADR-0009: OAuth M2M and GitHub OIDC workload identity; no personal access tokens outside dev

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

The agent and its pipelines call Databricks APIs from three places: developer
machines, GitHub Actions, and Databricks jobs and serving endpoints. Personal
access tokens (PATs) are long-lived bearer secrets tied to a human, commonly
leaked through CI secrets and logs, and hard to rotate. Static secrets stored in
GitHub are a supply-chain risk.

## Decision

- All code authenticates through the Databricks SDK unified-auth chain
  (`databricks/auth.py::build_workspace_client`): explicit arguments, then
  `DATABRICKS_*` environment variables (OAuth M2M with `DATABRICKS_CLIENT_ID` /
  `DATABRICKS_CLIENT_SECRET`), then a `~/.databrickscfg` profile, then ambient
  notebook or job credentials.
- PATs are rejected in `staging` and `prod` twice: by the
  `AppSettings._prod_requires_workspace` validator (a configured
  `databricks.token` raises "static tokens are forbidden outside dev") and by
  `build_workspace_client`, which for environments in `_PAT_FORBIDDEN` refuses a
  configured token or a `DATABRICKS_TOKEN` environment variable before building
  the client, and refuses a client whose resolved auth type is `pat` afterwards.
- GitHub Actions uses **OIDC workload identity federation**. `deployment/terraform/identity.tf`
  creates `databricks_service_principal_federation_policy.github` with issuer
  `https://token.actions.githubusercontent.com`, audience = account ID, and
  subject `repo:<owner>/<repo>:environment:<env>`. `cd.yml` sets
  `DATABRICKS_AUTH_TYPE=github-oidc` and `permissions: id-token: write`. No
  Databricks secret is stored in GitHub; only non-secret variables
  (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_ACCOUNT_ID`).
- Jobs run as the environment service principal (`run_as.service_principal_name`
  in the `staging` and `prod` bundle targets).
- Chat and embedding calls use a callable API key backed by
  `WorkspaceCredentials.bearer_token`, evaluated per request, so short-lived
  OAuth tokens refresh transparently.
- Remaining secrets are read through `security/secrets.py`
  (`EnvSecretProvider` for `{{secrets/<scope>/<key>}}` injection,
  `DatabricksSecretProvider`, `ChainedSecretProvider`) and returned as
  `SecretStr`. The scope is `databricks.secret_scope` (`client-research-agent`);
  the service principal has `READ`, engineers get `engineer_secret_permission`
  per environment (`MANAGE` dev, `WRITE` staging, `READ` prod).

## Consequences

- Positive: no long-lived credential exists for staging or prod; a leaked GitHub
  token is useless outside the matching repository environment.
- Positive: every action in staging and prod is attributable to a service
  principal and visible in audit logs.
- Negative: federation policies are account-level resources; Terraform needs an
  account-level provider (`provider = databricks.account`) and account admin
  rights for the first apply.
- Negative: developers need `databricks auth login` (U2M OAuth) for dev; a PAT is
  still tolerated there for convenience, and that exception must not be widened.
