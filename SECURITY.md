# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.x     | Yes       |
| < 1.0   | No        |

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/kogunlowo123/client-research-qualification-agent/security/advisories/new)
for this repository. Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- affected versions, commits or deployment configuration.

You will receive an acknowledgement within 3 business days and a status update
within 10 business days. Confirmed issues are fixed in a patch release and
disclosed through a GitHub Security Advisory once a fix is available. We are
happy to credit reporters who wish to be named.

## Scope

In scope:

- the `client_research_agent` Python package and `cra` CLI,
- prompt-injection, guardrail or citation-validation bypasses that cause the
  agent to emit unsupported "verified facts", leak data or exfiltrate secrets,
- infrastructure-as-code in `deployment/` and `infrastructure/` (Terraform,
  Databricks Asset Bundle, Unity Catalog grants, row filters and column masks),
- the container image and GitHub Actions workflows.

Out of scope: vulnerabilities in Databricks, MLflow or third-party services
themselves (report those to the vendor), and findings that require a
compromised Databricks workspace administrator.

## Security controls in this repository

- No secrets in code or configuration; Databricks access uses unified auth
  (OAuth M2M / GitHub OIDC workload identity federation) and secret scopes.
- Least-privilege Unity Catalog grants, PII column masks and row filters.
- AI Gateway rate limits, usage tracking, inference tables and PII/safety
  guardrails on model endpoints.
- CI runs bandit, pip-audit, Trivy (filesystem and image), gitleaks and CodeQL.
