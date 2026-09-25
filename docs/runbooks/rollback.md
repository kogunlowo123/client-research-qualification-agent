# Runbook: Rollback

Script: `deployment/workflows/rollback.sh <dev|staging|prod> [--to-version N] [--via-job] [--skip-smoke]`.
Registered model: `<catalog>.<schema>.client_research_agent` (default
`client_research.agent_<env>.client_research_agent`, override with `CRA_UC_MODEL`).
Endpoint: from `deployment/serving/agent_endpoint.<env>.json` (override with
`CRA_AGENT_ENDPOINT`).

## When to use

- Any prod incident whose onset coincides with a new `champion` version
  (quality, errors, latency).
- Automatically: `cd.yml` runs `rollback.sh prod --skip-smoke` when the prod
  smoke test fails after a deployment.

Rollback reverts the **agent model version** only. It does not revert data,
Terraform, bundle job definitions or FMAPI configuration.

## Preconditions

- `databricks` CLI and `jq` on PATH; authenticated as a principal with
  `CAN_MANAGE` on the endpoint and alias rights on the model (the environment
  service principal, or an engineer in dev).
- A `previous_champion` alias exists (maintained by `cra_agent_deploy` on every
  promotion), or a known-good version number.

## Procedure

1. Inspect aliases and versions:

```bash
MODEL=client_research.agent_prod.client_research_agent
databricks model-versions get-by-alias "$MODEL" champion --output json | jq '.version'
databricks model-versions get-by-alias "$MODEL" previous_champion --output json | jq '.version'
```

2. Roll back (default target is `previous_champion`):

```bash
deployment/workflows/rollback.sh prod
# or to a specific version
deployment/workflows/rollback.sh prod --to-version 12
# or through the deploy job (action=deploy_champion) instead of a direct config update
deployment/workflows/rollback.sh prod --via-job
```

   The script:
   1. refuses if the target is already champion or not `READY`;
   2. sets `rolled_back` on the current champion and moves `champion` to the target;
   3. updates the endpoint config rendered from `agent_endpoint.<env>.json` with the
      target version (`serving-endpoints update-config`, waits up to 30 minutes),
      or runs `cra_agent_deploy` with `action=deploy_champion` when `--via-job` is given;
   4. verifies the endpoint serves the target version;
   5. runs `smoke_test.sh <env>` unless `--skip-smoke`.

3. Verify:

```bash
deployment/workflows/smoke_test.sh prod
```

   and watch `error_rate_hourly` and `latency_percentiles_hourly` for 30 minutes.

## Roll forward

Fix the defect, merge, and release a new tag. `cra_agent_deploy` logs a new
version, evaluates it against the gate, and promotes it; the `rolled_back` alias
remains on the bad version for forensics.

## Failure modes

| Error | Action |
|---|---|
| `no champion alias` | Endpoint was never deployed through the job; deploy with `databricks bundle run -t <env> cra_agent_deploy` |
| `no previous_champion alias ... pass --to-version` | First promotion or alias removed; choose a version from `eval_quality_trend` that passed the gate |
| `version N is not READY` | Pick another version |
| `endpoint does not serve vN` | Endpoint update failed or another update was in progress; check `databricks serving-endpoints get`, then retry with `--via-job` |
| Smoke test fails after rollback | The cause is not the model version (FMAPI, Vector Search, data); continue with the matching runbook |

## Escalation

- If rollback does not restore service within 30 minutes, escalate as SEV1 per
  [runbooks/README.md](README.md).
