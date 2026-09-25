#!/usr/bin/env bash
# Validate, deploy and (optionally) run the Databricks Asset Bundle.
#
#   deployment/workflows/deploy.sh <dev|staging|prod> [options]
#
# Options:
#   --deploy-agent       run cra_agent_deploy (log -> evaluate -> promote -> deploy)
#   --evaluate           run cra_evaluation and fail if the quality gate fails
#   --run <job_key>      run any bundle job by resource key (repeatable)
#   --apply-fm-gateway   apply AI Gateway config to the Foundation Model endpoints
#
# Auth: Databricks unified auth (DATABRICKS_HOST + DATABRICKS_CLIENT_ID and
# DATABRICKS_AUTH_TYPE=github-oidc in CI, or a CLI profile via DATABRICKS_CONFIG_PROFILE).
set -euo pipefail

usage() {
  sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 1 ]] || usage
TARGET="$1"
shift
case "$TARGET" in
  dev | staging | prod) ;;
  -h | --help) usage ;;
  *)
    echo "error: unknown target '$TARGET' (expected dev, staging or prod)" >&2
    exit 2
    ;;
esac

DEPLOY_AGENT=false
EVALUATE=false
APPLY_FM_GATEWAY=false
RUN_JOBS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-agent) DEPLOY_AGENT=true ;;
    --evaluate) EVALUATE=true ;;
    --apply-fm-gateway) APPLY_FM_GATEWAY=true ;;
    --run)
      [[ $# -ge 2 ]] || usage
      RUN_JOBS+=("$2")
      shift
      ;;
    -h | --help) usage ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage
      ;;
  esac
  shift
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

for tool in databricks jq uv; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: '$tool' is required on PATH" >&2
    exit 1
  }
done

log() { printf '[deploy:%s] %s\n' "$TARGET" "$*"; }

# Jobs install ../dist/*.whl, so stale wheels from earlier builds must not linger.
rm -rf dist

log "validating bundle"
databricks bundle validate --target "$TARGET"

log "deploying bundle"
databricks bundle deploy --target "$TARGET"

if [[ "$APPLY_FM_GATEWAY" == "true" ]]; then
  gateway_file="deployment/serving/foundation_model_ai_gateway.${TARGET}.json"
  log "applying AI Gateway configuration from $gateway_file"
  count="$(jq '.endpoints | length' "$gateway_file")"
  for ((i = 0; i < count; i++)); do
    name="$(jq -r ".endpoints[$i].name" "$gateway_file")"
    payload="$(jq -c ".endpoints[$i].ai_gateway" "$gateway_file")"
    log "  $name"
    databricks serving-endpoints put-ai-gateway "$name" --json "$payload" >/dev/null
  done
fi

if [[ "$DEPLOY_AGENT" == "true" ]]; then
  log "running cra_agent_deploy"
  databricks bundle run --target "$TARGET" cra_agent_deploy --params action=log_and_deploy
fi

if [[ "$EVALUATE" == "true" ]]; then
  log "running cra_evaluation quality gate"
  databricks bundle run --target "$TARGET" cra_evaluation
fi

for job in "${RUN_JOBS[@]+"${RUN_JOBS[@]}"}"; do
  log "running $job"
  databricks bundle run --target "$TARGET" "$job"
done

log "done"
