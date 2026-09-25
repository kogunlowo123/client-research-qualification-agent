#!/usr/bin/env bash
# Roll the agent back to a previous Unity Catalog model version.
#
#   deployment/workflows/rollback.sh <dev|staging|prod> [--to-version N] [--via-job] [--skip-smoke]
#
# Default target version is the one the `previous_champion` alias points to
# (maintained by cra_agent_deploy on every promotion). The script:
#   1. moves `champion` to the target version and tags the bad version
#      with the `rolled_back` alias,
#   2. updates the serving endpoint to serve the target version, either
#      directly (serving-endpoints update-config, rendered from
#      deployment/serving/agent_endpoint.<env>.json) or with --via-job by
#      running cra_agent_deploy with action=deploy_champion,
#   3. runs smoke_test.sh.
#
# Environment overrides:
#   CRA_UC_MODEL        full UC model name (default client_research.agent_<env>.client_research_agent)
#   CRA_AGENT_ENDPOINT  endpoint name (default from agent_endpoint.<env>.json)
set -euo pipefail

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

[[ $# -ge 1 ]] || usage
TARGET="$1"
shift
case "$TARGET" in
  dev | staging | prod) ;;
  -h | --help) usage ;;
  *)
    echo "error: unknown target '$TARGET'" >&2
    exit 2
    ;;
esac

TO_VERSION=""
VIA_JOB=false
SKIP_SMOKE=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --to-version)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || {
        echo "error: --to-version needs a numeric version" >&2
        exit 2
      }
      TO_VERSION="$2"
      shift
      ;;
    --via-job) VIA_JOB=true ;;
    --skip-smoke) SKIP_SMOKE=true ;;
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

for tool in databricks jq; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: '$tool' is required on PATH" >&2
    exit 1
  }
done

CONFIG_FILE="deployment/serving/agent_endpoint.${TARGET}.json"
MODEL="${CRA_UC_MODEL:-client_research.agent_${TARGET}.client_research_agent}"
ENDPOINT="${CRA_AGENT_ENDPOINT:-$(jq -r '.name' "$CONFIG_FILE")}"

log() { printf '[rollback:%s] %s\n' "$TARGET" "$*"; }

alias_version() {
  databricks model-versions get-by-alias "$MODEL" "$1" --output json 2>/dev/null | jq -r '.version // empty'
}

CURRENT="$(alias_version champion)"
[[ -n "$CURRENT" ]] || {
  echo "error: $MODEL has no champion alias" >&2
  exit 1
}
if [[ -z "$TO_VERSION" ]]; then
  TO_VERSION="$(alias_version previous_champion)"
  [[ -n "$TO_VERSION" ]] || {
    echo "error: no previous_champion alias on $MODEL; pass --to-version" >&2
    exit 1
  }
fi
[[ "$TO_VERSION" != "$CURRENT" ]] || {
  echo "error: version $TO_VERSION is already champion" >&2
  exit 1
}

status="$(databricks model-versions get "$MODEL" "$TO_VERSION" --output json | jq -r '.status // empty')"
[[ "$status" == "READY" ]] || {
  echo "error: $MODEL version $TO_VERSION is not READY (status=$status)" >&2
  exit 1
}

log "moving champion of $MODEL: v$CURRENT -> v$TO_VERSION"
databricks registered-models set-alias "$MODEL" rolled_back "$CURRENT" >/dev/null
databricks registered-models set-alias "$MODEL" champion "$TO_VERSION" >/dev/null

if [[ "$VIA_JOB" == "true" ]]; then
  log "redeploying champion through cra_agent_deploy"
  databricks bundle run --target "$TARGET" cra_agent_deploy --params action=deploy_champion
else
  payload="$(
    jq -c --arg model "$MODEL" --arg version "$TO_VERSION" '
      .config
      | .served_entities |= map(.entity_name = $model | .entity_version = $version)
    ' "$CONFIG_FILE"
  )"
  log "updating $ENDPOINT to serve v$TO_VERSION (waits for NOT_UPDATING)"
  databricks serving-endpoints update-config "$ENDPOINT" --json "$payload" --timeout 30m >/dev/null
fi

served="$(databricks serving-endpoints get "$ENDPOINT" --output json \
  | jq -r '[.config.served_entities[]?.entity_version] | join(",")')"
log "endpoint $ENDPOINT now serves version(s): $served"
[[ ",$served," == *",$TO_VERSION,"* ]] || {
  echo "error: endpoint does not serve v$TO_VERSION" >&2
  exit 1
}

if [[ "$SKIP_SMOKE" != "true" ]]; then
  CRA_AGENT_ENDPOINT="$ENDPOINT" "$REPO_ROOT/deployment/workflows/smoke_test.sh" "$TARGET"
fi

log "rollback complete: champion=v$TO_VERSION, rolled_back=v$CURRENT"
