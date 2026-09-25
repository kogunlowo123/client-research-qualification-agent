#!/usr/bin/env bash
# Smoke-test the agent serving endpoint with a canned research request.
#
#   deployment/workflows/smoke_test.sh <dev|staging|prod>
#
# Environment overrides:
#   CRA_AGENT_ENDPOINT        endpoint name (default cra-agent-<env>)
#   SMOKE_MAX_LATENCY_SECONDS fail if the request takes longer (default 180)
#   DATABRICKS_HOST           workspace URL (default: resolved from the CLI auth config)
#
# The request is sent with curl using an OAuth access token from
# `databricks auth token`; when that auth type cannot mint a token (e.g. M2M
# via env vars on older CLIs) the script falls back to
# `databricks serving-endpoints query`, which uses the same unified auth.
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: $0 <dev|staging|prod>" >&2
  exit 2
}
TARGET="$1"
case "$TARGET" in
  dev | staging | prod) ;;
  *)
    echo "error: unknown target '$TARGET'" >&2
    exit 2
    ;;
esac

for tool in databricks jq curl; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: '$tool' is required on PATH" >&2
    exit 1
  }
done

ENDPOINT="${CRA_AGENT_ENDPOINT:-cra-agent-${TARGET}}"
MAX_LATENCY="${SMOKE_MAX_LATENCY_SECONDS:-180}"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

log() { printf '[smoke:%s] %s\n' "$TARGET" "$*"; }
fail() {
  printf '[smoke:%s] FAIL: %s\n' "$TARGET" "$*" >&2
  exit 1
}

log "checking endpoint $ENDPOINT state"
state_json="$(databricks serving-endpoints get "$ENDPOINT" --output json)"
ready="$(jq -r '.state.ready // "UNKNOWN"' <<<"$state_json")"
config_update="$(jq -r '.state.config_update // "UNKNOWN"' <<<"$state_json")"
[[ "$ready" == "READY" ]] || fail "endpoint not ready (ready=$ready, config_update=$config_update)"

cat >"$WORKDIR/request.json" <<'JSON'
{
  "input": [
    {
      "role": "user",
      "content": "Research Microsoft Corporation (ticker MSFT, domain microsoft.com) and return a qualification brief with cited public sources."
    }
  ],
  "custom_inputs": {
    "company_name": "Microsoft Corporation",
    "ticker": "MSFT",
    "domain": "microsoft.com",
    "max_documents": 10,
    "requested_by": "smoke-test"
  },
  "context": {
    "conversation_id": "smoke-test",
    "user_id": "smoke-test"
  }
}
JSON

HOST="${DATABRICKS_HOST:-}"
if [[ -z "$HOST" ]]; then
  HOST="$(databricks auth env --output json 2>/dev/null | jq -r '.env.DATABRICKS_HOST // empty')"
fi
HOST="${HOST%/}"

TOKEN=""
if [[ -n "$HOST" ]]; then
  TOKEN="$(databricks auth token --host "$HOST" --output json 2>/dev/null | jq -r '.access_token // empty' || true)"
fi

start="$(date +%s)"
if [[ -n "$TOKEN" ]]; then
  log "querying $HOST/serving-endpoints/$ENDPOINT/invocations via curl"
  http_code="$(
    curl --silent --show-error --fail-with-body \
      --max-time "$MAX_LATENCY" \
      --output "$WORKDIR/response.json" \
      --write-out '%{http_code}' \
      --header "Authorization: Bearer ${TOKEN}" \
      --header "Content-Type: application/json" \
      --data @"$WORKDIR/request.json" \
      "$HOST/serving-endpoints/$ENDPOINT/invocations"
  )" || fail "HTTP request failed (status ${http_code:-none}): $(head -c 2000 "$WORKDIR/response.json" 2>/dev/null)"
  [[ "$http_code" == "200" ]] || fail "unexpected HTTP status $http_code"
else
  log "querying $ENDPOINT via databricks serving-endpoints query"
  databricks serving-endpoints query "$ENDPOINT" --json @"$WORKDIR/request.json" --output json \
    >"$WORKDIR/response.json" || fail "query failed: $(head -c 2000 "$WORKDIR/response.json")"
fi
elapsed=$(($(date +%s) - start))

jq -e . "$WORKDIR/response.json" >/dev/null || fail "response is not JSON"
jq -e '(.output | type == "array") and (.output | length > 0)' "$WORKDIR/response.json" >/dev/null \
  || fail "response has no output items: $(head -c 2000 "$WORKDIR/response.json")"

text="$(jq -r '[.output[] | select(.type == "message") | .content[]? | select(.type == "output_text") | .text] | join("\n")' "$WORKDIR/response.json")"
[[ -n "$text" ]] || fail "response contains no output_text"
grep -Eq 'https?://' <<<"$text" || fail "brief contains no cited source URL"
grep -Eqi 'good[_ ]fit|potential[_ ]fit|not[_ ]enough[_ ]evidence' <<<"$text" \
  || fail "brief contains no qualification verdict"
((elapsed <= MAX_LATENCY)) || fail "latency ${elapsed}s exceeded ${MAX_LATENCY}s"

log "PASS (${elapsed}s, $(wc -c <<<"$text") chars of brief text)"
