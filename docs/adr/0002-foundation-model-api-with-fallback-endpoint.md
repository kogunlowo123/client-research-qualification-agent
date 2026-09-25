# ADR-0002: Databricks Foundation Model API as the default LLM, with a fallback endpoint

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

The agent needs a capable chat model for criterion scoring, opportunity
analysis, brief writing, citation judging, query rewriting, CRAG grading and
reranking, plus an embedding model. Data must stay inside the Databricks
security perimeter, calls must be governed (rate limits, usage tracking,
inference tables, PII guardrails) and billed through the existing platform
contract. A single model endpoint is a single point of failure: pay-per-token
Foundation Model API (FMAPI) endpoints are shared capacity and return 429 and
5xx under load.

## Decision

- Chat and embeddings go to Databricks Model Serving FMAPI endpoints through the
  OpenAI-compatible protocol at `{host}/serving-endpoints`
  (`databricks/auth.py::build_openai_client`, `databricks/model_serving.py`,
  `databricks/embeddings.py`). The API key is a callable backed by the SDK
  credential provider, so OAuth tokens refresh per request.
- Endpoint names are configuration, not code (`ModelServingSettings`):
  - `serving.chat_endpoint`: `databricks-claude-sonnet-4`
  - `serving.fallback_chat_endpoint`: `databricks-meta-llama-3-3-70b-instruct`
  - `serving.judge_endpoint`: `databricks-claude-sonnet-4`
  - `serving.embedding_endpoint`: `databricks-gte-large-en` (1024-d)
- Each endpoint call runs under a per-endpoint `CircuitBreaker` and a
  `RetryPolicy` built from `ResilienceSettings` (`max_attempts`,
  `initial_backoff_seconds`, `max_backoff_seconds`, `breaker_failure_threshold`,
  `breaker_reset_seconds`). `openai` exceptions are mapped to the typed hierarchy
  in `utils/errors.py` (`RateLimitedError`, `UpstreamServiceError`,
  `UpstreamTimeoutError`) by `map_openai_error`.
- `FallbackLLMClient` answers from the fallback endpoint only on
  `TransientError` or `CircuitOpenError`. Non-transient failures (bad request,
  auth, validation) are not masked, because the second endpoint would repeat
  them.
- AI Gateway configuration for the FMAPI endpoints (usage tracking, inference
  tables, rate limits, input/output safety and PII masking) is versioned in
  `deployment/serving/foundation_model_ai_gateway.<env>.json` and applied by
  `deployment/workflows/deploy.sh --apply-fm-gateway`.

## Consequences

- Positive: no data leaves the workspace; model calls are governed and visible in
  `system.serving.endpoint_usage` and AI Gateway inference tables.
- Positive: a model swap is a configuration change (`CRA_SERVING__CHAT_ENDPOINT`),
  and prompt and model versions are recorded on every brief (`model_versions`).
- Negative: the fallback model is weaker. Its outputs pass through the same
  schema validation (`services/structured.py::complete_structured`) and the same
  citation validation, so quality degrades but grounding guarantees do not.
- Negative: when both endpoints are unavailable, every LLM step degrades to its
  deterministic path (ADR-0005). The brief is still produced, but narrative
  quality drops; `warnings` on the brief records each degradation.
- Negative: FMAPI pay-per-token has no capacity guarantee. Provisioned
  throughput is the escalation path if sustained 429s appear
  (see `docs/runbooks/fm-api-rate-limiting-and-circuit-breaker.md`).
