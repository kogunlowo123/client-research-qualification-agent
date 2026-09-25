# ADR-0001: Hexagonal architecture with ports and adapters

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

The agent depends on five external systems: Databricks Model Serving (chat and
embeddings), Mosaic AI Vector Search, Unity Catalog Delta tables (via the SQL
Statement Execution API), the public web, and an audit destination. All of them
are slow, metered, or non-deterministic, and none of them is available in a
pull-request CI run.

We need the business logic (retrieval, qualification, scoring, briefing,
citation validation) to be testable offline and deterministically, and we need
to be able to swap a vendor SDK version without touching that logic.

## Decision

Every external system is reached through a `typing.Protocol` port defined in
`src/client_research_agent/services/ports.py`:

| Port | Production adapter | Local / CI adapter |
|---|---|---|
| `LLMClient` | `databricks.model_serving.DatabricksChatClient`, wrapped by `FallbackLLMClient` | `None` (every LLM step has a deterministic path, see ADR-0005) or test doubles |
| `EmbeddingClient` | `databricks.embeddings.DatabricksEmbeddingClient` | `retrieval.embeddings.HashingEmbeddingClient` |
| `VectorIndex` | `databricks.vector_search.DatabricksVectorIndex` | `services.local.InMemoryVectorIndex` |
| `DocumentStore` | `databricks.unity_catalog.DeltaDocumentStore` | `services.local.InMemoryDocumentStore` |
| `BriefRepository` | `databricks.unity_catalog.DeltaBriefRepository` | `services.local.JsonlBriefRepository` |
| `HttpFetcher` | `research.fetcher.PolicyEnforcingFetcher` over `HttpxFetcher` | same, with `respx` in tests |
| `AuditSink` | `databricks.audit_sink.FanOutAuditLogger` (hash-chained JSONL replicated to `audit_log` through `DeltaAuditSink`) | `services.local.JsonlAuditSink` |

Rules:

1. Vendor SDKs (`databricks-sdk`, `databricks-vectorsearch`, `mlflow`, `openai`)
   are imported only inside `client_research_agent.databricks` (and lazily in
   `observability` and `security.secrets`). They are an optional extra
   (`pip install .[databricks]`).
2. Domain types in `models/domain.py` are frozen Pydantic models; adapters
   translate to and from them at the boundary (`chunk_to_row`, `chunk_from_row`).
3. Adapter parity is enforced by the contract suite in `tests/contract/`, which
   parametrises the same tests over the local and the Databricks adapter (the
   latter against `tests/contract/fakes.py`).

## Consequences

- Positive: the full pipeline runs with no network model calls; unit and RAG
  quality tests are reproducible; vendor upgrades are contained in one package.
- Positive: the composition root is the only place that knows which adapter is
  bound, which makes environment differences explicit and reviewable.
- Negative: a second implementation of every port has to be maintained, and the
  in-memory adapters can drift from Databricks semantics (for example Vector
  Search filter dialect). The contract suite is the mitigation; a behaviour not
  covered there is not guaranteed to match.
- Negative: the Protocol surface is the lowest common denominator. Service
  features beyond it (for example `DatabricksVectorIndex.hybrid_search` with
  `query_type="HYBRID"`) are reachable only by depending on the concrete adapter.
