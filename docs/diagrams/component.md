# Component Diagram

Packages under `src/client_research_agent/` and their main dependencies. An
arrow `A --> B` means A imports B. Vendor SDKs are imported only inside
`databricks/` (and lazily by `observability/` and `security/secrets.py`).

```text
                          +-------------+      +-------------+
                          |   cli.py    |      | workflows/  |
                          +------+------+      +------+------+
                                 |                    |
                                 v                    v
 +------------------+     +------------------------------------+
 | agent/           |<----|        orchestration/              |
 |  serving_agent   |---->|  orchestrator  steps  state        |
 |  factory         |     |  review  rendering                 |
 |  metering        |     +--+------+------+------+------+-----+
 +--+---------+-----+        |      |      |      |      |
    |         |              |      |      |      |      v
    |         |              |      |      |      |  +---------------+
    |         |              |      |      |      |  | evaluation/   |
    |         |              |      |      |      |  | dataset,      |
    |         |              |      |      |      |  | harness, gate |
    |         |              |      |      |      |  +---------------+
    |         |              v      v      v      v
    |         |  +----------+ +------------+ +---------+ +-----------+
    |         |  | research/| |qualification| | scoring | | briefing/ |
    |         |  | sources  | | agent       | | engine  | | generator |
    |         |  | fetcher  | | criteria    | +----+----+ | opportunity
    |         |  | url_guard| | heuristic   |      |      | renderer  |
    |         |  | robots   | | evidence    |      |      +-----+-----+
    |         |  | parsing  | +------+------+      |            |
    |         |  +----+-----+        |             |            v
    |         |       |              v             |      +-----------+
    |         |       |       +------------+       |      | citations/|
    |         |       |       | ranking/   |       |      | validator |
    |         |       |       +------------+       |      | entailment|
    |         |       |              |             |      +-----+-----+
    |         |       |              v             |            |
    |         |       |       +------------------------------+  |
    |         |       |       | retrieval/                   |  |
    |         |       |       | pipeline hybrid bm25 dense   |  |
    |         |       |       | crag self_rag graph rerank   |  |
    |         |       |       | chunking enrichment indexing |  |
    |         |       |       +--------------+---------------+  |
    |         |       |                      |                  |
    v         v       v                      v                  v
 +---------------------------------------------------------------------+
 | services/ports.py (Protocols)   services/structured.py   services/local.py
 +---------------------------------------------------------------------+
 | models/domain.py   config/settings.py   prompts/registry.py (+templates)
 +---------------------------------------------------------------------+
 | security/  (sanitizer, prompt_injection, poisoning, pii, output_guard,
 |             rbac, rate_limiter, secrets)
 | governance/ (audit, lineage, responsible_ai, data_classification)
 | observability/ (logging, tracing, metrics, cost, mlflow_tracking, setup)
 | utils/ (errors, resilience)
 +---------------------------------------------------------------------+
            ^
            | implements ports
 +---------------------------------------------------------------------+
 | databricks/  auth  model_serving  embeddings  vector_search          |
 |              unity_catalog  mlflow_registry  jobs  errors            |
 |   -> databricks-sdk, databricks-vectorsearch, mlflow, openai         |
 +---------------------------------------------------------------------+
```

Key contracts:

| Contract | Defined in | Consumed by |
|---|---|---|
| `LLMClient`, `EmbeddingClient`, `VectorIndex`, `DocumentStore`, `BriefRepository`, `HttpFetcher`, `AuditSink` | `services/ports.py` | every agent; bound in `agent/factory.py::build_runtime` |
| `Retriever` (qualification view) | `qualification/contracts.py` | `QualificationAgent` |
| `Retriever`, `RetrievalOutcome` | `retrieval/pipeline.py` | orchestration steps (`GuardedRetriever`) |
| `ChunkRetriever` | `retrieval/base.py` | `HybridRetriever`, `MultiQueryRetriever`, `CorrectiveRetriever` |
| `Reranker` | `retrieval/reranking.py` | `RetrievalPipeline` |
| `SecretProvider` | `security/secrets.py` | adapters needing secrets |
