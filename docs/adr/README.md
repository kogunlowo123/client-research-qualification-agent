# Architecture Decision Records

Decisions are recorded in the Michael Nygard format (Status, Context, Decision,
Consequences). A record is immutable once accepted; a changed decision gets a new
ADR that supersedes the old one, and the old one's status is updated to
"Superseded by ADR-NNNN".

| ADR | Title | Status |
|---|---|---|
| [ADR-0001](0001-hexagonal-ports-and-adapters.md) | Hexagonal architecture with ports and adapters | Accepted |
| [ADR-0002](0002-foundation-model-api-with-fallback-endpoint.md) | Databricks Foundation Model API as the default LLM, with a fallback endpoint | Accepted |
| [ADR-0003](0003-delta-sync-index-self-managed-embeddings.md) | Delta Sync Vector Search index with self-managed embeddings and a separate `parent_chunks` table | Accepted |
| [ADR-0004](0004-hybrid-retrieval-with-rrf.md) | Hybrid retrieval (BM25 + dense) fused with weighted Reciprocal Rank Fusion | Accepted |
| [ADR-0005](0005-deterministic-fallback-for-every-llm-step.md) | A deterministic fallback for every LLM step | Accepted |
| [ADR-0006](0006-verbatim-compression-for-verifiable-citations.md) | Verbatim (extractive) context compression to keep citations verifiable | Accepted |
| [ADR-0007](0007-public-sources-only-and-analyst-compliance.md) | Public sources only; analyst-firm pages only when explicitly supplied | Accepted |
| [ADR-0008](0008-verdict-semantics.md) | Verdict semantics: low fit reported as NOT_ENOUGH_EVIDENCE with reason LOW_FIT | Accepted |
| [ADR-0009](0009-oauth-m2m-and-oidc-no-pats.md) | OAuth M2M and GitHub OIDC workload identity; no PATs outside dev | Accepted |
| [ADR-0010](0010-hash-chained-audit-log.md) | Tamper-evident, hash-chained audit log | Accepted |
| [ADR-0011](0011-mlflow-responsesagent-and-agents-deploy.md) | Serve the agent as an MLflow ResponsesAgent deployed with `databricks.agents.deploy` | Accepted |

## Writing a new ADR

1. Copy the structure of an existing record into `docs/adr/NNNN-short-title.md`
   with the next free number.
2. State the forces in Context without advocating; put the choice in Decision;
   list positive and negative Consequences honestly.
3. Reference code by module path and configuration by settings key so the record
   can be checked against the repository.
4. Add a row to the table above in the same pull request.
