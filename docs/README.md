# Documentation Index

Documentation for the Client Research & Qualification Agent. Start with the
repository [README](../README.md) for the summary, quickstart and production
readiness checklist; the documents below hold the detail.

## Architecture

| Document | Contents |
|---|---|
| [architecture/architecture.md](architecture/architecture.md) | Logical, component, deployment, sequence and Databricks architecture; agent flow; data model; resilience; human-in-the-loop |
| [architecture/rag_design.md](architecture/rag_design.md) | Ingestion, parsing, chunking, enrichment, embeddings, storage, retrieval stages, grounding, advanced RAG patterns, evaluation gates |
| [architecture/threat_model.md](architecture/threat_model.md) | Assets, trust boundaries, STRIDE, OWASP Top 10 for LLM Applications (2025) mapping, residual risks |

## Diagrams (standalone)

| Diagram | File |
|---|---|
| Logical architecture | [diagrams/logical.md](diagrams/logical.md) |
| Component diagram | [diagrams/component.md](diagrams/component.md) |
| Deployment diagram | [diagrams/deployment.md](diagrams/deployment.md) |
| Sequence diagram (request through the serving endpoint to a brief) | [diagrams/sequence.md](diagrams/sequence.md) |
| Databricks architecture (UC, Vector Search, Model Serving, MLflow, Workflows, inference tables) | [diagrams/databricks.md](diagrams/databricks.md) |

All diagrams are ASCII inside fenced code blocks so they render in any Markdown
viewer and diff cleanly in review.

## Architecture Decision Records

[adr/README.md](adr/README.md) indexes ADR-0001 to ADR-0011 (hexagonal ports,
FMAPI with fallback, Delta Sync index and `parent_chunks`, hybrid RRF,
deterministic fallbacks, verbatim compression, public sources and analyst
compliance, verdict semantics, OAuth/OIDC without PATs, hash-chained audit log,
ResponsesAgent with `agents.deploy`).

## Operations

| Document | Audience |
|---|---|
| [operations/deployment_guide.md](operations/deployment_guide.md) | Platform engineers: Terraform order, DDL, bundle targets, OIDC, secrets, smoke test, promotion, rollback |
| [operations/developer_guide.md](operations/developer_guide.md) | Contributors: setup with uv, make targets, conventions, extending sources, criteria and adapters, testing |
| [operations/operational_guide.md](operations/operational_guide.md) | On-call and service owners: SLOs, monitoring, alerts, capacity, cost, retention, access reviews |

## Runbooks

[runbooks/README.md](runbooks/README.md) lists incident runbooks for serving
errors and latency, FMAPI throttling and open circuit breakers, vector index sync
failures, ingestion blocked by policy, citation coverage and quality-gate
failures, prompt-injection alerts, and rollback.

## Other repository documents

- [CONTRIBUTING.md](../CONTRIBUTING.md) - workflow and commit conventions
- [SECURITY.md](../SECURITY.md) - vulnerability reporting
- [CHANGELOG.md](../CHANGELOG.md) - release history
