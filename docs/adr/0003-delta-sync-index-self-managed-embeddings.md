# ADR-0003: Delta Sync Vector Search index with self-managed embeddings and a separate `parent_chunks` table

- Status: Accepted
- Date: 2026-09-24
- Deciders: Kehinde Ogunlowo

## Context

Mosaic AI Vector Search offers two index types: Direct Vector Access (the
application writes vectors to the index) and Delta Sync (the index follows a
Delta table through Change Data Feed). Delta Sync supports either
Databricks-managed embeddings (the service calls an embedding endpoint on a text
column) or self-managed embeddings (the table already holds a vector column).

The retrieval design is small-to-big (parent-child, see `docs/architecture/rag_design.md`):
small child chunks are matched, large parent chunks are fed to the LLM. Parents
are too long to embed usefully and must never appear as retrieval hits.

A Delta Sync index with self-managed embeddings fails or skips rows whose vector
column is NULL.

## Decision

1. Production uses a **Delta Sync** index `<catalog>.<schema>.chunks_index` on
   the Unity Catalog table `<catalog>.<schema>.chunks`, primary key `chunk_id`,
   vector column `embedding` (`ARRAY<FLOAT>`, 1024-d), pipeline type `TRIGGERED`.
   The specification lives once in `infrastructure/vector_search/index_spec.json`
   and is read by Terraform (`deployment/terraform/vector_search.tf`), by
   `infrastructure/vector_search/provision_index.py` and by the contract tests.
2. Embeddings are **self-managed**: the ingestion pipeline embeds each child's
   `embedding_text` (contextual header plus body) with `databricks-gte-large-en`
   and writes the vector with the row.
3. `chunks` holds **embedded child chunks only**. This invariant is enforced by
   `embedding ARRAY<FLOAT> NOT NULL` and the CHECK constraint
   `chunks_embedded_children_only` (`strategy <> 'parent' AND size(embedding) > 0`).
4. Parent chunks live in `parent_chunks` (same columns minus `embedding`, CHECK
   `parent_chunks_parents_only`). They are read by id only, so the table has no
   Change Data Feed and is never indexed.
5. Write routing in `DeltaDocumentStore`: `save_chunks` sends parents to
   `parent_chunks` and only refreshes metadata of child rows already present;
   children become visible when `DatabricksVectorIndex.upsert` writes them with
   vectors through `save_chunks_with_embeddings` and then calls `index.sync()`.

## Consequences

- Positive: the Delta table is the single governed source of truth (UC grants,
  tags, lineage, time travel). The index can be rebuilt from the table at any
  time without re-billing the embedding endpoint.
- Positive: right-to-erasure and company deletion is one `DELETE` on `chunks`;
  CDF propagates it on the next sync.
- Positive: the NOT NULL plus CHECK constraints make the "NULL vector silently
  dropped by the index" failure mode impossible rather than merely unlikely.
- Negative: `TRIGGERED` sync is asynchronous. New evidence is not searchable until
  a sync completes (`cra_ingestion_refresh.sync_index`, or `upsert` with
  `wait_for_sync=True` where read-after-write is required). Brief generation in
  the same job must sync before retrieval (`--sync-index` on the ingest task).
- Negative: embeddings are pinned to one model and dimension. Changing the
  embedding model requires re-embedding every row and recreating the index
  (the dimension is fixed at index creation). `embedding_dimension` is validated
  by `DatabricksEmbeddingClient` before rows reach the table.
- Neutral: the service's native hybrid query (`query_type="HYBRID"`) is available
  through `DatabricksVectorIndex.hybrid_search`, but the default pipeline uses the
  application-side hybrid of ADR-0004 so behaviour is identical locally and in
  production.
