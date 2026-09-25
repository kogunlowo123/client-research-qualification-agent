# Runbook: Vector index sync failure

Index: `<catalog>.<schema>.chunks_index` on endpoint `cra-vs-endpoint-<env>`,
Delta Sync from `<catalog>.<schema>.chunks`, self-managed `embedding` column
(1024-d), pipeline `TRIGGERED`. Specification:
`infrastructure/vector_search/index_spec.json`.

## Symptoms

- `cra_ingestion_refresh` task `sync_index` (`cra-ingest --sync-index-only`)
  fails or exceeds its 3600 s timeout; `cra_brief_generation` task `ingest`
  (`--sync-index`) fails.
- Briefs cite only old evidence, or dense retrieval returns nothing for newly
  ingested companies while BM25 still finds them.
- `retrieval.hybrid.dense_fallback` increases (dense leg raising errors).
- Index status not `ONLINE` / not ready.

## Dashboards and queries

```bash
databricks vector-search-endpoints get-endpoint cra-vs-endpoint-prod --output json \
  | jq '.endpoint_status'
databricks vector-search-indexes get-index client_research.agent_prod.chunks_index --output json \
  | jq '.status'
```

Rows written since the last successful sync:

```sql
SELECT count(*) AS rows_since, max(updated_at) AS newest
FROM `${catalog}`.`${schema}`.`chunks`
WHERE updated_at >= current_timestamp() - INTERVAL 1 DAY;
```

Invariant checks (both must return 0):

```sql
SELECT count(*) FROM `${catalog}`.`${schema}`.`chunks`
WHERE embedding IS NULL OR size(embedding) <> 1024 OR strategy = 'parent';

SHOW TBLPROPERTIES `${catalog}`.`${schema}`.`chunks` ('delta.enableChangeDataFeed');
```

## Diagnosis

| Observation | Cause | Fix |
|---|---|---|
| Endpoint not `ONLINE` | Endpoint provisioning or platform incident | Wait; if prolonged, escalate |
| Index `FAILED` with schema error | `chunks` schema changed (column added, type changed) outside `index_spec.json` | Revert the DDL change; `infrastructure/unity_catalog/01_tables.sql` and `databricks/unity_catalog.py::DDL` must match (`tests/contract/test_ddl_consistency.py`) |
| Dimension mismatch | Embedding endpoint or `serving.embedding_dimension` changed | Revert the embedding config; a model change requires re-embedding every row and recreating the index |
| Change Data Feed disabled | Table recreated without `delta.enableChangeDataFeed` | Re-enable with `ALTER TABLE ... SET TBLPROPERTIES`, then full resync |
| Permission denied | Grants on `chunks` changed | Re-apply Terraform (`databricks_grant.schema_agent_sp`) or `infrastructure/unity_catalog/05_grants.sql` |
| Sync succeeded, results still stale | Brief ran before sync completed (asynchronous `TRIGGERED` sync) | Ensure the ingest task runs with `--sync-index`; for manual runs use `wait_for_sync=True` |

## Mitigation

1. Trigger a sync manually and wait:

```bash
python infrastructure/vector_search/provision_index.py --environment prod --sync
```

   The script verifies the existing index matches `index_spec.json`, creates
   missing objects idempotently, and triggers a sync.
2. If the index is corrupt or its spec drifted, recreate it: delete the index,
   then run `provision_index.py` (or `terraform apply` with
   `-var=create_vector_index=true`). The Delta table is the source of truth, so no
   evidence is lost and embeddings are not recomputed (ADR-0003).
3. While the index is unavailable, briefs continue on BM25 through the hybrid
   fallback. Communicate reduced recall to users if it lasts more than a day.

## Escalation

- SEV2 if prod dense retrieval is unavailable for more than 4 hours.
- Open a Databricks support case with endpoint name, index name and the index
  status message.
