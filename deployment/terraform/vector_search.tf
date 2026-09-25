resource "databricks_vector_search_endpoint" "agent" {
  name          = local.vs_endpoint_name
  endpoint_type = var.vector_search_endpoint_type
}

# Delta Sync index over the chunks table (embedded child chunks only) with
# self-managed embeddings: the adapter writes the `embedding` column
# (databricks-gte-large-en) and the index syncs from the Change Data Feed.
# Primary key, embedding column and synced columns come from the same
# index_spec.json that provision_index.py and the contract tests use.
locals {
  index_spec = jsondecode(file("${path.module}/../../infrastructure/vector_search/index_spec.json")).index
}

resource "databricks_vector_search_index" "chunks" {
  count = var.create_vector_index ? 1 : 0

  name          = local.vs_index_name
  endpoint_name = databricks_vector_search_endpoint.agent.name
  primary_key   = local.index_spec.primary_key
  index_type    = "DELTA_SYNC"

  delta_sync_index_spec {
    source_table    = local.chunks_table_name
    pipeline_type   = var.vector_index_pipeline_type
    columns_to_sync = local.index_spec.columns_to_sync

    embedding_vector_columns {
      name                = local.index_spec.embedding_vector_column.name
      embedding_dimension = var.embedding_dimension
    }
  }

  depends_on = [databricks_grant.schema_agent_sp]
}
