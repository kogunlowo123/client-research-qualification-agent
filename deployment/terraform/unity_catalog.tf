# Catalog: owned by exactly one state (prod, manage_catalog = true) and read by
# the others so that three independent environment states never fight over it.
resource "databricks_catalog" "this" {
  count = var.manage_catalog ? 1 : 0

  name           = var.catalog_name
  comment        = "Client Research & Qualification Agent: public-source evidence, chunks, briefs and audit data."
  storage_root   = var.catalog_storage_root
  isolation_mode = "OPEN"
  force_destroy  = false

  properties = {
    app        = "client-research-agent"
    managed_by = "terraform"
  }
}

data "databricks_catalog" "this" {
  count = var.manage_catalog ? 0 : 1

  name = var.catalog_name
}

resource "databricks_schema" "agent" {
  catalog_name  = local.catalog_name
  name          = local.schema_name
  comment       = "Client Research Agent ${var.environment} tables, vector index, functions and registered model."
  force_destroy = false

  properties = {
    app         = "client-research-agent"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Raw crawled artefacts (HTML, PDF, EDGAR filings) kept for provenance and re-chunking.
resource "databricks_volume" "raw_documents" {
  catalog_name = local.catalog_name
  schema_name  = databricks_schema.agent.name
  name         = "raw_documents"
  volume_type  = "MANAGED"
  comment      = "Raw public-source documents captured by the ingestion job, keyed by content hash."
}

# Rendered briefs (Markdown/JSON) exported for downstream CRM sync.
resource "databricks_volume" "brief_exports" {
  catalog_name = local.catalog_name
  schema_name  = databricks_schema.agent.name
  name         = "brief_exports"
  volume_type  = "MANAGED"
  comment      = "Rendered client briefs exported by cra_brief_generation."
}
