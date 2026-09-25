# Least-privilege Unity Catalog grants. databricks_grant (singular) manages one
# principal per securable, so the three environment states can each grant on
# the shared catalog without overwriting one another.

# ---------------------------------------------------------------- catalog
resource "databricks_grant" "catalog_agent_sp" {
  catalog    = local.catalog_name
  principal  = local.sp_application_id
  privileges = ["USE_CATALOG", "CREATE_SCHEMA"]
}

resource "databricks_grant" "catalog_engineers" {
  count = var.manage_catalog ? 1 : 0

  catalog    = local.catalog_name
  principal  = data.databricks_group.engineers.display_name
  privileges = ["USE_CATALOG"]
}

resource "databricks_grant" "catalog_analysts" {
  count = var.manage_catalog ? 1 : 0

  catalog    = local.catalog_name
  principal  = data.databricks_group.analysts.display_name
  privileges = ["USE_CATALOG"]
}

# ----------------------------------------------------------------- schema
resource "databricks_grant" "schema_agent_sp" {
  schema    = databricks_schema.agent.id
  principal = local.sp_application_id
  privileges = [
    "USE_SCHEMA",
    "SELECT",
    "MODIFY",
    "EXECUTE",
    "READ_VOLUME",
    "WRITE_VOLUME",
    "CREATE_TABLE",
    "CREATE_FUNCTION",
    "CREATE_MODEL",
  ]
}

resource "databricks_grant" "schema_engineers" {
  schema     = databricks_schema.agent.id
  principal  = data.databricks_group.engineers.display_name
  privileges = var.engineer_schema_privileges
}

# Analysts only get USE_SCHEMA here; SELECT is granted on the briefs table alone.
resource "databricks_grant" "schema_analysts" {
  schema     = databricks_schema.agent.id
  principal  = data.databricks_group.analysts.display_name
  privileges = ["USE_SCHEMA"]
}

# ----------------------------------------------------------------- volumes
resource "databricks_grant" "raw_documents_agent_sp" {
  volume     = databricks_volume.raw_documents.id
  principal  = local.sp_application_id
  privileges = ["READ_VOLUME", "WRITE_VOLUME"]
}

resource "databricks_grant" "brief_exports_analysts" {
  volume     = databricks_volume.brief_exports.id
  principal  = data.databricks_group.analysts.display_name
  privileges = ["READ_VOLUME"]
}

# ------------------------------------------------------------------ tables
# Requires the DDL in infrastructure/unity_catalog to have created the table.
resource "databricks_grant" "briefs_analysts" {
  count = var.tables_provisioned ? 1 : 0

  table      = local.briefs_table_name
  principal  = data.databricks_group.analysts.display_name
  privileges = ["SELECT"]
}
