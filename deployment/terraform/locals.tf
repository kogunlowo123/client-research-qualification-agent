locals {
  schema_name        = "agent_${var.environment}"
  catalog_name       = var.manage_catalog ? databricks_catalog.this[0].name : data.databricks_catalog.this[0].name
  schema_full_name   = "${local.catalog_name}.${local.schema_name}"
  vs_endpoint_name   = "cra-vs-endpoint-${var.environment}"
  vs_index_name      = "${local.schema_full_name}.chunks_index"
  chunks_table_name  = "${local.schema_full_name}.chunks"
  briefs_table_name  = "${local.schema_full_name}.briefs"
  agent_model_name   = "${local.schema_full_name}.client_research_agent"
  agent_endpoint     = "cra-agent-${var.environment}"
  warehouse_name     = "cra-sql-${var.environment}"
  sp_application_id  = databricks_service_principal.agent.application_id
  budget_enabled     = var.databricks_account_id != "" && var.workspace_id != "" && length(var.budget_alert_emails) > 0
  federation_enabled = var.databricks_account_id != ""

  common_tags = merge(
    {
      app         = "client-research-agent"
      environment = var.environment
      managed_by  = "terraform"
    },
    var.tags,
  )
}
