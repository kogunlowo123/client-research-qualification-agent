output "catalog_name" {
  description = "Unity Catalog catalog."
  value       = local.catalog_name
}

output "schema_full_name" {
  description = "Environment schema (catalog.schema)."
  value       = local.schema_full_name
}

output "service_principal_application_id" {
  description = "Application ID of the agent service principal (set as DATABRICKS_CLIENT_ID / BUNDLE_VAR_service_principal_application_id in CI)."
  value       = local.sp_application_id
}

output "vector_search_endpoint_name" {
  description = "Vector Search endpoint name."
  value       = databricks_vector_search_endpoint.agent.name
}

output "vector_search_index_name" {
  description = "Vector Search index full name."
  value       = local.vs_index_name
}

output "sql_warehouse_id" {
  description = "Serverless SQL warehouse ID."
  value       = databricks_sql_endpoint.agent.id
}

output "cluster_policy_id" {
  description = "Cluster policy for classic job clusters."
  value       = databricks_cluster_policy.jobs.id
}

output "secret_scope" {
  description = "Secret scope name."
  value       = databricks_secret_scope.agent.name
}

output "agent_endpoint_name" {
  description = "Model Serving endpoint name for the agent."
  value       = local.agent_endpoint
}
