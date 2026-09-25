environment        = "dev"
github_environment = "dev"
manage_catalog     = false

engineer_schema_privileges = [
  "USE_SCHEMA",
  "SELECT",
  "MODIFY",
  "EXECUTE",
  "READ_VOLUME",
  "WRITE_VOLUME",
  "CREATE_TABLE",
  "CREATE_FUNCTION",
  "CREATE_MODEL",
  "CREATE_VOLUME",
]
engineer_secret_permission    = "MANAGE"
engineer_warehouse_permission = "CAN_MANAGE"

vector_search_endpoint_type = "STANDARD"
vector_index_pipeline_type  = "TRIGGERED"

serving_workload_size          = "Small"
serving_scale_to_zero          = true
rate_limit_endpoint_per_minute = 120
rate_limit_user_per_minute     = 30

warehouse_size              = "2X-Small"
warehouse_max_clusters      = 1
warehouse_auto_stop_minutes = 5

cluster_policy_max_workers = 2
monthly_budget_usd         = 300

tags = {
  cost_center = "client-research"
}
