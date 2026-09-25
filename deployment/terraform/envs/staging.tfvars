environment        = "staging"
github_environment = "staging"
manage_catalog     = false

engineer_schema_privileges = [
  "USE_SCHEMA",
  "SELECT",
  "EXECUTE",
  "READ_VOLUME",
]
engineer_secret_permission    = "WRITE"
engineer_warehouse_permission = "CAN_USE"

vector_search_endpoint_type = "STANDARD"
vector_index_pipeline_type  = "TRIGGERED"

serving_workload_size          = "Small"
serving_scale_to_zero          = true
rate_limit_endpoint_per_minute = 300
rate_limit_user_per_minute     = 60

warehouse_size              = "2X-Small"
warehouse_max_clusters      = 1
warehouse_auto_stop_minutes = 10

cluster_policy_max_workers = 4
monthly_budget_usd         = 500

tags = {
  cost_center = "client-research"
}
