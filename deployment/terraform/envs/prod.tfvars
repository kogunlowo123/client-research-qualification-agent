environment        = "prod"
github_environment = "prod"

# prod owns the shared catalog; apply this state before dev and staging.
manage_catalog = true

engineer_schema_privileges = [
  "USE_SCHEMA",
  "SELECT",
]
engineer_secret_permission    = "READ"
engineer_warehouse_permission = "CAN_USE"

vector_search_endpoint_type = "STANDARD"
vector_index_pipeline_type  = "TRIGGERED"

serving_workload_size          = "Medium"
serving_scale_to_zero          = false
rate_limit_endpoint_per_minute = 1200
rate_limit_user_per_minute     = 60

warehouse_size              = "X-Small"
warehouse_max_clusters      = 2
warehouse_auto_stop_minutes = 15

cluster_policy_max_workers = 8
monthly_budget_usd         = 2000

tags = {
  cost_center = "client-research"
}
