variable "environment" {
  description = "Deployment environment: dev, staging or prod."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "The environment must be one of dev, staging, prod."
  }
}

variable "databricks_host" {
  description = "Workspace URL. Null reads DATABRICKS_HOST from the environment."
  type        = string
  default     = null
}

variable "databricks_account_host" {
  description = "Account console URL (https://accounts.azuredatabricks.net, https://accounts.cloud.databricks.com or https://accounts.gcp.databricks.com)."
  type        = string
  default     = "https://accounts.azuredatabricks.net"
}

variable "databricks_account_id" {
  description = "Databricks account ID. Empty disables account-level resources (budget, OIDC federation policy)."
  type        = string
  default     = ""
}

variable "workspace_id" {
  description = "Numeric workspace ID used to scope the budget filter."
  type        = string
  default     = ""
}

variable "catalog_name" {
  description = "Unity Catalog catalog shared by every environment schema."
  type        = string
  default     = "client_research"
}

variable "manage_catalog" {
  description = "Create the catalog in this state. Exactly one environment state (prod) owns it; the others read it."
  type        = bool
  default     = false
}

variable "catalog_storage_root" {
  description = "Managed storage location for the catalog. Null uses the metastore root."
  type        = string
  default     = null
}

variable "engineers_group" {
  description = "Existing workspace/account group of agent engineers."
  type        = string
  default     = "cra-engineers"
}

variable "analysts_group" {
  description = "Existing workspace/account group of business analysts who read briefs."
  type        = string
  default     = "cra-analysts"
}

variable "engineer_schema_privileges" {
  description = "Privileges engineers receive on the environment schema."
  type        = list(string)
}

variable "engineer_secret_permission" {
  description = "Secret scope ACL for engineers: READ, WRITE or MANAGE."
  type        = string
  default     = "WRITE"

  validation {
    condition     = contains(["READ", "WRITE", "MANAGE"], var.engineer_secret_permission)
    error_message = "The engineer_secret_permission must be READ, WRITE or MANAGE."
  }
}

variable "engineer_warehouse_permission" {
  description = "Warehouse permission for engineers: CAN_USE, CAN_MONITOR or CAN_MANAGE."
  type        = string
  default     = "CAN_USE"
}

variable "service_principal_application_id" {
  description = "Application (client) ID of an existing Entra ID application to register as the agent service principal (Azure). Empty creates a Databricks-managed service principal."
  type        = string
  default     = ""
}

variable "github_repository" {
  description = "owner/repo allowed to authenticate as the service principal through GitHub OIDC."
  type        = string
  default     = "kogunlowo123/client-research-qualification-agent"
}

variable "github_environment" {
  description = "GitHub Actions environment whose OIDC tokens may act as this environment's service principal."
  type        = string
}

variable "secret_scope_name" {
  description = "Secret scope read by jobs and the serving endpoint."
  type        = string
  default     = "client-research-agent"
}

variable "vector_search_endpoint_type" {
  description = "STANDARD or STORAGE_OPTIMIZED."
  type        = string
  default     = "STANDARD"
}

variable "create_vector_index" {
  description = "Create chunks_index. Enable after infrastructure/unity_catalog DDL has created the chunks table with Change Data Feed."
  type        = bool
  default     = false
}

variable "embedding_dimension" {
  description = "Dimension of the self-managed embedding column (databricks-gte-large-en = 1024)."
  type        = number
  default     = 1024
}

variable "vector_index_pipeline_type" {
  description = "TRIGGERED (synced by cra_ingestion_refresh) or CONTINUOUS."
  type        = string
  default     = "TRIGGERED"
}

variable "tables_provisioned" {
  description = "Set true once the DDL in infrastructure/unity_catalog has run, enabling table-level grants."
  type        = bool
  default     = false
}

variable "manage_serving_endpoint" {
  description = "Manage the agent serving endpoint in Terraform. Normally false: databricks.agents.deploy() in cra_agent_deploy owns it."
  type        = bool
  default     = false
}

variable "agent_model_version" {
  description = "UC model version served when manage_serving_endpoint is true (the version the champion alias points to)."
  type        = string
  default     = "1"
}

variable "serving_workload_size" {
  description = "Small, Medium or Large."
  type        = string
  default     = "Small"
}

variable "serving_scale_to_zero" {
  description = "Allow the agent endpoint to scale to zero."
  type        = bool
  default     = true
}

variable "rate_limit_endpoint_per_minute" {
  description = "AI Gateway endpoint-wide calls per minute."
  type        = number
  default     = 300
}

variable "rate_limit_user_per_minute" {
  description = "AI Gateway per-user calls per minute."
  type        = number
  default     = 60
}

variable "warehouse_size" {
  description = "Serverless SQL warehouse T-shirt size."
  type        = string
  default     = "2X-Small"
}

variable "warehouse_max_clusters" {
  description = "Maximum clusters for the serverless SQL warehouse."
  type        = number
  default     = 1
}

variable "warehouse_auto_stop_minutes" {
  description = "Idle minutes before the warehouse stops."
  type        = number
  default     = 10
}

variable "cluster_policy_max_workers" {
  description = "Upper bound on workers for classic job clusters that use the policy."
  type        = number
  default     = 4
}

variable "monthly_budget_usd" {
  description = "Monthly list-price budget for resources tagged app=client-research-agent."
  type        = number
  default     = 500
}

variable "budget_alert_emails" {
  description = "Recipients of budget alerts."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Extra custom tags applied to compute resources."
  type        = map(string)
  default     = {}
}
