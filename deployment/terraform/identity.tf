# Service principal that runs every job, owns the bundle deployment in
# staging/prod and is the identity GitHub Actions assumes through OIDC.
resource "databricks_service_principal" "agent" {
  application_id             = var.service_principal_application_id != "" ? var.service_principal_application_id : null
  display_name               = "cra-agent-${var.environment}"
  active                     = true
  workspace_access           = true
  databricks_sql_access      = true
  allow_cluster_create       = false
  allow_instance_pool_create = false
}

data "databricks_group" "engineers" {
  display_name = var.engineers_group
}

data "databricks_group" "analysts" {
  display_name = var.analysts_group
}

# GitHub Actions -> Databricks workload identity federation. Tokens issued for
# the configured GitHub environment of this repository may act as the SP.
resource "databricks_service_principal_federation_policy" "github" {
  count    = local.federation_enabled ? 1 : 0
  provider = databricks.account

  service_principal_id = tonumber(databricks_service_principal.agent.id)
  policy_id            = "github-${var.environment}"
  description          = "GitHub Actions OIDC for ${var.github_repository} environment ${var.github_environment}"

  oidc_policy = {
    issuer    = "https://token.actions.githubusercontent.com"
    audiences = [var.databricks_account_id]
    subject   = "repo:${var.github_repository}:environment:${var.github_environment}"
  }
}

resource "databricks_secret_scope" "agent" {
  name                     = var.secret_scope_name
  initial_manage_principal = null
}

resource "databricks_secret_acl" "agent_sp_read" {
  scope      = databricks_secret_scope.agent.name
  principal  = local.sp_application_id
  permission = "READ"
}

resource "databricks_secret_acl" "engineers" {
  scope      = databricks_secret_scope.agent.name
  principal  = data.databricks_group.engineers.display_name
  permission = var.engineer_secret_permission
}
