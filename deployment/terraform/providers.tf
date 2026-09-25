# Workspace-level provider. Authentication uses Databricks unified auth
# (DATABRICKS_HOST + DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET, Azure CLI,
# or GitHub OIDC via DATABRICKS_AUTH_TYPE=github-oidc); nothing is hard-coded.
provider "databricks" {
  host = var.databricks_host
}

# Account-level provider, used only for account-scoped resources (budget and
# the GitHub OIDC federation policy). Those resources are skipped when
# databricks_account_id is empty.
provider "databricks" {
  alias      = "account"
  host       = var.databricks_account_host
  account_id = var.databricks_account_id
}
