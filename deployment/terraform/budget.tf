# Account-level budget on everything tagged app=client-research-agent in this
# workspace. Enabled when databricks_account_id, workspace_id and
# budget_alert_emails are all set.
resource "databricks_budget" "agent" {
  count    = local.budget_enabled ? 1 : 0
  provider = databricks.account

  display_name = "client-research-agent-${var.environment}"

  alert_configurations {
    time_period        = "MONTH"
    trigger_type       = "CUMULATIVE_SPENDING_EXCEEDED"
    quantity_type      = "LIST_PRICE_DOLLARS_USD"
    quantity_threshold = tostring(var.monthly_budget_usd)

    dynamic "action_configurations" {
      for_each = var.budget_alert_emails
      content {
        action_type = "EMAIL_NOTIFICATION"
        target      = action_configurations.value
      }
    }
  }

  filter {
    workspace_id {
      operator = "IN"
      values   = [tonumber(var.workspace_id)]
    }

    tags {
      key = "app"
      value {
        operator = "IN"
        values   = ["client-research-agent"]
      }
    }
  }
}
