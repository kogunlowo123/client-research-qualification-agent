# The agent endpoint is normally created by databricks.agents.deploy() in the
# cra_agent_deploy job. Set manage_serving_endpoint = true to have Terraform own
# it instead (the registered model and version must already exist).
data "databricks_registered_model" "agent" {
  count = var.manage_serving_endpoint ? 1 : 0

  full_name = local.agent_model_name
}

resource "databricks_model_serving" "agent" {
  count = var.manage_serving_endpoint ? 1 : 0

  name = local.agent_endpoint

  config {
    served_entities {
      name                  = "client_research_agent"
      entity_name           = data.databricks_registered_model.agent[0].full_name
      entity_version        = var.agent_model_version
      workload_size         = var.serving_workload_size
      scale_to_zero_enabled = var.serving_scale_to_zero

      environment_vars = {
        CRA_ENVIRONMENT       = var.environment
        ENABLE_MLFLOW_TRACING = "true"
      }
    }

    traffic_config {
      routes {
        served_model_name  = "client_research_agent"
        traffic_percentage = 100
      }
    }
  }

  ai_gateway {
    usage_tracking_config {
      enabled = true
    }

    inference_table_config {
      enabled           = true
      catalog_name      = local.catalog_name
      schema_name       = databricks_schema.agent.name
      table_name_prefix = "cra_agent"
    }

    rate_limits {
      calls          = var.rate_limit_endpoint_per_minute
      key            = "endpoint"
      renewal_period = "minute"
    }

    rate_limits {
      calls          = var.rate_limit_user_per_minute
      key            = "user"
      renewal_period = "minute"
    }
  }

  dynamic "tags" {
    for_each = local.common_tags
    content {
      key   = tags.key
      value = tags.value
    }
  }
}

resource "databricks_permissions" "agent_endpoint" {
  count = var.manage_serving_endpoint ? 1 : 0

  serving_endpoint_id = databricks_model_serving.agent[0].serving_endpoint_id

  access_control {
    service_principal_name = local.sp_application_id
    permission_level       = "CAN_MANAGE"
  }

  access_control {
    group_name       = data.databricks_group.engineers.display_name
    permission_level = "CAN_VIEW"
  }

  access_control {
    group_name       = data.databricks_group.analysts.display_name
    permission_level = "CAN_QUERY"
  }
}
