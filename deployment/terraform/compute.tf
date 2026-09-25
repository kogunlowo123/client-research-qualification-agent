# Serverless SQL warehouse for alerts, dashboards, Lakehouse Monitoring and
# analyst access to briefs. The bundle looks it up by name (cra-sql-<env>).
resource "databricks_sql_endpoint" "agent" {
  name                      = local.warehouse_name
  cluster_size              = var.warehouse_size
  min_num_clusters          = 1
  max_num_clusters          = var.warehouse_max_clusters
  auto_stop_mins            = var.warehouse_auto_stop_minutes
  enable_serverless_compute = true
  warehouse_type            = "PRO"
  enable_photon             = true
  spot_instance_policy      = "COST_OPTIMIZED"

  tags {
    dynamic "custom_tags" {
      for_each = local.common_tags
      content {
        key   = custom_tags.key
        value = custom_tags.value
      }
    }
  }
}

resource "databricks_permissions" "warehouse" {
  sql_endpoint_id = databricks_sql_endpoint.agent.id

  access_control {
    service_principal_name = local.sp_application_id
    permission_level       = "CAN_USE"
  }

  access_control {
    group_name       = data.databricks_group.engineers.display_name
    permission_level = var.engineer_warehouse_permission
  }

  access_control {
    group_name       = data.databricks_group.analysts.display_name
    permission_level = "CAN_USE"
  }
}

# Guard rails for any classic job cluster (jobs default to serverless; this
# policy applies when an engineer attaches a classic cluster for debugging).
resource "databricks_cluster_policy" "jobs" {
  name        = "cra-jobs-${var.environment}"
  description = "Client Research Agent job clusters: UC single-user, autoscaling capped, tagged for chargeback."

  definition = jsonencode({
    "cluster_type" = {
      type  = "fixed"
      value = "job"
    }
    "data_security_mode" = {
      type  = "fixed"
      value = "SINGLE_USER"
    }
    "spark_version" = {
      type         = "unlimited"
      defaultValue = "auto:latest-lts"
    }
    "runtime_engine" = {
      type   = "allowlist"
      values = ["STANDARD", "PHOTON"]
    }
    "autoscale.min_workers" = {
      type     = "range"
      minValue = 1
      maxValue = 2
    }
    "autoscale.max_workers" = {
      type     = "range"
      minValue = 1
      maxValue = var.cluster_policy_max_workers
    }
    "custom_tags.app" = {
      type  = "fixed"
      value = "client-research-agent"
    }
    "custom_tags.environment" = {
      type  = "fixed"
      value = var.environment
    }
    "spark_conf.spark.databricks.cluster.profile" = {
      type   = "forbidden"
      hidden = true
    }
  })
}

resource "databricks_permissions" "jobs_policy" {
  cluster_policy_id = databricks_cluster_policy.jobs.id

  access_control {
    service_principal_name = local.sp_application_id
    permission_level       = "CAN_USE"
  }

  access_control {
    group_name       = data.databricks_group.engineers.display_name
    permission_level = "CAN_USE"
  }
}
