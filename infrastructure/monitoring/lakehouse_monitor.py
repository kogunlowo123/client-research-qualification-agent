"""Create or update Lakehouse Monitoring (data quality) monitors for the agent.

Two monitors are reconciled idempotently:

* ``cra_agent_payload`` (AI Gateway inference table of ``cra-agent-<env>``):
  time-series profile on ``request_time`` sliced by status code and served
  entity, so latency (``execution_duration_ms``) and error drift are tracked.
* ``briefs`` (adapter-owned schema): time-series profile on ``generated_at``
  sliced by ``verdict``, so verdict distribution and ``weighted_score`` drift
  are tracked over time.

Metric tables land in the bundle-managed ``<schema>_monitoring`` schema.

    python infrastructure/monitoring/lakehouse_monitor.py --environment prod \
        --notify-email "$ALERT_EMAIL"
"""

from __future__ import annotations

import argparse
import re
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorCronSchedule,
    MonitorCronSchedulePauseStatus,
    MonitorDestination,
    MonitorNotifications,
    MonitorTimeSeries,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GRANULARITIES = ["1 hour", "1 day"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision Lakehouse Monitoring for the agent.")
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--catalog", default="client_research")
    parser.add_argument("--schema", help="Defaults to agent_<environment>.")
    parser.add_argument("--output-schema", help="Defaults to <schema>_monitoring.")
    parser.add_argument("--warehouse-id", help="Warehouse for the generated dashboard.")
    parser.add_argument("--notify-email", action="append", default=[], help="Repeatable.")
    parser.add_argument("--profile", help="Databricks CLI profile.")
    parser.add_argument("--refresh", action="store_true", help="Trigger a metric refresh after reconciling.")
    return parser.parse_args(argv)


def reconcile(
    client: WorkspaceClient,
    table: str,
    *,
    output_schema: str,
    assets_dir: str,
    warehouse_id: str | None,
    notifications: MonitorNotifications | None,
    schedule: MonitorCronSchedule,
    slicing_exprs: list[str],
    time_series: MonitorTimeSeries | None = None,
) -> None:
    try:
        client.quality_monitors.get(table)
    except NotFound:
        client.quality_monitors.create(
            table_name=table,
            output_schema_name=output_schema,
            assets_dir=assets_dir,
            warehouse_id=warehouse_id,
            notifications=notifications,
            schedule=schedule,
            slicing_exprs=slicing_exprs,
            time_series=time_series,
        )
        sys.stdout.write(f"created monitor on {table}\n")
        return
    client.quality_monitors.update(
        table_name=table,
        output_schema_name=output_schema,
        notifications=notifications,
        schedule=schedule,
        slicing_exprs=slicing_exprs,
        time_series=time_series,
    )
    sys.stdout.write(f"updated monitor on {table}\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = args.schema or f"agent_{args.environment}"
    output = args.output_schema or f"{schema}_monitoring"
    for label, value in (("catalog", args.catalog), ("schema", schema), ("output-schema", output)):
        if not _IDENTIFIER.match(value):
            raise SystemExit(f"invalid {label}: {value!r}")

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    output_schema = f"{args.catalog}.{output}"
    assets_root = f"/Workspace/Shared/client-research-agent/{args.environment}/monitoring"
    notifications = (
        MonitorNotifications(on_failure=MonitorDestination(email_addresses=args.notify_email))
        if args.notify_email
        else None
    )
    schedule = MonitorCronSchedule(
        quartz_cron_expression="0 0 */6 * * ?",
        timezone_id="UTC",
        pause_status=MonitorCronSchedulePauseStatus.UNPAUSED,
    )

    payload_table = f"{args.catalog}.{schema}.cra_agent_payload"
    briefs_table = f"{args.catalog}.{schema}.briefs"
    reconcile(
        client,
        payload_table,
        output_schema=output_schema,
        assets_dir=f"{assets_root}/cra_agent_payload",
        warehouse_id=args.warehouse_id,
        notifications=notifications,
        schedule=schedule,
        slicing_exprs=["status_code", "served_entity_id"],
        time_series=MonitorTimeSeries(timestamp_col="request_time", granularities=GRANULARITIES),
    )
    reconcile(
        client,
        briefs_table,
        output_schema=output_schema,
        assets_dir=f"{assets_root}/briefs",
        warehouse_id=args.warehouse_id,
        notifications=notifications,
        schedule=schedule,
        slicing_exprs=["verdict", "weighted_score >= 3.5"],
        time_series=MonitorTimeSeries(timestamp_col="generated_at", granularities=GRANULARITIES),
    )
    if args.refresh:
        for table in (payload_table, briefs_table):
            client.quality_monitors.run_refresh(table)
            sys.stdout.write(f"refresh queued for {table}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
