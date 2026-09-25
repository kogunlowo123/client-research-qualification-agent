"""Create or update the Databricks SQL alerts defined in alerts.sql + alerts.json.

Alerts are matched by display name, so re-running updates them in place.

    python infrastructure/monitoring/provision_alerts.py --environment prod \
        --notify-email "$ALERT_EMAIL" --destination-id "$ONCALL_DESTINATION_ID"

Authentication follows Databricks unified auth; pass --profile to use a CLI profile.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    AlertEvaluationState,
    AlertV2,
    AlertV2Evaluation,
    AlertV2Notification,
    AlertV2Operand,
    AlertV2OperandColumn,
    AlertV2OperandValue,
    AlertV2Subscription,
    ComparisonOperator,
    CronSchedule,
    SchedulePauseStatus,
)

HERE = Path(__file__).resolve().parent
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_HEADER = re.compile(r"^-- alert: ([a-z0-9_]+)\s*$", re.MULTILINE)
_UPDATE_MASK = "display_name,query_text,warehouse_id,evaluation,schedule,custom_description"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision Client Research Agent SQL alerts.")
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--catalog", default="client_research")
    parser.add_argument("--schema", help="Defaults to agent_<environment>.")
    parser.add_argument("--warehouse-id", help="Defaults to the warehouse named cra-sql-<environment>.")
    parser.add_argument("--notify-email", action="append", default=[], help="Repeatable.")
    parser.add_argument("--destination-id", action="append", default=[], help="Notification destination.")
    parser.add_argument("--parent-path", help="Workspace folder for alerts.")
    parser.add_argument("--profile", help="Databricks CLI profile.")
    parser.add_argument("--paused", action="store_true", help="Create alerts with paused schedules.")
    return parser.parse_args(argv)


def load_queries(context: dict[str, str]) -> dict[str, str]:
    text = (HERE / "alerts.sql").read_text(encoding="utf-8")
    for key, value in context.items():
        text = text.replace("${" + key + "}", value)
    parts = _HEADER.split(text)
    # parts = [preamble, name1, body1, name2, body2, ...]
    return {parts[i]: parts[i + 1].strip().rstrip(";") for i in range(1, len(parts), 2)}


def resolve_warehouse(client: WorkspaceClient, warehouse_id: str | None, environment: str) -> str:
    if warehouse_id:
        return warehouse_id
    name = f"cra-sql-{environment}"
    for warehouse in client.warehouses.list():
        if warehouse.name == name and warehouse.id:
            return warehouse.id
    raise SystemExit(f"SQL warehouse {name!r} not found; pass --warehouse-id")


def build_alert(
    spec: dict[str, Any],
    query: str,
    *,
    warehouse_id: str,
    parent_path: str,
    subscriptions: list[AlertV2Subscription],
    environment: str,
    paused: bool,
) -> AlertV2:
    return AlertV2(
        display_name=spec["display_name"].format(environment=environment),
        custom_description=spec["description"].format(environment=environment),
        query_text=query,
        warehouse_id=warehouse_id,
        parent_path=parent_path,
        evaluation=AlertV2Evaluation(
            source=AlertV2OperandColumn(name="value"),
            comparison_operator=ComparisonOperator[spec["operator"]],
            threshold=AlertV2Operand(value=AlertV2OperandValue(double_value=float(spec["threshold"]))),
            empty_result_state=AlertEvaluationState.OK,
            notification=AlertV2Notification(
                notify_on_ok=True,
                retrigger_seconds=int(spec["retrigger_seconds"]),
                subscriptions=subscriptions,
            ),
        ),
        schedule=CronSchedule(
            quartz_cron_schedule=spec["schedule"],
            timezone_id="UTC",
            pause_status=SchedulePauseStatus.PAUSED if paused else SchedulePauseStatus.UNPAUSED,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = args.schema or f"agent_{args.environment}"
    for label, value in (("catalog", args.catalog), ("schema", schema)):
        if not _IDENTIFIER.match(value):
            raise SystemExit(f"invalid {label}: {value!r}")
    context = {"catalog": args.catalog, "schema": schema, "environment": args.environment}
    specs: dict[str, dict[str, Any]] = json.loads((HERE / "alerts.json").read_text(encoding="utf-8"))[
        "alerts"
    ]
    queries = load_queries(context)
    missing = set(specs) ^ set(queries)
    if missing:
        raise SystemExit(f"alerts.json and alerts.sql disagree on: {sorted(missing)}")

    subscriptions = [AlertV2Subscription(user_email=e) for e in args.notify_email] + [
        AlertV2Subscription(destination_id=d) for d in args.destination_id
    ]
    if not subscriptions:
        raise SystemExit("at least one --notify-email or --destination-id is required")

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    warehouse_id = resolve_warehouse(client, args.warehouse_id, args.environment)
    parent_path = args.parent_path or f"/Workspace/Shared/client-research-agent/{args.environment}/alerts"
    client.workspace.mkdirs(parent_path)
    existing = {a.display_name: a.id for a in client.alerts_v2.list_alerts() if a.display_name and a.id}

    for key, spec in sorted(specs.items()):
        alert = build_alert(
            spec,
            queries[key],
            warehouse_id=warehouse_id,
            parent_path=parent_path,
            subscriptions=subscriptions,
            environment=args.environment,
            paused=args.paused,
        )
        alert_id = existing.get(alert.display_name or "")
        if alert_id:
            client.alerts_v2.update_alert(id=alert_id, alert=alert, update_mask=_UPDATE_MASK)
            sys.stdout.write(f"updated alert {alert.display_name}\n")
        else:
            client.alerts_v2.create_alert(alert=alert)
            sys.stdout.write(f"created alert {alert.display_name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
