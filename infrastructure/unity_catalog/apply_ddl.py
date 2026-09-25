"""Apply the Unity Catalog DDL in this directory to a Databricks workspace.

Renders ``${catalog}``, ``${schema}``, ``${environment}``, ``${engineers_group}``,
``${analysts_group}`` and ``${agent_sp}`` into every ``NN_*.sql`` file (in
lexical order) and executes each statement on a SQL warehouse through the
Statement Execution API. Every statement is idempotent, so the script can be
re-run safely.

Authentication follows Databricks unified auth (DATABRICKS_HOST plus OAuth M2M,
Azure CLI, GitHub OIDC or a CLI profile via --profile).

    python infrastructure/unity_catalog/apply_ddl.py --environment dev \
        --agent-sp 00000000-0000-0000-0000-000000000000
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementResponse, StatementState

SQL_DIR = Path(__file__).resolve().parent
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
_PRINCIPAL = re.compile(r"^[A-Za-z0-9_.@+\- ]{1,255}$")
_PLACEHOLDER = re.compile(r"\$\{([a-z_]+)\}")
_TERMINAL = {StatementState.SUCCEEDED, StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--catalog", default="client_research")
    parser.add_argument("--schema", help="Defaults to agent_<environment>.")
    parser.add_argument("--warehouse-id", help="Defaults to the warehouse named cra-sql-<environment>.")
    parser.add_argument("--engineers-group", default="cra-engineers")
    parser.add_argument("--analysts-group", default="cra-analysts")
    parser.add_argument("--agent-sp", required=True, help="Application ID of the agent service principal.")
    parser.add_argument("--profile", help="Databricks CLI profile to authenticate with.")
    parser.add_argument("--only", nargs="*", default=None, help="Apply only these files, e.g. 01_tables.sql.")
    parser.add_argument("--dry-run", action="store_true", help="Print rendered SQL without executing it.")
    return parser.parse_args(argv)


def build_context(args: argparse.Namespace) -> dict[str, str]:
    schema = args.schema or f"agent_{args.environment}"
    for name, value in (("catalog", args.catalog), ("schema", schema)):
        if not _IDENTIFIER.match(value):
            raise SystemExit(f"invalid {name} identifier: {value!r}")
    for name, value in (
        ("engineers-group", args.engineers_group),
        ("analysts-group", args.analysts_group),
        ("agent-sp", args.agent_sp),
    ):
        if not _PRINCIPAL.match(value):
            raise SystemExit(f"invalid {name} principal: {value!r}")
    return {
        "catalog": args.catalog,
        "schema": schema,
        "environment": args.environment,
        "engineers_group": args.engineers_group,
        "analysts_group": args.analysts_group,
        "agent_sp": args.agent_sp,
    }


def render(sql: str, context: dict[str, str]) -> str:
    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise KeyError(f"unknown template variable ${{{key}}}")
        return context[key]

    return _PLACEHOLDER.sub(substitute, sql)


def split_statements(sql: str) -> list[str]:
    """Split on semicolons that end a line; comments are dropped."""
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [stmt.strip() for stmt in re.split(r";\s*(?:\n|$)", body) if stmt.strip()]


def sql_files(only: list[str] | None) -> list[Path]:
    files = sorted(SQL_DIR.glob("[0-9][0-9]_*.sql"))
    if only:
        wanted = set(only)
        files = [f for f in files if f.name in wanted]
        missing = wanted - {f.name for f in files}
        if missing:
            raise SystemExit(f"unknown SQL files: {sorted(missing)}")
    return files


def resolve_warehouse(client: WorkspaceClient, warehouse_id: str | None, environment: str) -> str:
    if warehouse_id:
        return warehouse_id
    name = f"cra-sql-{environment}"
    for warehouse in client.warehouses.list():
        if warehouse.name == name and warehouse.id:
            return warehouse.id
    raise SystemExit(f"SQL warehouse {name!r} not found; pass --warehouse-id")


def execute(client: WorkspaceClient, warehouse_id: str, statement: str) -> None:
    response: StatementResponse = client.statement_execution.execute_statement(
        statement=statement, warehouse_id=warehouse_id, wait_timeout="50s"
    )
    while response.status and response.status.state not in _TERMINAL:
        time.sleep(2)
        response = client.statement_execution.get_statement(response.statement_id or "")
    state = response.status.state if response.status else None
    if state is not StatementState.SUCCEEDED:
        error = response.status.error.message if response.status and response.status.error else "unknown"
        raise RuntimeError(f"statement failed ({state}): {error}\n{statement}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = build_context(args)
    files = sql_files(args.only)
    if args.dry_run:
        for path in files:
            sys.stdout.write(f"-- {path.name}\n")
            for statement in split_statements(render(path.read_text(encoding="utf-8"), context)):
                sys.stdout.write(f"{statement};\n\n")
        return 0

    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()
    warehouse_id = resolve_warehouse(client, args.warehouse_id, args.environment)
    for path in files:
        statements = split_statements(render(path.read_text(encoding="utf-8"), context))
        sys.stdout.write(f"applying {path.name} ({len(statements)} statements)\n")
        for statement in statements:
            execute(client, warehouse_id, statement)
    sys.stdout.write(f"applied {len(files)} files to {context['catalog']}.{context['schema']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
