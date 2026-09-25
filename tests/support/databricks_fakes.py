"""Workspace-level fakes for Databricks-environment wiring tests (no network)."""

from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from databricks.sdk.service.sql import StatementParameterListItem

from tests.contract.fakes import FakeStatementExecution


class PlatformStatementExecution(FakeStatementExecution):
    """Adds the platform-owned statements (audit, lineage, watchlist, evaluation) to the SQL fake."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.platform_rows: dict[str, list[dict[str, Any]]] = {}
        self.watchlist: list[dict[str, Any]] = []
        self.eval_rows: list[dict[str, Any]] = []

    def _run(
        self, op: str, params: dict[str, StatementParameterListItem]
    ) -> tuple[list[str], list[list[Any]]]:
        values = {name: item.value for name, item in params.items()}
        if op in ("insert_audit", "merge_lineage", "insert_eval_result", "mark_ingested"):
            self.platform_rows.setdefault(op, []).append(values)
            return self._affected(1)
        if op == "read_watchlist":
            columns = ["company_name", "domain", "ticker", "cik", "industry"]
            return columns, [[row.get(c) for c in columns] for row in self.watchlist]
        if op == "read_eval_set":
            columns = sorted({k for row in self.eval_rows for k in row})
            return columns, [[row.get(c) for c in columns] for row in self.eval_rows]
        return super()._run(op, params)


@dataclass
class FakeWarehouse:
    id: str
    name: str
    enable_serverless_compute: bool = True


@dataclass
class FakeWorkspace:
    statement_execution: Any = field(default_factory=PlatformStatementExecution)
    warehouse_list: list[FakeWarehouse] = field(
        default_factory=lambda: [FakeWarehouse("wh-dev", "cra-sql-dev")]
    )
    host: str = "https://adb-1.azuredatabricks.net"

    def __post_init__(self) -> None:
        self.config = types.SimpleNamespace(
            host=self.host,
            authenticate=lambda: {"Authorization": "Bearer test-token"},
            auth_type="oauth-m2m",
            client_id=None,
            client_secret=None,
            token=None,
        )
        self.warehouses = types.SimpleNamespace(list=lambda: list(self.warehouse_list))


class FakeExecutor:
    """Minimal ``StatementExecutor`` stand-in: records statements, returns canned rows per op."""

    def __init__(self, rows: Mapping[str, list[dict[str, Any]]] | None = None, *, fail: bool = False) -> None:
        self.rows = dict(rows or {})
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    def execute(
        self, statement: str, params: Mapping[str, Any] | None = None, *, op: str
    ) -> list[dict[str, Any]]:
        if self.fail:
            raise RuntimeError("warehouse unavailable")
        self.calls.append((statement, dict(params or {}), op))
        return list(self.rows.get(op, []))
