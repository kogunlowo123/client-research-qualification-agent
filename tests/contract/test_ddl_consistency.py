"""The platform DDL must match the Databricks adapter's canonical table schemas.

``infrastructure/unity_catalog/*.sql`` bootstraps a workspace (and adds
platform-owned tables); ``client_research_agent.databricks.unity_catalog``
reads and writes ``documents``, ``chunks``, ``parent_chunks``, ``briefs`` and
``audit_log`` with MERGE statements. If the two drift, the adapter's writes fail
against tables created by the platform DDL, so every shared table is compared
column by column (name, type, nullability, order), together with CHECK
constraints, Change Data Feed and the Vector Search index specification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.config.settings import ModelServingSettings, VectorSearchSettings
from client_research_agent.databricks.unity_catalog import (
    CHECK_CONSTRAINTS,
    CHUNK_COLUMNS,
    TABLES,
    render_ddl,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
UC_DIR = REPO_ROOT / "infrastructure" / "unity_catalog"
INDEX_SPEC = REPO_ROOT / "infrastructure" / "vector_search" / "index_spec.json"
TERRAFORM_INDEX = REPO_ROOT / "deployment" / "terraform" / "vector_search.tf"
CATALOG = "client_research"
SCHEMA = "agent_contract"

Column = tuple[str, str, bool]  # (name, normalised type, NOT NULL)

_CREATE = re.compile(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+(\S+)\s*\(", re.IGNORECASE)
_ADD_CHECK = re.compile(
    r"ALTER\s+TABLE\s+(\S+)\s+ADD\s+CONSTRAINT\s+(\w+)\s+CHECK\s*\((.*)\)\s*$", re.IGNORECASE | re.DOTALL
)


def _render_sql_file(name: str) -> str:
    text = (UC_DIR / name).read_text(encoding="utf-8")
    return text.replace("${catalog}", CATALOG).replace("${schema}", SCHEMA)


def _statements(sql: str) -> list[str]:
    body = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [part.strip() for part in re.split(r";\s*(?:\n|$)", body) if part.strip()]


def _table_name(qualified: str) -> str:
    return qualified.replace("`", "").split(".")[-1]


def _column_block(statement: str) -> tuple[str, str]:
    """Return (table, text between the column-list parentheses)."""
    match = _CREATE.search(statement)
    assert match is not None, statement[:120]
    depth, start = 1, match.end()
    in_quote = False
    for index in range(start, len(statement)):
        char = statement[index]
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char == "(":
            depth += 1
        elif not in_quote and char == ")":
            depth -= 1
            if depth == 0:
                return _table_name(match.group(1)), statement[start:index]
    raise AssertionError(f"unbalanced parentheses in: {statement[:120]}")


def _split_top_level(block: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth, in_quote = 0, False
    for char in block:
        if char == "'":
            in_quote = not in_quote
        elif not in_quote and char in "(<":
            depth += 1
        elif not in_quote and char in ")>":
            depth -= 1
        if char == "," and depth == 0 and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def _normalise_type(raw: str) -> str:
    return re.sub(r"\s+", "", raw).upper()


def _parse_column(definition: str) -> Column:
    without_comment = re.sub(r"\s+COMMENT\s+'(?:[^']|'')*'\s*$", "", definition, flags=re.IGNORECASE)
    not_null = bool(re.search(r"\s+NOT\s+NULL\s*$", without_comment, flags=re.IGNORECASE))
    without_null = re.sub(r"\s+NOT\s+NULL\s*$", "", without_comment, flags=re.IGNORECASE)
    name, _, column_type = without_null.strip().partition(" ")
    return name.strip("`"), _normalise_type(column_type), not_null


def _tables(statements: list[str]) -> dict[str, list[Column]]:
    tables: dict[str, list[Column]] = {}
    for statement in statements:
        if not _CREATE.search(statement):
            continue
        table, block = _column_block(statement)
        columns = [
            _parse_column(item)
            for item in _split_top_level(block)
            if not item.upper().startswith("CONSTRAINT")
        ]
        tables[table] = columns
    return tables


def _create_statement(statements: list[str], table: str) -> str:
    for statement in statements:
        match = _CREATE.search(statement)
        if match and _table_name(match.group(1)) == table:
            return statement
    raise AssertionError(f"no CREATE TABLE for {table}")


def _checks(statements: list[str]) -> dict[tuple[str, str], str]:
    checks: dict[tuple[str, str], str] = {}
    for statement in statements:
        match = _ADD_CHECK.search(statement)
        if match:
            key = (_table_name(match.group(1)), match.group(2))
            checks[key] = re.sub(r"\s+", " ", match.group(3)).strip()
    return checks


@pytest.fixture(scope="module")
def platform_statements() -> list[str]:
    statements: list[str] = []
    for path in sorted(UC_DIR.glob("[0-9][0-9]_*.sql")):
        statements.extend(_statements(_render_sql_file(path.name)))
    return statements


@pytest.fixture(scope="module")
def adapter_statements() -> list[str]:
    return render_ddl(CATALOG, SCHEMA)


@pytest.fixture(scope="module")
def index_spec() -> dict[str, Any]:
    spec: dict[str, dict[str, Any]] = json.loads(INDEX_SPEC.read_text(encoding="utf-8"))
    return spec["index"]


def test_parser_reads_every_adapter_table(adapter_statements: list[str]) -> None:
    tables = _tables(adapter_statements)
    assert set(tables) == set(TABLES)
    assert all(tables[name] for name in TABLES)


@pytest.mark.parametrize("table", TABLES)
def test_platform_ddl_matches_adapter_columns(
    table: str, platform_statements: list[str], adapter_statements: list[str]
) -> None:
    platform = _tables(platform_statements)
    adapter = _tables(adapter_statements)
    assert table in platform, f"infrastructure/unity_catalog is missing table {table}"
    assert platform[table] == adapter[table], (
        f"{table}: platform DDL columns differ from the adapter's canonical DDL\n"
        f"platform: {platform[table]}\nadapter:  {adapter[table]}"
    )


@pytest.mark.parametrize("table", TABLES)
def test_primary_keys_match(
    table: str, platform_statements: list[str], adapter_statements: list[str]
) -> None:
    def primary_key(statements: list[str]) -> str:
        found = re.search(r"PRIMARY\s+KEY\s*\(([^)]*)\)", _create_statement(statements, table), re.IGNORECASE)
        assert found is not None
        return re.sub(r"\s+", "", found.group(1))

    assert primary_key(platform_statements) == primary_key(adapter_statements)


@pytest.mark.parametrize("table", TABLES)
def test_change_data_feed_matches(
    table: str, platform_statements: list[str], adapter_statements: list[str]
) -> None:
    def cdf(statements: list[str]) -> bool:
        return "'delta.enableChangeDataFeed' = 'true'" in _create_statement(statements, table)

    assert cdf(platform_statements) == cdf(adapter_statements)


def test_chunks_is_a_change_data_feed_source(platform_statements: list[str]) -> None:
    assert "'delta.enableChangeDataFeed' = 'true'" in _create_statement(platform_statements, "chunks")


def test_check_constraints_on_adapter_tables_match(
    platform_statements: list[str], adapter_statements: list[str]
) -> None:
    platform = {key: value for key, value in _checks(platform_statements).items() if key[0] in TABLES}
    adapter = _checks(adapter_statements)
    expected = {
        (table, name): re.sub(r"\s+", " ", predicate).strip()
        for table, constraints in CHECK_CONSTRAINTS.items()
        for name, predicate in constraints.items()
    }
    assert adapter == expected
    assert platform == expected


def test_index_spec_columns_match_chunk_columns(index_spec: dict[str, Any]) -> None:
    assert index_spec["columns_to_sync"] == list(CHUNK_COLUMNS)


def test_index_spec_matches_vector_search_settings(index_spec: dict[str, Any]) -> None:
    settings = VectorSearchSettings()
    assert index_spec["primary_key"] == settings.primary_key
    assert index_spec["embedding_vector_column"]["name"] == settings.embedding_column
    assert index_spec["embedding_vector_column"]["embedding_dimension"] == (
        ModelServingSettings().embedding_dimension
    )
    assert index_spec["pipeline_type"] == settings.pipeline_type
    assert index_spec["name_template"].endswith(f".{settings.index_name}")
    assert index_spec["source_table_template"].endswith(f".{settings.source_table}")


def test_index_columns_exist_in_chunks_table(
    index_spec: dict[str, Any], adapter_statements: list[str]
) -> None:
    columns = {
        name: (column_type, not_null) for name, column_type, not_null in _tables(adapter_statements)["chunks"]
    }
    for name in index_spec["columns_to_sync"]:
        assert name in columns, f"columns_to_sync references missing chunks column {name}"
    embedding = index_spec["embedding_vector_column"]["name"]
    assert columns[embedding] == ("ARRAY<FLOAT>", True)
    assert columns[index_spec["primary_key"]][1] is True


def test_terraform_index_reads_the_shared_spec() -> None:
    text = TERRAFORM_INDEX.read_text(encoding="utf-8")
    assert "infrastructure/vector_search/index_spec.json" in text
    assert "local.index_spec.primary_key" in text
    assert "local.index_spec.columns_to_sync" in text
    assert "local.index_spec.embedding_vector_column.name" in text
