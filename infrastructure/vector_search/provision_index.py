"""Idempotently provision the Mosaic AI Vector Search endpoint and chunks_index.

Creates the endpoint (``cra-vs-endpoint-<env>``) and the Delta Sync index
(``<catalog>.<schema>.chunks_index``, self-managed ``embedding`` column) when
they do not exist, verifies an existing index matches ``index_spec.json`` and
optionally triggers a sync. Re-running against a matching index is a no-op.

    python infrastructure/vector_search/provision_index.py --environment dev --sync

Authentication follows Databricks unified auth; pass --profile to use a CLI profile.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingVectorColumn,
    EndpointInfo,
    EndpointStatusState,
    EndpointType,
    PipelineType,
    VectorIndex,
    VectorIndexType,
)

DEFAULT_SPEC = Path(__file__).resolve().parent / "index_spec.json"


@dataclass(frozen=True)
class IndexPlan:
    endpoint_name: str
    endpoint_type: EndpointType
    index_name: str
    source_table: str
    primary_key: str
    pipeline_type: PipelineType
    embedding_column: str
    embedding_dimension: int
    columns_to_sync: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provision the Client Research Agent vector index.")
    parser.add_argument("--environment", required=True, choices=["dev", "staging", "prod"])
    parser.add_argument("--catalog", default="client_research")
    parser.add_argument("--schema", help="Defaults to agent_<environment>.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--profile", help="Databricks CLI profile.")
    parser.add_argument("--sync", action="store_true", help="Trigger a sync once the index is ready.")
    parser.add_argument(
        "--recreate", action="store_true", help="Drop and recreate an index whose spec has drifted."
    )
    parser.add_argument("--timeout-minutes", type=int, default=60)
    return parser.parse_args(argv)


def load_plan(spec_path: Path, environment: str, catalog: str, schema: str) -> IndexPlan:
    spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
    names = {"environment": environment, "catalog": catalog, "schema": schema}
    endpoint, index = spec["endpoint"], spec["index"]
    return IndexPlan(
        endpoint_name=endpoint["name_template"].format(**names),
        endpoint_type=EndpointType(endpoint["endpoint_type"]),
        index_name=index["name_template"].format(**names),
        source_table=index["source_table_template"].format(**names),
        primary_key=index["primary_key"],
        pipeline_type=PipelineType(index["pipeline_type"]),
        embedding_column=index["embedding_vector_column"]["name"],
        embedding_dimension=int(index["embedding_vector_column"]["embedding_dimension"]),
        columns_to_sync=tuple(index["columns_to_sync"]),
    )


def log(message: str) -> None:
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def ensure_endpoint(client: WorkspaceClient, plan: IndexPlan, timeout_s: float) -> EndpointInfo:
    try:
        endpoint = client.vector_search_endpoints.get_endpoint(plan.endpoint_name)
        log(f"endpoint {plan.endpoint_name} exists")
    except NotFound:
        log(f"creating endpoint {plan.endpoint_name} ({plan.endpoint_type.value})")
        return client.vector_search_endpoints.create_endpoint_and_wait(
            name=plan.endpoint_name,
            endpoint_type=plan.endpoint_type,
            timeout=timedelta(seconds=timeout_s),
        )
    state = endpoint.endpoint_status.state if endpoint.endpoint_status else None
    if state is EndpointStatusState.PROVISIONING:
        log("waiting for endpoint to come online")
        return client.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
            plan.endpoint_name, timeout=timedelta(seconds=timeout_s)
        )
    if state is not EndpointStatusState.ONLINE:
        raise RuntimeError(f"endpoint {plan.endpoint_name} is {state}; resolve before provisioning")
    return endpoint


def drift(index: VectorIndex, plan: IndexPlan) -> list[str]:
    problems: list[str] = []
    spec = index.delta_sync_index_spec
    if index.index_type is not VectorIndexType.DELTA_SYNC or spec is None:
        return [f"index_type={index.index_type} (expected DELTA_SYNC)"]
    if index.endpoint_name != plan.endpoint_name:
        problems.append(f"endpoint_name={index.endpoint_name}")
    if index.primary_key != plan.primary_key:
        problems.append(f"primary_key={index.primary_key}")
    if spec.source_table != plan.source_table:
        problems.append(f"source_table={spec.source_table}")
    if spec.pipeline_type is not plan.pipeline_type:
        problems.append(f"pipeline_type={spec.pipeline_type}")
    columns = {c.name: c.embedding_dimension for c in spec.embedding_vector_columns or []}
    if columns.get(plan.embedding_column) != plan.embedding_dimension:
        problems.append(f"embedding_vector_columns={columns}")
    return problems


def create_index(client: WorkspaceClient, plan: IndexPlan) -> VectorIndex:
    log(f"creating index {plan.index_name} on {plan.source_table}")
    return client.vector_search_indexes.create_index(
        name=plan.index_name,
        endpoint_name=plan.endpoint_name,
        primary_key=plan.primary_key,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=plan.source_table,
            pipeline_type=plan.pipeline_type,
            embedding_vector_columns=[
                EmbeddingVectorColumn(
                    name=plan.embedding_column, embedding_dimension=plan.embedding_dimension
                )
            ],
            columns_to_sync=list(plan.columns_to_sync),
        ),
    )


def ensure_index(client: WorkspaceClient, plan: IndexPlan, recreate: bool) -> VectorIndex:
    try:
        index = client.vector_search_indexes.get_index(plan.index_name)
    except NotFound:
        return create_index(client, plan)
    problems = drift(index, plan)
    if not problems:
        log(f"index {plan.index_name} exists and matches spec")
        return index
    if not recreate:
        raise RuntimeError(f"index {plan.index_name} drifted from spec: {problems}; rerun with --recreate")
    log(f"recreating drifted index {plan.index_name}: {problems}")
    client.vector_search_indexes.delete_index(plan.index_name)
    return create_index(client, plan)


def wait_ready(client: WorkspaceClient, index_name: str, timeout_s: float) -> VectorIndex:
    deadline = time.monotonic() + timeout_s
    while True:
        index = client.vector_search_indexes.get_index(index_name)
        if index.status and index.status.ready:
            rows = index.status.indexed_row_count
            log(f"index {index_name} ready ({rows} rows)")
            return index
        if time.monotonic() > deadline:
            message = index.status.message if index.status else "no status"
            raise TimeoutError(f"index {index_name} not ready after {timeout_s:.0f}s: {message}")
        time.sleep(20)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    schema = args.schema or f"agent_{args.environment}"
    plan = load_plan(args.spec, args.environment, args.catalog, schema)
    timeout_s = float(args.timeout_minutes * 60)
    client = WorkspaceClient(profile=args.profile) if args.profile else WorkspaceClient()

    ensure_endpoint(client, plan, timeout_s)
    ensure_index(client, plan, args.recreate)
    if args.sync:
        wait_ready(client, plan.index_name, timeout_s)
        if plan.pipeline_type is PipelineType.TRIGGERED:
            client.vector_search_indexes.sync_index(plan.index_name)
            log(f"sync triggered for {plan.index_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
