"""In-memory fakes of the Databricks / OpenAI SDK surfaces used by the adapters.

They emulate the *wire behaviour* the adapters depend on (statement states,
JSON_ARRAY string rows, pagination, Vector Search response manifests, OpenAI
response models and exceptions) without any network access.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx2
import openai
from databricks.sdk.service.jobs import Run, RunLifeCycleState, RunNowResponse, RunResultState, RunState
from databricks.sdk.service.sql import (
    ColumnInfo,
    ResultData,
    ResultManifest,
    ResultSchema,
    ServiceError,
    StatementParameterListItem,
    StatementResponse,
    StatementState,
    StatementStatus,
)
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import Usage

from client_research_agent.models import (
    BriefStatement,
    ClientBrief,
    Criterion,
    CriterionScore,
    DocumentType,
    Evidence,
    FitVerdict,
    ProvenanceKind,
    QualificationResult,
)

# --------------------------------------------------------------------- SQL fake


def _to_wire(value: Any) -> str | None:
    """JSON_ARRAY disposition returns every value as a string (or null)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | dict):
        return json.dumps(value)
    return str(value)


@dataclass
class ExecutedStatement:
    statement: str
    op: str
    params: dict[str, StatementParameterListItem]
    warehouse_id: str
    kwargs: dict[str, Any]


class FakeStatementExecution:
    """Emulates ``WorkspaceClient.statement_execution`` over dict-backed Delta tables.

    Statements are dispatched on the ``cra_op`` query tag the adapter attaches.
    """

    def __init__(self, *, page_size: int = 1000) -> None:
        self.tables: dict[str, dict[str, dict[str, Any]]] = {
            "documents": {},
            "chunks": {},
            "parent_chunks": {},
            "briefs": {},
        }
        self.executed: list[ExecutedStatement] = []
        self.page_size = page_size
        self.pending_polls = 0
        self.fail_with: list[BaseException] = []
        self.final_state: StatementState | None = None
        self.cancelled: list[str] = []
        self.report_affected_rows = True
        self._pending: dict[str, tuple[int, StatementResponse]] = {}
        self._pages: dict[str, list[list[list[str | None]]]] = {}
        self._counter = 0

    # -- SDK surface -------------------------------------------------------
    def execute_statement(
        self,
        statement: str,
        warehouse_id: str,
        *,
        parameters: list[StatementParameterListItem] | None = None,
        query_tags: list[Any] | None = None,
        **kwargs: Any,
    ) -> StatementResponse:
        if self.fail_with:
            raise self.fail_with.pop(0)
        op = next((t.value for t in query_tags or [] if t.key == "cra_op"), "")
        params = {p.name: p for p in parameters or []}
        self.executed.append(ExecutedStatement(statement, op, params, warehouse_id, kwargs))
        self._counter += 1
        statement_id = f"stmt-{self._counter}"
        if self.final_state is not None:
            return StatementResponse(
                statement_id=statement_id,
                status=StatementStatus(
                    state=self.final_state, error=ServiceError(message="warehouse said no")
                ),
            )
        columns, rows = self._run(op, params)
        response = self._response(statement_id, columns, rows)
        if self.pending_polls:
            self._pending[statement_id] = (self.pending_polls, response)
            return StatementResponse(
                statement_id=statement_id, status=StatementStatus(state=StatementState.PENDING)
            )
        return response

    def get_statement(self, statement_id: str) -> StatementResponse:
        remaining, response = self._pending[statement_id]
        if remaining > 1:
            self._pending[statement_id] = (remaining - 1, response)
            return StatementResponse(
                statement_id=statement_id, status=StatementStatus(state=StatementState.RUNNING)
            )
        del self._pending[statement_id]
        return response

    def get_statement_result_chunk_n(self, statement_id: str, chunk_index: int) -> ResultData:
        pages = self._pages[statement_id]
        next_index = chunk_index + 1 if chunk_index + 1 < len(pages) else None
        return ResultData(chunk_index=chunk_index, data_array=pages[chunk_index], next_chunk_index=next_index)

    def cancel_execution(self, statement_id: str) -> None:
        self.cancelled.append(statement_id)

    # -- emulation ----------------------------------------------------------
    def _response(self, statement_id: str, columns: list[str], rows: list[list[Any]]) -> StatementResponse:
        manifest = ResultManifest(
            schema=ResultSchema(
                column_count=len(columns),
                columns=[ColumnInfo(name=c, position=i) for i, c in enumerate(columns)],
            ),
            total_row_count=len(rows),
        )
        wire = [[_to_wire(v) for v in row] for row in rows]
        pages = [wire[i : i + self.page_size] for i in range(0, len(wire), self.page_size)] or [[]]
        self._pages[statement_id] = pages
        first = ResultData(chunk_index=0, data_array=pages[0], next_chunk_index=1 if len(pages) > 1 else None)
        return StatementResponse(
            statement_id=statement_id,
            status=StatementStatus(state=StatementState.SUCCEEDED),
            manifest=manifest,
            result=first if columns else None,
        )

    @staticmethod
    def _value(params: dict[str, StatementParameterListItem], name: str) -> Any:
        return params[name].value

    def _merge(self, table: str, key: str, rows: list[dict[str, Any]], *, insert: bool = True) -> int:
        store = self.tables[table]
        touched = 0
        for row in rows:
            existing = store.get(row[key])
            if existing is None and not insert:
                continue  # MERGE without WHEN NOT MATCHED: unmatched source rows are ignored
            store[row[key]] = {**(existing or {}), **row}
            touched += 1
        return touched

    def _affected(self, count: int) -> tuple[list[str], list[list[Any]]]:
        if not self.report_affected_rows:
            return [], []
        return ["num_affected_rows", "num_inserted_rows"], [[count, count]]

    def _run(
        self, op: str, params: dict[str, StatementParameterListItem]
    ) -> tuple[list[str], list[list[Any]]]:
        chunk_cols = [
            "chunk_id", "doc_id", "text", "company", "url", "title", "document_type", "source_domain",
            "chunk_index", "strategy", "parent_id", "publication_date", "industry", "confidence",
            "token_count", "entities_json", "contextual_header", "metadata_json",
        ]  # fmt: skip
        if op == "ddl":
            return [], []
        if op == "merge_documents":
            rows = json.loads(self._value(params, "rows"))
            return self._affected(self._merge("documents", "doc_id", rows))
        if op == "merge_chunk_metadata":
            rows = json.loads(self._value(params, "rows"))
            return self._affected(self._merge("chunks", "chunk_id", rows, insert=False))
        if op == "merge_chunk_vectors":
            rows = json.loads(self._value(params, "rows"))
            # Emulate chunks' NOT NULL embedding + chunks_embedded_children_only CHECK.
            for row in rows:
                if not row.get("embedding") or row["strategy"] == "parent":
                    raise AssertionError(f"constraint violation for {row['chunk_id']}")
            return self._affected(self._merge("chunks", "chunk_id", rows))
        if op == "merge_parent_chunks":
            rows = json.loads(self._value(params, "rows"))
            assert all(r["strategy"] == "parent" and "embedding" not in r for r in rows)
            return self._affected(self._merge("parent_chunks", "chunk_id", rows))
        if op == "list_chunks":
            company = self._value(params, "company")
            found = [
                r
                for t in ("chunks", "parent_chunks")
                for r in self.tables[t].values()
                if r["company"] == company
            ]
            found.sort(key=lambda r: (r["doc_id"], r["chunk_index"], r["chunk_id"]))
            return chunk_cols, [[r.get(c) for c in chunk_cols] for r in found]
        if op in ("get_chunks", "get_parent_chunks"):
            ids = set(json.loads(self._value(params, "ids")))
            tables = ("chunks", "parent_chunks") if op == "get_chunks" else ("parent_chunks",)
            found = [r for t in tables for cid, r in self.tables[t].items() if cid in ids]
            return chunk_cols, [[r.get(c) for c in chunk_cols] for r in found]
        if op == "known_hashes":
            company = self._value(params, "company")
            hashes = sorted(
                {r["content_hash"] for r in self.tables["documents"].values() if r["company"] == company}
            )
            return ["content_hash"], [[h] for h in hashes]
        if op in ("delete_chunks", "delete_parent_chunks", "delete_documents"):
            table = op.removeprefix("delete_")
            company = self._value(params, "company")
            doomed = [k for k, r in self.tables[table].items() if r["company"] == company]
            for k in doomed:
                del self.tables[table][k]
            return self._affected(len(doomed))
        if op == "merge_brief":
            row = {name: item.value for name, item in params.items()}
            self.tables["briefs"][row["run_id"]] = row
            return self._affected(1)
        if op == "get_brief":
            row = self.tables["briefs"].get(self._value(params, "run_id"))
            return ["brief_json"], ([[row["brief_json"]]] if row else [])
        raise AssertionError(f"unexpected op {op!r}")


# ------------------------------------------------------------- Vector Search fake

_FILTER_OPS = (" >=", " <=", " >", " <", " NOT")


class FakeVectorSearchIndex:
    """Delta Sync index emulation: ``sync()`` snapshots the chunks table (rows with vectors)."""

    def __init__(self, sql: FakeStatementExecution, *, name: str = "cat.sch.chunks_index") -> None:
        self.name = name
        self._sql = sql
        self._snapshot: dict[str, dict[str, Any]] = {}
        self.sync_calls = 0
        self.wait_calls: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []
        self.fail_with: list[BaseException] = []
        self.response_override: Any = None

    def sync(self) -> None:
        if self.fail_with:
            raise self.fail_with.pop(0)
        self.sync_calls += 1
        self._snapshot = {
            cid: dict(row)
            for cid, row in self._sql.tables["chunks"].items()
            if row.get("embedding") is not None
        }

    def wait_until_ready(self, **kwargs: Any) -> None:
        self.wait_calls.append(kwargs)

    @staticmethod
    def _matches(row: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            column, op = key, "="
            for suffix in _FILTER_OPS:
                if key.endswith(suffix):
                    column, op = key[: -len(suffix)], suffix.strip()
                    break
            actual = row.get(column)
            if isinstance(expected, list):
                hit = actual in expected
                if (op == "=" and not hit) or (op == "NOT" and hit):
                    return False
                continue
            if op == "=" and actual != expected:
                return False
            if op == "NOT" and actual == expected:
                return False
            if op in (">=", "<=", ">", "<"):
                if actual is None:
                    return False
                ok = {
                    ">=": actual >= expected,
                    "<=": actual <= expected,
                    ">": actual > expected,
                    "<": actual < expected,
                }[op]
                if not ok:
                    return False
        return True

    def similarity_search(self, columns: list[str], **kwargs: Any) -> Any:
        self.queries.append({"columns": columns, **kwargs})
        if self.fail_with:
            raise self.fail_with.pop(0)
        if self.response_override is not None:
            return self.response_override
        query = kwargs["query_vector"]
        filters = kwargs.get("filters") or {}
        qn = math.sqrt(sum(v * v for v in query)) or 1.0
        scored = []
        for row in self._snapshot.values():
            if not self._matches(row, filters):
                continue
            vec = row["embedding"]
            vn = math.sqrt(sum(v * v for v in vec)) or 1.0
            score = sum(a * b for a, b in zip(query, vec, strict=True)) / (qn * vn)
            scored.append((score, row["chunk_id"], row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[: kwargs["num_results"]]
        return {
            "manifest": {
                "column_count": len(columns) + 1,
                "columns": [{"name": c} for c in [*columns, "score"]],
            },
            "result": {
                "row_count": len(top),
                "data_array": [[row.get(c) for c in columns] + [score] for score, _, row in top],
            },
        }


class FakeVectorSearchClient:
    """Provisioning surface of ``VectorSearchClient``."""

    def __init__(self, *, endpoints: set[str] | None = None, indexes: set[str] | None = None) -> None:
        self.endpoints = set(endpoints or ())
        self.indexes = set(indexes or ())
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.index_objects: dict[str, Any] = {}

    def endpoint_exists(self, name: str) -> bool:
        return name in self.endpoints

    def create_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("create_endpoint", kwargs))
        self.endpoints.add(kwargs["name"])
        return {}

    def create_endpoint_and_wait(self, **kwargs: Any) -> None:
        self.calls.append(("create_endpoint_and_wait", kwargs))
        self.endpoints.add(kwargs["name"])

    def index_exists(self, endpoint_name: str, index_name: str) -> bool:
        return index_name in self.indexes

    def create_delta_sync_index(self, **kwargs: Any) -> Any:
        self.calls.append(("create_delta_sync_index", kwargs))
        self.indexes.add(kwargs["index_name"])

    def create_delta_sync_index_and_wait(self, **kwargs: Any) -> Any:
        self.calls.append(("create_delta_sync_index_and_wait", kwargs))
        self.indexes.add(kwargs["index_name"])

    def get_index(self, endpoint_name: str, index_name: str) -> Any:
        self.calls.append(("get_index", {"endpoint_name": endpoint_name, "index_name": index_name}))
        return self.index_objects[index_name]


# --------------------------------------------------------------- OpenAI fakes

_REQUEST = httpx2.Request("POST", "https://example.cloud.databricks.com/serving-endpoints/chat/completions")


def status_error(cls: type[openai.APIStatusError], status: int, headers: dict[str, str] | None = None) -> Any:
    response = httpx2.Response(status, request=_REQUEST, headers=headers or {})
    return cls(f"HTTP {status}", response=response, body=None)


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=_REQUEST)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


def chat_completion(
    content: str | None, *, model: str = "endpoint", prompt: int = 11, completion: int = 7
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=int(datetime(2026, 1, 1, tzinfo=UTC).timestamp()),
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
    )


Script = list[Any]


class _Completions:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.chat_requests.append(kwargs)
        return self._owner.next_item(
            self._owner.chat_script, lambda: chat_completion("{}", model=kwargs["model"])
        )


class _Chat:
    def __init__(self, owner: FakeOpenAI) -> None:
        self.completions = _Completions(owner)


class _Embeddings:
    def __init__(self, owner: FakeOpenAI) -> None:
        self._owner = owner

    def create(self, *, model: str, input: list[str]) -> Any:
        self._owner.embedding_requests.append({"model": model, "input": list(input)})
        return self._owner.next_item(
            self._owner.embedding_script, lambda: self._owner.embed_default(model, input)
        )


@dataclass
class FakeOpenAI:
    """Stands in for ``openai.OpenAI``; each script entry is a response or an exception to raise."""

    chat_script: Script = field(default_factory=list)
    embedding_script: Script = field(default_factory=list)
    dimension: int = 8
    reverse_embedding_order: bool = False
    chat_requests: list[dict[str, Any]] = field(default_factory=list)
    embedding_requests: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    @staticmethod
    def next_item(script: Script, default: Callable[[], Any]) -> Any:
        item = script.pop(0) if script else default()
        if isinstance(item, BaseException):
            raise item
        return item

    def vector_for(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for i, ch in enumerate(text):
            vector[(ord(ch) + i) % self.dimension] += 1.0
        return vector

    def embed_default(self, model: str, inputs: list[str]) -> CreateEmbeddingResponse:
        data = [
            Embedding(object="embedding", index=i, embedding=self.vector_for(t)) for i, t in enumerate(inputs)
        ]
        if self.reverse_embedding_order:
            data.reverse()
        return CreateEmbeddingResponse(
            object="list",
            model=model,
            data=data,
            usage=Usage(prompt_tokens=len(inputs), total_tokens=len(inputs)),
        )


# ------------------------------------------------------------------- Jobs fake


@dataclass
class _Waiter:
    response: RunNowResponse
    run_id: int


class FakeJobsAPI:
    def __init__(self, states: list[tuple[RunLifeCycleState, RunResultState | None]]) -> None:
        self.states = list(states)
        self.run_now_calls: list[dict[str, Any]] = []
        self.cancelled: list[int] = []
        self.fail_with: list[BaseException] = []

    def run_now(self, **kwargs: Any) -> _Waiter:
        if self.fail_with:
            raise self.fail_with.pop(0)
        self.run_now_calls.append(kwargs)
        return _Waiter(response=RunNowResponse(run_id=4242, number_in_job=4242), run_id=4242)

    def get_run(self, run_id: int) -> Run:
        if self.fail_with:
            raise self.fail_with.pop(0)
        life, result = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return Run(
            run_id=run_id,
            run_page_url=f"https://example.cloud.databricks.com/jobs/runs/{run_id}",
            state=RunState(life_cycle_state=life, result_state=result, state_message=f"{life.value}"),
        )

    def cancel_run(self, run_id: int) -> None:
        if self.fail_with:
            raise self.fail_with.pop(0)
        self.cancelled.append(run_id)


# ---------------------------------------------------------------- brief factory


def make_brief(run_id: str = "run-001", company: str = "Acme Corp") -> ClientBrief:
    evidence = Evidence(
        evidence_id="ev-1",
        chunk_id="c1",
        url="https://acme.example.com/news/doc-1",
        title="Acme expands data platform",
        quote="Acme is migrating its analytics estate to a lakehouse.",
        document_type=DocumentType.PRESS_RELEASE,
    )
    fact = BriefStatement(
        text="Acme is migrating to a lakehouse.",
        provenance=ProvenanceKind.VERIFIED_FACT,
        evidence_ids=("ev-1",),
    )
    idea = BriefStatement(text="Propose a governance workshop.", provenance=ProvenanceKind.AI_RECOMMENDATION)
    score = CriterionScore(
        criterion=Criterion.AI_DATA_FOCUS,
        score=4,
        weight=0.25,
        confidence=0.8,
        rationale="Lakehouse migration",
        evidence_ids=("ev-1",),
    )
    return ClientBrief(
        run_id=run_id,
        company=company,
        company_overview=(fact,),
        qualification=QualificationResult(
            scores=(score,),
            weighted_score=3.9,
            overall_confidence=0.8,
            verdict=FitVerdict.GOOD_FIT,
            verdict_rationale="Strong data focus",
        ),
        evidence=(evidence,),
        technology_priorities=(fact,),
        gartner_relevant_insights=(),
        opportunities=(idea,),
        risks=(),
        executive_summary=(fact,),
        discovery_questions=("Q1?", "Q2?", "Q3?", "Q4?", "Q5?"),
        executive_talking_points=(idea,),
        recommended_next_actions=(idea,),
        model_versions={"chat": "databricks-claude-sonnet-4"},
    )
