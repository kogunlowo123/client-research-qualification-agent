from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from client_research_agent.config.settings import ResilienceSettings
from client_research_agent.databricks.unity_catalog import (
    DeltaBriefRepository,
    DeltaDocumentStore,
    StatementExecutor,
)
from client_research_agent.databricks.vector_search import DatabricksVectorIndex
from client_research_agent.models import Chunk, ChunkStrategy
from client_research_agent.services.local import (
    InMemoryDocumentStore,
    InMemoryVectorIndex,
    JsonlBriefRepository,
)
from client_research_agent.services.ports import BriefRepository, DocumentStore, VectorIndex
from tests.contract.fakes import FakeStatementExecution, FakeVectorSearchIndex

FAST = ResilienceSettings(max_attempts=2, initial_backoff_seconds=0.0, max_backoff_seconds=0.0)


def _executor(sql: FakeStatementExecution) -> StatementExecutor:
    return StatementExecutor(sql, "wh-123", resilience=FAST, sleep=lambda _s: None)


@pytest.fixture(params=["local", "databricks"])
def vector_index(request: pytest.FixtureRequest) -> Iterator[VectorIndex]:
    if request.param == "local":
        yield InMemoryVectorIndex()
        return
    sql = FakeStatementExecution()
    store = DeltaDocumentStore(_executor(sql), catalog="cat", schema="sch")
    yield DatabricksVectorIndex(FakeVectorSearchIndex(sql), store, resilience=FAST, sleep=lambda _s: None)


@dataclass(frozen=True)
class StoreAndIndex:
    """A document store plus the vector index it feeds, wired as production wires them."""

    store: DocumentStore
    index: VectorIndex

    def ingest(self, chunks: Sequence[Chunk]) -> None:
        """Mirror ``IndexingPipeline``: persist every chunk, then embed and index the children."""
        self.store.save_chunks(chunks)
        children = [c for c in chunks if c.strategy is not ChunkStrategy.PARENT]
        if children:
            self.index.upsert(children, [[1.0, float(i)] for i, _ in enumerate(children)])


@pytest.fixture(params=["local", "databricks"])
def store_and_index(request: pytest.FixtureRequest) -> StoreAndIndex:
    if request.param == "local":
        return StoreAndIndex(InMemoryDocumentStore(), InMemoryVectorIndex())
    # Page size 2 forces the adapter through result-chunk pagination.
    sql = FakeStatementExecution(page_size=2)
    store = DeltaDocumentStore(_executor(sql), catalog="cat", schema="sch")
    index = DatabricksVectorIndex(FakeVectorSearchIndex(sql), store, resilience=FAST, sleep=lambda _s: None)
    return StoreAndIndex(store, index)


@pytest.fixture
def document_store(store_and_index: StoreAndIndex) -> DocumentStore:
    return store_and_index.store


@pytest.fixture(params=["local", "databricks"])
def brief_repository(request: pytest.FixtureRequest, tmp_path: Path) -> BriefRepository:
    if request.param == "local":
        return JsonlBriefRepository(tmp_path / "briefs")
    return DeltaBriefRepository(_executor(FakeStatementExecution()), catalog="cat", schema="sch")


_CONTRACT_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Every test under tests/contract carries the ``contract`` marker."""
    for item in items:
        if _CONTRACT_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.contract)
