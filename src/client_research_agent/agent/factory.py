"""Composition root: wires every port to an adapter for the configured environment.

``build_runtime(settings)`` returns an :class:`AgentRuntime` holding every
component the orchestrator, the jobs, the CLI and the serving agent need.

Local (``Environment.LOCAL``)
    Fully offline-capable: :class:`HashingEmbeddingClient` (512-d),
    :class:`InMemoryVectorIndex`, :class:`InMemoryDocumentStore`,
    :class:`JsonlBriefRepository` under ``<var>/briefs`` and a hash-chained
    :class:`AuditLogger` at ``<var>/audit/audit.jsonl``. The LLM is the Databricks
    Foundation Model API (primary + fallback endpoint) only when a workspace host
    is configured (``DATABRICKS_HOST`` or ``databricks.host``); otherwise it is
    ``None`` and every agent answers with its deterministic fallback.

Dev / staging / prod
    Databricks adapters throughout: a unified-auth ``WorkspaceClient``, chat and
    embedding clients over Model Serving, :class:`DeltaDocumentStore` and
    :class:`DeltaBriefRepository` over a SQL warehouse, and the Delta Sync
    :class:`DatabricksVectorIndex`.

The public web is always reached through :class:`PolicyEnforcingFetcher`
(SSRF guard, robots.txt, rate limiting, retries, per-host breakers). Every
component can be injected, which is how tests bind doubles.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

import httpx

from client_research_agent import __version__
from client_research_agent.agent.metering import MeteredLLMClient
from client_research_agent.config.settings import AppSettings, Environment
from client_research_agent.governance.audit import AuditLogger
from client_research_agent.governance.responsible_ai import ResponsibleAIPolicy
from client_research_agent.models import Chunk, ResearchRequest
from client_research_agent.observability.logging import get_logger
from client_research_agent.orchestration.review import FeedbackStore, ReviewPolicy, ReviewQueue
from client_research_agent.prompts.registry import PromptRegistry, default_registry
from client_research_agent.research import IngestionResult, build_ingestion_pipeline
from client_research_agent.research.fetcher import PolicyEnforcingFetcher
from client_research_agent.research.ingestion import IngestionPipeline
from client_research_agent.research.parsing import (
    DocumentTypeClassifier,
    EntityExtractor,
    HtmlParser,
)
from client_research_agent.research.sources import CorporateSiteDiscoverer, PublicAnalystSource
from client_research_agent.retrieval.embeddings import HashingEmbeddingClient
from client_research_agent.retrieval.enrichment import MetadataEnricher, extract_candidate_entities
from client_research_agent.retrieval.indexing import IndexingPipeline
from client_research_agent.retrieval.pipeline import KnowledgeRefresh, RetrievalPipeline
from client_research_agent.scoring.engine import ScoringEngine
from client_research_agent.security.output_guard import OutputGuard
from client_research_agent.security.pii import PiiRedactor
from client_research_agent.security.poisoning import PoisoningPolicy, RetrievalPoisoningGuard
from client_research_agent.security.prompt_injection import PromptInjectionDetector
from client_research_agent.security.rate_limiter import RunBudget
from client_research_agent.security.sanitizer import ContentSanitizer
from client_research_agent.services.local import (
    InMemoryDocumentStore,
    InMemoryVectorIndex,
    JsonlBriefRepository,
)
from client_research_agent.services.ports import (
    BriefRepository,
    DocumentStore,
    EmbeddingClient,
    HttpFetcher,
    LLMClient,
    VectorIndex,
)
from client_research_agent.utils.errors import AgentError, ConfigurationError

LOCAL_EMBEDDING_DIMENSION = 512
#: Documents are screened whole (a 10-K primary document can exceed a million characters).
DOCUMENT_SANITIZER_MAX_CHARS = 2_000_000
_log = get_logger(__name__)


class _Default(Enum):
    SENTINEL = "default"


DEFAULT: Final = _Default.SENTINEL
Defaultable = Literal[_Default.SENTINEL]


@runtime_checkable
class IngestionRunner(Protocol):
    def run(self, request: ResearchRequest) -> IngestionResult: ...


def research_entity_extractor(extractor: EntityExtractor | None = None) -> Callable[[str], tuple[str, ...]]:
    """Graph entities for enrichment: candidate names plus technologies and executives found by research."""
    research = extractor or EntityExtractor()

    def extract(text: str) -> tuple[str, ...]:
        found = research.extract(text)
        names = [*extract_candidate_entities(text), *found.technologies, *(e.name for e in found.executives)]
        return tuple(dict.fromkeys(n for n in names if n))[:40]

    return extract


def default_var_dir(settings: AppSettings, environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get("CRA_VAR_DIR")
    if configured:
        return Path(configured)
    if settings.environment is Environment.LOCAL:
        return Path("var")
    return Path(tempfile.gettempdir()) / "client-research-agent"


def databricks_host_configured(settings: AppSettings, environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return bool(settings.databricks.host or env.get("DATABRICKS_HOST"))


def resolve_warehouse_id(
    settings: AppSettings, workspace_client: Any, environ: Mapping[str, str] | None = None
) -> str:
    """``databricks.warehouse_id`` -> ``DATABRICKS_WAREHOUSE_ID`` -> the ``cra-sql-<env>`` warehouse.

    The last step looks the warehouse up by the name the bundle and Terraform give it and,
    failing that, takes the first serverless warehouse visible to the principal.
    """
    env = os.environ if environ is None else environ
    if settings.databricks.warehouse_id:
        return settings.databricks.warehouse_id
    configured = env.get("DATABRICKS_WAREHOUSE_ID")
    if configured:
        return configured
    expected = f"cra-sql-{settings.environment.value}"
    try:
        warehouses = list(workspace_client.warehouses.list())
    except Exception as exc:
        raise ConfigurationError(f"cannot list SQL warehouses to resolve warehouse_id: {exc}") from exc
    named = [w for w in warehouses if getattr(w, "name", None) == expected and getattr(w, "id", None)]
    serverless = [
        w for w in warehouses if getattr(w, "enable_serverless_compute", False) and getattr(w, "id", None)
    ]
    chosen = (named or serverless or [None])[0]
    if chosen is None:
        raise ConfigurationError(
            "no SQL warehouse configured: set databricks.warehouse_id (CRA_DATABRICKS__WAREHOUSE_ID) "
            f"or DATABRICKS_WAREHOUSE_ID, or create the '{expected}' warehouse"
        )
    return str(chosen.id)


def build_databricks_llm(settings: AppSettings, workspace_client: Any) -> LLMClient:
    """Primary chat endpoint with an automatic fallback endpoint, both over the Foundation Model API."""
    from client_research_agent.databricks.auth import (  # noqa: PLC0415
        WorkspaceCredentials,
        build_openai_client,
    )
    from client_research_agent.databricks.model_serving import (  # noqa: PLC0415 - optional extra
        DatabricksChatClient,
        FallbackLLMClient,
    )

    credentials = WorkspaceCredentials.from_workspace_client(workspace_client)
    openai_client = build_openai_client(credentials, timeout_seconds=settings.serving.request_timeout_seconds)
    primary = DatabricksChatClient(
        settings.serving.chat_endpoint, openai_client, resilience=settings.resilience
    )
    fallback_endpoint = settings.serving.fallback_chat_endpoint
    if not fallback_endpoint or fallback_endpoint == settings.serving.chat_endpoint:
        return primary
    fallback = DatabricksChatClient(fallback_endpoint, openai_client, resilience=settings.resilience)
    return FallbackLLMClient(primary, fallback)


@dataclass
class AgentRuntime:
    """Every wired component of one agent process. Build with :func:`build_runtime`."""

    settings: AppSettings
    llm: LLMClient | None
    embedder: EmbeddingClient
    vector_index: VectorIndex
    document_store: DocumentStore
    brief_repository: BriefRepository
    audit: AuditLogger
    fetcher: HttpFetcher
    ingestion: IngestionRunner
    refresh_ingestion: IngestionRunner | None
    indexer: IndexingPipeline
    prompts: PromptRegistry
    scoring: ScoringEngine
    input_sanitizer: ContentSanitizer
    document_sanitizer: ContentSanitizer
    injection_detector: PromptInjectionDetector
    poisoning_guard: RetrievalPoisoningGuard
    pii_redactor: PiiRedactor
    output_guard: OutputGuard
    responsible_ai: ResponsibleAIPolicy
    review_policy: ReviewPolicy
    review_queue: ReviewQueue
    feedback_store: FeedbackStore
    var_dir: Path
    workspace_client: Any = None
    track_runs: bool = False
    llm_relevance_grading: bool = False
    max_knowledge_refreshes: int = 1
    budget_factory: Callable[[], RunBudget] = RunBudget
    _closers: list[Callable[[], None]] = field(default_factory=list, repr=False)

    @property
    def environment(self) -> Environment:
        return self.settings.environment

    def model_versions(self) -> dict[str, str]:
        return {
            "llm": self.llm.model_name if self.llm is not None else "none",
            "embedding": self.embedder.model_name,
            "agent": __version__,
            "environment": self.settings.environment.value,
        }

    def build_retriever(
        self, corpus: Sequence[Chunk], *, knowledge_refresh: KnowledgeRefresh | None = None
    ) -> RetrievalPipeline:
        """A retrieval pipeline over ``corpus`` (one company's chunks) for a single run."""
        return RetrievalPipeline.from_components(
            embedder=self.embedder,
            vector_index=self.vector_index,
            document_store=self.document_store,
            llm=self.llm,
            settings=self.settings.retrieval,
            corpus=corpus,
            use_llm_grader=self.llm_relevance_grading,
            knowledge_refresh=knowledge_refresh,
        )

    def close(self) -> None:
        while self._closers:
            closer = self._closers.pop()
            try:
                closer()
            except Exception as exc:  # closing must never mask the caller's outcome
                _log.warning("runtime.close_failed", error=type(exc).__name__)


def _refresh_pipeline(settings: AppSettings, fetcher: HttpFetcher, store: DocumentStore) -> IngestionPipeline:
    """Targeted re-ingestion for CRAG knowledge refresh: seeds and the corporate site, no EDGAR sweep."""
    parser = HtmlParser()
    return IngestionPipeline(
        fetcher,
        settings,
        parser,
        DocumentTypeClassifier(),
        EntityExtractor(),
        None,
        CorporateSiteDiscoverer(
            fetcher, parser=parser, max_pages=min(10, settings.crawler.max_pages_per_domain)
        ),
        PublicAnalystSource(fetcher),
        store,
        max_workers=4,
    )


def build_runtime(
    settings: AppSettings,
    *,
    llm: LLMClient | Defaultable | None = DEFAULT,
    embedder: EmbeddingClient | None = None,
    vector_index: VectorIndex | None = None,
    document_store: DocumentStore | None = None,
    brief_repository: BriefRepository | None = None,
    audit: AuditLogger | None = None,
    fetcher: HttpFetcher | None = None,
    ingestion: IngestionRunner | None = None,
    refresh_ingestion: IngestionRunner | Defaultable | None = DEFAULT,
    prompts: PromptRegistry | None = None,
    workspace_client: Any = None,
    var_dir: str | os.PathLike[str] | None = None,
    track_runs: bool | None = None,
    trigger_index_sync: bool | None = None,
    llm_relevance_grading: bool = False,
    ingestion_max_workers: int = 8,
    review_policy: ReviewPolicy | None = None,
    budget_factory: Callable[[], RunBudget] | None = None,
    environ: Mapping[str, str] | None = None,
) -> AgentRuntime:
    """Wire an :class:`AgentRuntime` for ``settings.environment``; any component may be injected."""
    env = os.environ if environ is None else environ
    root = Path(var_dir) if var_dir is not None else default_var_dir(settings, env)
    closers: list[Callable[[], None]] = []
    databricks = settings.uses_databricks

    workspace = workspace_client
    if databricks and workspace is None:
        from client_research_agent.databricks.auth import build_workspace_client  # noqa: PLC0415

        workspace = build_workspace_client(settings, environ=env)

    resolved_llm = _resolve_llm(settings, llm, workspace, env)
    metered_llm: LLMClient | None = MeteredLLMClient(resolved_llm) if resolved_llm is not None else None

    resolved_audit = audit
    if databricks:
        store, index, briefs, resolved_embedder, executor = _databricks_adapters(
            settings,
            workspace,
            env,
            embedder=embedder,
            vector_index=vector_index,
            document_store=document_store,
            brief_repository=brief_repository,
            trigger_index_sync=trigger_index_sync,
        )
        if resolved_audit is None:
            resolved_audit = _replicated_audit(settings, root, executor, closers)
    else:
        resolved_embedder = embedder or HashingEmbeddingClient(LOCAL_EMBEDDING_DIMENSION)
        store = document_store or InMemoryDocumentStore()
        index = vector_index or InMemoryVectorIndex(resolved_embedder.dimension)
        briefs = brief_repository or JsonlBriefRepository(root / "briefs")

    if fetcher is None:
        client = httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(
                settings.crawler.request_timeout_seconds,
                connect=min(10.0, settings.crawler.request_timeout_seconds),
            ),
        )
        closers.append(client.close)
        resolved_fetcher: HttpFetcher = PolicyEnforcingFetcher.from_settings(settings, client=client)
    else:
        resolved_fetcher = fetcher

    resolved_ingestion = ingestion or build_ingestion_pipeline(
        settings, resolved_fetcher, document_store=store, max_workers=ingestion_max_workers
    )
    resolved_refresh = (
        _refresh_pipeline(settings, resolved_fetcher, store)
        if refresh_ingestion is DEFAULT
        else refresh_ingestion
    )

    guardrails = settings.guardrails
    detector = PromptInjectionDetector.from_settings(guardrails)
    redactor = PiiRedactor()
    indexer = IndexingPipeline(
        chunking=settings.chunking,
        enricher=MetadataEnricher(entity_extractor=research_entity_extractor()),
        embedder=resolved_embedder,
        vector_index=index,
        document_store=store,
    )
    return AgentRuntime(
        settings=settings,
        llm=metered_llm,
        embedder=resolved_embedder,
        vector_index=index,
        document_store=store,
        brief_repository=briefs,
        audit=resolved_audit or AuditLogger(root / "audit" / "audit.jsonl", redactor=redactor),
        fetcher=resolved_fetcher,
        ingestion=resolved_ingestion,
        refresh_ingestion=resolved_refresh,
        indexer=indexer,
        prompts=prompts or default_registry(),
        scoring=ScoringEngine(settings.scoring),
        input_sanitizer=ContentSanitizer.from_settings(guardrails),
        document_sanitizer=ContentSanitizer(DOCUMENT_SANITIZER_MAX_CHARS),
        injection_detector=detector,
        poisoning_guard=RetrievalPoisoningGuard(
            detector,
            PoisoningPolicy(allowed_domain_suffixes=settings.crawler.allowed_domain_suffixes),
        ),
        pii_redactor=redactor,
        output_guard=OutputGuard(redactor=redactor),
        responsible_ai=ResponsibleAIPolicy(),
        review_policy=review_policy
        or ReviewPolicy(min_citation_coverage=guardrails.review_min_citation_coverage),
        review_queue=ReviewQueue(root / "reviews" / "queue.jsonl"),
        feedback_store=FeedbackStore(root / "feedback" / "feedback.jsonl"),
        var_dir=root,
        workspace_client=workspace,
        track_runs=databricks if track_runs is None else track_runs,
        llm_relevance_grading=llm_relevance_grading,
        budget_factory=budget_factory or RunBudget,
        _closers=closers,
    )


def _resolve_llm(
    settings: AppSettings,
    llm: LLMClient | Defaultable | None,
    workspace: Any,
    environ: Mapping[str, str],
) -> LLMClient | None:
    if llm is not DEFAULT:
        return llm
    if settings.uses_databricks:
        return build_databricks_llm(settings, workspace)
    if not databricks_host_configured(settings, environ):
        return None
    try:
        from client_research_agent.databricks.auth import build_workspace_client  # noqa: PLC0415

        return build_databricks_llm(settings, workspace or build_workspace_client(settings, environ=environ))
    except (AgentError, ImportError) as exc:
        _log.warning("runtime.local_llm_unavailable", error=f"{type(exc).__name__}: {exc}"[:300])
        return None


def _databricks_adapters(
    settings: AppSettings,
    workspace: Any,
    environ: Mapping[str, str],
    *,
    embedder: EmbeddingClient | None,
    vector_index: VectorIndex | None,
    document_store: DocumentStore | None,
    brief_repository: BriefRepository | None,
    trigger_index_sync: bool | None,
) -> tuple[DocumentStore, VectorIndex, BriefRepository, EmbeddingClient, Callable[[], Any]]:
    from client_research_agent.databricks.auth import (  # noqa: PLC0415
        WorkspaceCredentials,
        build_openai_client,
    )
    from client_research_agent.databricks.embeddings import DatabricksEmbeddingClient  # noqa: PLC0415
    from client_research_agent.databricks.unity_catalog import (  # noqa: PLC0415
        DeltaBriefRepository,
        DeltaDocumentStore,
        StatementExecutor,
    )
    from client_research_agent.databricks.vector_search import (  # noqa: PLC0415
        DatabricksVectorIndex,
        build_vector_search_client,
    )

    resolved_embedder = embedder
    if resolved_embedder is None:
        credentials = WorkspaceCredentials.from_workspace_client(workspace)
        openai_client = build_openai_client(
            credentials, timeout_seconds=settings.serving.request_timeout_seconds
        )
        resolved_embedder = DatabricksEmbeddingClient(
            openai_client,
            endpoint=settings.serving.embedding_endpoint,
            dimension=settings.serving.embedding_dimension,
            resilience=settings.resilience,
        )

    catalog, schema = settings.databricks.catalog, settings.databricks.schema_
    cache: dict[str, Any] = {}

    def executor() -> StatementExecutor:
        if "executor" not in cache:
            warehouse_id = resolve_warehouse_id(settings, workspace, environ)
            cache["executor"] = StatementExecutor(
                workspace.statement_execution, warehouse_id, resilience=settings.resilience
            )
        result: StatementExecutor = cache["executor"]
        return result

    def delta_store() -> DeltaDocumentStore:
        if "store" not in cache:
            cache["store"] = DeltaDocumentStore(executor(), catalog=catalog, schema=schema)
        result: DeltaDocumentStore = cache["store"]
        return result

    store: DocumentStore = document_store if document_store is not None else delta_store()
    briefs: BriefRepository = (
        brief_repository
        if brief_repository is not None
        else DeltaBriefRepository(executor(), catalog=catalog, schema=schema)
    )
    if vector_index is None:
        kwargs: dict[str, Any] = {}
        if trigger_index_sync is not None:
            kwargs["trigger_sync"] = trigger_index_sync
        index: VectorIndex = DatabricksVectorIndex.from_settings(
            build_vector_search_client(workspace), settings, delta_store(), **kwargs
        )
    else:
        index = vector_index
    return store, index, briefs, resolved_embedder, executor


def _replicated_audit(
    settings: AppSettings, root: Path, executor: Callable[[], Any], closers: list[Callable[[], None]]
) -> AuditLogger:
    """Local hash-chained JSONL log replicated to the Unity Catalog ``audit_log`` table."""
    from client_research_agent.databricks.audit_sink import DeltaAuditSink, FanOutAuditLogger  # noqa: PLC0415

    path = root / "audit" / "audit.jsonl"
    try:
        sink = DeltaAuditSink(
            executor(), catalog=settings.databricks.catalog, schema=settings.databricks.schema_
        )
    except AgentError as exc:
        _log.warning("runtime.delta_audit_unavailable", error=f"{type(exc).__name__}: {exc}"[:300])
        return AuditLogger(path)
    closers.append(sink.close)
    return FanOutAuditLogger(path, [sink])
