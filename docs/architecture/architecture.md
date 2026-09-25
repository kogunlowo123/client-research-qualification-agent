# Architecture

This document describes how the Client Research & Qualification Agent is built
and deployed. It is written against the code in `src/client_research_agent/`,
the Asset Bundle, Terraform and `infrastructure/`. Every diagram is also
available standalone in [docs/diagrams](../diagrams/).

Related: [rag_design.md](rag_design.md), [threat_model.md](threat_model.md),
[ADR index](../adr/README.md), [deployment guide](../operations/deployment_guide.md).

## 1. Context and quality attributes

The agent takes a company (`models/domain.py::ResearchRequest`) and returns a
`ClientBrief`: five weighted criterion scores, a verdict, and narrative
sections in which every statement is either a `VERIFIED_FACT` citing evidence
or an `AI_RECOMMENDATION`.

| Quality attribute | How the architecture addresses it |
|---|---|
| Groundedness | Evidence ids, verbatim quotes, citation validation with number and entity hard rules; unsupported facts removed or relabelled |
| Availability | Retries and breakers per dependency, fallback chat endpoint, deterministic path for every LLM step; each orchestration step can degrade without failing the run |
| Compliance | Public sources only, robots.txt, SEC fair-access identification, analyst pages only when supplied |
| Security | Six trust boundaries with controls at each (see [threat_model.md](threat_model.md)) |
| Auditability | Hash-chained audit log, per-run `RunState`, lineage from statement to URL, MLflow traces, inference tables |
| Portability and testability | Hexagonal ports; the same code runs offline with local adapters and on Databricks |
| Cost control | Per-run budget enforced on every LLM call, AI Gateway limits, persisted embeddings |

## 2. Logical architecture

```text
+----------------------------------------------------------------------------------------------+
| Interfaces                                                                                   |
|   Model Serving endpoint cra-agent-<env>  (agent/serving_agent.py, MLflow ResponsesAgent)    |
|   CLI cra research | ingest | evaluate | serve-check   (cli.py)                              |
|   Workflows entry points cra-ingest, cra-brief, cra-evaluate, cra-deploy-agent (workflows/)  |
+----------------------------------------------+-----------------------------------------------+
                                               |
+----------------------------------------------v-----------------------------------------------+
| Orchestration  (orchestration/)                                                              |
|   ClientResearchOrchestrator.run(request, principal=...)  -> RunState (per-step StepRecord)  |
|   steps: company_input -> research_plan -> evidence_gathering -> retrieval -> qualification  |
|          -> scoring -> brief_generation -> validation -> output                              |
|   ReviewPolicy / ReviewQueue / FeedbackStore (human in the loop)                             |
+----------------------------------------------+-----------------------------------------------+
                                               |
+----------------------------------------------v-----------------------------------------------+
| Agents and pipelines                                                                         |
|   Research Agent            ResearchPlanner (research_planner prompt) + IngestionPipeline    |
|   Evidence Gathering Agent  EvidenceGatherer: sanitise, injection screen, PII redact,        |
|                             poisoning guard, IndexingPipeline (parent/child, enrich, embed)  |
|   Retrieval Pipeline        RetrievalPipeline (rewrite, multi-query, hybrid RRF, GraphRAG,   |
|                             CRAG, rerank, MMR, parent expansion, verbatim compression)       |
|   Qualification Agent       QualificationAgent + EvidenceRanker + EvidenceRegistry +         |
|                             HeuristicQualifier (criterion_qualifier prompt)                  |
|   Scoring Engine            ScoringEngine (weights, confidence, verdict, sensitivity)        |
|   Opportunity Analysis      OpportunityAnalysisAgent (opportunity_analysis prompt)           |
|   Brief Generation          BriefGenerationAgent (brief_writer, discovery_questions)         |
|   Citation Validation       CitationValidator (lexical entailment + citation_judge)          |
|                             OutputGuard, ResponsibleAIPolicy                                 |
+----------------------------------------------+-----------------------------------------------+
                                               |
+-----------------------+----------------------v-----------------+-----------------------------+
| Cross-cutting         | Domain model (models/domain.py)        | Configuration               |
|  security/            |  ResearchRequest, SourceDocument,      |  config/settings.py         |
|  governance/          |  Chunk, Evidence, CriterionScore,      |  config/environments/*.yaml |
|  observability/       |  QualificationResult, ClientBrief ...  |  CRA_* environment vars     |
|  prompts/             |                                        |                             |
+-----------------------+----------------------+-----------------+-----------------------------+
                                               |
+----------------------------------------------v-----------------------------------------------+
| Ports (services/ports.py)                                                                    |
|   LLMClient  EmbeddingClient  VectorIndex  DocumentStore  BriefRepository  HttpFetcher       |
|   AuditSink                                                                                  |
+-----------------------+----------------------------------------------+-----------------------+
                        |                                              |
+-----------------------v-------------------------+  +-----------------v-----------------------+
| Databricks adapters (databricks/)               |  | Local adapters (services/local.py,      |
|  DatabricksChatClient + FallbackLLMClient       |  |  retrieval/embeddings.py, governance/)  |
|  DatabricksEmbeddingClient                      |  |  LLM = None (deterministic paths)       |
|  DatabricksVectorIndex (Delta Sync)             |  |  HashingEmbeddingClient                 |
|  DeltaDocumentStore, DeltaBriefRepository       |  |  InMemoryVectorIndex                    |
|  (SQL Statement Execution API)                  |  |  InMemoryDocumentStore                  |
|  mlflow_registry, jobs, auth                    |  |  JsonlBriefRepository, AuditLogger      |
+-------------------------------------------------+  +-----------------------------------------+
        HttpFetcher is PolicyEnforcingFetcher over HttpxFetcher in every environment.
```

The orchestration layer is the only component that knows the order of steps.
Agents are plain classes with explicit dependencies, constructed from an
`AgentRuntime` (`agent/factory.py::build_runtime`), which is the composition
root: it chooses local or Databricks adapters from `settings.environment`, wraps
the LLM in `MeteredLLMClient`, and builds the security, governance and review
components.

## 3. Component diagram

```text
                          +-------------+      +-------------+
                          |   cli.py    |      | workflows/  |
                          +------+------+      +------+------+
                                 |                    |
                                 v                    v
 +------------------+     +------------------------------------+
 | agent/           |<----|        orchestration/              |
 |  serving_agent   |---->|  orchestrator  steps  state        |
 |  factory         |     |  review  rendering                 |
 |  metering        |     +--+------+------+------+------+-----+
 +--+---------+-----+        |      |      |      |      |
    |         |              |      |      |      |      v
    |         |              |      |      |      |  +---------------+
    |         |              |      |      |      |  | evaluation/   |
    |         |              |      |      |      |  | dataset,      |
    |         |              |      |      |      |  | harness, gate |
    |         |              |      |      |      |  +---------------+
    |         |              v      v      v      v
    |         |  +----------+ +------------+ +---------+ +-----------+
    |         |  | research/| |qualification| | scoring | | briefing/ |
    |         |  | sources  | | agent       | | engine  | | generator |
    |         |  | fetcher  | | criteria    | +----+----+ | opportunity
    |         |  | url_guard| | heuristic   |      |      | renderer  |
    |         |  | robots   | | evidence    |      |      +-----+-----+
    |         |  | parsing  | +------+------+      |            |
    |         |  +----+-----+        |             |            v
    |         |       |              v             |      +-----------+
    |         |       |       +------------+       |      | citations/|
    |         |       |       | ranking/   |       |      | validator |
    |         |       |       +------------+       |      | entailment|
    |         |       |              |             |      +-----+-----+
    |         |       |              v             |            |
    |         |       |       +------------------------------+  |
    |         |       |       | retrieval/                   |  |
    |         |       |       | pipeline hybrid bm25 dense   |  |
    |         |       |       | crag self_rag graph rerank   |  |
    |         |       |       | chunking enrichment indexing |  |
    |         |       |       +--------------+---------------+  |
    |         |       |                      |                  |
    v         v       v                      v                  v
 +---------------------------------------------------------------------+
 | services/ports.py (Protocols)   services/structured.py   services/local.py
 +---------------------------------------------------------------------+
 | models/domain.py   config/settings.py   prompts/registry.py (+templates)
 +---------------------------------------------------------------------+
 | security/  (sanitizer, prompt_injection, poisoning, pii, output_guard,
 |             rbac, rate_limiter, secrets)
 | governance/ (audit, lineage, responsible_ai, data_classification)
 | observability/ (logging, tracing, metrics, cost, mlflow_tracking, setup)
 | utils/ (errors, resilience)
 +---------------------------------------------------------------------+
            ^
            | implements ports
 +---------------------------------------------------------------------+
 | databricks/  auth  model_serving  embeddings  vector_search          |
 |              unity_catalog  mlflow_registry  jobs  errors            |
 |   -> databricks-sdk, databricks-vectorsearch, mlflow, openai         |
 +---------------------------------------------------------------------+
```

## 4. End-to-end flow

`orchestration/orchestrator.py::ClientResearchOrchestrator.run(request, *, principal, run_id=None, options=None) -> RunResult`

| Step (`StepName`) | Agent / component | Output | Degrades to |
|---|---|---|---|
| `company_input` | `authorize(principal, Permission.RUN_RESEARCH)`, `ContentSanitizer`, `PromptInjectionDetector`, `RunAccounting(RunBudget)` | Validated request, budget | Fails the run (only fatal step besides unrecoverable brief generation) |
| `research_plan` | `orchestration/steps.py::ResearchPlanner` (`research_planner` prompt) | `ResearchPlan`: queries, focus questions, source priorities | Deterministic plan from the criteria catalogue |
| `evidence_gathering` | `EvidenceGatherer`: `IngestionPipeline` -> sanitise -> `screen_document` (windowed injection screening) -> `PiiRedactor` -> `RetrievalPoisoningGuard` -> `DocumentStore` -> `IndexingPipeline` -> lineage | `GatheringResult` (documents, chunks, quarantined) | Stored evidence only; skipped entirely with `IngestMode.NEVER` or `IF_MISSING` when evidence exists |
| `retrieval` | Chunk-level poisoning guard -> `ChunkBlocklist`; `AgentRuntime.build_retriever` -> `RetrievalPipeline`; `GuardedRetriever`; `KnowledgeRefresher` for CRAG; plan probe queries | Guarded retriever for the run | `NoEvidenceRetriever` when nothing is stored |
| `qualification` | `QualificationAgent.qualify` | Criterion scores, `EvidenceRegistry` | Heuristic-only scoring |
| `scoring` | `ScoringEngine.breakdown`, `sensitivity` | `ScoreBreakdown` (verdict, reason), `SensitivityReport` | n/a (deterministic) |
| `brief_generation` | `BriefGenerationAgent.generate` (runs `OpportunityAnalysisAgent` and `CitationValidator` internally) | `ClientBrief` with `CitationReport` | Regenerated with `llm=None` |
| `validation` | `OutputGuard.check_brief`, `ResponsibleAIPolicy.evaluate`, `strip_statements`, `SelfRagCritic.critique_answer` on cited executive-summary facts | Cleaned brief, reports, reflection verdict | Unresolved findings mark the step degraded and force review |
| `output` | `ReviewPolicy.decide`, `BriefRepository.save`, `render_run_markdown`, lineage rows, `ReviewQueue.enqueue`, audit, `RunTracker` | `RunResult(brief, markdown, state, review, lineage_rows)` | Each side effect can fail independently without losing the brief |

Each step runs inside `orchestrator.<step>` spans, records a `StepRecord`
(status `ok`, `degraded`, `failed` or `skipped`, duration, warnings, detail) and
writes an audit event `step.<step>`. The run itself writes `run.started`,
`run.completed` or `run.failed`, `evidence.quarantined`, `brief.generated`, and
guardrail events `guardrail.injection_blocked`, `guardrail.pii_redacted`,
`guardrail.chunk_quarantined` and `guardrail.output_violation`.

## 5. Sequence: request through the serving endpoint to a brief

```text
Caller        AI Gateway /       ClientResearch     ClientResearch      Agents and          Ports / Databricks
(Review App,  cra-agent-<env>    ResponsesAgent     Orchestrator        pipelines           services
 CRM, curl)                      (agent/serving_    (orchestration/
                                 agent.py)          orchestrator.py)
  |                 |                  |                  |                   |                     |
  | POST /serving-endpoints/cra-agent-<env>/invocations                       |                     |
  | {input:[...], custom_inputs:{company_name, ticker, domain,                |                     |
  |  max_documents, requested_by}, context:{user_id, ...}}                    |                     |
  |---------------->|                  |                  |                   |                     |
  |                 | OAuth, CAN_QUERY, rate limits (endpoint, per user),     |                     |
  |                 | request logged to inference table cra_agent_payload     |                     |
  |                 |----------------->|                  |                   |                     |
  |                 |                  | parse_request: custom_inputs, else company/ticker/domain     |
  |                 |                  | extracted from the last user message (LLM, then regex)       |
  |                 |                  | PrincipalRateLimiter.acquire(asserted_requester)             |
  |                 |                  | Principal = service identity (CRA_SERVICE_PRINCIPAL_ID or    |
  |                 |                  | model-serving:<env>), roles={SERVICE}; requested_by/user_id  |
  |                 |                  | recorded as asserted_requester (unverified)                  |
  |                 |                  | runtime built lazily on first request (default_runtime_factory)|
  |                 |                  |---- run(request, principal=...) ---->|                      |
  |                 |                  |                  |                   |                     |
  |                 |                  |        [company_input] authorize(RUN_RESEARCH) -> audit      |
  |                 |                  |        sanitise company name, injection check on fields,     |
  |                 |                  |        RunAccounting(RunBudget) opened for MeteredLLMClient  |
  |                 |                  |                  |                   |                     |
  |                 |                  |        [research_plan] ResearchPlanner.plan ----------------->| FMAPI chat
  |                 |                  |                  |<------------ plan (or criteria catalogue) |
  |                 |                  |                  |                   |                     |
  |                 |                  |        [evidence_gathering] EvidenceGatherer.gather          |
  |                 |                  |                  |-- IngestionPipeline.run --> PolicyEnforcingFetcher --> SEC EDGAR,
  |                 |                  |                  |                   |     (UrlGuard, robots, rate,     corporate site,
  |                 |                  |                  |                   |      breaker)                     seed URLs
  |                 |                  |                  |-- ContentSanitizer, screen_document (PromptInjectionDetector),
  |                 |                  |                  |   PiiRedactor, RetrievalPoisoningGuard (quarantine | flag)
  |                 |                  |                  |-- DocumentStore.save_documents --------------------->| documents (MERGE)
  |                 |                  |                  |-- IndexingPipeline.index: ParentChildChunker, MetadataEnricher,
  |                 |                  |                  |   EmbeddingClient.embed ------------------------------->| FMAPI gte-large-en
  |                 |                  |                  |   save_chunks (parents) ------------------------------>| parent_chunks
  |                 |                  |                  |   VectorIndex.upsert (children + vectors, sync) ------>| chunks -> chunks_index
  |                 |                  |                  |                   |                     |
  |                 |                  |        [retrieval] DocumentStore.list_chunks(company) ------>| chunks, parent_chunks
  |                 |                  |        poisoning guard on child chunks -> ChunkBlocklist     |
  |                 |                  |        runtime.build_retriever(corpus, knowledge_refresh)    |
  |                 |                  |        GuardedRetriever(RetrievalPipeline); plan probe queries
  |                 |                  |                  |                   |                     |
  |                 |                  |        [qualification] QualificationAgent.qualify(company)   |
  |                 |                  |                  |-- per criterion (parallel): 3 queries --> RetrievalPipeline.retrieve
  |                 |                  |                  |     rewrite, multi-query, BM25 + dense (VectorIndex.search) ---->| chunks_index
  |                 |                  |                  |     GraphRAG, CRAG (grade / correct / KnowledgeRefresher), rerank,
  |                 |                  |                  |     MMR, ParentExpander (get_chunks) ------------------------->| parent_chunks
  |                 |                  |                  |     ContextCompressor                  |
  |                 |                  |                  |-- EvidenceRanker -> EvidenceRegistry (E1..En)
  |                 |                  |                  |-- HeuristicQualifier + criterion_qualifier LLM ------------->| FMAPI chat
  |                 |                  |                  |   reconcile: drop unknown ids, disagreement penalty
  |                 |                  |                  |                   |                     |
  |                 |                  |        [scoring] ScoringEngine.breakdown + sensitivity       |
  |                 |                  |                  |                   |                     |
  |                 |                  |        [brief_generation] BriefGenerationAgent.generate      |
  |                 |                  |                  |-- OpportunityAnalysisAgent.analyze --------------------------->| FMAPI chat
  |                 |                  |                  |-- brief_writer, discovery_questions ------------------------->| FMAPI chat
  |                 |                  |                  |-- CitationValidator.validate (lexical + citation_judge) ----->| FMAPI judge
  |                 |                  |                  |                   |                     |
  |                 |                  |        [validation] OutputGuard.check_brief, ResponsibleAIPolicy.evaluate,
  |                 |                  |        strip_statements for removable findings; audit        |
  |                 |                  |        guardrail.output_violation; SelfRagCritic reflection  |
  |                 |                  |        on executive-summary verified facts                    |
  |                 |                  |                  |                   |                     |
  |                 |                  |        [output] ReviewPolicy.decide                          |
  |                 |                  |                  |-- BriefRepository.save -------------------------------------->| briefs (MERGE)
  |                 |                  |                  |-- render_run_markdown; LineageRecorder; evidence_lineage_rows
  |                 |                  |                  |-- ReviewQueue.enqueue (if needs_review)
  |                 |                  |                  |-- audit "brief.generated"; RunTracker (MLflow params, metrics,
  |                 |                  |                  |   brief.json, lineage.json, run_state.json) ------------------>| MLflow
  |                 |                  |<-------------- RunResult(brief, markdown, state, review, lineage_rows)
  |                 |                  | output: message item with output_text (Markdown brief) and   |
  |                 |                  | custom_outputs {run_id, verdict, weighted_score, confidence, |
  |                 |                  | citation_coverage, brief_json}; predict_stream sends deltas  |
  |                 |<-----------------|                  |                   |                     |
  |                 | response logged to cra_agent_payload; trace to MLflow   |                     |
  |<----------------|                  |                  |                   |                     |
```

## 6. Deployment architecture

```text
 Developer workstation                     GitHub
 +---------------------------+             +--------------------------------------------------+
 | uv / make / pre-commit    |  PR, tag    | ci.yml: lint, mypy, unit (3.11, 3.12, cov>=90),  |
 | cra CLI (local adapters)  |-----------> | marked suites, bandit, pip-audit, Trivy,         |
 | docker compose: MLflow,   |             | gitleaks, build wheel/sdist/image                |
 | OTel Collector, Jaeger    |             | codeql.yml: python, actions                      |
 | databricks auth login     |             | cd.yml: main -> staging; vX.Y.Z tag -> prod      |
 +---------------------------+             |   (environment approval), OIDC id-token          |
             |                             | Release: wheel/sdist, GHCR image (SBOM, provenance)
             | terraform apply             +------------------------+-------------------------+
             | (per env, prod first)                                 | OIDC token exchange
             v                                                       v (federation policy, no secrets)
 +------------------------------------------------------------------------------------------------+
 | Databricks account                                                                             |
 |   service principal cra-agent-<env> + federation policy github-<env>; budget                  |
 |                                                                                                |
 |  Databricks workspace (one per environment or shared; schema agent_<env> isolates data)       |
 |  +------------------------------------------------------------------------------------------+ |
 |  | Asset Bundle target <env>  (databricks bundle deploy; deploy.sh)                          | |
 |  |   wheel dist/*.whl  ->  serverless jobs (environment_version 3)                           | |
 |  |     cra_ingestion_refresh   cra_brief_generation   cra_evaluation   cra_agent_deploy      | |
 |  |   MLflow experiment, UC registered model, monitoring schema                               | |
 |  |                                                                                           | |
 |  | Model Serving                                                                             | |
 |  |   cra-agent-<env>  <- databricks.agents.deploy() from cra_agent_deploy (champion alias)   | |
 |  |   FMAPI: databricks-claude-sonnet-4 | databricks-meta-llama-3-3-70b-instruct |            | |
 |  |          databricks-gte-large-en            (AI Gateway via deploy.sh --apply-fm-gateway) | |
 |  |                                                                                           | |
 |  | Vector Search endpoint cra-vs-endpoint-<env> / index chunks_index      (Terraform)        | |
 |  | SQL warehouse cra-sql-<env>                                            (Terraform)        | |
 |  | Unity Catalog client_research.agent_<env>.*   tables via apply_ddl.py  (Terraform + DDL)  | |
 |  | Secret scope client-research-agent                                     (Terraform)        | |
 |  +------------------------------------------------------------------------------------------+ |
 +------------------------------------------------------------------------------------------------+
             ^                                      |
             | HTTPS (OAuth, CAN_QUERY)             | HTTPS egress (UrlGuard, robots.txt, rate limit)
             |                                      v
 +---------------------------+          +-----------------------------------------------+
 | Callers                   |          | Public internet                               |
 | Review App / Playground   |          | www.sec.gov, data.sec.gov                     |
 | CRM integrations          |          | corporate domains in the request scope        |
 | analysts (SQL on briefs)  |          | analyst public pages supplied as seed URLs    |
 +---------------------------+          +-----------------------------------------------+
```

## 7. Databricks architecture

```text
Unity Catalog metastore
`-- catalog client_research                                              [T, prod state owns it]
    |-- schema agent_<env>                                               [T]
    |   |-- documents            Delta, CDF        one row per fetched source document        [D]
    |   |-- chunks               Delta, CDF        EMBEDDED CHILD CHUNKS ONLY                  [D]
    |   |                        embedding ARRAY<FLOAT> NOT NULL (1024-d)
    |   |                        CHECK chunks_embedded_children_only
    |   |                        (strategy <> 'parent' AND size(embedding) > 0)
    |   |-- parent_chunks        Delta, no CDF     parent windows, read by id only             [D]
    |   |                        CHECK parent_chunks_parents_only (strategy = 'parent')
    |   |-- briefs               Delta, CDF        run_id, verdict, weighted_score, brief_json [D]
    |   |-- audit_log            Delta, CDF, delta.appendOnly = true, hash-chained,           [D]
    |   |                        row filter audit_log_row_filter
    |   |-- lineage              run -> evidence -> chunk -> doc -> url -> content_hash      [D]
    |   |-- companies_watchlist  Delta, CDF        owner column masked by mask_email          [D]
    |   |-- eval_set             Delta, CDF        curated evaluation rows                    [D]
    |   |-- eval_results         per-metric results and gate_passed                          [D]
    |   |-- chunks_index         Vector Search Delta Sync index (see below)                  [T]
    |   |-- client_research_agent  UC registered model                                        [B]
    |   |                        aliases: challenger, champion, previous_champion, rolled_back
    |   |-- cra_agent_payload    AI Gateway inference table of cra-agent-<env>               [G]
    |   |-- fm_claude_sonnet_4_*, fm_llama_3_3_70b_*, fm_gte_large_en_*                       [G]
    |   |                        AI Gateway inference tables of the FMAPI endpoints
    |   |-- volume raw_documents  raw artefacts keyed by content hash                         [T]
    |   `-- volume brief_exports  rendered briefs for CRM sync                                [T]
    `-- schema agent_<env>_monitoring                                                       [B]
        `-- Lakehouse Monitoring profile and drift metric tables
            (monitors on cra_agent_payload and briefs)

Mosaic AI Vector Search
`-- endpoint cra-vs-endpoint-<env>  (STANDARD)                                           [T]
    `-- index client_research.agent_<env>.chunks_index                                   [T]
        type DELTA_SYNC, source chunks, primary key chunk_id,
        embedding column embedding (self-managed, 1024-d), pipeline TRIGGERED
        synced by: cra_ingestion_refresh.sync_index, cra-ingest --sync-index,
                   DatabricksVectorIndex.upsert -> index.sync()

Model Serving
|-- cra-agent-<env>                    MLflow ResponsesAgent, served entity client_research_agent [A]
|     AI Gateway: usage tracking, inference table prefix cra_agent,
|     rate limits (prod: 1200/min endpoint, 60/min per user)
|-- databricks-claude-sonnet-4         FMAPI chat (serving.chat_endpoint, judge_endpoint)
|-- databricks-meta-llama-3-3-70b-instruct   FMAPI chat fallback (serving.fallback_chat_endpoint)
`-- databricks-gte-large-en            FMAPI embeddings, 1024-d (serving.embedding_endpoint)
      FMAPI AI Gateway: usage tracking, inference tables, rate limits,
      input/output safety and PII MASK (chat endpoints)   [deploy.sh --apply-fm-gateway]

MLflow
|-- experiment /Shared/client-research-agent-<env>   traces, GenAI evaluation runs,      [B]
|                                                    logged ResponsesAgent models
`-- registry databricks-uc (models:/client_research.agent_<env>.client_research_agent@champion)

Workflows (serverless python_wheel_task, wheel cra_wheel)                                [B]
|-- cra-ingestion-refresh-<env>   03:00 UTC  ingest_watchlist (cra-ingest) -> sync_index (cra-ingest --sync-index-only)
|-- cra-brief-generation-<env>    on demand  ingest (cra-ingest --sync-index) -> brief (cra-brief)
|                                            -> evaluate (cra-evaluate --mode brief --fail-on-gate)
|-- cra-evaluation-<env>          06:00 UTC  evaluate_champion (cra-evaluate --mode dataset --fail-on-gate)
`-- cra-agent-deploy-<env>        on demand  route_action -> log_and_register -> evaluate_candidate
                                             -> promote_and_deploy | redeploy_champion

SQL warehouse cra-sql-<env> (serverless PRO)    alerts, dashboards, Lakehouse Monitoring,     [T]
                                                Statement Execution API for the Delta adapter
Secret scope client-research-agent               SP READ, engineers per environment           [T]
Service principal cra-agent-<env>                runs jobs; GitHub OIDC federation policy     [T]
Budget client-research-agent-<env>               monthly, tag app = client-research-agent     [T]
```

### Data model

| Table | Key | Written by | Read by |
|---|---|---|---|
| `documents` | `doc_id` | `DeltaDocumentStore.save_documents` | ingestion de-duplication (`known_hashes`) |
| `chunks` | `chunk_id` | `DatabricksVectorIndex.upsert` via `save_chunks_with_embeddings` | Vector Search (Delta Sync), `list_chunks` (BM25 corpus) |
| `parent_chunks` | `chunk_id` | `DeltaDocumentStore.save_chunks` (parents) | `ParentExpander` via `get_chunks` |
| `briefs` | `run_id` | `DeltaBriefRepository.save` | analysts, dashboards, alerts |
| `audit_log` | `event_id` | `DeltaAuditSink` via `FanOutAuditLogger` | engineers (row filter) |
| `lineage` | `lineage_id` | rows from `evidence_lineage_rows` | grounding incident response |
| `companies_watchlist` | `company_name` | account teams | `cra_ingestion_refresh` |
| `eval_set`, `eval_results` | `eval_id`; (`eval_run_id`, `metric_name`) | curators; evaluation jobs | quality gate, dashboards |

The adapter-owned tables (`documents`, `chunks`, `parent_chunks`, `briefs`,
`audit_log`) are defined once in `databricks/unity_catalog.py::DDL` and copied
into `infrastructure/unity_catalog/01_tables.sql`;
`tests/contract/test_ddl_consistency.py` fails if they drift.

## 8. Resilience model

| Mechanism | Scope | Configuration |
|---|---|---|
| Typed errors | `TransientError` (retried, counts toward breakers) vs everything else (fail fast) | `utils/errors.py` |
| Retry with backoff | Every remote call | `resilience.max_attempts`, `initial_backoff_seconds`, `max_backoff_seconds` |
| Circuit breaker | Per FMAPI endpoint, per crawled host, per adapter | `resilience.breaker_failure_threshold`, `breaker_reset_seconds` |
| Fallback endpoint | Chat completions | `serving.fallback_chat_endpoint` |
| Deterministic fallbacks | Every LLM step | [ADR-0005](../adr/0005-deterministic-fallback-for-every-llm-step.md) |
| Dense-leg degradation | Hybrid retrieval | BM25 only when embeddings or the index fail |
| Step degradation | Orchestrator | Optional steps record a warning and continue |
| Budget | Per run, every LLM call | `RunBudget` via `MeteredLLMClient`; exhaustion behaves like an open breaker |

## 9. Human in the loop

`orchestration/review.py`:

- `ReviewPolicy.decide` requires review for `NOT_ENOUGH_EVIDENCE` verdicts,
  citation coverage below `min_citation_coverage` (set from
  `guardrails.review_min_citation_coverage`, default 0.8), output-guard violations,
  responsible-AI findings, verdicts that a single +/-1 criterion change would
  flip, and runs with degraded steps. Priority is `high` for guard or policy
  findings, `medium` for low coverage or sensitive verdicts, `low` otherwise.
- `ReviewQueue` is an append-only JSON Lines log (`<var>/reviews/queue.jsonl`);
  an item's state is its latest event (`pending`, `approved`, `rejected`).
- `FeedbackStore` (`<var>/feedback/feedback.jsonl`) records verdict overrides,
  statement corrections and 1-5 ratings, and `export_eval_examples` turns them
  into `evaluation/dataset.py::EvalExample` rows, closing the loop between
  production review and the regression suite.

## 10. Configuration and environments

Layered settings (`config/settings.py`): explicit overrides > `CRA_*`
environment variables > `config/environments/<env>.yaml` over `base.yaml` >
model defaults. Staging and prod require a workspace host and reject static
tokens. See the Configuration Strategy section of the [README](../../README.md#configuration-strategy).

## 11. Architectural decisions

| Decision | ADR |
|---|---|
| Ports and adapters | [ADR-0001](../adr/0001-hexagonal-ports-and-adapters.md) |
| FMAPI with fallback endpoint | [ADR-0002](../adr/0002-foundation-model-api-with-fallback-endpoint.md) |
| Delta Sync index, self-managed embeddings, `parent_chunks` | [ADR-0003](../adr/0003-delta-sync-index-self-managed-embeddings.md) |
| Hybrid retrieval with RRF | [ADR-0004](../adr/0004-hybrid-retrieval-with-rrf.md) |
| Deterministic fallbacks | [ADR-0005](../adr/0005-deterministic-fallback-for-every-llm-step.md) |
| Verbatim compression | [ADR-0006](../adr/0006-verbatim-compression-for-verifiable-citations.md) |
| Public sources and analyst compliance | [ADR-0007](../adr/0007-public-sources-only-and-analyst-compliance.md) |
| Verdict semantics | [ADR-0008](../adr/0008-verdict-semantics.md) |
| OAuth M2M / OIDC, no PATs | [ADR-0009](../adr/0009-oauth-m2m-and-oidc-no-pats.md) |
| Hash-chained audit log | [ADR-0010](../adr/0010-hash-chained-audit-log.md) |
| ResponsesAgent and `agents.deploy` | [ADR-0011](../adr/0011-mlflow-responsesagent-and-agents-deploy.md) |
