# Sequence Diagram: request through the serving endpoint to a Client Brief

One synchronous research request to `cra-agent-<env>`. Step names are
`orchestration/state.py::StepName` values; every step is an MLflow/OTel span
(`orchestrator.<step>`), emits `orchestrator.step.duration_ms` and writes an
audit event `step.<step>`.

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

Failure behaviour along the path:

| Where | Failure | Behaviour |
|---|---|---|
| `company_input` | Principal lacks `RUN_RESEARCH`; injection in request fields | Run fails (`AccessDeniedError`, `SecurityViolationError`); audit `run.failed` |
| `research_plan` | LLM error | Step `degraded`; deterministic plan from the criteria catalogue |
| `evidence_gathering` | Ingestion or indexing outage | Step `degraded`; run continues with evidence already stored |
| `retrieval` | No stored chunks | `NoEvidenceRetriever`; brief reports insufficient evidence |
| Any LLM call | 429 / 5xx / timeout / breaker open | `FallbackLLMClient` answers from the fallback endpoint; if that fails too, the step's deterministic path |
| Any LLM call | Run budget exhausted (`LLMBudgetExhaustedError`, a `CircuitOpenError`, raised by `MeteredLLMClient` before the call) | The step's deterministic path |
| `brief_generation` | LLM path raises | Regenerated with `llm=None` (deterministic) |
| `output` | Persistence, lineage, review queue or MLflow unavailable | Step `degraded`; the brief is still returned |
