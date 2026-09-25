# Logical Architecture

Layers from caller to infrastructure. Dependencies point downwards only; the
domain model and ports are the only things every layer shares.

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
