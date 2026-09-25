# Databricks Architecture Diagram

Unity Catalog objects, Vector Search, Model Serving, MLflow, Workflows and AI
Gateway inference tables for one environment (`<env>` is `dev`, `staging` or
`prod`; the schema is `agent_<env>`). Owners: T = Terraform, D = UC DDL
(`infrastructure/unity_catalog`), B = Asset Bundle, A = `databricks.agents.deploy`
in `cra_agent_deploy`, G = AI Gateway (created on first inference).

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
