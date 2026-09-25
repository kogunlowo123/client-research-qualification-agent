-- Delta tables for the Client Research Agent.
--
-- documents, chunks, parent_chunks, briefs and audit_log are owned by the
-- Databricks adapter (client_research_agent.databricks.unity_catalog.DDL),
-- which is the canonical schema: the definitions below are an exact copy so
-- the adapter's MERGE statements work against tables created here. The
-- contract test tests/contract/test_ddl_consistency.py fails if they drift.
--   chunks        embedded child chunks only; Change Data Feed source of the
--                 Vector Search Delta Sync index chunks_index
--   parent_chunks parent windows for small-to-big expansion (no embedding)
--
-- companies_watchlist, lineage, eval_set and eval_results are owned by the
-- platform (workflows, evaluation and governance) and defined only here.

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`documents` (
  doc_id STRING NOT NULL COMMENT 'Stable document identifier',
  company STRING NOT NULL COMMENT 'Company the document is about',
  url STRING NOT NULL COMMENT 'Canonical source URL',
  title STRING COMMENT 'Document title',
  text STRING COMMENT 'Extracted plain text',
  document_type STRING NOT NULL COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  content_hash STRING NOT NULL COMMENT 'SHA-256 of normalised text, used for dedupe',
  publication_date DATE COMMENT 'Publication date when known',
  retrieved_at TIMESTAMP COMMENT 'When the crawler fetched the document',
  industry STRING COMMENT 'Industry label',
  language STRING COMMENT 'ISO language code',
  trust_score DOUBLE COMMENT 'Source trust score in [0,1]',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT documents_pk PRIMARY KEY (doc_id)
) USING DELTA
COMMENT 'Public company evidence documents collected by the research agent'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`chunks` (
  chunk_id STRING NOT NULL COMMENT 'Stable chunk identifier (Vector Search primary key)',
  doc_id STRING NOT NULL COMMENT 'Parent document id',
  text STRING COMMENT 'Chunk text',
  company STRING NOT NULL COMMENT 'Company the chunk is about',
  url STRING COMMENT 'Source URL',
  title STRING COMMENT 'Source title',
  document_type STRING COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  chunk_index INT COMMENT 'Position of the chunk within its document',
  strategy STRING COMMENT 'ChunkStrategy enum value',
  parent_id STRING COMMENT 'Parent chunk id for child chunks',
  publication_date DATE COMMENT 'Publication date of the source',
  industry STRING COMMENT 'Industry label',
  confidence DOUBLE COMMENT 'Extraction confidence in [0,1]',
  token_count INT COMMENT 'Token count of the chunk text',
  entities_json STRING COMMENT 'Extracted entities as a JSON array',
  contextual_header STRING COMMENT 'Contextual-retrieval header',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  embedding ARRAY<FLOAT> NOT NULL COMMENT 'Self-managed embedding vector (never NULL)',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT chunks_pk PRIMARY KEY (chunk_id)
) USING DELTA
COMMENT 'Embedded child chunks only; source table of the Vector Search Delta Sync index'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`parent_chunks` (
  chunk_id STRING NOT NULL COMMENT 'Stable parent chunk identifier',
  doc_id STRING NOT NULL COMMENT 'Parent document id',
  text STRING COMMENT 'Parent chunk text (LLM context unit)',
  company STRING NOT NULL COMMENT 'Company the chunk is about',
  url STRING COMMENT 'Source URL',
  title STRING COMMENT 'Source title',
  document_type STRING COMMENT 'DocumentType enum value',
  source_domain STRING COMMENT 'Registered domain of the source',
  chunk_index INT COMMENT 'Position of the chunk within its document',
  strategy STRING COMMENT 'ChunkStrategy enum value (always parent)',
  parent_id STRING COMMENT 'Unused for parents; kept for a uniform chunk schema',
  publication_date DATE COMMENT 'Publication date of the source',
  industry STRING COMMENT 'Industry label',
  confidence DOUBLE COMMENT 'Extraction confidence in [0,1]',
  token_count INT COMMENT 'Token count of the chunk text',
  entities_json STRING COMMENT 'Extracted entities as a JSON array',
  contextual_header STRING COMMENT 'Contextual-retrieval header',
  metadata_json STRING COMMENT 'Additional metadata as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT parent_chunks_pk PRIMARY KEY (chunk_id)
) USING DELTA
COMMENT 'Parent chunks for small-to-big expansion; never indexed';

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`briefs` (
  run_id STRING NOT NULL COMMENT 'Agent run identifier',
  company STRING NOT NULL COMMENT 'Company researched',
  generated_at TIMESTAMP COMMENT 'When the brief was generated',
  verdict STRING COMMENT 'FitVerdict enum value',
  weighted_score DOUBLE COMMENT 'Weighted qualification score in [0,5]',
  brief_json STRING NOT NULL COMMENT 'Full ClientBrief as JSON',
  updated_at TIMESTAMP COMMENT 'Last write time',
  CONSTRAINT briefs_pk PRIMARY KEY (run_id)
) USING DELTA
COMMENT 'Cited, scored client briefs produced by the agent'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`audit_log` (
  event_id STRING NOT NULL COMMENT 'Unique event identifier',
  sequence BIGINT NOT NULL COMMENT 'Monotonic sequence within the chain',
  event_type STRING NOT NULL COMMENT 'Audit event type',
  payload_json STRING COMMENT 'Event payload as JSON (PII-scrubbed)',
  recorded_at TIMESTAMP NOT NULL COMMENT 'Event time (UTC)',
  prev_hash STRING COMMENT 'Hash of the previous event',
  hash STRING NOT NULL COMMENT 'SHA-256 over prev_hash and this event',
  CONSTRAINT audit_log_pk PRIMARY KEY (event_id)
) USING DELTA
COMMENT 'Append-only, hash-chained audit trail of agent actions'
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true', 'delta.appendOnly' = 'true');

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`companies_watchlist` (
  company_name      STRING    NOT NULL COMMENT 'Legal or commonly used company name.',
  domain            STRING             COMMENT 'Corporate web domain, e.g. example.com.',
  ticker            STRING             COMMENT 'Exchange ticker symbol.',
  cik               STRING             COMMENT 'SEC Central Index Key (digits only).',
  industry          STRING             COMMENT 'Industry label used for trend analysis.',
  priority          INT       NOT NULL COMMENT '1 (highest) to 5 (lowest) refresh priority.',
  active            BOOLEAN   NOT NULL COMMENT 'Only active companies are refreshed by cra_ingestion_refresh.',
  owner             STRING             COMMENT 'Account owner email (PII, masked).',
  added_at          TIMESTAMP NOT NULL,
  last_ingested_at  TIMESTAMP,
  CONSTRAINT companies_watchlist_pk PRIMARY KEY (company_name)
)
USING DELTA
COMMENT 'Companies whose public evidence is refreshed daily.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true',
  'delta.enableDeletionVectors' = 'true'
);

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`lineage` (
  lineage_id           STRING    NOT NULL,
  run_id               STRING    NOT NULL COMMENT 'briefs.run_id of the brief that cited the evidence.',
  evidence_id          STRING    NOT NULL,
  chunk_id             STRING    NOT NULL,
  doc_id               STRING    NOT NULL,
  url                  STRING    NOT NULL,
  content_hash         STRING    NOT NULL,
  statement_provenance STRING    NOT NULL COMMENT 'verified_fact | ai_recommendation',
  supported            BOOLEAN            COMMENT 'Citation validator outcome.',
  support_score        DOUBLE,
  created_at           TIMESTAMP NOT NULL,
  CONSTRAINT lineage_pk PRIMARY KEY (lineage_id),
  CONSTRAINT lineage_briefs_fk FOREIGN KEY (run_id) REFERENCES `${catalog}`.`${schema}`.`briefs` (run_id)
)
USING DELTA
CLUSTER BY (run_id)
COMMENT 'Evidence lineage: every cited statement in a brief traced to chunk, document, URL and content hash.';

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`eval_set` (
  eval_id          STRING        NOT NULL,
  company          STRING        NOT NULL,
  domain           STRING,
  ticker           STRING,
  cik              STRING,
  request          STRING        NOT NULL COMMENT 'Agent request JSON (ResponsesAgent input).',
  expected_verdict STRING,
  expected_facts   ARRAY<STRING>          COMMENT 'Facts a correct brief must state (MLflow expectations).',
  guidelines       ARRAY<STRING>          COMMENT 'Natural-language guidelines judged by MLflow Guidelines scorer.',
  tags             MAP<STRING, STRING>,
  active           BOOLEAN       NOT NULL,
  created_by       STRING        NOT NULL,
  created_at       TIMESTAMP     NOT NULL,
  CONSTRAINT eval_set_pk PRIMARY KEY (eval_id)
)
USING DELTA
COMMENT 'Curated evaluation set for cra_evaluation and the cra_agent_deploy quality gate.'
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

CREATE TABLE IF NOT EXISTS `${catalog}`.`${schema}`.`eval_results` (
  eval_run_id     STRING    NOT NULL,
  mlflow_run_id   STRING    NOT NULL,
  model_uri       STRING    NOT NULL,
  model_version   STRING,
  dataset_table   STRING    NOT NULL,
  dataset_version BIGINT             COMMENT 'Delta version of eval_set used for the run.',
  metric_name     STRING    NOT NULL,
  metric_value    DOUBLE    NOT NULL,
  threshold       DOUBLE,
  passed          BOOLEAN   NOT NULL,
  gate_passed     BOOLEAN   NOT NULL COMMENT 'Overall quality gate outcome for the eval run.',
  environment     STRING    NOT NULL,
  evaluated_at    TIMESTAMP NOT NULL,
  CONSTRAINT eval_results_pk PRIMARY KEY (eval_run_id, metric_name)
)
USING DELTA
CLUSTER BY (evaluated_at)
COMMENT 'Per-metric results of MLflow GenAI evaluation runs and quality-gate outcomes.';
