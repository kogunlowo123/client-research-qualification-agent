-- Catalog, environment schema and volumes.
-- Rendered by apply_ddl.py: ${catalog} and ${schema} are replaced with
-- validated identifiers before execution. Idempotent (IF NOT EXISTS).
-- Terraform (deployment/terraform) creates the same objects; this file lets a
-- workspace be bootstrapped without Terraform and is a no-op afterwards.

CREATE CATALOG IF NOT EXISTS `${catalog}`
  COMMENT 'Client Research & Qualification Agent: public-source evidence, chunks, briefs and audit data.';

CREATE SCHEMA IF NOT EXISTS `${catalog}`.`${schema}`
  COMMENT 'Client Research Agent environment schema: tables, vector index, governance functions and registered model.';

CREATE VOLUME IF NOT EXISTS `${catalog}`.`${schema}`.`raw_documents`
  COMMENT 'Raw public-source documents captured by the ingestion job, keyed by content hash.';

CREATE VOLUME IF NOT EXISTS `${catalog}`.`${schema}`.`brief_exports`
  COMMENT 'Rendered client briefs exported by cra_brief_generation.';
