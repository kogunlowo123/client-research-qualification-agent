-- Governance tags used for discovery, data classification and cost attribution.
-- Adapter-owned tables get the same values as
-- client_research_agent.databricks.unity_catalog.TABLE_TAGS (so neither writer
-- flips them) plus the platform-wide 'app' tag.

ALTER SCHEMA `${catalog}`.`${schema}` SET TAGS ('app' = 'client-research-agent', 'environment' = '${environment}');

ALTER TABLE `${catalog}`.`${schema}`.`documents` SET TAGS ('domain' = 'client_research', 'data_classification' = 'public', 'layer' = 'silver', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`chunks` SET TAGS ('domain' = 'client_research', 'data_classification' = 'public', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`parent_chunks` SET TAGS ('domain' = 'client_research', 'data_classification' = 'public', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`briefs` SET TAGS ('domain' = 'client_research', 'data_classification' = 'internal', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`audit_log` SET TAGS ('domain' = 'client_research', 'data_classification' = 'confidential', 'layer' = 'audit', 'app' = 'client-research-agent');

ALTER TABLE `${catalog}`.`${schema}`.`lineage` SET TAGS ('domain' = 'client_research', 'data_classification' = 'internal', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`eval_set` SET TAGS ('domain' = 'client_research', 'data_classification' = 'internal', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`eval_results` SET TAGS ('domain' = 'client_research', 'data_classification' = 'internal', 'layer' = 'gold', 'app' = 'client-research-agent');
ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` SET TAGS ('domain' = 'client_research', 'data_classification' = 'internal', 'layer' = 'silver', 'contains_pii' = 'true', 'app' = 'client-research-agent');

ALTER TABLE `${catalog}`.`${schema}`.`audit_log` ALTER COLUMN payload_json SET TAGS ('pii_scrubbed' = 'true');
ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` ALTER COLUMN owner SET TAGS ('pii' = 'email');
