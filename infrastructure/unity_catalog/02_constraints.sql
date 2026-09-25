-- CHECK constraints, applied with ALTER TABLE so the file stays idempotent
-- (DROP CONSTRAINT IF EXISTS + ADD CONSTRAINT).
--
-- Adapter-owned tables carry exactly the constraints in
-- client_research_agent.databricks.unity_catalog.CHECK_CONSTRAINTS; stricter
-- checks here could reject rows the adapter legitimately writes.

ALTER TABLE `${catalog}`.`${schema}`.`chunks` DROP CONSTRAINT IF EXISTS chunks_embedded_children_only;
ALTER TABLE `${catalog}`.`${schema}`.`chunks` ADD CONSTRAINT chunks_embedded_children_only CHECK (strategy <> 'parent' AND size(embedding) > 0);

ALTER TABLE `${catalog}`.`${schema}`.`parent_chunks` DROP CONSTRAINT IF EXISTS parent_chunks_parents_only;
ALTER TABLE `${catalog}`.`${schema}`.`parent_chunks` ADD CONSTRAINT parent_chunks_parents_only CHECK (strategy = 'parent');

-- Platform-owned tables.

ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` DROP CONSTRAINT IF EXISTS watchlist_priority_range;
ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` ADD CONSTRAINT watchlist_priority_range CHECK (priority BETWEEN 1 AND 5);

ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` DROP CONSTRAINT IF EXISTS watchlist_cik_digits;
ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist` ADD CONSTRAINT watchlist_cik_digits CHECK (cik IS NULL OR cik RLIKE '^[0-9]{1,10}$');

ALTER TABLE `${catalog}`.`${schema}`.`lineage` DROP CONSTRAINT IF EXISTS lineage_provenance_valid;
ALTER TABLE `${catalog}`.`${schema}`.`lineage` ADD CONSTRAINT lineage_provenance_valid CHECK (statement_provenance IN ('verified_fact', 'ai_recommendation'));

ALTER TABLE `${catalog}`.`${schema}`.`eval_set` DROP CONSTRAINT IF EXISTS eval_set_verdict_valid;
ALTER TABLE `${catalog}`.`${schema}`.`eval_set` ADD CONSTRAINT eval_set_verdict_valid CHECK (expected_verdict IS NULL OR expected_verdict IN ('good_fit', 'potential_fit', 'not_enough_evidence'));
