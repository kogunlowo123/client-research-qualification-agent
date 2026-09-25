-- Least-privilege grants (mirrors deployment/terraform/grants.tf for
-- workspaces bootstrapped without Terraform). GRANT is idempotent.
--   analysts : read briefs only
--   agent SP : read/write inside the environment schema, register models
--   engineers: read everything in the schema, execute governance functions

GRANT USE CATALOG ON CATALOG `${catalog}` TO `${analysts_group}`;
GRANT USE SCHEMA ON SCHEMA `${catalog}`.`${schema}` TO `${analysts_group}`;
GRANT SELECT ON TABLE `${catalog}`.`${schema}`.`briefs` TO `${analysts_group}`;
GRANT READ VOLUME ON VOLUME `${catalog}`.`${schema}`.`brief_exports` TO `${analysts_group}`;

GRANT USE CATALOG, CREATE SCHEMA ON CATALOG `${catalog}` TO `${agent_sp}`;
GRANT USE SCHEMA, SELECT, MODIFY, EXECUTE, READ VOLUME, WRITE VOLUME, CREATE TABLE, CREATE FUNCTION, CREATE MODEL
  ON SCHEMA `${catalog}`.`${schema}` TO `${agent_sp}`;

-- Engineers' schema-level SELECT covers documents, parent_chunks, chunks and the
-- rest; the explicit grants below keep retrieval tables readable if schema-level
-- grants are later narrowed. Analysts never read chunks or parent_chunks.
GRANT SELECT ON TABLE `${catalog}`.`${schema}`.`chunks` TO `${engineers_group}`;
GRANT SELECT ON TABLE `${catalog}`.`${schema}`.`parent_chunks` TO `${engineers_group}`;

GRANT USE CATALOG ON CATALOG `${catalog}` TO `${engineers_group}`;
GRANT USE SCHEMA, SELECT, EXECUTE, READ VOLUME ON SCHEMA `${catalog}`.`${schema}` TO `${engineers_group}`;
