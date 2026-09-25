-- Row filters and column masks.
-- ${engineers_group} and ${agent_sp} are rendered by apply_ddl.py.
--
-- audit_log is confidential: a row filter limits it to engineers and the agent
-- service principal (the adapter writes it as that principal), on top of the
-- fact that no other group is granted SELECT. Its payload_json is PII-scrubbed
-- by the adapter before it is written.
-- companies_watchlist.owner holds account-owner emails and is masked for
-- everyone except engineers, the agent and the owner themself.
-- briefs carry no personal columns, so analysts read them unfiltered.

CREATE OR REPLACE FUNCTION `${catalog}`.`${schema}`.`mask_email`(email STRING)
  RETURNS STRING
  COMMENT 'Column mask: keeps the domain, hides the mailbox for non-privileged readers.'
  RETURN CASE
    WHEN email IS NULL THEN NULL
    WHEN is_account_group_member('${engineers_group}') OR current_user() = '${agent_sp}' THEN email
    WHEN email = current_user() THEN email
    ELSE regexp_replace(email, '^[^@]+', '***')
  END;

CREATE OR REPLACE FUNCTION `${catalog}`.`${schema}`.`audit_log_row_filter`(event_type STRING)
  RETURNS BOOLEAN
  COMMENT 'Row filter: audit events are visible to engineers and the agent service principal only.'
  RETURN is_account_group_member('${engineers_group}') OR current_user() = '${agent_sp}';

ALTER TABLE `${catalog}`.`${schema}`.`audit_log`
  SET ROW FILTER `${catalog}`.`${schema}`.`audit_log_row_filter` ON (event_type);

ALTER TABLE `${catalog}`.`${schema}`.`companies_watchlist`
  ALTER COLUMN owner SET MASK `${catalog}`.`${schema}`.`mask_email`;
