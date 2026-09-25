# Runbook: Ingestion blocked by robots.txt or SSRF policy

Components: `research/fetcher.py::PolicyEnforcingFetcher`,
`research/url_guard.py::UrlGuard`, `research/robots.py::RobotsPolicy`,
`research/rate_limit.py::HostRateLimiter`,
`research/sources/analyst_public.py::PublicAnalystSource`.

## Symptoms

- Briefs for a company have little or no evidence; verdict
  `not_enough_evidence` with reason `MISSING_EVIDENCE` or `LOW_CONFIDENCE`.
- Alert `not_enough_evidence_share`: more than 40% of briefs in 7 days.
- `ingestion.sources_skipped` rises; `IngestionResult.skipped_by_reason()` is
  dominated by `policy_violation` or `circuit_open`.
- Log events carrying `CrawlPolicyViolationError` messages (robots disallow,
  host not in allow-list, non-public address, gated analyst path).

## Dashboards and queries

- `verdict_distribution` (daily verdict mix, mean confidence).
- Skips recorded per run: the `evidence_gathering` step detail in the run
  state (`sources_skipped` by `SkipReason`, `documents_ingested`,
  `documents_quarantined`), logged as `run_state.json` on the MLflow run
  `brief-<run_id>`; the job's task values (`documents_ingested`,
  `chunks_written`); and the metric `ingestion.sources_skipped{reason=...}`.
- Step status from the audit trail (event type `step.evidence_gathering`,
  payload `status`, `warnings`, `error`):

```sql
SELECT recorded_at, get_json_object(payload_json, '$.status') AS status,
       get_json_object(payload_json, '$.warnings') AS warnings
FROM `${catalog}`.`${schema}`.`audit_log`
WHERE recorded_at >= current_timestamp() - INTERVAL 7 DAYS
  AND event_type = 'step.evidence_gathering'
ORDER BY recorded_at DESC;
```

  In dev, staging and prod the audit trail is replicated to `audit_log` by
  `FanOutAuditLogger` / `DeltaAuditSink` (`databricks/audit_sink.py`).

Reproduce locally for one company (offline adapters, real public fetches):

```bash
uv run cra ingest --company "Example Corp" --domain example.com --ticker EXMP --max-documents 20
```

## Diagnosis

| `SkipReason` | Meaning | Typical cause |
|---|---|---|
| `policy_violation` | `UrlGuard` or `RobotsPolicy` refused the URL | Site disallows our product token in robots.txt; URL outside the request scope (wrong `domain` in the request); redirect to a CDN host outside scope; analyst URL on a gated path |
| `circuit_open` | Per-host breaker open after repeated failures | Site rate-limiting us (429) or down |
| `http_status` | Non-2xx final status | 403 from bot protection, 404 from stale sitemap |
| `fetch_failed` | Network error or timeout | Site slow; `crawler.request_timeout_seconds` (20 s) exceeded |
| `unsupported_content` | Content type outside the allow-list | PDFs or binaries |
| `max_documents_reached` | `request.max_documents` hit | Expected |

Specific checks:

- **robots.txt fail-closed.** `RobotsPolicy` treats 429, 5xx and network errors
  on robots.txt as disallow-all (RFC 9309 section 2.3.1.4), cached for a shorter
  failure TTL. A transient robots.txt outage therefore blocks a host briefly;
  retry after the failure TTL.
- **Scope.** Only `crawler.allowed_domain_suffixes` (default `sec.gov`) plus the
  request's `domain` and `seed_urls` hosts are allowed. A company whose investor
  site is on a separate registrable domain (for example `investors.example-ir.com`)
  needs that URL passed as a seed.
- **SSRF refusal** of a public site usually means its DNS resolved to a private
  or reserved range from the workspace network (split-horizon DNS). Do not bypass;
  investigate network configuration.
- **SEC EDGAR.** `crawler.contact_email` must be a monitored mailbox. Jobs get it
  from the bundle variable `sec_contact_email` (`--contact-email`), the endpoint
  from the secret `client-research-agent/sec-contact-email`; staging and prod
  settings refuse the `example.org` placeholder.

## Mitigation

1. Correct the request: supply the right `domain`, `ticker` or `cik`, and add
   investor-relations URLs as `seed_urls`.
2. For sites that disallow crawling, accept the outcome: the verdict
   `NOT_ENOUGH_EVIDENCE` is correct. Do not disable `crawler.respect_robots_txt`.
3. For rate limiting by a site, lower the load: `crawler.requests_per_second_per_host`
   (default 1.0) and fewer concurrent brief runs for that company.
4. Adding a domain to `crawler.allowed_domain_suffixes` or an analyst path to
   `DEFAULT_ANALYST_POLICIES` requires a reviewed pull request (and legal review
   for analyst domains, per [ADR-0007](../adr/0007-public-sources-only-and-analyst-compliance.md)).

## Escalation

- SEV3 for a single company. SEV2 if EDGAR is blocked for all companies (check
  User-Agent and contact e-mail first; SEC publishes its fair-access policy).
