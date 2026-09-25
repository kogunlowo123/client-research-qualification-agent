# Threat Model

Scope: the `client_research_agent` package, its Databricks deployment (Asset
Bundle jobs, Model Serving endpoint, Unity Catalog objects, Vector Search), the
Terraform and CI/CD that create them, and the public web it reads.

Method: data-flow decomposition into trust boundaries, STRIDE per boundary, and
a mapping to the OWASP Top 10 for LLM Applications (2025). Each threat is tied to
a concrete control in the repository and, where one exists, the test that
exercises it. Residual risks are listed explicitly at the end.

Related: [architecture.md](architecture.md), [ADR-0007](../adr/0007-public-sources-only-and-analyst-compliance.md),
[ADR-0009](../adr/0009-oauth-m2m-and-oidc-no-pats.md), [ADR-0010](../adr/0010-hash-chained-audit-log.md),
[SECURITY.md](../../SECURITY.md).

## 1. Assets

| Asset | Classification (`governance/data_classification.py`) | Why it matters |
|---|---|---|
| Client briefs, scores, verdicts (`briefs`) | Tagged `internal` on the table; derived sales intelligence is treated as CONFIDENTIAL by `DataClassifier` | Commercial decisions; reputational harm if wrong or leaked |
| Audit trail (`audit_log`) | `confidential` | Accountability and forensics |
| Watchlist owners (`companies_watchlist.owner`) | `internal`, `contains_pii = true`, column masked | Employee e-mail addresses |
| Public evidence (`documents`, `chunks`, `parent_chunks`) | `public` | Integrity matters more than confidentiality: poisoned evidence poisons briefs |
| Prompts and system instructions (`prompts/templates`) | Internal | Leakage eases targeted injection |
| Service-principal credentials, OAuth tokens, secret scope `client-research-agent` | Restricted | Workspace compromise |
| Model endpoints and budget | n/a | Cost and availability |

## 2. Trust boundaries

```text
+------------------------------------------------------------------------------------+
| TB0  Public internet (UNTRUSTED)                                                   |
|   sec.gov / data.sec.gov      corporate sites (robots.txt)      analyst public pages|
+----------------------------------------+-------------------------------------------+
                                         | HTTP(S) responses: attacker-controllable
                                         v
+------------------------------------------------------------------------------------+
| TB1  Crawl boundary  research/fetcher.py::PolicyEnforcingFetcher                   |
|   UrlGuard (SSRF, allow-list, per-hop)  RobotsPolicy  HostRateLimiter  size cap    |
|   HtmlParser (hidden-text removal)  ContentSanitizer  PromptInjectionDetector      |
|   RetrievalPoisoningGuard (quarantine / flag)                                      |
+----------------------------------------+-------------------------------------------+
                                         | sanitised SourceDocument / Chunk
                                         v
+------------------------------------------------------------------------------------+
| TB2  Data plane (Unity Catalog, governed)                                          |
|   documents  chunks(+CDF) -> Vector Search chunks_index  parent_chunks             |
|   briefs  audit_log (append-only)  companies_watchlist (masked)  eval_*            |
|   Grants: least privilege per group/SP; row filter on audit_log                    |
+----------------------------------------+-------------------------------------------+
                                         | retrieved evidence (still untrusted text)
                                         v
+------------------------------------------------------------------------------------+
| TB3  Model boundary  Model Serving FMAPI (AI Gateway: rate limits, PII/safety,     |
|      inference tables)                                                             |
|   Prompts: evidence in <evidence> delimiters, spotlighting, JSON schema output     |
+----------------------------------------+-------------------------------------------+
                                         | model output (UNTRUSTED)
                                         v
+------------------------------------------------------------------------------------+
| TB4  Output boundary  CitationValidator  OutputGuard  ResponsibleAIPolicy          |
|      renderer escaping (briefing/renderer.py)                                      |
+----------------------------------------+-------------------------------------------+
                                         | ClientBrief (Verified Facts / AI Recommendations)
                                         v
+------------------------------------------------------------------------------------+
| TB5  Caller boundary  Agent endpoint cra-agent-<env> (OAuth, CAN_QUERY, AI Gateway |
|      per-user rate limit)  RBAC Principal/Permission  PrincipalRateLimiter RunBudget|
|   Callers: analysts (Review App / Playground), CRM integrations, Workflows jobs     |
+------------------------------------------------------------------------------------+

+------------------------------------------------------------------------------------+
| TB6  Control plane  GitHub Actions (OIDC -> SP)  Terraform state  Asset Bundle     |
|      CODEOWNERS on deployment/, infrastructure/, .github/  prod environment review |
+------------------------------------------------------------------------------------+
```

## 3. STRIDE

| Threat | Boundary | Scenario | Controls | Evidence |
|---|---|---|---|---|
| **S**poofing | TB5 | Caller impersonates an analyst to run research or read briefs | Endpoint OAuth; `CAN_QUERY` only for `cra-analysts` (Terraform `databricks_permissions.agent_endpoint` when Terraform owns the endpoint); `security/rbac.py` maps SCIM groups (`cra-viewers`, `cra-analysts`, `cra-operators`, `cra-admins`, `cra-service-principals`) to `Role` and checks `Permission` with `authorize` / `@requires` | `tests/unit/security/test_rbac.py`, `test_llm06_viewer_cannot_run_or_deploy` |
| Spoofing | TB6 | Stolen CI credential deploys to prod | GitHub OIDC federation bound to `repo:<repo>:environment:<env>`; no Databricks secrets in GitHub; PATs refused outside dev (`AppSettings`, `build_workspace_client`) | ADR-0009; `tests/unit/databricks/test_errors_and_auth.py` |
| Spoofing | TB0 | DNS or redirect points the crawler at an internal host | `UrlGuard` resolves hosts and rejects non-global addresses, IP literals, numeric host spellings, non-default ports, embedded credentials; re-checked on every redirect hop | `tests/unit/research/test_url_guard.py`, `test_fetcher.py` |
| **T**ampering | TB0/TB1 | Web page carries hidden instructions or fabricated claims | `HtmlParser` drops invisible elements; `ContentSanitizer` (NFKC, zero-width, bidi, Unicode tag characters, markdown image beacons); `PromptInjectionDetector`; `RetrievalPoisoningGuard` (quarantine on injection, low trust, SEO-spam clusters, severe keyword stuffing; flag on off-domain company claims) | `tests/unit/security/*`, `test_llm04_llm08_poisoned_chunks_quarantined` |
| Tampering | TB2 | Audit records altered after the fact | Hash chain (`governance/audit.py`), `verify_chain`; `audit_log` `delta.appendOnly = true` | `test_audit_chain_tamper_detection`, `tests/unit/governance/test_audit.py` |
| Tampering | TB2 | NULL or parent vectors silently dropped from the index | `chunks.embedding NOT NULL`, CHECK `chunks_embedded_children_only` | `tests/contract/test_ddl_consistency.py` |
| Tampering | TB6 | Malicious change to IaC or workflows | CODEOWNERS on `/deployment/`, `/infrastructure/`, `/.github/`, `databricks.yml`; CodeQL for `actions`; prod behind environment approval; tag must be on `main` | `.github/CODEOWNERS`, `cd.yml` |
| **R**epudiation | TB5 | Requester denies having requested a brief | Audit events carry the service principal and `run_id`; the caller's claimed identity is recorded as `asserted_requester` with `asserted_requester_verified: false`; the AI Gateway inference table `cra_agent_payload` holds the authenticated requester; UC audit logs | ADR-0010 |
| **I**nformation disclosure | TB3/TB4 | PII from a web page appears in a brief or a log | `PiiRedactor` (personal e-mail, phone, SSN, Luhn-valid card, mod-97-valid IBAN, IPv4, residential address) with public business info kept; AI Gateway PII masking on FMAPI endpoints; `observability/logging.py::scrub`; OTel collector deletes identity attributes | `tests/unit/security/test_pii.py`, `test_llm02_*` |
| Information disclosure | TB4 | Model leaks system prompt, spotlight delimiters or credentials | `OutputGuard` violations `PROMPT_LEAK`, `SECRET`, `PII`, `UNSAFE_URL`, `UNCITED_URL`, `CODE_EXECUTION` | `tests/unit/security/test_output_guard.py`, `test_llm05_*` |
| Information disclosure | TB2 | Analysts read raw evidence or audit data | Grants: analysts read `briefs` only; `audit_log` row filter; `companies_watchlist.owner` masked by `mask_email` | `infrastructure/unity_catalog/03_governance.sql`, `05_grants.sql`, `deployment/terraform/grants.tf` |
| **D**enial of service | TB0 | Compression bomb or huge 10-K exhausts memory | Streamed body cap `crawler.max_response_bytes` on the decompressed stream; content-type allow-list | `tests/unit/research/test_fetcher.py` |
| Denial of service | TB3/TB5 | Request flood or reasoning loop burns tokens | AI Gateway endpoint and per-user rate limits; `PrincipalRateLimiter` in `ClientResearchResponsesAgent` (keyed on the asserted requester, raises `RateLimitExceededError`); `RunBudget` (enforced by `MeteredLLMClient`) (default 400,000 tokens, 200 LLM calls, 25.0 cost units per run); per-endpoint circuit breakers; `databricks_budget` in Terraform | `tests/unit/security/test_rate_limiter.py`, `test_llm10_*` |
| **E**levation of privilege | TB5 | Injected instruction makes the agent take an action (send e-mail, call a tool, deploy) | The agent has no write tools toward external systems; outputs are text only; `Permission.DEPLOY` and `MANAGE_SOURCES` are not granted to `ANALYST` or `SERVICE`; jobs run as a least-privilege SP (`allow_cluster_create = false`) | `security/rbac.py::POLICY`, `deployment/terraform/identity.tf` |

## 4. OWASP Top 10 for LLM Applications (2025)

| ID | Risk | Controls in this repository | Tests |
|---|---|---|---|
| LLM01 | Prompt Injection (direct and indirect) | Hidden-text removal (`research/parsing/html.py`); `security/sanitizer.py::ContentSanitizer`; `security/prompt_injection.py::PromptInjectionDetector` (weighted signal families combined by noisy-OR, scanning normalised, de-obfuscated, despaced (uniformly single-spaced letters collapsed by `despaced_runs`), Unicode-tag-decoded, ROT13, URL-decoded and base64/hex-decoded views; block threshold `guardrails.injection_block_threshold` 0.6, 0.55 staging, 0.5 prod); `spotlight()` datamarking with nonce delimiters; evidence rendered inside `<evidence>` blocks with delimiter spoofing neutralised (`qualification/evidence.py::sanitize_untrusted`); system prompts instruct the model to never follow instructions in evidence; the citation judge can raise support by at most 0.2 so evidence text cannot argue itself into "supported" | `tests/security/test_owasp_llm_top10.py::test_llm01_*`, `tests/unit/security/test_prompt_injection.py`, `test_sanitizer.py` |
| LLM02 | Sensitive Information Disclosure | `security/pii.py` policy-driven redaction; `governance/audit.py` redacts before hashing; `observability/logging.py` scrubs credential-shaped values and sensitive keys; AI Gateway PII `MASK` on FMAPI input and output; UC column mask on watchlist owners | `test_llm02_*`, `tests/unit/security/test_pii.py`, `tests/unit/observability/test_logging.py` |
| LLM03 | Supply Chain | Pinned `requirements*.txt` compiled by `uv pip compile`; Dependabot; `pip-audit --strict`; Trivy filesystem and image scans; gitleaks; CodeQL (python, actions); SBOM and provenance on the release image; model endpoints are first-party FMAPI | `.github/workflows/ci.yml`, `codeql.yml`, `.github/dependabot.yml` |
| LLM04 | Data and Model Poisoning | Public sources only with trust scores; robots and allow-list; `security/poisoning.py::RetrievalPoisoningGuard` (cross-domain near-duplicate shingles, SEO-spam clusters on >= 3 domains, keyword stuffing, off-domain claims about the company, low trust); content-hash de-duplication; no fine-tuning on collected data | `test_llm04_llm08_poisoned_chunks_quarantined`, `tests/unit/security/test_poisoning.py` |
| LLM05 | Improper Output Handling | `security/output_guard.py::OutputGuard.check_brief` and `enforce`; Markdown escaping and http(s)-only link targets in `briefing/renderer.py`; structured output validated by Pydantic | `test_llm05_*`, `tests/unit/briefing/test_renderer.py` |
| LLM06 | Excessive Agency | No side-effecting tools exposed to the model; RBAC least privilege (`POLICY`); jobs as SP without cluster-create; human review queue for briefs (`orchestration/review.py`) | `test_llm06_*`, `tests/unit/security/test_rbac.py` |
| LLM07 | System Prompt Leakage | `OutputGuard` `PROMPT_LEAK` (chat-template tokens, spotlight delimiters, datamark characters, configured canaries); prompts contain no secrets or credentials; templates are versioned and fingerprinted | `tests/unit/security/test_output_guard.py` |
| LLM08 | Vector and Embedding Weaknesses | Delta Sync index governed by UC grants on `chunks`; company filter on every dense query and cross-company chunks discarded by the qualification agent; poisoning guard before indexing; `chunks` accepts only embedded child rows | `tests/contract/test_vector_index_contract.py`, `test_llm04_llm08_*` |
| LLM09 | Misinformation | Verified facts must cite registered evidence (`BriefStatement` validator); `CitationValidator` with number and entity hard rules; LLM scores without valid citations discarded; heuristic cross-check reduces confidence on disagreement; Gartner attribution requires an `ANALYST_PUBLIC` source; `governance/responsible_ai.py::ResponsibleAIPolicy` (grounding, recommendation labelling, protected attributes, personal speculation) | `tests/rag_eval/test_hallucination.py`, `test_citation_verification.py`, `test_llm09_*` |
| LLM10 | Unbounded Consumption | AI Gateway rate limits (agent endpoint prod: 1200/min endpoint, 60/min per user); `RunBudget` via `MeteredLLMClient`; `PrincipalRateLimiter` in the serving agent; `request.max_documents` <= 500; crawler page and byte caps; circuit breakers; Terraform `databricks_budget`; `serving_cost_usd_daily` alert | `test_llm10_budget_and_rate_limits` |

## 5. Security-relevant configuration

| Key | Default | Effect |
|---|---|---|
| `guardrails.injection_block_threshold` | 0.6 (staging 0.55, prod 0.5) | Injection score at which text is blocked |
| `guardrails.min_citation_support` | 0.3 (prod 0.35) | Support needed to keep a verified fact |
| `guardrails.redact_pii` | true | PII redaction enabled |
| `guardrails.max_input_chars` | 20,000 | Sanitiser length cap |
| `crawler.allowed_domain_suffixes` | `["sec.gov"]` | Static allow-list (request scope adds the company domain and seed hosts) |
| `crawler.respect_robots_txt` | true | robots.txt enforcement |
| `crawler.max_response_bytes` | 15,000,000 | Response size cap |
| `databricks.token` | unset | Must stay unset in staging and prod |

## 6. Residual risks

| Risk | Detail | Current mitigation | Recommended follow-up |
|---|---|---|---|
| Unverified caller identity | The endpoint runs as its service identity; `requested_by` / `context.user_id` from the request are recorded as `asserted_requester` (unverified) and key the per-requester rate limit, so a caller can choose its rate-limit bucket and the label in the audit trail. | Endpoint OAuth and `CAN_QUERY`; AI Gateway per-user limits keyed on the authenticated identity; inference tables | Forward the authenticated identity from Model Serving into the agent when the platform exposes it. |
| DNS rebinding | Documented in `research/url_guard.py`: DNS answers can change between the guard's resolution and the socket connect. | Re-validation on every redirect hop; IP literals refused; per-request allow-list | Enforce egress at the network layer (serverless egress control or firewall allow-list for `sec.gov` and approved domains); pin the resolved IP for the connection. |
| Heuristic injection detection | Regex and signal families can be evaded by novel phrasing or non-English payloads. | Defence in depth: evidence is data-marked, outputs are validated, the model has no tools | Add an LLM-based or classifier-based second opinion on flagged-but-not-blocked text; track `guardrail_blocks_hourly`. |
| Audit chain truncation | Hash chaining detects edits, deletions and reorders but not removal of the tail by a principal able to rewrite the file or table. | `delta.appendOnly`; row filter; grants | Anchor the latest `record_hash` externally (MLflow run tag or write-once storage) at the end of each run. |
| Concurrent audit writers | `AuditLogger` / `FanOutAuditLogger` serialise writes per process; multiple serving replicas and jobs produce independent chains replicated into the same `audit_log` table. | Each chain is independently verifiable | One chain per replica keyed by replica id, or a single sequencer job that ingests per-run records. |
| Lexical entailment gaps | A fact supported only through paraphrase or synonym can pass the lexical rule when content words overlap but meaning differs (for example negation). | Number and entity hard rules; optional LLM judge with bounded uplift | Enable the judge (`judge_endpoint`) in prod; add negation-aware checks. |
| Public-source integrity | A compromised corporate site or newsroom is a trusted first-party source (trust 0.85). | Cross-source corroboration via per-domain caps in `EvidenceRanker`; XBRL facts preferred for scale | Flag single-source facts in the brief; human review before external use. |
| Analyst ToS | A user can seed a URL that is public today and gated tomorrow. | Path allow-list and gated-path deny-list per analyst domain | Periodic legal review of `DEFAULT_ANALYST_POLICIES`. |
| Model provider behaviour | FMAPI model updates can change output style and pass rates. | Schema validation, citation validation, nightly evaluation gate | Pin model versions where the platform allows; re-baseline evaluation on provider changes. |
