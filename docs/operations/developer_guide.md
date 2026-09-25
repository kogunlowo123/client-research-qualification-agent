# Developer Guide

For contribution workflow and commit conventions see
[CONTRIBUTING.md](../../CONTRIBUTING.md). This guide covers local setup, the
build targets, the conventions the codebase relies on, and how to extend it.

## 1. Setup

Requirements: Python 3.11 or 3.12 (`requires-python = ">=3.11,<3.14"`),
[uv](https://docs.astral.sh/uv/), GNU make. Optional: Docker, Terraform >= 1.6,
Databricks CLI >= 0.250.

```bash
make install            # uv venv --python 3.12; install requirements-dev.txt; editable install (--no-deps)
pre-commit install      # ruff, mypy, shellcheck, terraform fmt, gitleaks on commit
cp .env.example .env    # non-secret settings only
```

`requirements.txt` and `requirements-dev.txt` are compiled from `pyproject.toml`
by `make lock` (`uv pip compile --universal --python-version 3.11`); commit both
after changing dependencies. The dev lock includes the `dev`, `databricks` and
`otlp` extras, so the Databricks adapters are importable and type-checked
locally.

On Windows without make, the equivalent is:

```powershell
uv venv --python 3.12
uv pip install -r requirements-dev.txt
uv pip install --no-deps -e .
.venv\Scripts\python.exe -m pytest -q
```

### Running locally

With `CRA_ENVIRONMENT=local` (the default), `agent/factory.py::build_runtime`
binds offline adapters: `HashingEmbeddingClient` (512-d), `InMemoryVectorIndex`,
`InMemoryDocumentStore`, `JsonlBriefRepository` under `var/briefs`, and a
hash-chained `AuditLogger` at `var/audit/audit.jsonl` (override the root with
`CRA_VAR_DIR`). The LLM is `None` (deterministic paths) unless a workspace host
is configured through `DATABRICKS_HOST` or `CRA_DATABRICKS__HOST`, in which case
the Foundation Model API endpoints are used with your CLI profile
(`databricks auth login`). Public sources are fetched over the network through
`PolicyEnforcingFetcher` in every environment.

```bash
uv run cra research --company "Microsoft Corporation" --ticker MSFT --domain microsoft.com
```

CLI reference (`src/client_research_agent/cli.py`; global options `--environment {local,dev,staging,prod}`, `--version`):

| Command | Options |
|---|---|
| `cra research` | `--company` (required), `--domain`, `--ticker`, `--cik`, `--max-documents` (40), `--industry`, `--format {md,json}`, `--output PATH` |
| `cra ingest` | `--company` (required), `--domain`, `--ticker`, `--cik`, `--max-documents` (40) |
| `cra evaluate` | `--dataset` (JSON Lines path or `golden`), `--min-pass-rate` (0.85), `--min-citation-coverage` (0.9), `--min-grounded-fact-ratio` (0.8), `--output PATH` |
| `cra serve-check` | none; exits 0 when the agent can start (container health check) |

Job entry points (`src/client_research_agent/workflows/`, console scripts in
`pyproject.toml`) share `--environment`, `--catalog`, `--schema`,
`--vs-endpoint`, `--experiment`, `--warehouse-id`, `--contact-email` from `workflows/common.py`:

| Script | Additional options |
|---|---|
| `cra-ingest` | `--company`, `--domain`, `--ticker`, `--max-documents`, `--watchlist-table`, `--sync-index`, `--sync-index-only` |
| `cra-brief` | `--run-id`, `--company`, `--domain`, `--ticker`, `--requested-by`, `--max-documents` |
| `cra-evaluate` | `--mode {brief,dataset}`, `--brief-id`, `--eval-table`, `--results-table`, `--model-uri`, `--min-pass-rate`, `--min-citation-coverage`, `--min-grounded-fact-ratio`, `--fail-on-gate` |
| `cra-deploy-agent` | `--environment`, `--stage {log,promote,deploy-champion}`, `--uc-model`, `--model-version`, `--experiment`, `--vs-endpoint`, `--vs-index`, `--endpoint-name`, `--endpoint-config`, `--warehouse-id`, `--catalog`, `--schema`, `--contact-email` |

Local observability stack (MLflow on :5000, OTel Collector on :4317/:4318,
Jaeger on :16686):

```bash
docker compose up -d mlflow otel-collector jaeger
docker compose run --rm agent research --company "Microsoft Corporation" --ticker MSFT --domain microsoft.com
```

## 2. Make targets

| Target | Does |
|---|---|
| `make install` | Create `.venv` with pinned dev dependencies and the package in editable mode |
| `make lock` | Recompile `requirements.txt` and `requirements-dev.txt` |
| `make lint` | `ruff check`, `ruff format --check`, `shellcheck` on `deployment/workflows/*.sh`, `terraform fmt -check` (tools skipped if absent) |
| `make format` | Apply ruff fixes and formatting, `terraform fmt` |
| `make typecheck` | `mypy` (strict, pydantic plugin) |
| `make test` | Full test suite |
| `make test-unit` | `tests/unit` |
| `make test-integration` | `-m "integration or contract"` |
| `make test-e2e` | `-m e2e` |
| `make test-security` | `-m security` |
| `make test-perf` | `-m performance --benchmark-only` |
| `make coverage` | Unit tests with branch coverage, XML and HTML reports, `--cov-fail-under=90` |
| `make security` | `bandit -c pyproject.toml -r src`, `pip-audit --strict` |
| `make build` | Wheel and sdist into `dist/` |
| `make docker-build` | Container image `client-research-agent:local` |
| `make bundle-validate` | `databricks bundle validate` for dev, staging, prod |
| `make deploy-dev`, `make deploy-staging` | `deployment/workflows/deploy.sh dev` / `staging --evaluate` |
| `make clean` | Remove build, test and cache artefacts |

## 3. Conventions

| Area | Convention |
|---|---|
| Layout | `src/` layout, package `client_research_agent`, typed (`py.typed`) |
| Style | ruff, line length 110, rule sets `E F W I B UP S C4 SIM RUF PL PT N ANN` |
| Types | `mypy --strict` over the package; vendor SDK stubs ignored (`databricks.*`, `mlflow.*`, `openai.*`, ...) |
| Domain types | Frozen Pydantic models with `extra="forbid"` (`models/domain.py`); evidence cannot be mutated after it is cited |
| Dependencies on vendors | Only `client_research_agent.databricks` imports vendor SDKs, and only lazily elsewhere ([ADR-0001](../adr/0001-hexagonal-ports-and-adapters.md)) |
| Errors | Raise from `utils/errors.py`. `TransientError` subclasses are retried and count toward circuit breakers; everything else fails fast. Map foreign exceptions at the adapter boundary (`map_databricks_error`, `map_openai_error`) |
| Resilience | Wrap remote calls with `utils/resilience.py` (`CircuitBreaker`, retry policy from `ResilienceSettings`) |
| LLM calls | Always through `services/structured.py::complete_structured` with a Pydantic schema; always provide a deterministic path ([ADR-0005](../adr/0005-deterministic-fallback-for-every-llm-step.md)) |
| Prompts | Markdown templates with YAML front-matter in `prompts/templates/`; bump `version` on any change; placeholders must be declared in `variables` (checked at load) |
| Untrusted text | Evidence enters prompts only through `EvidenceRegistry.render_block` (delimited and sanitised); never format untrusted text into instructions |
| Observability | Decorate units of work with `@traced(name, span_type=SpanType.X)` or `with span(...)`; metrics via `get_metrics().increment/observe` with dotted or `_total` names; log with `get_logger(__name__)` and key-value fields, never f-string secrets |
| Configuration | New settings go on the relevant model in `config/settings.py` with bounds (`Field(ge=..., le=...)`); environment differences go in `config/environments/<env>.yaml`; secrets never go in YAML |
| SQL | Parameter markers for every value; identifiers validated (`validate_identifier`) and back-quoted |

## 4. Extending the system

### 4.1 Add a research source

1. Create `src/client_research_agent/research/sources/<name>.py`. Take an
   `HttpFetcher` in the constructor and fetch only through it, so `UrlGuard`,
   robots.txt, rate limiting and breakers apply. Return candidate URLs (or
   documents) plus `SkippedSource` entries; never raise for a single bad URL.
2. If the source lives on a new domain, it must be in the request scope (seed
   URL or request domain) or in `crawler.allowed_domain_suffixes` in
   `config/environments/base.yaml`. Widening the static allow-list needs review;
   licensed or ToS-restricted content is out of scope ([ADR-0007](../adr/0007-public-sources-only-and-analyst-compliance.md)).
3. Wire it into `research/ingestion.py::IngestionPipeline` (`_collect_sources`
   runs sources concurrently and converts failures into
   `SkipReason.SOURCE_FAILED`) and into `research/__init__.py::build_ingestion_pipeline`.
   Assign a trust score (`trust_score` in `ingestion.py`) and, if needed, a
   `DocumentType` with a `SOURCE_TRUST` entry in `qualification/signals.py`.
4. Tests: `respx`-mocked HTTP with fixtures under `tests/unit/research/fixtures/`,
   covering robots disallow, redirects out of scope, oversize responses and a
   failing endpoint.

### 4.2 Add or change a qualification criterion

1. Add the member to `Criterion` in `models/domain.py`.
2. Add a `CriterionDefinition` to `_DEFINITIONS` in `qualification/criteria.py`:
   title, description, a rubric with levels 0-5 (validated), at least one
   retrieval query containing `{company}` (validated), positive and negative
   signal lexicons for the heuristic, and a discovery question.
3. Add its weight to `ScoringSettings.weights` in `config/settings.py`. The
   validator requires a weight for every criterion and a sum of 1.0; adjust the
   other weights and record the rationale in an ADR.
4. If it needs structured signals (like revenue bands for scale), extend
   `qualification/signals.py` and `qualification/heuristic.py`.
5. Update the evaluation data: `expected_scores` in the golden set and any
   `eval_set` rows.
6. Tests: `tests/unit/qualification/test_signals_and_criteria.py`,
   `tests/unit/scoring/test_engine.py` (threshold and sensitivity cases).

### 4.3 Add an adapter for a port

1. Implement the Protocol from `services/ports.py` (for example `VectorIndex`)
   in `client_research_agent/databricks/` (vendor-backed) or
   `services/local.py` (in-process). Map vendor exceptions to the typed
   hierarchy.
2. Add the adapter to the parametrised fixtures in `tests/contract/conftest.py`
   so the existing contract tests run against it; add a fake to
   `tests/contract/fakes.py` if it needs a remote service.
3. Bind it in `agent/factory.py::build_runtime` (or inject it through the
   `build_runtime` keyword arguments in tests).

### 4.4 Change a prompt

Edit the template, bump `version` in its front-matter, run the unit tests
(`tests/unit/prompts/test_registry.py` validates placeholders) and the RAG
evaluation suites. The template fingerprint changes automatically and is recorded
on every brief, so the change is traceable in production.

## 5. Testing

### Pyramid

| Layer | Location | Marker | Runs in CI |
|---|---|---|---|
| Unit | `tests/unit/<package>/` | none | `unit-tests` job (Python 3.11 and 3.12), coverage gate 90% |
| Contract (adapter parity) | `tests/contract/` | `contract` (applied to every test by `tests/contract/conftest.py`); parametrised over local and Databricks adapters | `integration-tests` |
| Integration | `tests/integration/` | `integration` | `integration-tests` |
| End-to-end (local adapters) | `tests/e2e/` | `e2e` | `integration-tests` |
| RAG quality | `tests/rag_eval/` | `rag_eval` | `integration-tests` |
| Security (OWASP LLM Top 10) | `tests/security/` | `security` | `integration-tests` |
| Performance budgets | `tests/performance/` | `performance` | not in CI (`make test-perf`) |
| Live workspace | any | `databricks` (requires `DATABRICKS_HOST`) | not in CI |

Markers are declared in `pyproject.toml` and `--strict-markers` is on, so an
undeclared marker fails collection.

Shared test support: `tests/support/doubles.py` (builders such as `make_chunk`),
`tests/contract/fakes.py` (Statement Execution and Vector Search fakes),
`tests/security/corpus.py` (injection payloads and benign paragraphs).

### Coverage

`[tool.coverage.report] fail_under = 90` with branch coverage over
`client_research_agent`. CI enforces it on `tests/unit` only
(`--cov-fail-under=90`), so new code needs unit tests even when it is covered by
integration tests.

### Determinism

- No test reaches the network: HTTP goes through `respx` or doubles; the hashing
  embedder replaces model embeddings; LLM tests use scripted doubles.
- Code that depends on "today" accepts a `today` argument (for example
  `QualificationAgent`, `EvidenceRanker`); tests pass a fixed date.
- `ResilienceSettings` in tests uses zero backoff and injected `sleep`.
