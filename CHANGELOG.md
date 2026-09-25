# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-24

### Added

- Agentic RAG pipeline that researches a company from SEC EDGAR and corporate
  investor-relations/newsroom pages and produces a scored, cited client brief.
- `cra` CLI with `research`, `ingest`, `evaluate` and `serve-check` commands.
- Databricks Asset Bundle with ingestion refresh, brief generation, nightly
  evaluation and agent deployment workflows (serverless, wheel entry points).
- Mosaic AI Agent Framework deployment of an MLflow ResponsesAgent registered
  in Unity Catalog, promoted through `challenger` / `champion` aliases.
- Terraform for Unity Catalog (catalog, schemas, volumes, least-privilege
  grants), service principal with GitHub OIDC federation, secret scope,
  Vector Search endpoint and Delta Sync index, serverless SQL warehouse,
  cluster policy and budget.
- Unity Catalog DDL with Change Data Feed on `chunks`, separate
  `parent_chunks`, PII column masks, row filters and governance tags.
- AI Gateway configuration (usage tracking, inference tables, rate limits,
  PII/safety guardrails), SQL alerts, dashboard queries and Lakehouse
  Monitoring.
- CI (lint, type check, unit tests with 90% coverage gate, integration tests,
  bandit, pip-audit, Trivy, gitleaks, CodeQL, wheel/sdist/image build) and CD
  (OIDC deploy to staging with evaluation gate and smoke test; prod on tags
  behind environment approval, with automatic rollback).

[Unreleased]: https://github.com/kogunlowo123/client-research-qualification-agent/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kogunlowo123/client-research-qualification-agent/releases/tag/v1.0.0
