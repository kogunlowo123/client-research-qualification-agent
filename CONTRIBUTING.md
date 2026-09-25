# Contributing

Thanks for helping improve the Client Research & Qualification Agent.

## Development setup

Requirements: Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), GNU make.
Optional: Docker, Terraform >= 1.6, Databricks CLI >= 0.250.

```bash
make install                 # .venv with pinned dev dependencies + editable package
pre-commit install           # ruff, mypy, shellcheck, terraform fmt, gitleaks on commit
cp .env.example .env         # local settings (no secrets)
```

## Workflow

1. Create a branch from `main` (`feat/...`, `fix/...`, `chore/...`).
2. Make focused changes with tests. Unit coverage must stay at or above 90%.
3. Run the checks CI runs:

   ```bash
   make lint typecheck coverage test-integration security
   ```

4. Update `CHANGELOG.md` under **Unreleased**.
5. Open a pull request using the template. CI (lint, type check, tests,
   security scans, build) must pass and a code owner must approve.

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`).

## Dependencies

Runtime and development dependencies are declared in `pyproject.toml` and
pinned in `requirements.txt` / `requirements-dev.txt`. After changing
`pyproject.toml`, run `make lock` and commit both files.

## Databricks changes

- Jobs, the MLflow experiment, the registered model and the monitoring schema
  live in the Asset Bundle (`databricks.yml`, `deployment/databricks/resources`).
  Validate with `make bundle-validate`.
- Platform resources (catalog, schemas, grants, Vector Search, warehouse,
  service principal) live in `deployment/terraform`; run `terraform fmt` and
  `terraform validate` before opening a PR.
- Table DDL, governance functions and grants live in `infrastructure/unity_catalog`;
  every statement must stay idempotent.

## Releases

Maintainers bump `version` in `pyproject.toml`, move **Unreleased** entries in
`CHANGELOG.md` under the new version and push a `vX.Y.Z` tag on `main`. CI
publishes the wheel/sdist to GitHub Releases and the image to GHCR; CD deploys
to prod after approval on the `prod` environment.

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
