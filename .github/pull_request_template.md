## Summary

<!-- What does this change do and why? Link issues with "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] Feature
- [ ] Refactor / tech debt
- [ ] Infrastructure / deployment (Terraform, bundle, UC DDL, CI/CD)
- [ ] Documentation

## How was this tested?

- [ ] `make lint typecheck`
- [ ] `make coverage` (unit coverage >= 90%)
- [ ] `make test-integration test-e2e test-security`
- [ ] `databricks bundle validate` / `terraform validate` (if deployment changed)
- [ ] Evaluated against `eval_set` (if prompts, retrieval or scoring changed)

## Checklist

- [ ] No secrets, tokens or workspace-specific identifiers committed
- [ ] New or changed UC DDL is idempotent and grants stay least-privilege
- [ ] `CHANGELOG.md` updated under **Unreleased**
- [ ] Docs updated where behavior or configuration changed
