from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from client_research_agent.security.rbac import (
    POLICY,
    AccessDeniedError,
    Permission,
    Principal,
    Role,
    acting_as,
    authorize,
    current_principal,
    map_groups_to_roles,
    principal_from_groups,
    requires,
)
from client_research_agent.utils.errors import SecurityViolationError

EXPECTED: dict[Role, set[Permission]] = {
    Role.VIEWER: {Permission.VIEW_BRIEF},
    Role.ANALYST: {Permission.RUN_RESEARCH, Permission.VIEW_BRIEF, Permission.VIEW_EVIDENCE},
    Role.OPERATOR: {Permission.VIEW_BRIEF, Permission.MANAGE_SOURCES, Permission.DEPLOY},
    Role.ADMIN: set(Permission),
    Role.SERVICE: {Permission.RUN_RESEARCH, Permission.VIEW_BRIEF, Permission.VIEW_EVIDENCE},
}


class RecordingAudit:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def record(self, event_type: str, payload: Mapping[str, Any]) -> None:
        self.events.append((event_type, dict(payload)))


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("permission", list(Permission))
def test_rbac_matrix(role: Role, permission: Permission) -> None:
    principal = Principal(id="u1", roles=frozenset({role}))
    if permission in EXPECTED[role]:
        authorize(principal, permission)
    else:
        with pytest.raises(AccessDeniedError):
            authorize(principal, permission)


def test_least_privilege_properties() -> None:
    assert Permission.READ_AUDIT in POLICY[Role.ADMIN]
    assert all(Permission.READ_AUDIT not in POLICY[r] for r in Role if r is not Role.ADMIN)
    assert Permission.DEPLOY not in POLICY[Role.SERVICE]
    assert Principal(id="nobody").permissions() == frozenset()


def test_multiple_roles_union() -> None:
    principal = Principal(id="u", roles=frozenset({Role.VIEWER, Role.OPERATOR}))
    assert principal.permissions() == {Permission.VIEW_BRIEF, Permission.MANAGE_SOURCES, Permission.DEPLOY}


def test_access_denied_is_permission_and_security_error() -> None:
    error = AccessDeniedError("u", Permission.DEPLOY)
    assert isinstance(error, PermissionError)
    assert isinstance(error, SecurityViolationError)
    assert error.permission == "deploy"
    assert error.principal_id == "u"


def test_group_mapping() -> None:
    assert map_groups_to_roles(["CRA-Analysts", "unrelated"]) == {Role.ANALYST}
    assert map_groups_to_roles(["admins"], {"admins": "admin"}) == {Role.ADMIN}
    principal = principal_from_groups("sp-1", ["cra-service-principals"])
    assert principal.roles == {Role.SERVICE}
    assert principal.groups == {"cra-service-principals"}


def test_authorize_audits_decisions() -> None:
    audit = RecordingAudit()
    viewer = Principal(id="v", roles=frozenset({Role.VIEWER}))
    authorize(viewer, Permission.VIEW_BRIEF, audit=audit)
    with pytest.raises(AccessDeniedError):
        authorize(viewer, Permission.DEPLOY, audit=audit)
    assert [e[1]["decision"] for e in audit.events] == ["allow", "deny"]
    assert audit.events[0][1]["roles"] == ["viewer"]


def test_requires_decorator_sources() -> None:
    analyst = Principal(id="a", roles=frozenset({Role.ANALYST}))
    viewer = Principal(id="v", roles=frozenset({Role.VIEWER}))

    @requires(Permission.RUN_RESEARCH)
    def run(company: str, *, principal: Principal | None = None) -> str:
        return company

    @requires(Permission.RUN_RESEARCH)
    def run_positional(principal: Principal, company: str) -> str:
        return company

    assert run("Acme", principal=analyst) == "Acme"
    assert run_positional(analyst, "Acme") == "Acme"
    with pytest.raises(AccessDeniedError):
        run("Acme", principal=viewer)
    with pytest.raises(AccessDeniedError, match="anonymous"):
        run("Acme")
    assert current_principal() is None
    with acting_as(analyst):
        assert current_principal() is analyst
        assert run("Acme") == "Acme"
    assert current_principal() is None
    assert run.__name__ == "run"
