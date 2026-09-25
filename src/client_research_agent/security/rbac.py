"""Role-based access control with least-privilege defaults (OWASP LLM06 excessive agency).

Roles are derived from Databricks workspace / account groups via a
configurable mapping, so access is administered where identities already live
(SCIM-synced groups) rather than in application config.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ParamSpec, TypeVar

from client_research_agent.services.ports import AuditSink
from client_research_agent.utils.errors import SecurityViolationError

P = ParamSpec("P")
T = TypeVar("T")


class Role(StrEnum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    OPERATOR = "operator"
    ADMIN = "admin"
    SERVICE = "service"


class Permission(StrEnum):
    RUN_RESEARCH = "run_research"
    VIEW_BRIEF = "view_brief"
    VIEW_EVIDENCE = "view_evidence"
    MANAGE_SOURCES = "manage_sources"
    DEPLOY = "deploy"
    READ_AUDIT = "read_audit"


POLICY: Mapping[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.VIEW_BRIEF}),
    Role.ANALYST: frozenset({Permission.RUN_RESEARCH, Permission.VIEW_BRIEF, Permission.VIEW_EVIDENCE}),
    Role.OPERATOR: frozenset({Permission.VIEW_BRIEF, Permission.MANAGE_SOURCES, Permission.DEPLOY}),
    Role.ADMIN: frozenset(Permission),
    Role.SERVICE: frozenset({Permission.RUN_RESEARCH, Permission.VIEW_BRIEF, Permission.VIEW_EVIDENCE}),
}

DEFAULT_GROUP_ROLE_MAPPING: Mapping[str, Role] = {
    "cra-viewers": Role.VIEWER,
    "cra-analysts": Role.ANALYST,
    "cra-operators": Role.OPERATOR,
    "cra-admins": Role.ADMIN,
    "cra-service-principals": Role.SERVICE,
}


class AccessDeniedError(SecurityViolationError, PermissionError):
    def __init__(self, principal_id: str, permission: Permission | str) -> None:
        super().__init__(f"principal '{principal_id}' lacks permission '{permission}'")
        self.principal_id = principal_id
        self.permission = str(permission)


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    roles: frozenset[Role] = field(default_factory=frozenset)
    groups: frozenset[str] = field(default_factory=frozenset)

    def permissions(self, policy: Mapping[Role, frozenset[Permission]] = POLICY) -> frozenset[Permission]:
        granted: set[Permission] = set()
        for role in self.roles:
            granted |= policy.get(role, frozenset())
        return frozenset(granted)

    def has(self, permission: Permission, policy: Mapping[Role, frozenset[Permission]] = POLICY) -> bool:
        return permission in self.permissions(policy)


def map_groups_to_roles(
    groups: Iterable[str], mapping: Mapping[str, Role | str] | None = None
) -> frozenset[Role]:
    """Translate workspace group names (case-insensitive) to roles; unknown groups grant nothing."""
    table = {name.lower(): Role(role) for name, role in (mapping or DEFAULT_GROUP_ROLE_MAPPING).items()}
    return frozenset(table[g.lower()] for g in groups if g.lower() in table)


def principal_from_groups(
    principal_id: str, groups: Iterable[str], mapping: Mapping[str, Role | str] | None = None
) -> Principal:
    group_set = frozenset(groups)
    return Principal(id=principal_id, roles=map_groups_to_roles(group_set, mapping), groups=group_set)


def authorize(
    principal: Principal,
    permission: Permission,
    *,
    audit: AuditSink | None = None,
    policy: Mapping[Role, frozenset[Permission]] = POLICY,
) -> None:
    """Raise :class:`AccessDeniedError` unless ``principal`` holds ``permission``; denials are audited."""
    allowed = principal.has(permission, policy)
    if audit is not None:
        audit.record(
            "authorization",
            {
                "principal": principal.id,
                "permission": permission.value,
                "decision": "allow" if allowed else "deny",
                "roles": sorted(r.value for r in principal.roles),
            },
        )
    if not allowed:
        raise AccessDeniedError(principal.id, permission)


_current_principal: ContextVar[Principal | None] = ContextVar("cra_principal", default=None)


@contextmanager
def acting_as(principal: Principal) -> Iterator[Principal]:
    token = _current_principal.set(principal)
    try:
        yield principal
    finally:
        _current_principal.reset(token)


def current_principal() -> Principal | None:
    return _current_principal.get()


def requires(
    permission: Permission, *, audit: AuditSink | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Guard a function; the principal comes from a ``principal`` kwarg, a positional arg or ``acting_as``."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            candidate = kwargs.get("principal")
            principal = candidate if isinstance(candidate, Principal) else None
            if principal is None:
                principal = next((a for a in args if isinstance(a, Principal)), None)
            if principal is None:
                principal = current_principal()
            if principal is None:
                raise AccessDeniedError("anonymous", permission)
            authorize(principal, permission, audit=audit)
            return func(*args, **kwargs)

        return wrapper

    return decorator
