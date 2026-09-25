"""Workspace authentication via Databricks unified auth.

Resolution order is the SDK's own unified-auth chain: explicit arguments, then
``DATABRICKS_*`` environment variables (``DATABRICKS_HOST`` plus
``DATABRICKS_CLIENT_ID``/``DATABRICKS_CLIENT_SECRET`` for an OAuth M2M service
principal), then a ``~/.databrickscfg`` profile, then ambient notebook/job
credentials. Personal access tokens are accepted only in ``local`` and ``dev``;
staging and production must authenticate as a service principal, matching the
``AppSettings`` validator that already rejects ``databricks.token`` there.

The Foundation Model API is OpenAI-compatible, so chat and embedding adapters
use the ``openai`` client with ``base_url={host}/serving-endpoints`` and a
*callable* API key backed by :meth:`WorkspaceCredentials.bearer_token`. The
callable is evaluated per request, so short-lived OAuth tokens are refreshed
transparently by the SDK's credential provider.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from client_research_agent.config.settings import AppSettings, Environment
from client_research_agent.utils.errors import ConfigurationError

_PAT_FORBIDDEN = frozenset({Environment.STAGING, Environment.PROD})
_PAT_AUTH_TYPE = "pat"

WorkspaceClientFactory = Callable[..., Any]


def _default_factory(**kwargs: Any) -> Any:
    return importlib.import_module("databricks.sdk").WorkspaceClient(**kwargs)


def normalize_host(host: str) -> str:
    cleaned = host.strip().rstrip("/")
    if not cleaned:
        raise ConfigurationError("Databricks host is empty")
    if not cleaned.startswith(("https://", "http://")):
        cleaned = f"https://{cleaned}"
    return cleaned


def build_workspace_client(
    settings: AppSettings,
    *,
    profile: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    factory: WorkspaceClientFactory | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Build a ``WorkspaceClient`` through unified auth, refusing PATs outside local/dev.

    Explicit ``client_id``/``client_secret`` select OAuth M2M; otherwise the SDK
    discovers credentials from the environment or ``profile``.
    """
    env = os.environ if environ is None else environ
    strict = settings.environment in _PAT_FORBIDDEN
    if strict and (settings.databricks.token is not None or env.get("DATABRICKS_TOKEN")):
        raise ConfigurationError(
            f"personal access tokens are forbidden in {settings.environment.value}; "
            "authenticate as a service principal (DATABRICKS_CLIENT_ID/DATABRICKS_CLIENT_SECRET)"
        )
    if (client_id is None) != (client_secret is None):
        raise ConfigurationError("client_id and client_secret must be provided together")

    kwargs: dict[str, Any] = {"product": "client-research-agent", "product_version": "1.0.0"}
    if settings.databricks.host:
        kwargs["host"] = normalize_host(settings.databricks.host)
    if profile:
        kwargs["profile"] = profile
    if client_id is not None and client_secret is not None:
        kwargs.update(client_id=client_id, client_secret=client_secret, auth_type="oauth-m2m")
    elif settings.databricks.token is not None:
        kwargs.update(token=settings.databricks.token.get_secret_value(), auth_type=_PAT_AUTH_TYPE)

    try:
        client = (factory or _default_factory)(**kwargs)
    except ValueError as exc:  # the SDK raises ValueError when no credential chain resolves
        raise ConfigurationError(f"Databricks authentication failed: {exc}") from exc

    auth_type = getattr(client.config, "auth_type", None)
    if strict and auth_type == _PAT_AUTH_TYPE:
        raise ConfigurationError(
            f"resolved auth type 'pat' is forbidden in {settings.environment.value}; use a service principal"
        )
    return client


@dataclass(frozen=True, slots=True)
class WorkspaceCredentials:
    """Host plus a fresh-token source derived from a ``WorkspaceClient``'s config."""

    host: str
    header_factory: Callable[[], Mapping[str, str]]

    @classmethod
    def from_workspace_client(cls, client: Any) -> WorkspaceCredentials:
        config = client.config
        host = getattr(config, "host", None)
        if not host:
            raise ConfigurationError("WorkspaceClient has no resolved host")
        return cls(host=normalize_host(str(host)), header_factory=config.authenticate)

    def auth_headers(self) -> dict[str, str]:
        return dict(self.header_factory())

    def bearer_token(self) -> str:
        """Return the current OAuth/bearer token (refreshed by the SDK when near expiry)."""
        headers = self.auth_headers()
        value = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
        scheme, _, token = value.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise ConfigurationError("workspace credentials did not yield a bearer token")
        return token.strip()

    @property
    def serving_base_url(self) -> str:
        """Base URL of the OpenAI-compatible Foundation Model API."""
        return f"{self.host}/serving-endpoints"


OpenAIFactory = Callable[..., Any]


def _default_openai_factory(**kwargs: Any) -> Any:
    return importlib.import_module("openai").OpenAI(**kwargs)


def build_openai_client(
    credentials: WorkspaceCredentials,
    *,
    timeout_seconds: float = 60.0,
    factory: OpenAIFactory | None = None,
) -> Any:
    """OpenAI client against ``{host}/serving-endpoints`` with per-request token refresh.

    SDK-level retries are disabled (``max_retries=0``) because the adapters apply
    their own retry policy and circuit breaker.
    """
    return (factory or _default_openai_factory)(
        base_url=credentials.serving_base_url,
        api_key=credentials.bearer_token,
        timeout=timeout_seconds,
        max_retries=0,
    )
