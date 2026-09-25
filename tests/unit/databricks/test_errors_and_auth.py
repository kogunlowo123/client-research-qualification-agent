from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from databricks.sdk.errors import platform
from pydantic import SecretStr

from client_research_agent.config.settings import AppSettings, DatabricksSettings, Environment
from client_research_agent.databricks.auth import (
    WorkspaceCredentials,
    build_openai_client,
    build_workspace_client,
    normalize_host,
)
from client_research_agent.databricks.errors import DatabricksRequestError, map_databricks_error
from client_research_agent.utils.errors import (
    ConfigurationError,
    OutputValidationError,
    RateLimitedError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)

# ------------------------------------------------------------------- errors


class _StatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"status {status}")
        self.status_code = status


class _Response:
    status_code = 502


class _ResponseError(Exception):
    response = _Response()


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (platform.TooManyRequests("slow down", retry_after_secs=7), RateLimitedError),
        (_StatusError(429), RateLimitedError),
        (platform.DeadlineExceeded("late"), UpstreamTimeoutError),
        (TimeoutError("late"), UpstreamTimeoutError),
        (_StatusError(504), UpstreamTimeoutError),
        (platform.TemporarilyUnavailable("busy"), UpstreamServiceError),
        (platform.InternalError("boom"), UpstreamServiceError),
        (ConnectionError("reset"), UpstreamServiceError),
        (_ResponseError("bad gateway"), UpstreamServiceError),
        (platform.PermissionDenied("no"), ConfigurationError),
        (platform.Unauthenticated("who"), ConfigurationError),
        (_StatusError(401), ConfigurationError),
        (platform.BadRequest("bad"), DatabricksRequestError),
        (platform.NotFound("gone"), DatabricksRequestError),
        (ValueError("weird"), DatabricksRequestError),
    ],
)
def test_map_databricks_error(exc: BaseException, expected: type[Exception]) -> None:
    mapped = map_databricks_error(exc, "ctx")
    assert isinstance(mapped, expected)
    assert "ctx" in str(mapped)


def test_map_databricks_error_keeps_retry_after_and_passthrough() -> None:
    mapped = map_databricks_error(platform.TooManyRequests("x", retry_after_secs=7), "ctx")
    assert isinstance(mapped, RateLimitedError)
    assert mapped.retry_after_seconds == 7.0
    original = OutputValidationError("already typed")
    assert map_databricks_error(original, "ctx") is original
    assert map_databricks_error(_StatusError(400), "ctx").status_code == 400  # type: ignore[attr-defined]


# --------------------------------------------------------------------- auth


@dataclass
class _Config:
    host: str | None = "https://adb-1.azuredatabricks.net"
    auth_type: str = "oauth-m2m"
    headers: dict[str, str] = field(default_factory=lambda: {"Authorization": "Bearer tok-1"})

    def authenticate(self) -> dict[str, str]:
        return dict(self.headers)


@dataclass
class _Client:
    config: _Config
    kwargs: dict[str, Any]


def _factory(auth_type: str = "oauth-m2m") -> Any:
    def build(**kwargs: Any) -> _Client:
        return _Client(config=_Config(auth_type=auth_type), kwargs=kwargs)

    return build


def _settings(env: Environment, **databricks: Any) -> AppSettings:
    return AppSettings.model_construct(environment=env, databricks=DatabricksSettings(**databricks))


def test_normalize_host() -> None:
    assert normalize_host("adb-1.azuredatabricks.net/") == "https://adb-1.azuredatabricks.net"
    assert normalize_host("https://x.cloud.databricks.com") == "https://x.cloud.databricks.com"
    with pytest.raises(ConfigurationError):
        normalize_host("  ")


def test_build_workspace_client_oauth_m2m_in_prod() -> None:
    settings = _settings(Environment.PROD, host="adb-1.azuredatabricks.net")
    client = build_workspace_client(
        settings, client_id="sp-id", client_secret="sp-secret", factory=_factory(), environ={}
    )
    assert client.kwargs["host"] == "https://adb-1.azuredatabricks.net"
    assert client.kwargs["auth_type"] == "oauth-m2m"
    assert client.kwargs["client_id"] == "sp-id"
    assert "token" not in client.kwargs


def test_build_workspace_client_uses_ambient_chain_and_profile() -> None:
    client = build_workspace_client(_settings(Environment.DEV), profile="dev", factory=_factory(), environ={})
    assert client.kwargs == {"product": "client-research-agent", "product_version": "1.0.0", "profile": "dev"}


def test_build_workspace_client_allows_pat_in_dev() -> None:
    settings = _settings(Environment.DEV, host="h.example", token=SecretStr("dapi-x"))
    client = build_workspace_client(settings, factory=_factory("pat"), environ={})
    assert client.kwargs["token"] == "dapi-x"
    assert client.kwargs["auth_type"] == "pat"


@pytest.mark.parametrize("env", [Environment.STAGING, Environment.PROD])
def test_pat_refused_in_staging_and_prod(env: Environment) -> None:
    with pytest.raises(ConfigurationError, match="forbidden"):
        build_workspace_client(_settings(env, token=SecretStr("dapi-x")), factory=_factory(), environ={})
    with pytest.raises(ConfigurationError, match="forbidden"):
        build_workspace_client(_settings(env), factory=_factory(), environ={"DATABRICKS_TOKEN": "dapi-y"})
    with pytest.raises(ConfigurationError, match="resolved auth type 'pat'"):
        build_workspace_client(_settings(env), factory=_factory("pat"), environ={})


def test_build_workspace_client_errors() -> None:
    with pytest.raises(ConfigurationError, match="together"):
        build_workspace_client(
            _settings(Environment.DEV), client_id="only-id", factory=_factory(), environ={}
        )

    def failing(**_kwargs: Any) -> Any:
        raise ValueError("default auth: cannot configure default credentials")

    with pytest.raises(ConfigurationError, match="authentication failed"):
        build_workspace_client(_settings(Environment.DEV), factory=failing, environ={})


def test_build_workspace_client_default_factory_uses_sdk_class(monkeypatch: pytest.MonkeyPatch) -> None:
    import databricks.sdk

    monkeypatch.setattr(databricks.sdk, "WorkspaceClient", lambda **kw: _Client(config=_Config(), kwargs=kw))
    client = build_workspace_client(_settings(Environment.PROD), environ={})
    assert isinstance(client, _Client)
    assert client.kwargs["product"] == "client-research-agent"


def test_credentials_bearer_token_and_openai_client() -> None:
    client = _Client(config=_Config(), kwargs={})
    creds = WorkspaceCredentials.from_workspace_client(client)
    assert creds.serving_base_url == "https://adb-1.azuredatabricks.net/serving-endpoints"
    assert creds.bearer_token() == "tok-1"
    client.config.headers["Authorization"] = "Bearer tok-2"
    assert creds.bearer_token() == "tok-2"  # re-evaluated per call: refreshed tokens are picked up

    captured: dict[str, Any] = {}
    build_openai_client(creds, timeout_seconds=12.5, factory=lambda **kw: captured.update(kw))
    assert captured["base_url"] == creds.serving_base_url
    assert captured["api_key"]() == "tok-2"
    assert (captured["timeout"], captured["max_retries"]) == (12.5, 0)


def test_credentials_errors() -> None:
    with pytest.raises(ConfigurationError, match="host"):
        WorkspaceCredentials.from_workspace_client(_Client(config=_Config(host=None), kwargs={}))
    creds = WorkspaceCredentials(host="https://h", header_factory=lambda: {"Authorization": "Basic abc"})
    with pytest.raises(ConfigurationError, match="bearer"):
        creds.bearer_token()


def test_default_openai_factory_builds_real_client() -> None:
    creds = WorkspaceCredentials(
        host="https://h.example", header_factory=lambda: {"Authorization": "Bearer t"}
    )
    client = build_openai_client(creds)
    assert str(client.base_url).rstrip("/") == "https://h.example/serving-endpoints"
    assert client.max_retries == 0
