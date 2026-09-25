from __future__ import annotations

import base64
import sys
import types
from dataclasses import dataclass

import pytest

from client_research_agent.security.secrets import (
    ChainedSecretProvider,
    DatabricksSecretProvider,
    EnvSecretProvider,
    SecretProvider,
    require_secret,
)
from client_research_agent.utils.errors import ConfigurationError


class NotFound(Exception):  # noqa: N818 - mirrors databricks.sdk.errors.NotFound by name
    pass


@dataclass
class _Response:
    value: str | None


class FakeSecretsAPI:
    def __init__(self, values: dict[str, str | None], error: Exception | None = None) -> None:
        self.values = values
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def get_secret(self, *, scope: str, key: str) -> _Response:
        self.calls.append((scope, key))
        if self.error is not None:
            raise self.error
        if key not in self.values:
            raise NotFound(key)
        return _Response(self.values[key])


class FakeWorkspaceClient:
    def __init__(self, api: FakeSecretsAPI) -> None:
        self.secrets = api


def _b64(value: str) -> str:
    return base64.b64encode(value.encode()).decode()


def test_env_provider() -> None:
    provider = EnvSecretProvider(environ={"CRA_SECRET_OPENAI_API_KEY": "sk-value", "CRA_SECRET_EMPTY": ""})
    secret = provider.get("openai.api-key")
    assert secret is not None
    assert secret.get_secret_value() == "sk-value"
    assert "sk-value" not in repr(secret)
    assert provider.get("empty") is None
    assert provider.get("missing") is None
    assert provider.env_name("a-b") == "CRA_SECRET_A_B"
    assert "prefix" in repr(provider)
    assert isinstance(provider, SecretProvider)


def test_env_provider_reads_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_TOKEN", "abc")
    secret = EnvSecretProvider(prefix="X_").get("token")
    assert secret is not None
    assert secret.get_secret_value() == "abc"


def test_invalid_key_rejected() -> None:
    with pytest.raises(ValueError, match="secret keys"):
        EnvSecretProvider(environ={}).get("../etc/passwd")


def test_databricks_provider_decodes_and_caches() -> None:
    now = [0.0]
    api = FakeSecretsAPI({"serving-token": _b64("s3cr3t")})
    provider = DatabricksSecretProvider("scope-a", client=FakeWorkspaceClient(api), clock=lambda: now[0])
    first = provider.get("serving-token")
    assert first is not None
    assert first.get_secret_value() == "s3cr3t"
    provider.get("serving-token")
    assert len(api.calls) == 1
    now[0] = 301.0
    provider.get("serving-token")
    assert len(api.calls) == 2
    provider.invalidate("serving-token")
    provider.get("serving-token")
    provider.invalidate()
    provider.get("serving-token")
    assert len(api.calls) == 4
    assert provider.scope == "scope-a"
    assert "s3cr3t" not in repr(provider)


def test_databricks_provider_not_found_and_empty() -> None:
    api = FakeSecretsAPI({"empty": None})
    provider = DatabricksSecretProvider("s", client_factory=lambda: FakeWorkspaceClient(api))
    assert provider.get("missing") is None
    assert provider.get("empty") is None


def test_databricks_provider_errors() -> None:
    failing = DatabricksSecretProvider(
        "s", client=FakeWorkspaceClient(FakeSecretsAPI({}, RuntimeError("boom")))
    )
    with pytest.raises(ConfigurationError, match="RuntimeError"):
        failing.get("k")
    invalid = DatabricksSecretProvider(
        "s", client=FakeWorkspaceClient(FakeSecretsAPI({"k": "!!not-base64!!"}))
    )
    with pytest.raises(ConfigurationError, match="base64"):
        invalid.get("k")
    with pytest.raises(ValueError, match="scope"):
        DatabricksSecretProvider("")


def test_databricks_provider_lazy_sdk_import(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeSecretsAPI({"k": _b64("v")})
    fake_sdk = types.ModuleType("databricks.sdk")
    fake_sdk.WorkspaceClient = lambda: FakeWorkspaceClient(api)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "databricks.sdk", fake_sdk)
    secret = DatabricksSecretProvider("s").get("k")
    assert secret is not None
    assert secret.get_secret_value() == "v"


def test_databricks_provider_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "databricks.sdk", None)
    with pytest.raises(ConfigurationError, match="databricks-sdk"):
        DatabricksSecretProvider("s").get("k")


def test_chained_provider_and_require() -> None:
    first = EnvSecretProvider(environ={})
    second = EnvSecretProvider(environ={"CRA_SECRET_K": "v2"})
    chain = ChainedSecretProvider(first, second)
    value = require_secret(chain, "k")
    assert value.get_secret_value() == "v2"
    assert chain.get("none") is None
    with pytest.raises(ConfigurationError, match="none"):
        require_secret(chain, "none")
    assert "EnvSecretProvider" in repr(chain)
    with pytest.raises(ValueError, match="provider"):
        ChainedSecretProvider()
