"""Secret resolution without leaking values.

Secrets are returned as :class:`pydantic.SecretStr` so ``repr``/``str`` and
structured logging never reveal them. Providers:

* :class:`EnvSecretProvider` - environment variables (Databricks injects
  ``{{secrets/<scope>/<key>}}`` references into job / serving env vars);
* :class:`DatabricksSecretProvider` - the Databricks Secrets API through
  ``databricks.sdk.WorkspaceClient`` (imported lazily; the client can be
  injected);
* :class:`ChainedSecretProvider` - first provider that knows the key wins.
"""

from __future__ import annotations

import base64
import binascii
import importlib
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import SecretStr

from client_research_agent.utils.errors import ConfigurationError

_NOT_FOUND_ERRORS = frozenset({"NotFound", "ResourceDoesNotExist"})


@runtime_checkable
class SecretProvider(Protocol):
    def get(self, key: str) -> SecretStr | None: ...


def require_secret(provider: SecretProvider, key: str) -> SecretStr:
    value = provider.get(key)
    if value is None:
        raise ConfigurationError(f"required secret '{key}' is not configured")
    return value


def _validate_key(key: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", key):
        raise ValueError("secret keys may only contain letters, digits, '.', '_' and '-' (max 128)")
    return key


class EnvSecretProvider:
    """Reads ``<prefix><KEY>`` where the key is upper-cased and non-alphanumerics become ``_``."""

    def __init__(self, prefix: str = "CRA_SECRET_", environ: Mapping[str, str] | None = None) -> None:
        self._prefix = prefix
        self._environ = environ

    def env_name(self, key: str) -> str:
        return self._prefix + re.sub(r"[^A-Za-z0-9]", "_", _validate_key(key)).upper()

    def get(self, key: str) -> SecretStr | None:
        source = os.environ if self._environ is None else self._environ
        value = source.get(self.env_name(key))
        return SecretStr(value) if value else None

    def __repr__(self) -> str:
        return f"EnvSecretProvider(prefix={self._prefix!r})"


class DatabricksSecretProvider:
    """Databricks secret scope reader with a short TTL cache (values stay wrapped in SecretStr)."""

    def __init__(
        self,
        scope: str,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        cache_ttl_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not scope:
            raise ValueError("scope is required")
        self._scope = scope
        self._client = client
        self._factory = client_factory
        self._ttl = cache_ttl_seconds
        self._clock = clock
        self._cache: dict[str, tuple[float, SecretStr | None]] = {}
        self._lock = threading.Lock()

    @property
    def scope(self) -> str:
        return self._scope

    def _workspace_client(self) -> Any:
        if self._client is None:
            if self._factory is not None:
                self._client = self._factory()
            else:
                try:
                    sdk = importlib.import_module("databricks.sdk")
                except ImportError as exc:
                    raise ConfigurationError(
                        "databricks-sdk is required for DatabricksSecretProvider "
                        "(install the 'databricks' extra)"
                    ) from exc
                self._client = sdk.WorkspaceClient()
        return self._client

    def get(self, key: str) -> SecretStr | None:
        _validate_key(key)
        now = self._clock()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and now - cached[0] < self._ttl:
                return cached[1]
        value = self._fetch(key)
        with self._lock:
            self._cache[key] = (now, value)
        return value

    def _fetch(self, key: str) -> SecretStr | None:
        client = self._workspace_client()
        try:
            response = client.secrets.get_secret(scope=self._scope, key=key)
        except Exception as exc:
            if type(exc).__name__ in _NOT_FOUND_ERRORS:
                return None
            raise ConfigurationError(
                f"failed to read secret '{key}' from scope '{self._scope}': {type(exc).__name__}"
            ) from None
        encoded = getattr(response, "value", None)
        if not encoded:
            return None
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            raise ConfigurationError(
                f"secret '{key}' in scope '{self._scope}' is not valid base64 UTF-8"
            ) from None
        return SecretStr(decoded)

    def invalidate(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

    def __repr__(self) -> str:
        return f"DatabricksSecretProvider(scope={self._scope!r})"


class ChainedSecretProvider:
    def __init__(self, *providers: SecretProvider) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self._providers = providers

    def get(self, key: str) -> SecretStr | None:
        for provider in self._providers:
            value = provider.get(key)
            if value is not None:
                return value
        return None

    def __repr__(self) -> str:
        return f"ChainedSecretProvider({', '.join(repr(p) for p in self._providers)})"
