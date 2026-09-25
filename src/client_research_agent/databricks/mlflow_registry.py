"""Unity Catalog model registry helpers.

Models are registered under three-level names (``catalog.schema.model``) in the
Unity Catalog registry (``registry_uri="databricks-uc"``). Deployment targets
resolve versions through aliases rather than stages: ``champion`` serves
production traffic and ``challenger`` is the candidate under evaluation.

All functions accept the ``mlflow`` module (and an ``MlflowClient``) as
parameters so they can be exercised without a tracking server; by default the
real ``mlflow`` package is imported lazily.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from client_research_agent.databricks.unity_catalog import validate_identifier
from client_research_agent.utils.errors import ConfigurationError

UC_REGISTRY_URI = "databricks-uc"
CHAMPION = "champion"
CHALLENGER = "challenger"
_ALIAS = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


def _mlflow(module: Any | None) -> Any:
    return module if module is not None else importlib.import_module("mlflow")


def uc_model_name(catalog: str, schema: str, model: str) -> str:
    """Validated three-level Unity Catalog model name."""
    return ".".join(validate_identifier(part) for part in (catalog, schema, model))


def _check_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) != 3:
        raise ConfigurationError(f"Unity Catalog model name must be catalog.schema.model, got {name!r}")
    for part in parts:
        validate_identifier(part)
    return name


def _check_alias(alias: str) -> str:
    if not _ALIAS.fullmatch(alias):
        raise ConfigurationError(f"invalid model alias {alias!r}")
    return alias


def model_uri_for_alias(name: str, alias: str) -> str:
    return f"models:/{_check_name(name)}@{_check_alias(alias)}"


@dataclass(frozen=True, slots=True)
class ModelVersionInfo:
    name: str
    version: str
    source: str | None
    run_id: str | None
    aliases: tuple[str, ...]
    tags: Mapping[str, str]


def _to_info(version: Any) -> ModelVersionInfo:
    return ModelVersionInfo(
        name=str(version.name),
        version=str(version.version),
        source=getattr(version, "source", None),
        run_id=getattr(version, "run_id", None),
        aliases=tuple(getattr(version, "aliases", None) or ()),
        tags=dict(getattr(version, "tags", None) or {}),
    )


def registry_client(mlflow_module: Any | None = None) -> Any:
    """An ``MlflowClient`` bound to the Unity Catalog registry."""
    mlflow = _mlflow(mlflow_module)
    mlflow.set_registry_uri(UC_REGISTRY_URI)
    return mlflow.MlflowClient(registry_uri=UC_REGISTRY_URI)


def register_model(
    model_uri: str,
    name: str,
    *,
    tags: Mapping[str, str] | None = None,
    mlflow_module: Any | None = None,
) -> ModelVersionInfo:
    """Register ``model_uri`` (e.g. ``runs:/<id>/agent``) as a new version of ``name`` in UC."""
    mlflow = _mlflow(mlflow_module)
    mlflow.set_registry_uri(UC_REGISTRY_URI)
    version = mlflow.register_model(model_uri=model_uri, name=_check_name(name), tags=dict(tags or {}))
    return _to_info(version)


def set_alias(name: str, alias: str, version: str | int, *, client: Any) -> None:
    client.set_registered_model_alias(_check_name(name), _check_alias(alias), str(version))


def get_version_by_alias(name: str, alias: str, *, client: Any) -> ModelVersionInfo | None:
    """Return the version behind ``alias``, or ``None`` when the alias is unset."""
    try:
        version = client.get_model_version_by_alias(_check_name(name), _check_alias(alias))
    except Exception as exc:
        if _is_missing(exc):
            return None
        raise
    return _to_info(version)


def _is_missing(exc: BaseException) -> bool:
    code = getattr(exc, "error_code", None)
    text = f"{code} {exc}".upper()
    return "RESOURCE_DOES_NOT_EXIST" in text or "NOT_FOUND" in text or "NOT FOUND" in text


@dataclass(frozen=True, slots=True)
class Promotion:
    name: str
    new_champion: str
    previous_champion: str | None


def promote_challenger(
    name: str, *, client: Any, keep_previous_alias: str | None = "previous_champion"
) -> Promotion:
    """Point ``champion`` at the current ``challenger`` and clear ``challenger``.

    The outgoing champion keeps a ``previous_champion`` alias so rollback is a
    single alias move.
    """
    challenger = get_version_by_alias(name, CHALLENGER, client=client)
    if challenger is None:
        raise ConfigurationError(f"model {name!r} has no '{CHALLENGER}' alias to promote")
    champion = get_version_by_alias(name, CHAMPION, client=client)
    if champion is not None and keep_previous_alias and champion.version != challenger.version:
        set_alias(name, keep_previous_alias, champion.version, client=client)
    set_alias(name, CHAMPION, challenger.version, client=client)
    client.delete_registered_model_alias(name, CHALLENGER)
    return Promotion(
        name=name,
        new_champion=challenger.version,
        previous_champion=champion.version if champion is not None else None,
    )


def load_model_by_alias(name: str, alias: str = CHAMPION, *, mlflow_module: Any | None = None) -> Any:
    """Load the pyfunc model currently behind ``alias``."""
    mlflow = _mlflow(mlflow_module)
    mlflow.set_registry_uri(UC_REGISTRY_URI)
    return mlflow.pyfunc.load_model(model_uri_for_alias(name, alias))
