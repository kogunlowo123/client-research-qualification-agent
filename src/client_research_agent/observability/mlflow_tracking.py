"""MLflow run tracking that degrades to a no-op.

``RunTracker`` wraps ``mlflow.start_run`` when MLflow is importable and
tracking is enabled, logging params, metrics and dict artifacts (brief JSON,
lineage graph, policy reports). When MLflow is absent or disabled every method
is a cheap no-op, so the agent code never branches on the environment.
Logging failures are reported through structured logs and never fail a run.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from types import ModuleType, TracebackType
from typing import Any, Self

from client_research_agent.observability.logging import get_logger, scrub

_MAX_PARAM_CHARS = 500
_log = get_logger(__name__)


def _load_mlflow() -> ModuleType | None:
    try:
        return importlib.import_module("mlflow")
    except ImportError:
        return None


class RunTracker:
    def __init__(
        self,
        *,
        enabled: bool = True,
        experiment: str | None = None,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
        nested: bool = False,
    ) -> None:
        self._enabled = enabled
        self._experiment = experiment
        self._run_name = run_name
        self._tags = {k: str(v) for k, v in (tags or {}).items()}
        self._nested = nested
        self._mlflow: ModuleType | None = None
        self._run: Any = None

    @property
    def active(self) -> bool:
        return self._run is not None

    @property
    def run_id(self) -> str | None:
        if self._run is None:
            return None
        info = getattr(self._run, "info", None)
        value = getattr(info, "run_id", None)
        return str(value) if value is not None else None

    def __enter__(self) -> Self:
        if not self._enabled:
            return self
        mlflow = _load_mlflow()
        if mlflow is None:
            return self
        try:
            if self._experiment:
                mlflow.set_experiment(self._experiment)
            self._run = mlflow.start_run(
                run_name=self._run_name, tags=self._tags or None, nested=self._nested
            )
            self._mlflow = mlflow
        except Exception as exc:  # tracking backend outages must not fail research runs
            _log.warning("mlflow_start_run_failed", error=type(exc).__name__)
            self._run = None
            self._mlflow = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._mlflow is None or self._run is None:
            return
        status = "FAILED" if exc_type is not None else "FINISHED"
        try:
            self._mlflow.end_run(status=status)
        except Exception as error:
            _log.warning("mlflow_end_run_failed", error=type(error).__name__)
        finally:
            self._run = None
            self._mlflow = None

    def _call(self, method: str, *args: Any, **kwargs: Any) -> bool:
        if self._mlflow is None or self._run is None:
            return False
        try:
            getattr(self._mlflow, method)(*args, **kwargs)
        except Exception as exc:
            _log.warning("mlflow_logging_failed", method=method, error=type(exc).__name__)
            return False
        return True

    def log_params(self, params: Mapping[str, Any]) -> bool:
        cleaned = {str(k): str(scrub(v))[:_MAX_PARAM_CHARS] for k, v in params.items() if v is not None}
        return self._call("log_params", cleaned) if cleaned else False

    def log_metrics(self, metrics: Mapping[str, float | int], *, step: int | None = None) -> bool:
        cleaned = {
            str(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(float(v))
        }
        return self._call("log_metrics", cleaned, step=step) if cleaned else False

    def log_dict(self, data: Mapping[str, Any] | list[Any], artifact_file: str) -> bool:
        if not artifact_file.endswith((".json", ".yaml", ".yml")):
            raise ValueError("artifact_file must end with .json, .yaml or .yml")
        return self._call("log_dict", scrub(data), artifact_file)

    def set_tags(self, tags: Mapping[str, Any]) -> bool:
        return self._call("set_tags", {str(k): str(v) for k, v in tags.items()}) if tags else False
