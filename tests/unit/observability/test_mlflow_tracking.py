from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from client_research_agent.observability.mlflow_tracking import RunTracker


@dataclass
class _Info:
    run_id: str


@dataclass
class _Run:
    info: _Info


@dataclass
class FakeMlflow:
    calls: list[tuple[str, Any]] = field(default_factory=list)
    fail_on: set[str] = field(default_factory=set)

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail_on:
            raise RuntimeError(f"{name} failed")

    def set_experiment(self, name: str) -> None:
        self._maybe_fail("set_experiment")
        self.calls.append(("set_experiment", name))

    def start_run(self, *, run_name: str | None, tags: dict[str, str] | None, nested: bool) -> _Run:
        self._maybe_fail("start_run")
        self.calls.append(("start_run", (run_name, tags, nested)))
        return _Run(_Info("run-abc"))

    def end_run(self, *, status: str) -> None:
        self.calls.append(("end_run", status))
        self._maybe_fail("end_run")

    def log_params(self, params: dict[str, str]) -> None:
        self._maybe_fail("log_params")
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.calls.append(("log_metrics", (metrics, step)))

    def log_dict(self, data: Any, artifact_file: str) -> None:
        self.calls.append(("log_dict", (data, artifact_file)))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", tags))


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    impl = FakeMlflow()
    module = types.ModuleType("mlflow")
    for name in (
        "set_experiment",
        "start_run",
        "end_run",
        "log_params",
        "log_metrics",
        "log_dict",
        "set_tags",
    ):
        setattr(module, name, getattr(impl, name))
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return impl


def test_full_run_logging(fake: FakeMlflow) -> None:
    with RunTracker(experiment="/Shared/cra", run_name="acme", tags={"env": "dev"}) as tracker:
        assert tracker.active
        assert tracker.run_id == "run-abc"
        assert tracker.log_params(
            {"company": "Acme", "token": "sk-" + "x" * 30, "skip": None, "long": "y" * 900}
        )
        assert tracker.log_metrics(
            {"score": 3.5, "n": 2, "flag": True, "nan": float("nan"), "s": "x"}, step=1
        )  # type: ignore[dict-item]
        assert tracker.log_dict({"brief": {"company": "Acme"}}, "brief.json")
        assert tracker.set_tags({"verdict": "good_fit"})
    names = [c[0] for c in fake.calls]
    assert names == [
        "set_experiment",
        "start_run",
        "log_params",
        "log_metrics",
        "log_dict",
        "set_tags",
        "end_run",
    ]
    params = fake.calls[2][1]
    assert params["token"] == "[REDACTED]"
    assert len(params["long"]) == 500
    assert "skip" not in params
    assert fake.calls[3][1] == ({"score": 3.5, "n": 2.0}, 1)
    assert fake.calls[-1] == ("end_run", "FINISHED")
    assert not tracker.active
    assert tracker.run_id is None


def test_failed_block_marks_run_failed(fake: FakeMlflow) -> None:
    with pytest.raises(ValueError, match="bad"), RunTracker():
        raise ValueError("bad")
    assert fake.calls[-1] == ("end_run", "FAILED")


def test_empty_inputs_are_noops(fake: FakeMlflow) -> None:
    with RunTracker() as tracker:
        assert not tracker.log_params({})
        assert not tracker.log_metrics({"x": float("inf")})
        assert not tracker.set_tags({})
        with pytest.raises(ValueError, match="artifact_file"):
            tracker.log_dict({}, "brief.txt")


def test_logging_failures_are_swallowed(fake: FakeMlflow) -> None:
    fake.fail_on = {"log_params", "end_run"}
    with RunTracker() as tracker:
        assert not tracker.log_params({"a": 1})
    assert not tracker.active


def test_start_failure_degrades_to_noop(fake: FakeMlflow) -> None:
    fake.fail_on = {"start_run"}
    with RunTracker(experiment="/x") as tracker:
        assert not tracker.active
        assert not tracker.log_params({"a": 1})
    assert ("end_run", "FINISHED") not in fake.calls


def test_disabled_and_missing_mlflow(monkeypatch: pytest.MonkeyPatch, fake: FakeMlflow) -> None:
    with RunTracker(enabled=False) as tracker:
        assert not tracker.log_metrics({"a": 1.0})
    assert fake.calls == []
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with RunTracker() as tracker:
        assert not tracker.active


def test_run_id_without_info(fake: FakeMlflow) -> None:
    tracker = RunTracker()
    tracker._run = object()
    assert tracker.run_id is None
