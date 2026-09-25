from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from client_research_agent import cli
from client_research_agent.agent import factory
from client_research_agent.security.rbac import Role
from tests.support.world import northwind_fetcher

REAL_BUILD = factory.build_runtime


@pytest.fixture
def offline_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    var_dir = tmp_path / "var"

    def build(settings: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("fetcher", northwind_fetcher())
        kwargs.setdefault("llm", None)
        kwargs.setdefault("var_dir", var_dir)
        return REAL_BUILD(settings, **kwargs)

    monkeypatch.setattr(factory, "build_runtime", build)
    monkeypatch.setattr(cli, "configure_observability", lambda *a, **k: None)
    return var_dir


def test_principal_from_environment() -> None:
    local = cli.cli_principal({})
    assert local.id.startswith("cli:")
    assert local.roles == frozenset({Role.ANALYST})
    mapped = cli.cli_principal({"CRA_PRINCIPAL_ID": "sp-1", "CRA_PRINCIPAL_GROUPS": "cra-viewers, unknown"})
    assert mapped.id == "sp-1"
    assert mapped.roles == frozenset({Role.VIEWER})
    assert mapped.groups == frozenset({"cra-viewers", "unknown"})


def test_research_markdown_to_stdout(offline_world: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["research", "--company", "Northwind Industries", "--domain", "northwind.example", "--ticker", "NWND"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "# Client Brief: Northwind Industries" in captured.out
    assert "verdict=" in captured.err


def test_research_json_to_file(
    offline_world: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "brief.json"
    code = cli.main(
        ["research", "--company", "Northwind Industries", "--domain", "northwind.example", "--format", "json",
         "--output", str(output), "--industry", ""]
    )  # fmt: skip
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["company"] == "Northwind Industries"
    assert len(payload["discovery_questions"]) == 5
    assert f"wrote {output}" in capsys.readouterr().err


def test_research_denied_for_viewer(
    offline_world: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CRA_PRINCIPAL_GROUPS", "cra-viewers")
    assert cli.main(["research", "--company", "Northwind Industries"]) == 1
    assert "AccessDeniedError" in capsys.readouterr().err


def test_research_invalid_input(offline_world: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["research", "--company", "Acme", "--max-documents", "0"]) == 2
    assert "error:" in capsys.readouterr().err


def test_ingest_command(offline_world: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        ["ingest", "--company", "Northwind Industries", "--domain", "northwind.example", "--ticker", "NWND"]
    )
    out = capsys.readouterr().out
    summary = json.loads(out[out.index("{\n") :])
    assert code == 0
    assert summary["documents_ingested"] > 0
    assert summary["run_id"]


def test_evaluate_command(offline_world: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    assert cli.main(["evaluate", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["gate_passed"] is True
    strict = tmp_path / "strict.jsonl"
    from client_research_agent.evaluation.dataset import EvalExample, write_jsonl

    write_jsonl(
        [
            EvalExample.model_validate(
                {
                    "eval_id": "x",
                    "inputs": {"company_name": "Nobody Ltd"},
                    "expectations": {"expected_verdict": "good_fit"},
                }
            )
        ],
        strict,
    )
    assert cli.main(["evaluate", "--dataset", str(strict)]) == 1


def test_serve_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CRA_VAR_DIR", str(tmp_path / "var"))
    assert cli.main(["serve-check"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "healthy"
    assert status["prompts"] >= 6


def test_serve_check_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CRA_VAR_DIR", str(blocker / "sub"))
    assert cli.main(["serve-check"]) == 1
    assert "error:" in capsys.readouterr().err
