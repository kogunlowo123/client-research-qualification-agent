from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.agent import serving_agent
from client_research_agent.agent.serving_agent import (
    ClientResearchResponsesAgent,
    ServingInputError,
    as_mapping,
    extract_company,
    last_user_message,
    message_text,
    parse_request,
)
from client_research_agent.config.settings import AppSettings
from client_research_agent.models import ClientBrief
from client_research_agent.security.rate_limiter import PrincipalRateLimiter
from tests.support.doubles import ScriptedLLM
from tests.support.world import local_runtime


@pytest.fixture(autouse=True)
def _no_mlflow_traces() -> Iterator[None]:
    mlflow = importlib.import_module("mlflow")
    mlflow.tracing.disable()
    yield
    mlflow.tracing.enable()


def request_payload(**custom: Any) -> dict[str, Any]:
    return {
        "input": [
            {"role": "user", "content": "Research Northwind Industries and return a qualification brief."}
        ],
        "custom_inputs": {
            "company_name": "Northwind Industries",
            "ticker": "NWND",
            "domain": "northwind.example",
            "max_documents": 10,
            "requested_by": "smoke-test",
            **custom,
        },
        "context": {"conversation_id": "c1", "user_id": "caller@example.com"},
    }


def make_agent(settings: AppSettings, tmp_path: Path, **kwargs: Any) -> ClientResearchResponsesAgent:
    return ClientResearchResponsesAgent(lambda: local_runtime(settings, tmp_path), **kwargs)


# ------------------------------------------------------------------ parsing
def test_extract_company_from_message() -> None:
    parsed = extract_company(
        "Research Microsoft Corporation (ticker MSFT, domain microsoft.com) and return a qualification brief."
    )
    assert parsed is not None
    assert (parsed.company_name, parsed.ticker, parsed.domain) == (
        "Microsoft Corporation",
        "MSFT",
        "microsoft.com",
    )
    assert extract_company("company: Contoso Ltd").company_name == "Contoso Ltd"  # type: ignore[union-attr]
    assert extract_company("Please qualify Fabrikam for us, website fabrikam.io").domain == "fabrikam.io"  # type: ignore[union-attr]
    assert extract_company("hello there") is None


def test_message_helpers() -> None:
    assert message_text([{"type": "input_text", "text": "a"}, {"type": "x"}, "junk"]) == "a"
    assert message_text(None) == ""
    items = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "last"},
    ]
    assert last_user_message(items) == "last"
    assert last_user_message([]) == ""
    assert as_mapping(types.SimpleNamespace(model_dump=lambda exclude_none: {"a": 1})) == {"a": 1}
    with pytest.raises(ServingInputError):
        as_mapping(42)


def test_parse_request_custom_inputs_and_asserted_identity() -> None:
    parsed = parse_request(request_payload(domain="", ticker=" "))
    assert parsed.source == "custom_inputs"
    assert parsed.request.domain is None
    assert parsed.request.ticker is None
    assert parsed.asserted_requester == "smoke-test"
    assert parsed.request.requested_by == "unverified:smoke-test"
    anonymous = parse_request({"input": [{"role": "user", "content": "Research Contoso Ltd"}]})
    assert anonymous.asserted_requester == "anonymous"
    assert anonymous.source == "message"


def test_parse_request_uses_llm_then_regex() -> None:
    llm = ScriptedLLM(default={"company_name": "Tailspin Toys", "ticker": "TSPN"})
    parsed = parse_request({"input": [{"role": "user", "content": "tell me about tailspin"}]}, llm=llm)
    assert parsed.request.company_name == "Tailspin Toys"
    assert parsed.request.ticker == "TSPN"
    fallback = parse_request(
        {"input": [{"role": "user", "content": "Research Wingtip Partners"}]}, llm=ScriptedLLM(default="nope")
    )
    assert fallback.request.company_name == "Wingtip Partners"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"input": [{"role": "user", "content": "hi"}]}, "no company"),
        ({"input": [], "custom_inputs": "x"}, "custom_inputs"),
        (
            {"input": [], "custom_inputs": {"company_name": "Acme", "max_documents": 9999}},
            "invalid research request",
        ),
        (
            {"input": [], "custom_inputs": {"company_name": "Acme", "max_documents": "many"}},
            "invalid research request",
        ),
    ],
)
def test_parse_request_errors(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ServingInputError, match=message):
        parse_request(payload)


# ------------------------------------------------------------------ predict
def test_predict_returns_markdown_and_custom_outputs(settings: AppSettings, tmp_path: Path) -> None:
    agent = make_agent(settings, tmp_path)
    response = agent.predict(request_payload())
    payload = response.model_dump()
    text = payload["output"][0]["content"][0]["text"]
    assert "https://" in text
    custom = payload["custom_outputs"]
    assert custom["verdict"] in text.lower().replace(" ", "_") or custom["verdict"] in {
        "good_fit",
        "potential_fit",
        "not_enough_evidence",
    }
    assert ClientBrief.model_validate_json(custom["brief_json"]).run_id == custom["run_id"]
    events = [r for r in agent._runtime.audit.records() if r.event_type == "serving.request"]  # type: ignore[union-attr]
    assert events[0].payload["asserted_requester"] == "smoke-test"
    assert events[0].payload["asserted_requester_verified"] is False
    assert events[0].principal == "model-serving:local"


def test_predict_stream_emits_deltas_then_done(settings: AppSettings, tmp_path: Path) -> None:
    agent = make_agent(settings, tmp_path, service_principal_id="sp-123")
    events = [e.model_dump() for e in agent.predict_stream(request_payload())]
    deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    done = events[-1]
    assert done["type"] == "response.output_item.done"
    assert "".join(d["delta"] for d in deltas) == done["item"]["content"][0]["text"]
    assert done["custom_outputs"]["run_id"]


def test_rate_limit_returns_error_output(settings: AppSettings, tmp_path: Path) -> None:
    agent = make_agent(
        settings, tmp_path, rate_limiter=PrincipalRateLimiter(capacity=1, refill_per_second=0.001)
    )
    agent.predict(request_payload())
    limited = agent.predict(request_payload()).model_dump()
    assert limited["custom_outputs"]["error"] == "rate_limited"
    assert "Rate limit exceeded" in limited["output"][0]["content"][0]["text"]
    other = agent.predict(request_payload(requested_by="someone-else")).model_dump()
    assert "error" not in other["custom_outputs"]


def test_service_principal_from_environment(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRA_SERVICE_PRINCIPAL_ID", "app-id-42")
    agent = make_agent(settings, tmp_path)
    principal = agent.service_principal(local_runtime(settings, tmp_path))
    assert principal.id == "app-id-42"
    assert {r.value for r in principal.roles} == {"service"}


def test_audit_failure_does_not_block_serving(settings: AppSettings, tmp_path: Path) -> None:
    runtime = local_runtime(settings, tmp_path)
    original = runtime.audit.append

    def flaky(event_type: str, *args: Any, **kwargs: Any) -> Any:
        if event_type == "serving.request":
            raise OSError("audit offline")
        return original(event_type, *args, **kwargs)

    runtime.audit.append = flaky  # type: ignore[method-assign]
    agent = ClientResearchResponsesAgent(lambda: runtime)
    assert agent.predict(request_payload()).model_dump()["custom_outputs"]["run_id"]


def test_dict_responses_without_mlflow_types(
    settings: AppSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serving_agent, "_response_types", lambda: None)
    agent = make_agent(settings, tmp_path)
    response = agent.research(request_payload())
    assert response.brief.company == "Northwind Industries"
    plain = ClientResearchResponsesAgent.predict.__wrapped__(agent, request_payload())  # type: ignore[attr-defined]
    assert isinstance(plain, dict)
    assert plain["custom_outputs"]["verdict"]


def test_loaders_without_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow.pyfunc", None)
    monkeypatch.setitem(sys.modules, "mlflow.types.responses", None)
    assert serving_agent._load_base() is object
    assert serving_agent._response_types() is None


def test_default_runtime_factory_builds_local_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CRA_VAR_DIR", str(tmp_path))
    runtime = serving_agent.default_runtime_factory()
    assert runtime.track_runs is False
    runtime.close()


def test_agent_model_sets_model(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[Any] = []
    fake_models = types.ModuleType("mlflow.models")
    fake_models.set_model = captured.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow.models", fake_models)
    monkeypatch.delitem(sys.modules, "client_research_agent.agent.agent_model", raising=False)
    module = importlib.import_module("client_research_agent.agent.agent_model")
    assert captured == [module.AGENT]
    assert isinstance(module.AGENT, ClientResearchResponsesAgent)
    json.dumps({"ok": True})
