from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.evaluation.dataset import (
    EvalExample,
    EvalExpectations,
    EvalInputs,
    SnapshotDocument,
    example_from_row,
    examples_from_rows,
    load_golden_set,
    load_jsonl,
    parse_examples,
    write_jsonl,
)
from client_research_agent.evaluation.harness import (
    EvaluationHarness,
    QualityGate,
    brief_from_response,
    model_producer,
    offline_producer,
    responses_request,
)
from client_research_agent.evaluation.metrics import (
    citation_coverage,
    compute_metrics,
    criterion_score_mae,
    discovery_questions_ok,
    expected_fact_recall,
    grounded_fact_ratio,
    section_completeness,
    summarize,
    verdict_accuracy,
)
from client_research_agent.evaluation.mlflow_eval import (
    build_scorers,
    evaluation_records,
    log_to_mlflow_genai,
    metric_functions,
)
from client_research_agent.evaluation.snapshot import SnapshotIngestion
from client_research_agent.models import CitationReport, Criterion, DocumentType, FitVerdict, ResearchRequest
from client_research_agent.services.local import InMemoryDocumentStore
from tests.unit.orchestration.test_review_state_rendering import make_brief


# ------------------------------------------------------------------ dataset
def test_golden_set_is_valid_and_labelled() -> None:
    examples = load_golden_set()
    assert len(examples) >= 5
    assert all(e.expectations.expected_verdict is not None for e in examples)
    assert any(not e.snapshot for e in examples)
    assert {e.inputs.company_name for e in examples} >= {"Harborline Logistics", "Meridian Mutual Insurance"}
    record = examples[0].to_mlflow_record()
    assert set(record) == {"inputs", "expectations", "tags"}


def test_jsonl_round_trip_and_errors(tmp_path: Path) -> None:
    example = EvalExample(
        eval_id="x1",
        inputs=EvalInputs(company_name="Acme", domain=" ", ticker="ac"),
        expectations=EvalExpectations(expected_scores={Criterion.AI_DATA_FOCUS: 4}),
    )
    path = write_jsonl([example], tmp_path / "sub" / "set.jsonl")
    loaded = load_jsonl(path)
    assert loaded == [example]
    assert loaded[0].inputs.domain is None
    assert loaded[0].inputs.to_request().ticker == "AC"
    with pytest.raises(ValueError, match="duplicate eval_id"):
        parse_examples([example.model_dump_json(), "", example.model_dump_json()])
    with pytest.raises(ValueError, match="invalid evaluation example"):
        parse_examples(['{"eval_id": "bad"}'])
    with pytest.raises(ValueError, match="0-5"):
        EvalExpectations(expected_scores={Criterion.AI_DATA_FOCUS: 9})
    assert EvalExpectations().is_empty


def test_examples_from_table_rows() -> None:
    rows = [
        {
            "eval_id": "e1",
            "company": "Acme",
            "domain": "acme.example.com",
            "ticker": "",
            "request": json.dumps(
                {
                    "input": [],
                    "custom_inputs": {"company_name": "Ignored", "max_documents": 7, "industry": "Retail"},
                }
            ),
            "expected_verdict": "good_fit",
            "expected_facts": '["Acme has 10 stores", " "]',
            "guidelines": ["Cite sources"],
            "tags": '{"slice": "retail"}',
            "active": "true",
        },
        {"eval_id": "e2", "company": "Beta", "request": "not json", "active": False},
        {
            "eval_id": "e3",
            "company": "Gamma",
            "request": {"company_name": "Gamma"},
            "tags": "[]",
            "active": True,
        },
    ]
    examples = examples_from_rows(rows)
    assert [e.eval_id for e in examples] == ["e1", "e3"]
    first = examples[0]
    assert first.inputs.company_name == "Acme"
    assert first.inputs.max_documents == 7
    assert first.inputs.industry == "Retail"
    assert first.inputs.ticker is None
    assert first.expectations.expected_verdict is FitVerdict.GOOD_FIT
    assert first.expectations.expected_facts == ("Acme has 10 stores",)
    assert first.tags == {"slice": "retail"}
    assert examples[1].tags == {}
    assert (
        example_from_row({"eval_id": "e4", "company": "Delta", "request": None}).inputs.company_name
        == "Delta"
    )


# ------------------------------------------------------------------ metrics
def test_metrics_on_brief() -> None:
    brief = make_brief()
    assert citation_coverage(brief) == 1.0
    assert citation_coverage(brief.model_copy(update={"citation_report": CitationReport()})) == 1.0
    assert grounded_fact_ratio(brief) == 1.0
    assert (
        grounded_fact_ratio(brief.model_copy(update={"company_overview": (), "executive_summary": ()})) == 1.0
    )
    assert verdict_accuracy(brief, None) is None
    assert verdict_accuracy(brief, FitVerdict.GOOD_FIT) == 1.0
    assert verdict_accuracy(brief, FitVerdict.POTENTIAL_FIT) == 0.0
    assert criterion_score_mae(brief, {}) is None
    assert criterion_score_mae(brief, {Criterion.AI_DATA_FOCUS: 2, Criterion.INDUSTRY_TRENDS: 4}) == 1.0
    assert section_completeness(brief) == 6 / 8
    assert discovery_questions_ok(brief) == 1.0
    assert expected_fact_recall(brief, []) is None
    assert (
        expected_fact_recall(brief, ["Acme Corp employs 12,000 people", "Acme sells rockets to Mars"]) == 0.5
    )
    metrics = compute_metrics(brief, EvalExpectations(expected_verdict=FitVerdict.GOOD_FIT))
    assert metrics.as_dict()["verdict_accuracy"] == 1.0
    assert "criterion_score_mae" not in metrics.as_dict()
    assert summarize([{"a": 1.0, "b": True}, {"a": 3}]) == {"a": 2.0}


def test_ungrounded_fact_lowers_ratio() -> None:
    brief = make_brief()
    evidence = brief.evidence[0].model_copy(update={"quote": "Unrelated sentence about weather."})
    assert grounded_fact_ratio(brief.model_copy(update={"evidence": (evidence,)})) == 0.0


# ------------------------------------------------------------------ gate + harness
def example(eval_id: str = "e1", **expectations: Any) -> EvalExample:
    return EvalExample(
        eval_id=eval_id,
        inputs=EvalInputs(company_name="Acme Corp"),
        expectations=EvalExpectations(**expectations),
    )


def test_gate_example_and_run_failures() -> None:
    gate = QualityGate(min_citation_coverage=0.9, min_verdict_accuracy=1.0, max_criterion_mae=0.5)
    brief = make_brief(total=4, supported=2)
    brief = brief.model_copy(update={"discovery_questions": ("q",) * 5})
    metrics = compute_metrics(brief, EvalExpectations(expected_verdict=FitVerdict.POTENTIAL_FIT))
    failures = gate.example_failures(metrics, EvalExpectations())
    assert any("citation_coverage" in f for f in failures)
    assert any("verdict" in f for f in failures)
    assert gate.example_failures(metrics, EvalExpectations(min_citation_coverage=0.4)) == [
        "verdict does not match the label"
    ]
    run = gate.run_failures(
        0.5, {"citation_coverage": 0.5, "verdict_accuracy": 0.0, "criterion_score_mae": 2.0}
    )
    assert len(run) == 4
    assert set(gate.thresholds()) == {
        "pass_rate",
        "citation_coverage",
        "grounded_fact_ratio",
        "verdict_accuracy",
        "criterion_score_mae",
    }
    with pytest.raises(ValueError, match="min_pass_rate"):
        QualityGate(min_pass_rate=2.0)


def test_harness_aggregates_and_isolates_producer_errors() -> None:
    brief = make_brief()

    def producer(ex: EvalExample) -> Any:
        if ex.eval_id == "boom":
            raise RuntimeError("model endpoint down")
        return brief

    harness = EvaluationHarness(producer, QualityGate(min_pass_rate=0.5, max_criterion_mae=1.0))
    report = harness.run(
        [
            example(expected_verdict=FitVerdict.GOOD_FIT, expected_scores={Criterion.AI_DATA_FOCUS: 4}),
            example("boom"),
        ]
    )
    assert report.pass_rate == 0.5
    assert report.gate_passed
    assert report.results[1].error
    assert "producer error" in report.results[1].failures[0]
    rows = {r["metric_name"]: r for r in report.metric_rows()}
    assert rows["pass_rate"]["passed"]
    assert rows["criterion_score_mae"]["passed"]
    assert rows["section_completeness"]["threshold"] is None
    payload = report.to_dict()
    assert payload["results"][0]["verdict"] == "good_fit"
    assert payload["results"][1]["metrics"] == {}
    assert harness.gate.min_pass_rate == 0.5
    with pytest.raises(ValueError, match="at least one"):
        harness.run([])


def test_offline_producer_runs_golden_set(settings: AppSettings, tmp_path: Path) -> None:
    report = EvaluationHarness(offline_producer(settings, var_dir=str(tmp_path))).run(load_golden_set())
    assert report.gate_passed, report.gate_failures
    assert report.aggregates["verdict_accuracy"] == 1.0


def test_model_producer_parses_custom_outputs() -> None:
    brief = make_brief()

    class Model:
        def predict(self, request: Any) -> Any:
            assert request["custom_inputs"]["company_name"] == "Acme Corp"
            return {"output": [], "custom_outputs": {"brief_json": brief.model_dump_json()}}

    assert model_producer(Model())(example()) == brief
    assert responses_request(example())["custom_inputs"]["requested_by"] == "evaluation"
    as_object = types.SimpleNamespace(
        model_dump=lambda: {"custom_outputs": {"brief_json": brief.model_dump(mode="json")}}
    )
    assert brief_from_response(as_object) == brief
    with pytest.raises(ValueError, match="brief_json"):
        brief_from_response({"custom_outputs": {}})
    with pytest.raises(ValueError, match="unexpected model response"):
        brief_from_response(["not", "a", "mapping"])


# ------------------------------------------------------------------ snapshot
def test_snapshot_ingestion_dedupes_and_limits() -> None:
    docs = [
        SnapshotDocument(
            url="https://www.sec.gov/a",
            title="10-K",
            text="Annual report text",
            document_type=DocumentType.SEC_FILING,
        ),
        SnapshotDocument(
            url="https://acme.example.com/b",
            title="News",
            text="Annual report text",
            document_type=DocumentType.PRESS_RELEASE,
        ),
        SnapshotDocument(
            url="https://acme.example.com/c",
            title="News 2",
            text="Second story",
            document_type=DocumentType.PRESS_RELEASE,
        ),
        SnapshotDocument(
            url="https://acme.example.com/d",
            title="News 3",
            text="Third story",
            document_type=DocumentType.PRESS_RELEASE,
        ),
    ]
    store = InMemoryDocumentStore()
    ingestion = SnapshotIngestion({"ACME CORP": docs}, document_store=store)
    request = ResearchRequest(company_name="Acme Corp", domain="acme.example.com", max_documents=2)
    result = ingestion.run(request)
    assert [d.url for d in result.documents] == ["https://www.sec.gov/a", "https://acme.example.com/c"]
    assert result.documents[0].trust_score == 0.95
    assert result.documents[1].trust_score == 0.85
    assert {s.reason.value for s in result.skipped} == {"duplicate_content", "max_documents_reached"}
    store.save_documents(result.documents)
    again = ingestion.run(request)
    assert {s.reason.value for s in again.skipped} >= {"already_ingested"}
    assert SnapshotIngestion({}).run(request).documents == []


# ------------------------------------------------------------------ mlflow genai
def test_scorers_wrap_deterministic_metrics() -> None:
    brief = make_brief()
    registered: list[tuple[str, Any]] = []

    def fake_scorer(func: Any, *, name: str) -> Any:
        registered.append((name, func))
        return func

    scorers = build_scorers(fake_scorer)
    assert [n for n, _ in registered] == list(metric_functions())
    outputs = {"brief_json": brief.model_dump_json()}
    values = {name: fn(outputs, {"expected_verdict": "good_fit"}) for name, fn in registered}
    assert values["verdict_accuracy"] == 1.0
    assert values["criterion_score_mae"] is None
    assert len(scorers) == 7


def test_log_to_mlflow_genai_with_fake_module() -> None:
    brief = make_brief()
    report = EvaluationHarness(lambda _e: brief).run([example()])
    captured: dict[str, Any] = {}

    def evaluate(*, data: Any, scorers: Any) -> Any:
        captured["data"] = data
        captured["scorers"] = scorers
        return types.SimpleNamespace(run_id="genai-run")

    genai = types.SimpleNamespace(scorer=lambda f, name: f, evaluate=evaluate)
    fake = types.SimpleNamespace(genai=genai, set_experiment=lambda name: captured.setdefault("exp", name))
    assert log_to_mlflow_genai(report, experiment="/Shared/x", mlflow_module=fake) == "genai-run"
    assert captured["exp"] == "/Shared/x"
    assert captured["data"][0]["outputs"]["verdict"] == "good_fit"
    assert len(evaluation_records(report)) == 1

    def failing(**_kw: Any) -> Any:
        raise RuntimeError("tracking server down")

    broken = types.SimpleNamespace(genai=types.SimpleNamespace(scorer=lambda f, name: f, evaluate=failing))
    assert log_to_mlflow_genai(report, mlflow_module=broken) is None
    empty = EvaluationHarness(lambda _e: (_ for _ in ()).throw(RuntimeError("x"))).run([example()])
    assert log_to_mlflow_genai(empty, mlflow_module=fake) is None
