"""Evaluation dataset schema and loaders.

One :class:`EvalExample` mirrors a row of the Unity Catalog ``eval_set`` table
and the ``inputs`` / ``expectations`` / ``tags`` shape consumed by
``mlflow.genai.evaluate`` (see ``infrastructure/mlflow/eval_dataset_schema.json``).
Two optional extensions serve the offline harness:

* ``expectations.expected_scores`` - per-criterion rubric labels (0-5) for the
  criterion-score MAE metric;
* ``snapshot`` - captured public documents for the company, so the example
  can be replayed without network access (see :mod:`.snapshot`).

Examples load from JSON Lines files (including the packaged golden set) or from
``eval_set`` rows returned by the SQL Statement Execution API, where ARRAY and
MAP columns arrive either as native lists/dicts or as JSON strings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from client_research_agent.models import Criterion, DocumentType, FitVerdict, ResearchRequest

GOLDEN_SET_RESOURCE = "golden_set.jsonl"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class EvalInputs(_Model):
    company_name: str = Field(min_length=1, max_length=200)
    domain: str | None = Field(default=None, max_length=253)
    ticker: str | None = Field(default=None, max_length=10)
    cik: str | None = Field(default=None, pattern=r"^\d{1,10}$")
    industry: str | None = Field(default=None, max_length=120)
    max_documents: int = Field(default=40, ge=1, le=500)

    @field_validator("domain", "ticker", "cik", "industry", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    def to_request(self, *, requested_by: str = "evaluation") -> ResearchRequest:
        return ResearchRequest(
            company_name=self.company_name,
            domain=self.domain,
            ticker=self.ticker,
            cik=self.cik,
            industry=self.industry,
            max_documents=self.max_documents,
            requested_by=requested_by,
        )


class EvalExpectations(_Model):
    expected_verdict: FitVerdict | None = None
    expected_facts: tuple[str, ...] = ()
    guidelines: tuple[str, ...] = ()
    min_citation_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    expected_scores: dict[Criterion, int] = Field(default_factory=dict)

    @field_validator("expected_scores")
    @classmethod
    def _scores_on_rubric(cls, value: dict[Criterion, int]) -> dict[Criterion, int]:
        for criterion, score in value.items():
            if not 0 <= score <= 5:
                raise ValueError(f"expected score for {criterion.value} must be 0-5, got {score}")
        return value

    @property
    def is_empty(self) -> bool:
        return (
            self.expected_verdict is None
            and not self.expected_facts
            and not self.guidelines
            and not self.expected_scores
            and self.min_citation_coverage is None
        )


class SnapshotDocument(_Model):
    """A captured public document replayed by :class:`~.snapshot.SnapshotIngestion`."""

    url: str = Field(pattern=r"^https?://")
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    document_type: DocumentType
    publication_date: date | None = None


class EvalExample(_Model):
    eval_id: str = Field(min_length=1, max_length=200)
    inputs: EvalInputs
    expectations: EvalExpectations = Field(default_factory=EvalExpectations)
    tags: dict[str, str] = Field(default_factory=dict)
    snapshot: tuple[SnapshotDocument, ...] = ()

    def to_mlflow_record(self) -> dict[str, Any]:
        """``{"inputs", "expectations", "tags"}`` as consumed by ``mlflow.genai.evaluate``."""
        return {
            "inputs": self.inputs.model_dump(mode="json", exclude_none=True),
            "expectations": self.expectations.model_dump(mode="json", exclude_none=True),
            "tags": dict(self.tags),
        }


def parse_examples(lines: Iterable[str], *, source: str = "<lines>") -> list[EvalExample]:
    examples: list[EvalExample] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            example = EvalExample.model_validate_json(line)
        except ValueError as exc:
            raise ValueError(f"{source}:{number}: invalid evaluation example: {exc}") from exc
        if example.eval_id in seen:
            raise ValueError(f"{source}:{number}: duplicate eval_id {example.eval_id!r}")
        seen.add(example.eval_id)
        examples.append(example)
    return examples


def load_jsonl(path: str | os.PathLike[str]) -> list[EvalExample]:
    location = Path(path)
    return parse_examples(location.read_text(encoding="utf-8").splitlines(), source=str(location))


def load_golden_set() -> list[EvalExample]:
    """The packaged golden set (fictional companies with captured public-style documents)."""
    resource = resources.files("client_research_agent.evaluation") / "data" / GOLDEN_SET_RESOURCE
    return parse_examples(resource.read_text(encoding="utf-8").splitlines(), source=GOLDEN_SET_RESOURCE)


def write_jsonl(examples: Sequence[EvalExample], path: str | os.PathLike[str]) -> Path:
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(example.model_dump_json(exclude_defaults=True) + "\n" for example in examples)
    location.write_text(body, encoding="utf-8")
    return location


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return value


def _request_inputs(raw_request: Any) -> dict[str, Any]:
    """Extract research inputs from the ``request`` column (a ResponsesAgent request or plain inputs)."""
    request = _json_value(raw_request, {})
    if not isinstance(request, Mapping):
        return {}
    custom = request.get("custom_inputs")
    source: Mapping[str, Any] = custom if isinstance(custom, Mapping) else request
    keys = ("company_name", "domain", "ticker", "cik", "industry", "max_documents")
    return {key: source[key] for key in keys if source.get(key) not in (None, "")}


def example_from_row(row: Mapping[str, Any]) -> EvalExample:
    """Build an example from an ``eval_set`` row (Statement Execution API or Spark ``Row.asDict()``)."""
    inputs = _request_inputs(row.get("request"))
    for column, key in (
        ("company", "company_name"),
        ("domain", "domain"),
        ("ticker", "ticker"),
        ("cik", "cik"),
    ):
        value = row.get(column)
        if value not in (None, ""):
            inputs[key] = value
    facts = _json_value(row.get("expected_facts"), [])
    guidelines = _json_value(row.get("guidelines"), [])
    tags = _json_value(row.get("tags"), {})
    verdict = row.get("expected_verdict")
    return EvalExample(
        eval_id=str(row["eval_id"]),
        inputs=EvalInputs.model_validate(inputs),
        expectations=EvalExpectations(
            expected_verdict=FitVerdict(verdict) if verdict else None,
            expected_facts=tuple(str(f) for f in facts if str(f).strip()),
            guidelines=tuple(str(g) for g in guidelines if str(g).strip()),
        ),
        tags={str(k): str(v) for k, v in tags.items()} if isinstance(tags, Mapping) else {},
    )


def examples_from_rows(rows: Iterable[Mapping[str, Any]]) -> list[EvalExample]:
    return [example_from_row(row) for row in rows if _is_active(row.get("active", True))]


def _is_active(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", ""}
    return bool(value)
