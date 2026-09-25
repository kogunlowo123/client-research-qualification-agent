from __future__ import annotations

import pytest

from client_research_agent.governance.data_classification import (
    DataClassification,
    DataClassifier,
    table_tag_statements,
    uc_tags,
)

C = DataClassification


def test_ordering_and_highest() -> None:
    assert C.PUBLIC.rank < C.INTERNAL.rank < C.CONFIDENTIAL.rank < C.RESTRICTED.rank
    assert C.highest([C.INTERNAL, C.CONFIDENTIAL, C.PUBLIC]) is C.CONFIDENTIAL
    assert C.highest([]) is C.PUBLIC


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("text", C.PUBLIC),
        ("URL", C.PUBLIC),
        ("run_id", C.INTERNAL),
        ("weighted_score", C.CONFIDENTIAL),
        ("payload_json", C.RESTRICTED),
        ("client_secret", C.RESTRICTED),
        ("contact_email", C.RESTRICTED),
        ("fit_score_v2", C.CONFIDENTIAL),
        ("reviewer_id", C.INTERNAL),
        ("misc", C.INTERNAL),
    ],
)
def test_classify_field(name: str, expected: DataClassification) -> None:
    assert DataClassifier().classify_field(name) is expected


def test_classify_values_and_records() -> None:
    classifier = DataClassifier()
    assert classifier.classify_value("jane.doe@gmail.com") is C.RESTRICTED
    assert classifier.classify_value("Acme grew") is C.PUBLIC
    assert classifier.classify_value(["a", "123-45-6789"]) is C.RESTRICTED
    assert classifier.classify_value(42, baseline=C.INTERNAL) is C.INTERNAL
    assert classifier.classify_record({"text": "Acme grew", "company": "Acme"}) is C.PUBLIC
    assert classifier.classify_record({"text": "Acme", "verdict": "good_fit"}) is C.CONFIDENTIAL
    assert classifier.classify_record({"text": "x", "nested": {"note": "ssn 123-45-6789"}}) is C.RESTRICTED
    assert classifier.column_classifications(["url", "run_id"]) == {"url": C.PUBLIC, "run_id": C.INTERNAL}


def test_custom_field_map_and_default() -> None:
    classifier = DataClassifier({"notes": C.CONFIDENTIAL}, default=C.PUBLIC)
    assert classifier.classify_field("notes") is C.CONFIDENTIAL
    assert classifier.classify_field("other") is C.PUBLIC


def test_uc_tags() -> None:
    assert uc_tags(C.RESTRICTED) == {
        "data_classification": "restricted",
        "data_contains_pii": "true",
        "data_retention": "7y",
    }
    assert uc_tags(C.PUBLIC, contains_pii=True)["data_contains_pii"] == "true"
    assert uc_tags(C.INTERNAL)["data_contains_pii"] == "false"


def test_table_tag_statements() -> None:
    statements = table_tag_statements(
        "main.cra.briefs", C.CONFIDENTIAL, {"brief_json": C.CONFIDENTIAL, "requested_by": C.INTERNAL}
    )
    assert statements[0] == (
        "ALTER TABLE `main`.`cra`.`briefs` SET TAGS ('data_classification' = 'confidential', "
        "'data_contains_pii' = 'false', 'data_retention' = '3y')"
    )
    assert statements[2].startswith("ALTER TABLE `main`.`cra`.`briefs` ALTER COLUMN `requested_by` SET TAGS")
    assert len(table_tag_statements("t", C.PUBLIC)) == 1


@pytest.mark.parametrize("table", ["a.b.c.d", "x; DROP TABLE y", "1abc"])
def test_rejects_bad_table(table: str) -> None:
    with pytest.raises(ValueError, match="table"):
        table_tag_statements(table, C.PUBLIC)


def test_rejects_bad_column() -> None:
    with pytest.raises(ValueError, match="identifier"):
        table_tag_statements("t", C.PUBLIC, {"bad`col": C.PUBLIC})
