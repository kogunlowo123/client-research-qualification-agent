"""Data classification and Unity Catalog tag mapping.

Levels are ordered PUBLIC < INTERNAL < CONFIDENTIAL < RESTRICTED. Source text
scraped from the public web is PUBLIC; derived sales intelligence (scores,
briefs) is CONFIDENTIAL; identities and audit payloads are INTERNAL/RESTRICTED;
anything containing personal data or credentials is RESTRICTED.

:func:`table_tag_statements` renders ``ALTER TABLE ... SET TAGS`` SQL for Unity
Catalog governed tags, validating identifiers to avoid SQL injection.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from client_research_agent.security.pii import PiiRedactor


class DataClassification(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

    @property
    def rank(self) -> int:
        return _RANK[self]

    @classmethod
    def highest(cls, levels: list[DataClassification]) -> DataClassification:
        return max(levels, key=lambda level: level.rank) if levels else cls.PUBLIC


_RANK: Mapping[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}

DEFAULT_FIELD_CLASSIFICATIONS: Mapping[str, DataClassification] = {
    "company": DataClassification.PUBLIC,
    "company_name": DataClassification.PUBLIC,
    "domain": DataClassification.PUBLIC,
    "ticker": DataClassification.PUBLIC,
    "cik": DataClassification.PUBLIC,
    "industry": DataClassification.PUBLIC,
    "url": DataClassification.PUBLIC,
    "title": DataClassification.PUBLIC,
    "text": DataClassification.PUBLIC,
    "quote": DataClassification.PUBLIC,
    "publication_date": DataClassification.PUBLIC,
    "document_type": DataClassification.PUBLIC,
    "source_domain": DataClassification.PUBLIC,
    "content_hash": DataClassification.PUBLIC,
    "doc_id": DataClassification.INTERNAL,
    "chunk_id": DataClassification.INTERNAL,
    "evidence_id": DataClassification.INTERNAL,
    "run_id": DataClassification.INTERNAL,
    "embedding": DataClassification.INTERNAL,
    "trust_score": DataClassification.INTERNAL,
    "requested_by": DataClassification.INTERNAL,
    "principal": DataClassification.INTERNAL,
    "model_versions": DataClassification.INTERNAL,
    "score": DataClassification.CONFIDENTIAL,
    "weighted_score": DataClassification.CONFIDENTIAL,
    "verdict": DataClassification.CONFIDENTIAL,
    "rationale": DataClassification.CONFIDENTIAL,
    "brief_json": DataClassification.CONFIDENTIAL,
    "opportunities": DataClassification.CONFIDENTIAL,
    "recommended_next_actions": DataClassification.CONFIDENTIAL,
    "executive_talking_points": DataClassification.CONFIDENTIAL,
    "payload_json": DataClassification.RESTRICTED,
}

_NAME_RULES: tuple[tuple[re.Pattern[str], DataClassification], ...] = (
    (
        re.compile(r"(?:secret|token|password|passwd|credential|api_?key|private_?key)", re.I),
        DataClassification.RESTRICTED,
    ),
    (
        re.compile(r"(?:ssn|social_security|card_number|iban|phone|email|home_address|dob|birth)", re.I),
        DataClassification.RESTRICTED,
    ),
    (
        re.compile(r"(?:score|verdict|brief|recommend|opportunit|qualification)", re.I),
        DataClassification.CONFIDENTIAL,
    ),
    (re.compile(r"(?:_id$|^id$|user|principal|owner|audit)", re.I), DataClassification.INTERNAL),
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2}$")


def uc_tags(classification: DataClassification, *, contains_pii: bool | None = None) -> dict[str, str]:
    """Governed-tag key/values for a classification (keys follow a ``data_*`` convention)."""
    pii = classification is DataClassification.RESTRICTED if contains_pii is None else contains_pii
    return {
        "data_classification": classification.value,
        "data_contains_pii": "true" if pii else "false",
        "data_retention": "7y" if classification is DataClassification.RESTRICTED else "3y",
    }


def _quote_ident(name: str) -> str:
    if not _IDENT.fullmatch(name):
        raise ValueError(f"invalid identifier {name!r}")
    return f"`{name}`"


def _quote_table(name: str) -> str:
    if not _TABLE.fullmatch(name):
        raise ValueError(f"invalid table name {name!r}")
    return ".".join(f"`{part}`" for part in name.split("."))


def _tag_clause(tags: Mapping[str, str]) -> str:
    parts = []
    for key, value in tags.items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", key) or not re.fullmatch(
            r"[A-Za-z0-9_. -]{0,256}", value
        ):
            raise ValueError(f"invalid tag {key!r}={value!r}")
        parts.append(f"'{key}' = '{value}'")
    return ", ".join(parts)


def table_tag_statements(
    table: str,
    table_classification: DataClassification,
    column_classifications: Mapping[str, DataClassification] | None = None,
) -> list[str]:
    target = _quote_table(table)
    statements = [f"ALTER TABLE {target} SET TAGS ({_tag_clause(uc_tags(table_classification))})"]
    for column, level in (column_classifications or {}).items():
        column_tags = _tag_clause(uc_tags(level))
        statements.append(
            f"ALTER TABLE {target} ALTER COLUMN {_quote_ident(column)} SET TAGS ({column_tags})"
        )
    return statements


class DataClassifier:
    def __init__(
        self,
        field_classifications: Mapping[str, DataClassification] | None = None,
        *,
        redactor: PiiRedactor | None = None,
        default: DataClassification = DataClassification.INTERNAL,
    ) -> None:
        self._fields = dict(
            DEFAULT_FIELD_CLASSIFICATIONS if field_classifications is None else field_classifications
        )
        self._redactor = redactor or PiiRedactor()
        self._default = default

    def classify_field(self, name: str) -> DataClassification:
        key = name.lower()
        if key in self._fields:
            return self._fields[key]
        for pattern, level in _NAME_RULES:
            if pattern.search(key):
                return level
        return self._default

    def classify_value(
        self, value: Any, *, baseline: DataClassification = DataClassification.PUBLIC
    ) -> DataClassification:
        if isinstance(value, str):
            return DataClassification.RESTRICTED if self._redactor.contains_pii(value) else baseline
        if isinstance(value, Mapping):
            return DataClassification.highest([baseline, self.classify_record(value)])
        if isinstance(value, list | tuple):
            return DataClassification.highest([baseline, *(self.classify_value(v) for v in value)])
        return baseline

    def classify_record(self, record: Mapping[str, Any]) -> DataClassification:
        levels = [
            self.classify_value(value, baseline=self.classify_field(str(key)))
            for key, value in record.items()
        ]
        return DataClassification.highest(levels)

    def column_classifications(self, columns: list[str]) -> dict[str, DataClassification]:
        return {column: self.classify_field(column) for column in columns}
