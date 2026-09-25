from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pytest

from client_research_agent.models import DocumentType, SourceDocument
from client_research_agent.services.ports import ChatMessage, LLMResponse
from client_research_agent.utils.errors import UpstreamTimeoutError

LONG_DOCUMENT = """# Acme Corp Investor Day 2026

## Cloud Migration

Acme Corp completed the migration of 70 percent of its enterprise workloads to Microsoft Azure. \
The company closed two data centers in Ohio during fiscal 2026. Chief Information Officer Maria Chen \
said the remaining legacy systems will move by 2027. The program is expected to save $40 million a year.

## Artificial Intelligence

Acme launched machine learning models for predictive maintenance across 14 factories. \
A generative AI copilot now supports 1,200 field service engineers. The company plans to expand \
the copilot to 3,000 technicians next year. Responsible AI reviews are run by a governance council.

## Financial Results

Fourth quarter revenue was $4.2 billion, up 8 percent year over year. Operating margin expanded \
to 14.1 percent. Capital expenditure will rise to $600 million to fund automation and digital investments.
"""


def make_document(
    text: str = LONG_DOCUMENT,
    *,
    doc_id: str = "doc-acme-investor-day",
    company: str = "Acme Corp",
    title: str = "Acme Corp Investor Day 2026",
    trust: float = 0.9,
) -> SourceDocument:
    return SourceDocument(
        doc_id=doc_id,
        company=company,
        url="https://acme.example.com/ir/investor-day-2026",
        title=title,
        text=text,
        document_type=DocumentType.INVESTOR_RELATIONS,
        source_domain="acme.example.com",
        content_hash=f"hash-{doc_id}",
        publication_date=date(2026, 3, 12),
        industry="Industrial Manufacturing",
        trust_score=trust,
    )


class FailingLLM:
    """LLM whose endpoint is down: every call raises a transient error."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "failing-llm"

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls += 1
        raise UpstreamTimeoutError("model serving timed out")


@pytest.fixture
def document() -> SourceDocument:
    return make_document()


@pytest.fixture
def failing_llm() -> FailingLLM:
    return FailingLLM()
