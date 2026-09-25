"""Core domain model shared by every layer of the agent.

These types are the contract between ingestion, retrieval, qualification and
briefing. They are deliberately immutable (``frozen=True``) so that evidence
cannot be mutated after it has been cited, which keeps citation validation and
audit trails trustworthy.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class DocumentType(StrEnum):
    PRESS_RELEASE = "press_release"
    EARNINGS_RELEASE = "earnings_release"
    SEC_FILING = "sec_filing"
    INVESTOR_RELATIONS = "investor_relations"
    LEADERSHIP_ANNOUNCEMENT = "leadership_announcement"
    ANALYST_PUBLIC = "analyst_public"
    CORPORATE_WEBPAGE = "corporate_webpage"


class ChunkStrategy(StrEnum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    PARENT = "parent"
    CHILD = "child"


class Criterion(StrEnum):
    COMPANY_SCALE = "company_size_and_scale"
    TECH_MODERNIZATION = "technology_modernization"
    AI_DATA_FOCUS = "ai_and_data_focus"
    INDUSTRY_TRENDS = "industry_trends"
    NEAR_TERM_OPPORTUNITY = "near_term_opportunity"


class FitVerdict(StrEnum):
    GOOD_FIT = "good_fit"
    POTENTIAL_FIT = "potential_fit"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class ProvenanceKind(StrEnum):
    """Separates what a source said from what the model inferred."""

    VERIFIED_FACT = "verified_fact"
    AI_RECOMMENDATION = "ai_recommendation"


class ResearchRequest(_Frozen):
    company_name: str = Field(min_length=1, max_length=200)
    domain: str | None = Field(default=None, max_length=253, description="Corporate domain, e.g. example.com")
    ticker: str | None = Field(default=None, max_length=10)
    cik: str | None = Field(default=None, pattern=r"^\d{1,10}$", description="SEC Central Index Key")
    industry: str | None = Field(default=None, max_length=120)
    seed_urls: tuple[HttpUrl, ...] = Field(default=())
    max_documents: int = Field(default=40, ge=1, le=500)
    requested_by: str = Field(default="system", max_length=200)

    @field_validator("domain")
    @classmethod
    def _normalize_domain(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")
        return cleaned.split("/", 1)[0]

    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class SourceDocument(_Frozen):
    doc_id: str
    company: str
    url: str
    title: str
    text: str
    document_type: DocumentType
    source_domain: str
    content_hash: str
    publication_date: date | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    industry: str | None = None
    language: str = "en"
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(_Frozen):
    chunk_id: str
    doc_id: str
    text: str
    company: str
    url: str
    title: str
    document_type: DocumentType
    source_domain: str
    chunk_index: int = Field(ge=0)
    strategy: ChunkStrategy
    parent_id: str | None = None
    publication_date: date | None = None
    industry: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    token_count: int = Field(default=0, ge=0)
    entities: tuple[str, ...] = ()
    contextual_header: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """Contextual-retrieval text: situating header prepended to the chunk body."""
        return f"{self.contextual_header}\n\n{self.text}" if self.contextual_header else self.text


class RetrievedChunk(_Frozen):
    chunk: Chunk
    score: float
    retriever: str
    rank: int = Field(ge=0)


class Evidence(_Frozen):
    evidence_id: str
    chunk_id: str
    url: str
    title: str
    quote: str = Field(min_length=1)
    document_type: DocumentType
    publication_date: date | None = None
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class CriterionScore(_Frozen):
    criterion: Criterion
    score: int = Field(ge=0, le=5)
    weight: float = Field(gt=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    reasoning_trace: tuple[str, ...] = ()

    @property
    def weighted(self) -> float:
        return self.score * self.weight


class QualificationResult(_Frozen):
    scores: tuple[CriterionScore, ...]
    weighted_score: float = Field(ge=0.0, le=5.0)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    verdict: FitVerdict
    verdict_rationale: str

    @model_validator(mode="after")
    def _unique_criteria(self) -> QualificationResult:
        seen = [s.criterion for s in self.scores]
        if len(seen) != len(set(seen)):
            raise ValueError("duplicate criterion in qualification scores")
        return self


class BriefStatement(_Frozen):
    """A single brief line tagged with provenance and the evidence it rests on."""

    text: str = Field(min_length=1)
    provenance: ProvenanceKind
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _facts_need_evidence(self) -> BriefStatement:
        if self.provenance is ProvenanceKind.VERIFIED_FACT and not self.evidence_ids:
            raise ValueError("a verified fact must cite at least one evidence id")
        return self


class CitationCheck(_Frozen):
    evidence_id: str
    url: str
    supported: bool
    support_score: float = Field(ge=0.0, le=1.0)
    reason: str


class CitationReport(_Frozen):
    checks: tuple[CitationCheck, ...] = ()
    total_statements: int = 0
    supported_statements: int = 0
    removed_statements: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        return self.supported_statements / self.total_statements if self.total_statements else 0.0


class ClientBrief(_Frozen):
    run_id: str
    company: str
    generated_at: datetime = Field(default_factory=utc_now)
    company_overview: tuple[BriefStatement, ...]
    qualification: QualificationResult
    evidence: tuple[Evidence, ...]
    technology_priorities: tuple[BriefStatement, ...]
    gartner_relevant_insights: tuple[BriefStatement, ...]
    opportunities: tuple[BriefStatement, ...]
    risks: tuple[BriefStatement, ...]
    executive_summary: tuple[BriefStatement, ...]
    discovery_questions: tuple[str, ...] = Field(min_length=5, max_length=5)
    executive_talking_points: tuple[BriefStatement, ...]
    recommended_next_actions: tuple[BriefStatement, ...]
    citation_report: CitationReport = Field(default_factory=CitationReport)
    model_versions: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def all_statements(self) -> tuple[BriefStatement, ...]:
        return (
            *self.company_overview,
            *self.technology_priorities,
            *self.gartner_relevant_insights,
            *self.opportunities,
            *self.risks,
            *self.executive_summary,
            *self.executive_talking_points,
            *self.recommended_next_actions,
        )

    @property
    def verified_facts(self) -> tuple[BriefStatement, ...]:
        return tuple(s for s in self.all_statements() if s.provenance is ProvenanceKind.VERIFIED_FACT)

    @property
    def recommendations(self) -> tuple[BriefStatement, ...]:
        return tuple(s for s in self.all_statements() if s.provenance is ProvenanceKind.AI_RECOMMENDATION)
