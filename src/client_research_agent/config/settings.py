"""Layered configuration.

Precedence (highest wins):
    1. Environment variables prefixed ``CRA_`` (nested with ``__``)
    2. ``config/environments/<env>.yaml`` shipped in the wheel
    3. Defaults declared on the models below

Secrets are never read from YAML. They come from environment variables that
Databricks injects from a secret scope (``{{secrets/<scope>/<key>}}``) or from
the ambient Databricks unified auth chain.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from client_research_agent.models import Criterion

#: Domain of the shipped placeholder contact address; rejected in staging and production.
PLACEHOLDER_CONTACT_DOMAIN = "@example.org"


class Environment(StrEnum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class DatabricksSettings(BaseModel):
    host: str | None = None
    token: SecretStr | None = None
    catalog: str = "client_research"
    schema_: str = Field(default="agent_dev", alias="schema")
    secret_scope: str = "client-research-agent"  # noqa: S105 - scope name, not a secret
    warehouse_id: str | None = None

    model_config = {"populate_by_name": True}

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema_}.{name}"


class ModelServingSettings(BaseModel):
    chat_endpoint: str = "databricks-claude-sonnet-4"
    fallback_chat_endpoint: str = "databricks-meta-llama-3-3-70b-instruct"
    judge_endpoint: str = "databricks-claude-sonnet-4"
    embedding_endpoint: str = "databricks-gte-large-en"
    embedding_dimension: int = 1024
    request_timeout_seconds: float = 60.0
    max_output_tokens: int = 2048


class VectorSearchSettings(BaseModel):
    endpoint_name: str = "cra-vs-endpoint"
    index_name: str = "chunks_index"
    source_table: str = "chunks"
    primary_key: str = "chunk_id"
    embedding_column: str = "embedding"
    pipeline_type: str = "TRIGGERED"


class ChunkingSettings(BaseModel):
    child_chunk_tokens: int = Field(default=256, ge=64, le=2048)
    parent_chunk_tokens: int = Field(default=1024, ge=256, le=8192)
    chunk_overlap_tokens: int = Field(default=32, ge=0, le=512)
    semantic_breakpoint_percentile: float = Field(default=90.0, ge=50.0, le=99.9)

    @model_validator(mode="after")
    def _parent_larger_than_child(self) -> ChunkingSettings:
        if self.parent_chunk_tokens <= self.child_chunk_tokens:
            raise ValueError("parent_chunk_tokens must exceed child_chunk_tokens")
        if self.chunk_overlap_tokens >= self.child_chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than child_chunk_tokens")
        return self


class RetrievalSettings(BaseModel):
    top_k: int = Field(default=8, ge=1, le=50)
    candidate_pool: int = Field(default=40, ge=5, le=500)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    multi_query_count: int = Field(default=3, ge=1, le=8)
    rerank_enabled: bool = True
    compression_max_sentences: int = Field(default=6, ge=1, le=40)
    crag_min_relevance: float = Field(default=0.35, ge=0.0, le=1.0)
    crag_max_corrections: int = Field(default=2, ge=0, le=5)
    recency_half_life_days: int = Field(default=365, ge=30)


class CrawlerSettings(BaseModel):
    user_agent: str = (
        "ClientResearchAgent/1.0 (+https://github.com/kogunlowo123/client-research-qualification-agent)"
    )
    contact_email: str = "research-agent@example.org"
    requests_per_second_per_host: float = Field(default=1.0, gt=0.0, le=10.0)
    request_timeout_seconds: float = 20.0
    max_response_bytes: int = 15_000_000  # large 10-K primary documents exceed 5 MB
    max_pages_per_domain: int = 60
    allowed_domain_suffixes: tuple[str, ...] = ("sec.gov",)
    respect_robots_txt: bool = True


class ScoringSettings(BaseModel):
    weights: dict[Criterion, float] = Field(
        default_factory=lambda: {
            Criterion.COMPANY_SCALE: 0.20,
            Criterion.TECH_MODERNIZATION: 0.20,
            Criterion.AI_DATA_FOCUS: 0.25,
            Criterion.INDUSTRY_TRENDS: 0.10,
            Criterion.NEAR_TERM_OPPORTUNITY: 0.25,
        }
    )
    good_fit_threshold: float = Field(default=3.5, ge=0.0, le=5.0)
    potential_fit_threshold: float = Field(default=2.25, ge=0.0, le=5.0)
    min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    min_evidence_per_criterion: int = Field(default=1, ge=0)

    @field_validator("weights")
    @classmethod
    def _weights_sum_to_one(cls, value: dict[Criterion, float]) -> dict[Criterion, float]:
        if set(value) != set(Criterion):
            raise ValueError("weights must be provided for every criterion")
        total = sum(value.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"criterion weights must sum to 1.0, got {total:.4f}")
        return value


class GuardrailSettings(BaseModel):
    injection_block_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    redact_pii: bool = True
    min_citation_support: float = Field(default=0.3, ge=0.0, le=1.0)
    max_input_chars: int = Field(default=20_000, ge=100)
    #: Briefs whose fact statements are supported below this share go to analyst review.
    review_min_citation_coverage: float = Field(default=0.8, ge=0.0, le=1.0)


class ResilienceSettings(BaseModel):
    max_attempts: int = Field(default=4, ge=1, le=10)
    initial_backoff_seconds: float = Field(default=0.5, ge=0.0)
    max_backoff_seconds: float = Field(default=20.0, ge=0.0)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_reset_seconds: float = Field(default=30.0, gt=0.0)


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    json_logs: bool = True
    otlp_endpoint: str | None = None
    service_name: str = "client-research-agent"
    mlflow_experiment: str = "/Shared/client-research-agent"
    mlflow_tracing: bool = True


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRA_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    environment: Environment = Environment.LOCAL
    databricks: DatabricksSettings = Field(default_factory=DatabricksSettings)
    serving: ModelServingSettings = Field(default_factory=ModelServingSettings)
    vector_search: VectorSearchSettings = Field(default_factory=VectorSearchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    crawler: CrawlerSettings = Field(default_factory=CrawlerSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def uses_databricks(self) -> bool:
        return self.environment is not Environment.LOCAL

    @model_validator(mode="after")
    def _prod_requires_workspace(self) -> AppSettings:
        if self.environment in (Environment.STAGING, Environment.PROD):
            if not self.databricks.host and not os.environ.get("DATABRICKS_HOST"):
                raise ValueError(f"{self.environment} requires databricks.host or DATABRICKS_HOST")
            if self.databricks.token is not None:
                raise ValueError(
                    "static tokens are forbidden outside dev; use a service principal via unified auth"
                )
            if self.crawler.contact_email.lower().endswith(PLACEHOLDER_CONTACT_DOMAIN):
                raise ValueError(
                    f"{self.environment} requires a real SEC contact email: set crawler.contact_email "
                    "(CRA_CRAWLER__CONTACT_EMAIL); the example.org placeholder violates the SEC "
                    "fair-access policy"
                )
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_environment_overlay(environment: Environment) -> dict[str, Any]:
    package = resources.files("client_research_agent.config") / "environments"
    overlay: dict[str, Any] = {}
    for name in ("base.yaml", f"{environment.value}.yaml"):
        resource = package / name
        if resource.is_file():
            loaded = yaml.safe_load(resource.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"environment overlay {name} must be a mapping")
            overlay = _deep_merge(overlay, loaded)
    return overlay


class _EnvironmentSnapshot(AppSettings):
    """Reads ``CRA_*`` variables only; cross-field checks run once on the fully merged settings.

    Validating this intermediate snapshot would reject values that explicit overrides
    (``build_settings(..., databricks={"host": ...})``) are about to supply.
    """

    @model_validator(mode="after")
    def _prod_requires_workspace(self) -> _EnvironmentSnapshot:
        return self


def build_settings(environment: Environment | str | None = None, **overrides: Any) -> AppSettings:
    """Build settings for ``environment``: YAML overlay, then env vars, then explicit overrides."""
    env_value = environment or os.environ.get("CRA_ENVIRONMENT", Environment.LOCAL.value)
    env = Environment(env_value)
    overlay = load_environment_overlay(env)
    env_settings = _EnvironmentSnapshot(environment=env)
    explicitly_set = env_settings.model_dump(exclude_unset=True, by_alias=True)
    merged = _deep_merge(_deep_merge(overlay, explicitly_set), overrides)
    merged["environment"] = env
    return AppSettings.model_validate(merged)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return build_settings()
