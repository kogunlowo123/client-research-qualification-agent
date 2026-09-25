from __future__ import annotations

from dataclasses import dataclass

import pytest

from client_research_agent.config.settings import AppSettings
from client_research_agent.models import QualificationResult
from client_research_agent.prompts.registry import default_registry
from client_research_agent.qualification.agent import QualificationAgent
from client_research_agent.qualification.evidence import EvidenceRegistry
from client_research_agent.scoring.engine import ScoringEngine
from tests.unit.qualification.helpers import COMPANY, TODAY, KeywordRetriever, acme_corpus


@dataclass(frozen=True)
class Qualified:
    result: QualificationResult
    evidence: EvidenceRegistry


@pytest.fixture
def qualified(settings: AppSettings) -> Qualified:
    output = QualificationAgent(
        None, KeywordRetriever(acme_corpus()), default_registry(), settings, today=TODAY
    ).qualify(COMPANY)
    return Qualified(ScoringEngine(settings.scoring).evaluate(output.scores), output.evidence)
