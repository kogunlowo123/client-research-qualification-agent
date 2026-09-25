"""Behavioural contract every ``BriefRepository`` adapter must honour."""

from __future__ import annotations

import pytest

from client_research_agent.services.ports import BriefRepository
from tests.contract.fakes import make_brief


def test_satisfies_protocol(brief_repository: BriefRepository) -> None:
    assert isinstance(brief_repository, BriefRepository)


def test_round_trip(brief_repository: BriefRepository) -> None:
    brief = make_brief("run-42")
    assert brief_repository.save(brief) == "run-42"
    assert brief_repository.get("run-42") == brief


def test_missing_returns_none(brief_repository: BriefRepository) -> None:
    assert brief_repository.get("never-saved") is None


def test_save_overwrites(brief_repository: BriefRepository) -> None:
    brief_repository.save(make_brief("run-1", company="Acme Corp"))
    brief_repository.save(make_brief("run-1", company="Globex"))
    loaded = brief_repository.get("run-1")
    assert loaded is not None
    assert loaded.company == "Globex"


@pytest.mark.parametrize("run_id", ["../escape", "a/b", "", "..", "with space"])
def test_unsafe_run_ids_rejected(brief_repository: BriefRepository, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        brief_repository.get(run_id)
