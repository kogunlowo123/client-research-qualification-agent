from __future__ import annotations

import pytest

from tests.unit.research.helpers import ScriptedFetcher


@pytest.fixture
def scripted() -> ScriptedFetcher:
    return ScriptedFetcher()
