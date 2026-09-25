from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog


@pytest.fixture(autouse=True)
def _reset_structlog() -> Iterator[None]:
    """configure_logging binds sys.stdout; reset so later tests never write to a closed capture."""
    yield
    structlog.reset_defaults()
