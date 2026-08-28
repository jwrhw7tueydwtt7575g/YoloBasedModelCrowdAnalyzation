"""Shared pytest fixtures.

Helpers that are not fixtures live in ``_helpers.py`` and are imported
directly by the test modules.
"""

from __future__ import annotations

import pytest

from mctracker.types import Frame

from ._helpers import make_frame  # re-export for fixtures


@pytest.fixture
def blank_frame() -> Frame:
    return make_frame()
