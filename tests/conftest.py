"""Fixtures shared by several test files. pytest loads this file automatically."""

from pathlib import Path

import pytest


@pytest.fixture
def config_path() -> Path:
    """The real push-up config, found from this file so tests work from any directory."""
    return Path(__file__).resolve().parents[1] / "configs" / "pushup.yaml"
