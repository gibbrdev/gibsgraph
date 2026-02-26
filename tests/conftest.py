"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

# Make tests/ importable so `from tests.fixtures.golden_schemas import ...` works
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set minimal env vars so Settings can be instantiated in tests."""
    monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_READ_ONLY", "true")
