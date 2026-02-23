"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set minimal env vars so Settings can be instantiated in tests."""
    monkeypatch.setenv("NEO4J_PASSWORD", "testpassword")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_READ_ONLY", "true")
