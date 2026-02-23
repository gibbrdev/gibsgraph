"""Integration tests — uses mock Neo4j driver."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.agent import GibsGraphAgent, AgentState
from gibsgraph.config import Settings


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        NEO4J_READ_ONLY=True,
    )


@pytest.fixture
def mock_neo4j():
    """Patch GraphDatabase.driver to avoid real Neo4j connection."""
    with patch("gibsgraph.kg_builder.builder.GraphDatabase.driver") as mock_driver, \
         patch("gibsgraph.retrieval.retriever.GraphDatabase.driver") as mock_ret_driver:
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.run.return_value = []
        mock_driver.return_value.session.return_value = mock_session
        mock_ret_driver.return_value.session.return_value = mock_session
        yield mock_driver


def test_agent_ask_returns_state(mock_settings, mock_neo4j):
    agent = GibsGraphAgent(settings=mock_settings)
    result = agent.ask("What is the relationship between Apple and Beats?")
    assert isinstance(result, AgentState)
    assert result.query == "What is the relationship between Apple and Beats?"
    assert result.steps > 0


def test_agent_ask_no_crash_on_empty_graph(mock_settings, mock_neo4j):
    agent = GibsGraphAgent(settings=mock_settings)
    result = agent.ask("random question about nothing")
    # Should complete without exception
    assert result is not None


def test_kg_builder_raises_in_read_only(mock_settings, mock_neo4j):
    agent = GibsGraphAgent(settings=mock_settings)
    with pytest.raises(RuntimeError, match="read-only"):
        agent.kg_builder.ingest("Some text to ingest")
