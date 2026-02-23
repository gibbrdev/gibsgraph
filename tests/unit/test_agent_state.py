"""Unit tests for AgentState and agent node functions."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.agent import AgentState, generate_explanation
from gibsgraph.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
    )


def test_agent_state_defaults():
    state = AgentState(query="test question")
    assert state.query == "test question"
    assert state.steps == 0
    assert state.errors == []
    assert state.subgraph is None
    assert state.requires_human_review is False


def test_generate_explanation_no_context(settings):
    state = AgentState(query="test", retrieved_context="")
    result = generate_explanation(state, settings=settings)
    assert "No relevant information" in result["explanation"]


@patch("gibsgraph.agent._make_llm")
def test_generate_explanation_with_context(mock_make_llm, settings):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="Apple acquired Beats Electronics for $3 billion in 2014."
    )
    mock_make_llm.return_value = mock_llm

    state = AgentState(query="test", retrieved_context="Apple acquired Beats for $3B")
    result = generate_explanation(state, settings=settings)
    assert len(result["explanation"]) > 0
    assert result["steps"] == 1


def test_agent_state_error_accumulation():
    state = AgentState(query="test", errors=["error1"])
    new_errors = [*state.errors, "error2"]
    updated = state.model_copy(update={"errors": new_errors})
    assert len(updated.errors) == 2
