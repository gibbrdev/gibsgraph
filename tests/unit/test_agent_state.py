"""Unit tests for AgentState and agent node functions."""

import pytest

from gibsgraph.agent import AgentState, classify_usecase, generate_explanation


def test_agent_state_defaults():
    state = AgentState(query="test question")
    assert state.query == "test question"
    assert state.steps == 0
    assert state.errors == []
    assert state.subgraph is None
    assert state.requires_human_review is False


def test_classify_usecase_increments_steps():
    state = AgentState(query="What companies acquired Tesla?")
    result = classify_usecase(state)
    assert result["steps"] == 1
    assert result["usecase"] == "general"


def test_generate_explanation_no_context():
    state = AgentState(query="test", retrieved_context="")
    result = generate_explanation(state)
    assert "No relevant subgraph" in result["explanation"]


def test_generate_explanation_with_context():
    state = AgentState(query="test", retrieved_context="Apple acquired Beats for $3B")
    result = generate_explanation(state)
    assert len(result["explanation"]) > 0
    assert result["steps"] == 1


def test_agent_state_error_accumulation():
    state = AgentState(query="test", errors=["error1"])
    new_errors = [*state.errors, "error2"]
    updated = state.model_copy(update={"errors": new_errors})
    assert len(updated.errors) == 2
