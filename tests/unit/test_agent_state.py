"""Unit tests for AgentState and agent node functions."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.agent import AgentState, IntentClassification, classify_intent, generate_explanation
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


# --- IntentClassification ---


def test_intent_classification_defaults():
    intent = IntentClassification()
    assert intent.action == "ask"
    assert intent.industry == ""
    assert intent.region == ""
    assert intent.regulations == []
    assert intent.data_type == ""
    assert intent.goal == ""
    assert intent.enriched_query == ""


def test_intent_classification_full():
    intent = IntentClassification(
        action="build",
        industry="insurance",
        region="sweden",
        regulations=["GDPR", "IDD", "Solvency II"],
        data_type="TOS documents",
        goal="map customer psychology",
        enriched_query="Build a knowledge graph for a Swedish insurance company "
        "processing TOS documents to map customer psychology patterns, "
        "considering GDPR, IDD, and Solvency II regulations.",
    )
    assert intent.action == "build"
    assert "GDPR" in intent.regulations
    assert intent.industry == "insurance"


def test_agent_state_has_intent():
    state = AgentState(query="test")
    assert isinstance(state.intent, IntentClassification)
    assert state.intent.action == "ask"


def test_agent_state_with_intent():
    intent = IntentClassification(
        action="ask",
        industry="fintech",
        region="eu",
        regulations=["PSD2", "GDPR"],
        goal="detect fraud",
        enriched_query="What fraud detection patterns exist in EU fintech?",
    )
    state = AgentState(query="fraud patterns eu fintech", intent=intent)
    assert state.intent.industry == "fintech"
    assert state.intent.enriched_query == "What fraud detection patterns exist in EU fintech?"


# --- classify_intent node ---


@patch("gibsgraph.agent._make_llm")
def test_classify_intent_success(mock_make_llm, settings):
    """classify_intent extracts structured intent from NL input."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = IntentClassification(
        action="build",
        industry="insurance",
        region="sweden",
        regulations=["GDPR", "IDD"],
        data_type="TOS documents",
        goal="map customer psychology",
        enriched_query="Build insurance graph for Swedish TOS analysis",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_make_llm.return_value = mock_llm

    state = AgentState(
        query="running an insurance company in sweden, processing TOS docs, "
        "want to map out customer psychology"
    )
    result = classify_intent(state, settings=settings)

    assert result["intent"].industry == "insurance"
    assert result["intent"].region == "sweden"
    assert "GDPR" in result["intent"].regulations
    assert result["intent"].action == "build"
    assert "insurance" in result["usecase"]
    assert result["steps"] == 1


@patch("gibsgraph.agent._make_llm")
def test_classify_intent_fallback_on_error(mock_make_llm, settings):
    """classify_intent gracefully handles LLM failures."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.side_effect = Exception("API timeout")
    mock_make_llm.return_value = mock_llm

    state = AgentState(query="some query")
    result = classify_intent(state, settings=settings)

    # Should not crash, just increment steps
    assert result["steps"] == 1
    assert "intent" not in result  # no intent on failure


@patch("gibsgraph.agent._make_llm")
def test_classify_intent_unexpected_return_type(mock_make_llm, settings):
    """classify_intent handles non-IntentClassification return."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = "not a pydantic model"
    mock_llm.with_structured_output.return_value = mock_structured
    mock_make_llm.return_value = mock_llm

    state = AgentState(query="some query")
    result = classify_intent(state, settings=settings)

    assert result["steps"] == 1
    assert "intent" not in result


@patch("gibsgraph.agent._make_llm")
def test_classify_intent_minimal_input(mock_make_llm, settings):
    """classify_intent handles simple queries that don't have much context."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = IntentClassification(
        action="ask",
        enriched_query="Who acquired Beats?",
    )
    mock_llm.with_structured_output.return_value = mock_structured
    mock_make_llm.return_value = mock_llm

    state = AgentState(query="Who acquired Beats?")
    result = classify_intent(state, settings=settings)

    assert result["intent"].action == "ask"
    assert result["intent"].industry == ""  # nothing to extract
    assert result["usecase"] == ""  # no industry = no usecase


# --- generate_explanation with intent context ---


@patch("gibsgraph.agent._make_llm")
def test_generate_explanation_uses_intent_context(mock_make_llm, settings):
    """Explanation prompt includes industry/regulation context from intent."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="Answer with context.")
    mock_make_llm.return_value = mock_llm

    intent = IntentClassification(
        industry="insurance",
        region="sweden",
        regulations=["GDPR", "IDD"],
        goal="map customer psychology",
    )
    state = AgentState(
        query="test",
        intent=intent,
        retrieved_context="Some graph data",
    )
    result = generate_explanation(state, settings=settings)

    # Verify the LLM was called with intent context in the prompt
    call_args = mock_llm.invoke.call_args[0][0]
    assert "insurance" in call_args
    assert "GDPR" in call_args
    assert "customer psychology" in call_args
    assert result["explanation"] == "Answer with context."
