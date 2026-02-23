"""Unit tests for agent module — _make_llm, build_graph, nodes, GibsGraphAgent."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.agent import (
    AgentState,
    GibsGraphAgent,
    _make_llm,
    build_graph,
    generate_explanation,
    retrieve_subgraph,
    should_continue,
    validate_output,
    visualize,
)
from gibsgraph.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
    )


# --- _make_llm ---


@patch("langchain_openai.ChatOpenAI")
def test_make_llm_openai(mock_openai, settings):
    settings_oa = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        LLM_MODEL="gpt-4o-mini",
    )
    _make_llm(settings_oa)
    mock_openai.assert_called_once()


def test_make_llm_anthropic():
    settings_ant = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        LLM_MODEL="claude-3-haiku-20240307",
    )
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        _make_llm(settings_ant)
        mock_anthropic.assert_called_once()


def test_make_llm_mistral():
    import sys

    settings_m = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        LLM_MODEL="mistral-small-latest",
    )
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"langchain_mistralai": mock_module}):
        _make_llm(settings_m)
        mock_module.ChatMistralAI.assert_called_once()


@patch("langchain_openai.ChatOpenAI")
@patch.dict("os.environ", {"XAI_API_KEY": "xai-test-key"})
def test_make_llm_xai_grok(mock_openai):
    settings_xai = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        LLM_MODEL="grok-3",
    )
    _make_llm(settings_xai)
    mock_openai.assert_called_once()
    call_kwargs = mock_openai.call_args[1]
    assert call_kwargs["base_url"] == "https://api.x.ai/v1"
    assert call_kwargs["api_key"] == "xai-test-key"
    assert call_kwargs["model"] == "grok-3"


# --- should_continue ---


def test_should_continue_to_visualize():
    state = AgentState(query="test", steps=2, requires_human_review=False)
    assert should_continue(state) == "visualize"


def test_should_continue_max_steps():
    from langgraph.graph import END

    state = AgentState(query="test", steps=10)
    assert should_continue(state) == END


def test_should_continue_too_many_errors():
    from langgraph.graph import END

    state = AgentState(query="test", errors=["e1", "e2", "e3"])
    assert should_continue(state) == END


def test_should_continue_human_review():
    state = AgentState(query="test", requires_human_review=True)
    assert should_continue(state) == "human_review"


# --- validate_output ---


def test_validate_output_no_cypher():
    state = AgentState(query="test", cypher_used="", steps=0)
    result = validate_output(state)
    assert result["requires_human_review"] is False
    assert result["steps"] == 1


def test_validate_output_valid_cypher():
    state = AgentState(query="test", cypher_used="MATCH (n) RETURN n", steps=0)
    result = validate_output(state)
    assert result["requires_human_review"] is False


def test_validate_output_invalid_cypher():
    state = AgentState(query="test", cypher_used="MATCH (n) DELETE n", steps=0)
    result = validate_output(state)
    assert result["requires_human_review"] is True


def test_validate_output_with_errors():
    state = AgentState(query="test", cypher_used="MATCH (n) RETURN n", errors=["err"], steps=0)
    result = validate_output(state)
    assert result["requires_human_review"] is True


# --- visualize ---


def test_visualize_no_subgraph(settings):
    state = AgentState(query="test", subgraph=None, steps=0)
    result = visualize(state, settings=settings)
    assert result["steps"] == 1
    assert "visualization_url" not in result


def test_visualize_with_subgraph(settings):
    subgraph = {"nodes": [{"id": "n1", "name": "A"}], "edges": []}
    state = AgentState(query="test", subgraph=subgraph, steps=0)
    result = visualize(state, settings=settings)
    assert result["steps"] == 1
    assert "visualization_url" in result


# --- retrieve_subgraph ---


def test_retrieve_subgraph_success(settings):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = MagicMock(
        subgraph={"nodes": [{"id": "1"}], "edges": []},
        context="Some context",
        cypher="MATCH (n) RETURN n",
    )
    state = AgentState(query="test", steps=0)
    result = retrieve_subgraph(state, settings=settings, retriever=mock_retriever)
    assert result["subgraph"]["nodes"][0]["id"] == "1"
    assert result["retrieved_context"] == "Some context"
    assert result["steps"] == 1


def test_retrieve_subgraph_failure(settings):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.side_effect = RuntimeError("Connection failed")
    state = AgentState(query="test", steps=0)
    result = retrieve_subgraph(state, settings=settings, retriever=mock_retriever)
    assert "Connection failed" in result["errors"][-1]
    assert result["steps"] == 1


# --- generate_explanation ---


def test_generate_explanation_no_context(settings):
    state = AgentState(query="test", retrieved_context="")
    result = generate_explanation(state, settings=settings)
    assert "No relevant information" in result["explanation"]


def test_generate_explanation_no_results(settings):
    state = AgentState(query="test", retrieved_context="No results found.")
    result = generate_explanation(state, settings=settings)
    assert "No relevant information" in result["explanation"]


@patch("gibsgraph.agent._make_llm")
def test_generate_explanation_with_context(mock_make_llm, settings):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The answer is 42.")
    mock_make_llm.return_value = mock_llm

    state = AgentState(query="What is the answer?", retrieved_context="Data says 42", steps=0)
    result = generate_explanation(state, settings=settings)
    assert result["explanation"] == "The answer is 42."
    assert result["steps"] == 1


# --- build_graph ---


@patch("gibsgraph.agent.GraphRetriever")
def test_build_graph_compiles(mock_retriever_cls, settings):
    graph = build_graph(settings)
    assert graph is not None


# --- GibsGraphAgent ---


@patch("gibsgraph.agent.GraphRetriever")
@patch("gibsgraph.agent.KGBuilder")
def test_agent_init(mock_kb, mock_ret, settings):
    agent = GibsGraphAgent(settings=settings)
    assert agent._graph is not None
    agent.close()
    mock_ret.return_value.close.assert_called_once()
    mock_kb.return_value.close.assert_called_once()
