"""Unit tests for the Graph public facade (_graph.py)."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph._graph import Answer, Graph, IngestResult, _resolve_llm
from gibsgraph.config import PROVIDERS

# --- Answer dataclass ---


def test_answer_str():
    a = Answer(question="q", answer="The answer is 42.")
    assert str(a) == "The answer is 42."


def test_answer_repr():
    a = Answer(question="q", answer="short", confidence=0.85)
    r = repr(a)
    assert "0.85" in r
    assert "short" in r


def test_answer_defaults():
    a = Answer(question="q", answer="a")
    assert a.cypher == ""
    assert a.confidence == 0.0
    assert a.visualization == ""
    assert a.bloom_url == ""
    assert a.nodes_retrieved == 0
    assert a.errors == []


# --- IngestResult dataclass ---


def test_ingest_result_str():
    r = IngestResult(
        source="test.txt", nodes_created=5, relationships_created=3, chunks_processed=1
    )
    s = str(r)
    assert "test.txt" in s
    assert "5 nodes" in s
    assert "3 relationships" in s


# --- _resolve_llm ---


def test_resolve_llm_explicit():
    assert _resolve_llm("gpt-4o") == "gpt-4o"
    assert _resolve_llm("claude-3-5-sonnet") == "claude-3-5-sonnet"


@patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False)
def test_resolve_llm_auto_openai():
    openai_provider = PROVIDERS[0]
    assert _resolve_llm("auto") == openai_provider.default_model


def test_resolve_llm_auto_anthropic():
    import os

    anthropic_provider = PROVIDERS[1]
    # Remove all provider keys, then set only Anthropic
    saved = {p.env_key: os.environ.pop(p.env_key, None) for p in PROVIDERS}
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    try:
        with patch("dotenv.load_dotenv", side_effect=lambda: None):
            assert _resolve_llm("auto") == anthropic_provider.default_model
    finally:
        for key, val in saved.items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


def test_resolve_llm_auto_mistral():
    import os

    mistral_provider = PROVIDERS[2]
    saved = {p.env_key: os.environ.pop(p.env_key, None) for p in PROVIDERS}
    os.environ["MISTRAL_API_KEY"] = "test-mistral-key"
    try:
        with patch("dotenv.load_dotenv", side_effect=lambda: None):
            assert _resolve_llm("auto") == mistral_provider.default_model
    finally:
        for key, val in saved.items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


def test_resolve_llm_auto_no_keys():
    import os

    saved = {p.env_key: os.environ.pop(p.env_key, None) for p in PROVIDERS}
    try:
        with patch("dotenv.load_dotenv", side_effect=lambda: None):
            with pytest.raises(RuntimeError, match="No LLM API key found"):
                _resolve_llm("auto")
    finally:
        for key, val in saved.items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)


# --- Graph construction ---


def test_graph_requires_password():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="password is required"):
            Graph()


def test_graph_repr():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        r = repr(g)
        assert "bolt://localhost:7687" in r
        assert "gpt-4o-mini" in r
        assert "read_only=True" in r


def test_graph_context_manager():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        # No agent created yet, so close() should be safe
        with g:
            pass  # __enter__ and __exit__ work


# --- Graph.ask() ---


def test_graph_ask():
    mock_agent = MagicMock()
    mock_agent.ask.return_value = MagicMock(
        explanation="Tom Hanks acted in Forrest Gump.",
        cypher_used="MATCH (p:Person)-[:ACTED_IN]->(m) RETURN m",
        errors=[],
        visualization_url="https://bloom.neo4j.io/...",
        subgraph={"nodes": [{"id": "1"}, {"id": "2"}], "edges": []},
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g._agent = mock_agent

        result = g.ask("What movies did Tom Hanks act in?")
        assert isinstance(result, Answer)
        assert "Forrest Gump" in result.answer
        assert result.cypher.startswith("MATCH")
        assert result.confidence == 1.0
        assert result.nodes_retrieved == 2


def test_graph_ask_with_errors():
    mock_agent = MagicMock()
    mock_agent.ask.return_value = MagicMock(
        explanation="Partial answer.",
        cypher_used="MATCH (n) RETURN n",
        errors=["timeout"],
        visualization_url="",
        subgraph=None,
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g._agent = mock_agent

        result = g.ask("test")
        assert result.confidence == 0.3
        assert result.visualization == ""
        assert result.nodes_retrieved == 0


# --- Graph.visualize() and Graph.cypher() ---


def test_graph_visualize_delegates():
    mock_agent = MagicMock()
    mock_agent.ask.return_value = MagicMock(
        explanation="answer",
        cypher_used="MATCH (n) RETURN n",
        errors=[],
        visualization_url="",
        subgraph={"nodes": [{"id": "1", "name": "A"}], "edges": []},
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g._agent = mock_agent

        viz = g.visualize("test")
        assert "graph LR" in viz


def test_graph_cypher_delegates():
    mock_agent = MagicMock()
    mock_agent.ask.return_value = MagicMock(
        explanation="answer",
        cypher_used="MATCH (n:Movie) RETURN n",
        errors=[],
        visualization_url="",
        subgraph=None,
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g._agent = mock_agent

        cypher = g.cypher("test")
        assert cypher == "MATCH (n:Movie) RETURN n"


# --- Graph.close() ---


def test_graph_close_with_agent():
    mock_agent = MagicMock()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g._agent = mock_agent
        g.close()
        mock_agent.close.assert_called_once()


def test_graph_close_without_agent():
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw")
        g.close()  # Should not raise


# --- Graph.ingest() ---


def test_graph_ingest_delegates():
    """Graph.ingest() delegates to KGBuilder and wraps result with source."""
    from gibsgraph.kg_builder.builder import IngestResult as BuilderIngestResult

    mock_agent = MagicMock()
    mock_agent.kg_builder.ingest.return_value = BuilderIngestResult(
        nodes_created=5, relationships_created=3, chunks_processed=1
    )

    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False):
        g = Graph("bolt://localhost:7687", password="testpw", read_only=False)
        g._agent = mock_agent

        result = g.ingest("Apple acquired Beats.", source="test")

    assert isinstance(result, IngestResult)
    assert result.source == "test"
    assert result.nodes_created == 5
    assert result.relationships_created == 3
    assert result.chunks_processed == 1
    mock_agent.kg_builder.ingest.assert_called_once_with("Apple acquired Beats.", source="test")
