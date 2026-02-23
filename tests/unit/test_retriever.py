"""Unit tests for GraphRetriever and GraphSchema."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.config import Settings
from gibsgraph.retrieval.retriever import GraphRetriever, GraphSchema, RetrievalResult


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
    )


def test_graph_schema_to_prompt():
    schema = GraphSchema(
        labels=["Article", "Regulation"],
        relationship_types=["HAS_ARTICLE"],
        relationship_patterns=["(:Regulation)-[:HAS_ARTICLE]->(:Article)"],
        property_keys={"Article": ["title", "number"], "Regulation": ["name"]},
        node_count=100,
    )
    prompt = schema.to_prompt()
    assert "Article" in prompt
    assert "Regulation" in prompt
    assert "HAS_ARTICLE" in prompt
    assert "100" in prompt


def test_graph_schema_to_prompt_with_samples():
    schema = GraphSchema(
        labels=["Article"],
        relationship_types=[],
        relationship_patterns=[],
        property_keys={"Article": ["title"]},
        sample_values={"Article": {"title": ["Art. 1", "Art. 2"]}},
        node_count=10,
    )
    prompt = schema.to_prompt()
    assert "sample title" in prompt
    assert "Art. 1" in prompt


def test_retrieval_result_defaults():
    result = RetrievalResult()
    assert result.subgraph == {}
    assert result.context == ""
    assert result.cypher == ""
    assert result.strategy == ""


@patch("gibsgraph.retrieval.retriever.GraphDatabase")
def test_retriever_close(mock_gdb, settings):
    retriever = GraphRetriever(settings=settings)
    retriever.close()
    mock_gdb.driver.return_value.close.assert_called_once()


@patch("gibsgraph.retrieval.retriever.GraphDatabase")
def test_retriever_serialize_context_no_results(mock_gdb, settings):
    retriever = GraphRetriever(settings=settings)
    result = retriever._serialize_context({"nodes": [], "edges": []})
    assert result == "No results found."


@patch("gibsgraph.retrieval.retriever.GraphDatabase")
def test_retriever_serialize_context_with_records(mock_gdb, settings):
    retriever = GraphRetriever(settings=settings)
    subgraph = {
        "nodes": [],
        "edges": [],
        "records": [{"count": 42}],
    }
    result = retriever._serialize_context(subgraph)
    assert "1 rows" in result
    assert "count=42" in result
