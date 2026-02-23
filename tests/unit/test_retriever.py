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


@pytest.fixture
def mock_retriever(settings):
    """Create a GraphRetriever with a mocked Neo4j driver."""
    with patch("gibsgraph.retrieval.retriever.GraphDatabase") as mock_gdb:
        retriever = GraphRetriever(settings=settings)
        retriever._mock_gdb = mock_gdb
        yield retriever


# ---------------------------------------------------------------------------
# GraphSchema
# ---------------------------------------------------------------------------


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


def test_graph_schema_defaults():
    schema = GraphSchema()
    assert schema.labels == []
    assert schema.has_vector_index is False
    assert schema.vector_index_name == ""
    assert schema.node_count == 0


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


def test_retrieval_result_defaults():
    result = RetrievalResult()
    assert result.subgraph == {}
    assert result.context == ""
    assert result.cypher == ""
    assert result.strategy == ""
    assert result.nodes_count == 0
    assert result.edges_count == 0


# ---------------------------------------------------------------------------
# GraphRetriever basics
# ---------------------------------------------------------------------------


@patch("gibsgraph.retrieval.retriever.GraphDatabase")
def test_retriever_close(mock_gdb, settings):
    retriever = GraphRetriever(settings=settings)
    retriever.close()
    mock_gdb.driver.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# _serialize_context
# ---------------------------------------------------------------------------


def test_serialize_context_no_results(mock_retriever):
    result = mock_retriever._serialize_context({"nodes": [], "edges": []})
    assert result == "No results found."


def test_serialize_context_with_records(mock_retriever):
    subgraph = {
        "nodes": [],
        "edges": [],
        "records": [{"count": 42}],
    }
    result = mock_retriever._serialize_context(subgraph)
    assert "1 rows" in result
    assert "count=42" in result


def test_serialize_context_with_node_records(mock_retriever):
    subgraph = {
        "nodes": [],
        "edges": [],
        "records": [{"p": {"_labels": ["Person"], "name": "Alice", "_id": "1"}}],
    }
    result = mock_retriever._serialize_context(subgraph)
    assert "Person" in result
    assert "Alice" in result


def test_serialize_context_with_edge_records(mock_retriever):
    subgraph = {
        "nodes": [],
        "edges": [],
        "records": [{"r": {"type": "KNOWS"}}],
    }
    result = mock_retriever._serialize_context(subgraph)
    assert "KNOWS" in result


def test_serialize_context_with_nodes_no_records(mock_retriever):
    subgraph = {
        "nodes": [{"name": "Alice"}, {"name": "Bob"}],
        "edges": [{"start": "1", "end": "2", "type": "KNOWS"}],
    }
    result = mock_retriever._serialize_context(subgraph)
    assert "Nodes (2)" in result
    assert "Relationships (1)" in result
    assert "KNOWS" in result


# ---------------------------------------------------------------------------
# _clean_props
# ---------------------------------------------------------------------------


def test_clean_props_removes_embeddings():
    props = {
        "name": "Alice",
        "embedding": [0.1] * 100,
        "age": 30,
        "tags": ["a", "b"],
    }
    result = GraphRetriever._clean_props(props)
    assert "name" in result
    assert "age" in result
    assert "tags" in result  # short list, kept
    assert "embedding" not in result  # long list, removed


def test_clean_props_keeps_short_lists():
    props = {"items": [1, 2, 3]}
    result = GraphRetriever._clean_props(props)
    assert "items" in result


# ---------------------------------------------------------------------------
# _execute_read_cypher
# ---------------------------------------------------------------------------


def test_execute_read_cypher_validator_rejects(mock_retriever):
    subgraph, error = mock_retriever._execute_read_cypher("MATCH (n) DELETE n")
    assert "rejected by validator" in error
    assert subgraph["nodes"] == []


def test_execute_read_cypher_success(mock_retriever):
    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.keys.return_value = ["count"]
    mock_record.__getitem__ = lambda self, key: 42

    mock_session.execute_read.return_value = [mock_record]
    mock_retriever._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_retriever._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    subgraph, error = mock_retriever._execute_read_cypher("MATCH (n) RETURN count(n) AS count")
    assert error == ""
    assert len(subgraph["records"]) == 1


def test_execute_read_cypher_exception(mock_retriever):
    mock_session = MagicMock()
    mock_session.execute_read.side_effect = RuntimeError("Neo4j connection failed")
    mock_retriever._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_retriever._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    subgraph, error = mock_retriever._execute_read_cypher("MATCH (n) RETURN n")
    assert "Neo4j connection failed" in error
    assert subgraph["nodes"] == []


# ---------------------------------------------------------------------------
# _generate_cypher
# ---------------------------------------------------------------------------


def test_generate_cypher_basic(mock_retriever):
    schema = GraphSchema(
        labels=["Movie", "Person"],
        relationship_types=["ACTED_IN"],
        relationship_patterns=["(:Person)-[:ACTED_IN]->(:Movie)"],
        property_keys={"Person": ["name"], "Movie": ["title"]},
        node_count=100,
    )
    mock_llm = MagicMock()
    cypher_response = "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN m.title LIMIT 25"
    mock_llm.invoke.return_value = MagicMock(content=cypher_response)

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        cypher = mock_retriever._generate_cypher("What movies are there?", schema=schema)

    assert "MATCH" in cypher
    assert "Movie" in cypher


def test_generate_cypher_strips_markdown_fences(mock_retriever):
    schema = GraphSchema(labels=["X"], node_count=1)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="```cypher\nMATCH (n) RETURN n\n```")

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        cypher = mock_retriever._generate_cypher("test", schema=schema)

    assert not cypher.startswith("```")
    assert "MATCH (n) RETURN n" in cypher


def test_generate_cypher_with_retry_context(mock_retriever):
    schema = GraphSchema(labels=["X"], node_count=1)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n LIMIT 10")

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        mock_retriever._generate_cypher(
            "test",
            schema=schema,
            error="SyntaxError: blah",
            previous_cypher="MATCH (n RETURN n",
        )

    # Should include error context in the prompt
    call_args = mock_llm.invoke.call_args[0][0]
    assert "SyntaxError" in call_args
    assert "MATCH (n RETURN n" in call_args


# ---------------------------------------------------------------------------
# retrieve (integration of strategies)
# ---------------------------------------------------------------------------


def test_retrieve_uses_cypher_when_no_vector(mock_retriever):
    schema = GraphSchema(labels=["Person"], has_vector_index=False, node_count=10)
    mock_retriever._schema = schema

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="MATCH (n:Person) RETURN n.name LIMIT 10")

    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.keys.return_value = ["name"]
    mock_record.__getitem__ = lambda self, key: "Alice"
    mock_session.execute_read.return_value = [mock_record]
    mock_retriever._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_retriever._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        result = mock_retriever.retrieve("Who are the people?")

    assert result.strategy == "cypher"
    assert result.cypher != ""


def test_retrieve_uses_vector_when_available(mock_retriever):
    schema = GraphSchema(
        labels=["Person"],
        has_vector_index=True,
        vector_index_name="my_index",
        node_count=10,
    )
    mock_retriever._schema = schema

    # Mock embedding
    mock_retriever._embed = MagicMock(return_value=[0.1] * 1536)

    # Mock vector search to return candidate IDs
    mock_retriever._vector_search = MagicMock(return_value=["elem:1", "elem:2"])

    # Mock neighbourhood fetch
    mock_retriever._fetch_neighbourhood = MagicMock(
        return_value={
            "nodes": [{"name": "Alice"}, {"name": "Bob"}],
            "edges": [{"start": "elem:1", "end": "elem:2", "type": "KNOWS"}],
        }
    )

    result = mock_retriever.retrieve("test")
    assert result.strategy == "vector"
    assert result.nodes_count == 2
    assert result.edges_count == 1


def test_retrieve_vector_fallback_to_cypher(mock_retriever):
    schema = GraphSchema(
        labels=["Person"],
        has_vector_index=True,
        vector_index_name="my_index",
        node_count=10,
    )
    mock_retriever._schema = schema

    # Mock vector path returning empty
    mock_retriever._embed = MagicMock(return_value=[0.1] * 1536)
    mock_retriever._vector_search = MagicMock(return_value=[])

    # Mock cypher fallback
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="MATCH (n) RETURN n.name LIMIT 10")

    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.keys.return_value = ["name"]
    mock_record.__getitem__ = lambda self, key: "Alice"
    mock_session.execute_read.return_value = [mock_record]
    mock_retriever._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_retriever._driver.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        result = mock_retriever.retrieve("test")

    # Should fall back to cypher when vector returns nothing
    assert result.strategy == "cypher"


# ---------------------------------------------------------------------------
# _retrieve_cypher (empty generation)
# ---------------------------------------------------------------------------


def test_retrieve_cypher_empty_generation(mock_retriever):
    schema = GraphSchema(labels=["X"], node_count=1)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="")

    with patch("gibsgraph.agent._make_llm", return_value=mock_llm):
        result = mock_retriever._retrieve_cypher("test", schema=schema)

    assert "Could not generate" in result.context
    assert result.strategy == "cypher"
