"""Unit tests for KGBuilder."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.config import Settings
from gibsgraph.kg_builder.builder import (
    IngestResult,
    KGBuilder,
    _make_kg_embedder,
    _make_kg_llm,
)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
    )


@pytest.fixture
def read_only_settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        NEO4J_READ_ONLY=True,
    )


@pytest.fixture
def writable_settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        NEO4J_READ_ONLY=False,
        OPENAI_API_KEY="sk-test-key",
    )


# --- IngestResult ---


def test_ingest_result_fields():
    r = IngestResult(nodes_created=10, relationships_created=5, chunks_processed=3)
    assert r.nodes_created == 10
    assert r.relationships_created == 5
    assert r.chunks_processed == 3


# --- KGBuilder construction ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_builder_init(mock_gdb, settings):
    builder = KGBuilder(settings=settings)
    mock_gdb.driver.assert_called_once()
    builder.close()


# --- ingest ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_read_only_raises(mock_gdb, read_only_settings):
    builder = KGBuilder(settings=read_only_settings)
    with pytest.raises(RuntimeError, match="read-only mode"):
        builder.ingest("some text")


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_empty_text_returns_zero(mock_gdb, writable_settings):
    builder = KGBuilder(settings=writable_settings)
    result = builder.ingest("")
    assert result.nodes_created == 0
    assert result.relationships_created == 0
    assert result.chunks_processed == 0


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_whitespace_only_returns_zero(mock_gdb, writable_settings):
    builder = KGBuilder(settings=writable_settings)
    result = builder.ingest("   \n\t  ")
    assert result.nodes_created == 0


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_calls_pipeline(mock_gdb, writable_settings):
    """Full ingest path: mock pipeline + count queries, verify IngestResult."""
    builder = KGBuilder(settings=writable_settings)

    mock_pipeline = MagicMock()

    # Mock _count_graph_entities to return before/after counts
    with (
        patch.object(builder, "_build_pipeline", return_value=mock_pipeline),
        patch.object(builder, "_count_graph_entities", side_effect=[(0, 0), (5, 3)]),
        patch("gibsgraph.kg_builder.builder.asyncio") as mock_asyncio,
        patch("gibsgraph.kg_builder.builder._make_kg_llm") as mock_llm,
        patch("gibsgraph.kg_builder.builder._make_kg_embedder") as mock_emb,
    ):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_asyncio.run.return_value = None

        result = builder.ingest("Apple acquired Beats for $3B.")

    assert result.nodes_created == 5
    assert result.relationships_created == 3
    assert result.chunks_processed == 1
    mock_asyncio.run.assert_called_once()


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_event_loop_error(mock_gdb, writable_settings):
    """Running ingest inside an existing event loop gives a friendly error."""
    builder = KGBuilder(settings=writable_settings)

    with (
        patch.object(builder, "_build_pipeline", return_value=MagicMock()),
        patch.object(builder, "_count_graph_entities", return_value=(0, 0)),
        patch(
            "gibsgraph.kg_builder.builder.asyncio.run",
            side_effect=RuntimeError("asyncio.run() cannot be called from a running event loop"),
        ),
        patch("gibsgraph.kg_builder.builder._make_kg_llm") as mock_llm,
        patch("gibsgraph.kg_builder.builder._make_kg_embedder") as mock_emb,
    ):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()

        with pytest.raises(RuntimeError, match="nest_asyncio"):
            builder.ingest("some text")


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_ingest_chunks_long_text(mock_gdb, writable_settings):
    """Chunk count for long text uses ceiling division by 4000."""
    builder = KGBuilder(settings=writable_settings)

    with (
        patch.object(builder, "_build_pipeline", return_value=MagicMock()),
        patch.object(builder, "_count_graph_entities", side_effect=[(0, 0), (10, 5)]),
        patch("gibsgraph.kg_builder.builder.asyncio") as mock_asyncio,
        patch("gibsgraph.kg_builder.builder._make_kg_llm") as mock_llm,
        patch("gibsgraph.kg_builder.builder._make_kg_embedder") as mock_emb,
    ):
        mock_llm.return_value = MagicMock()
        mock_emb.return_value = MagicMock()
        mock_asyncio.run.return_value = None

        result = builder.ingest("x" * 10000)

    # ceil(10000 / 4000) = 3
    assert result.chunks_processed == 3


# --- _make_kg_llm ---


def test_make_kg_llm_openai():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
        LLM_MODEL="gpt-4o",
        OPENAI_API_KEY="sk-test",
    )
    with patch("gibsgraph.kg_builder.builder.provider_for_model") as mock_pfm:
        mock_pfm.return_value = MagicMock(name="openai", base_url=None)
        mock_pfm.return_value.name = "openai"
        with patch("neo4j_graphrag.llm.OpenAILLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _make_kg_llm(s)
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value


def test_make_kg_llm_anthropic():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
        LLM_MODEL="claude-3-haiku-20240307",
        ANTHROPIC_API_KEY="sk-ant-test",
    )
    with patch("gibsgraph.kg_builder.builder.provider_for_model") as mock_pfm:
        mock_pfm.return_value = MagicMock()
        mock_pfm.return_value.name = "anthropic"
        with patch("neo4j_graphrag.llm.AnthropicLLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _make_kg_llm(s)
            mock_cls.assert_called_once()
            assert result is mock_cls.return_value


def test_make_kg_llm_xai():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
        LLM_MODEL="grok-3",
        XAI_API_KEY="xai-test",
    )
    with patch("gibsgraph.kg_builder.builder.provider_for_model") as mock_pfm:
        provider = MagicMock()
        provider.name = "xai"
        provider.base_url = "https://api.x.ai/v1"
        provider.env_key = "XAI_API_KEY"
        mock_pfm.return_value = provider
        with patch("neo4j_graphrag.llm.OpenAILLM") as mock_cls:
            mock_cls.return_value = MagicMock()
            result = _make_kg_llm(s)
            mock_cls.assert_called_once()
            # Verify base_url was passed
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["base_url"] == "https://api.x.ai/v1"
            assert result is mock_cls.return_value


def test_make_kg_llm_missing_key():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
        LLM_MODEL="gpt-4o",
    )
    s.openai_api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        _make_kg_llm(s)


# --- _make_kg_embedder ---


def test_make_kg_embedder():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
        OPENAI_API_KEY="sk-test",
    )
    with patch("neo4j_graphrag.embeddings.OpenAIEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        result = _make_kg_embedder(s)
        mock_cls.assert_called_once()
        assert result is mock_cls.return_value


def test_make_kg_embedder_no_key():
    s = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpw",
    )
    s.openai_api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        _make_kg_embedder(s)


# --- _count_graph_entities ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_count_graph_entities(mock_gdb, writable_settings):
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # First call returns node count, second returns rel count
    mock_node_result = MagicMock()
    mock_node_result.single.return_value = {"c": 42}
    mock_rel_result = MagicMock()
    mock_rel_result.single.return_value = {"c": 17}
    mock_session.run.side_effect = [mock_node_result, mock_rel_result]

    builder = KGBuilder(settings=writable_settings)
    nodes, rels = builder._count_graph_entities()

    assert nodes == 42
    assert rels == 17


# --- clear_graph ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_clear_graph_read_only_raises(mock_gdb, read_only_settings):
    builder = KGBuilder(settings=read_only_settings)
    with pytest.raises(RuntimeError, match="read-only mode"):
        builder.clear_graph()


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_clear_graph_executes(mock_gdb):
    settings = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        NEO4J_READ_ONLY=False,
    )
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    builder = KGBuilder(settings=settings)
    builder.clear_graph()

    mock_driver.session.return_value.__enter__.return_value.run.assert_called_once_with(
        "MATCH (n) DETACH DELETE n"
    )


# --- context manager ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_context_manager(mock_gdb, settings):
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    with KGBuilder(settings=settings) as builder:
        assert builder is not None

    mock_driver.close.assert_called_once()


# --- close ---


@patch("gibsgraph.kg_builder.builder.GraphDatabase")
def test_close(mock_gdb, settings):
    mock_driver = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    builder = KGBuilder(settings=settings)
    builder.close()
    mock_driver.close.assert_called_once()
