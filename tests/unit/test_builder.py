"""Unit tests for KGBuilder."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.config import Settings
from gibsgraph.kg_builder.builder import IngestResult, KGBuilder


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
def test_ingest_not_implemented(mock_gdb, settings):
    settings_writable = Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
        NEO4J_READ_ONLY=False,
    )
    builder = KGBuilder(settings=settings_writable)
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        builder.ingest("some text")


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
