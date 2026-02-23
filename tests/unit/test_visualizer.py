"""Unit tests for GraphVisualizer."""

from unittest.mock import MagicMock, patch

import pytest

from gibsgraph.config import Settings
from gibsgraph.tools.visualizer import GraphVisualizer


@pytest.fixture
def settings() -> Settings:
    return Settings(
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_PASSWORD="testpassword",
    )


@pytest.fixture
def viz(settings: Settings) -> GraphVisualizer:
    return GraphVisualizer(settings=settings)


@pytest.fixture
def sample_subgraph() -> dict:
    return {
        "nodes": [
            {"id": "n1", "name": "Alice"},
            {"id": "n2", "name": "Bob"},
            {"id": "n3", "name": "Charlie"},
        ],
        "edges": [
            {"start": "n1", "end": "n2", "type": "KNOWS"},
            {"start": "n2", "end": "n3", "type": "WORKS_WITH"},
        ],
    }


# --- to_mermaid ---


def test_to_mermaid_basic(viz, sample_subgraph):
    result = viz.to_mermaid(sample_subgraph)
    assert result.startswith("graph LR")
    assert 'n1["Alice"]' in result
    assert 'n2["Bob"]' in result
    assert "n1 -->|KNOWS| n2" in result
    assert "n2 -->|WORKS_WITH| n3" in result


def test_to_mermaid_empty(viz):
    result = viz.to_mermaid({"nodes": [], "edges": []})
    assert result == "graph LR"


def test_to_mermaid_max_nodes(viz):
    nodes = [{"id": f"n{i}", "name": f"Node{i}"} for i in range(30)]
    result = viz.to_mermaid({"nodes": nodes, "edges": []}, max_nodes=5)
    # Should only have 5 node lines + the header
    lines = [ln for ln in result.split("\n") if ln.strip().startswith("n")]
    assert len(lines) == 5


def test_to_mermaid_special_chars_in_id(viz):
    subgraph = {
        "nodes": [{"id": "node-with-dashes!", "name": "Test"}],
        "edges": [],
    }
    result = viz.to_mermaid(subgraph)
    # Special chars should be replaced with underscores
    assert "node_with_dashes_" in result


def test_to_mermaid_quotes_in_label(viz):
    subgraph = {
        "nodes": [{"id": "n1", "name": 'Say "hello"'}],
        "edges": [],
    }
    result = viz.to_mermaid(subgraph)
    # Double quotes in label should be escaped to single quotes
    assert "Say 'hello'" in result


def test_to_mermaid_skips_edges_for_missing_nodes(viz):
    subgraph = {
        "nodes": [{"id": "n1", "name": "Alice"}],
        "edges": [{"start": "n1", "end": "n99", "type": "KNOWS"}],
    }
    result = viz.to_mermaid(subgraph)
    assert "KNOWS" not in result


def test_to_mermaid_uses_name_as_fallback_id(viz):
    subgraph = {
        "nodes": [{"name": "OnlyName"}],
        "edges": [],
    }
    result = viz.to_mermaid(subgraph)
    assert "OnlyName" in result


# --- bloom_url ---


def test_bloom_url_basic(viz, sample_subgraph):
    url = viz.bloom_url(sample_subgraph)
    assert url.startswith("https://bloom.neo4j.io/index.html#search=")
    assert "n1" in url
    assert "n2" in url
    assert "n3" in url


def test_bloom_url_empty_nodes(viz):
    url = viz.bloom_url({"nodes": []})
    assert url == ""


def test_bloom_url_nodes_without_ids(viz):
    url = viz.bloom_url({"nodes": [{"name": "NoId"}]})
    assert url == ""


def test_bloom_url_uses_elementid(viz):
    subgraph = {"nodes": [{"id": "4:abc:123"}]}
    url = viz.bloom_url(subgraph)
    assert "elementId" in url
    assert "4%3Aabc%3A123" in url or "4:abc:123" in url


# --- to_html_pyvis ---


def test_to_html_pyvis_without_pyvis(viz, sample_subgraph):
    """Should raise ImportError when pyvis is not installed."""
    with patch.dict("sys.modules", {"pyvis": None, "pyvis.network": None}):
        with pytest.raises(ImportError, match="pyvis"):
            viz.to_html_pyvis(sample_subgraph)


def test_to_html_pyvis_with_mock(viz, sample_subgraph):
    """Test pyvis integration with mocked Network."""
    mock_network_cls = MagicMock()
    mock_net = MagicMock()
    mock_network_cls.return_value = mock_net
    mock_net.generate_html.return_value = "<html>graph</html>"

    with patch.dict("sys.modules", {"pyvis": MagicMock(), "pyvis.network": MagicMock()}):
        with patch("gibsgraph.tools.visualizer.Network", mock_network_cls, create=True):
            # We need to reimport to pick up the mock — just call directly
            from pyvis.network import Network as MockNet

            MockNet.return_value = mock_net

            # The import inside to_html_pyvis will use the mocked module
            result = viz.to_html_pyvis(sample_subgraph)
            assert "html" in result.lower() or mock_net.generate_html.called
