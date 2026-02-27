"""Unit tests for PCST subgraph pruner.

Tests use realistic graph topologies (clusters, hubs, parallel edges) and
verify *which* nodes/edges survive pruning — not just counts.
"""

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from gibsgraph.retrieval.pcst_pruner import (
    _cosine_similarity,
    _pcst_available,
    compute_node_prizes,
    node_text,
    pcst_prune,
)

# ---------------------------------------------------------------------------
# Helpers — build realistic subgraphs
# ---------------------------------------------------------------------------


def _make_node(nid, name, labels=None, **props):
    """Build a node dict matching _fetch_neighbourhood output."""
    node = {"_id": nid, "_labels": labels or ["Entity"], "name": name}
    node.update(props)
    return node


def _make_edge(start, end, rel_type="RELATED", **props):
    """Build an edge dict matching _fetch_neighbourhood output."""
    return {"start": start, "end": end, "type": rel_type, "props": props}


# Two-cluster graph: a "relevant" cluster and a "noise" cluster
# connected by a single bridge edge.
#
#   [Apple]---ACQUIRED--->[Beats]---PRODUCED--->[Headphones]
#       \                                          (relevant cluster)
#        BRIDGE
#       /
#   [Zebra]---GRAZES--->[Savanna]---CONTAINS--->[Grass]
#                                                   (noise cluster)
#
RELEVANT_IDS = {"apple", "beats", "headphones"}
NOISE_IDS = {"zebra", "savanna", "grass"}

CLUSTER_NODES = [
    _make_node("apple", "Apple Inc.", ["Company"], description="Technology company"),
    _make_node("beats", "Beats Electronics", ["Company"], description="Audio products"),
    _make_node(
        "headphones", "Studio Headphones", ["Product"], description="Noise-cancelling headphones"
    ),
    _make_node("zebra", "Zebra", ["Animal"]),
    _make_node("savanna", "African Savanna", ["Location"]),
    _make_node("grass", "Grass", ["Plant"]),
]

CLUSTER_EDGES = [
    _make_edge("apple", "beats", "ACQUIRED"),
    _make_edge("beats", "headphones", "PRODUCED"),
    _make_edge("apple", "zebra", "BRIDGE"),
    _make_edge("zebra", "savanna", "GRAZES"),
    _make_edge("savanna", "grass", "CONTAINS"),
]

# Embeddings: relevant cluster points toward [1, 0], noise toward [0, 1]
CLUSTER_EMBEDDINGS = [
    [0.95, 0.05],  # apple — very relevant
    [0.90, 0.10],  # beats — very relevant
    [0.85, 0.15],  # headphones — relevant
    [0.05, 0.95],  # zebra — noise
    [0.10, 0.90],  # savanna — noise
    [0.08, 0.92],  # grass — noise
]

QUERY_EMBEDDING = [1.0, 0.0]  # "Tell me about Apple's acquisitions"


def _mock_pcst_context():
    """Context manager that injects a fake pcst_fast returning highest-prize nodes."""
    fake_pcst = types.ModuleType("pcst_fast")

    def _solver(edges, prizes, costs, root, num_clusters, pruning, verbosity):
        """Mimic real PCST: select nodes with prize above median, keep edges between them."""
        threshold = float(np.median(prizes[prizes > 0])) if np.any(prizes > 0) else 0
        sel_nodes = [i for i in range(len(prizes)) if prizes[i] >= threshold]
        sel_node_set = set(sel_nodes)
        sel_edges = [
            i
            for i in range(len(edges))
            if int(edges[i][0]) in sel_node_set and int(edges[i][1]) in sel_node_set
        ]
        return np.array(sel_nodes), np.array(sel_edges)

    fake_pcst.pcst_fast = _solver  # type: ignore[attr-defined]
    return (
        patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=True),
        patch.dict(sys.modules, {"pcst_fast": fake_pcst}),
    )


# ---------------------------------------------------------------------------
# node_text — realistic Neo4j node shapes
# ---------------------------------------------------------------------------


class TestNodeText:
    """Tests for node_text with realistic Neo4j node shapes."""

    def test_labels_and_name(self):
        node = _make_node("4:abc:1", "Apple Inc.", ["Company"])
        text = node_text(node)
        assert "Company" in text
        assert "Apple Inc." in text

    def test_title_and_description(self):
        node = _make_node(
            "4:abc:2",
            "Beats",
            ["Company"],
            title="CEO",
            description="Chief Executive Officer",
        )
        text = node_text(node)
        assert "CEO" in text
        assert "Chief Executive Officer" in text

    def test_includes_short_string_props(self):
        node = _make_node("4:abc:3", "Bob", ["Person"], status="active", role="admin")
        text = node_text(node)
        assert "active" in text
        assert "admin" in text

    def test_skips_long_strings(self):
        node = _make_node("4:abc:4", "Doc", ["Document"], body="x" * 300)
        text = node_text(node)
        assert "Doc" in text
        assert "x" * 300 not in text

    def test_empty_node(self):
        assert node_text({}) == ""

    def test_skips_internal_keys(self):
        node = {
            "_id": "4:abc:5",
            "_labels": ["Movie"],
            "embedding": [0.1] * 100,
            "vector": [0.2] * 100,
            "name": "Inception",
        }
        text = node_text(node)
        assert "4:abc:5" not in text
        assert "0.1" not in text
        assert "Movie" in text
        assert "Inception" in text

    def test_non_string_props_ignored(self):
        node = _make_node("4:abc:6", "Alice", ["Person"], age=30, scores=[1, 2, 3])
        text = node_text(node)
        assert "Alice" in text
        assert "30" not in text

    def test_multiple_labels(self):
        node = {"_labels": ["Person", "Employee"], "name": "Bob"}
        text = node_text(node)
        assert "Person" in text
        assert "Employee" in text


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_both_zero(self):
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_high_dimensional(self):
        """Real embeddings are 1536-dim, not 2-dim."""
        rng = np.random.default_rng(42)
        a = rng.normal(size=1536).tolist()
        b = rng.normal(size=1536).tolist()
        sim = _cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0
        # Random high-dim vectors should be near-orthogonal
        assert abs(sim) < 0.15


# ---------------------------------------------------------------------------
# compute_node_prizes
# ---------------------------------------------------------------------------


class TestComputeNodePrizes:
    def test_correct_length(self):
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        prizes = compute_node_prizes(embeddings, [1.0, 0.0])
        assert len(prizes) == 3

    def test_relevant_node_gets_higher_prize(self):
        embeddings = CLUSTER_EMBEDDINGS
        prizes = compute_node_prizes(embeddings, QUERY_EMBEDDING)
        # Apple (idx 0) should beat Zebra (idx 3)
        assert prizes[0] > prizes[3]
        # All relevant nodes should beat all noise nodes
        for rel_idx in (0, 1, 2):
            for noise_idx in (3, 4, 5):
                assert prizes[rel_idx] > prizes[noise_idx]

    def test_negative_similarity_clamped_to_zero(self):
        prizes = compute_node_prizes([[-1.0, 0.0]], [1.0, 0.0])
        assert prizes[0] == 0.0

    def test_all_zero_embeddings(self):
        """Nodes with zero embeddings should get zero prize."""
        prizes = compute_node_prizes([[0.0, 0.0], [0.0, 0.0]], [1.0, 0.0])
        assert all(p == 0.0 for p in prizes)

    def test_prizes_bounded_zero_to_one(self):
        rng = np.random.default_rng(99)
        embeddings = [rng.normal(size=10).tolist() for _ in range(50)]
        query = rng.normal(size=10).tolist()
        prizes = compute_node_prizes(embeddings, query)
        assert all(0.0 <= p <= 1.0 for p in prizes)


# ---------------------------------------------------------------------------
# pcst_prune — short-circuits
# ---------------------------------------------------------------------------


class TestPcstPruneShortCircuits:
    def test_returns_same_object_when_under_max(self):
        subgraph = {"nodes": [_make_node("1", "A"), _make_node("2", "B")], "edges": []}
        result = pcst_prune(subgraph, [[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0], max_nodes=5)
        assert result is subgraph

    def test_returns_same_object_when_exactly_max(self):
        nodes = [_make_node(str(i), f"N{i}") for i in range(10)]
        subgraph = {"nodes": nodes, "edges": []}
        result = pcst_prune(subgraph, [[1.0, 0.0]] * 10, [1.0, 0.0], max_nodes=10)
        assert result is subgraph

    def test_empty_subgraph(self):
        subgraph = {"nodes": [], "edges": []}
        result = pcst_prune(subgraph, [], [1.0, 0.0], max_nodes=10)
        assert result is subgraph

    def test_graceful_when_pcst_not_installed(self):
        """If pcst-fast is missing, return subgraph unchanged (no crash)."""
        nodes = [_make_node(str(i), f"N{i}") for i in range(25)]
        subgraph = {"nodes": nodes, "edges": []}
        embeddings = [[1.0, 0.0]] * 25
        with patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=False):
            result = pcst_prune(subgraph, embeddings, [1.0, 0.0], max_nodes=10)
        assert result is subgraph


# ---------------------------------------------------------------------------
# pcst_prune — real pruning behavior
# ---------------------------------------------------------------------------


class TestPcstPruneBehavior:
    """Tests that verify WHICH nodes survive, not just counts."""

    def test_relevant_cluster_survives_noise_dropped(self):
        """The Apple/Beats/Headphones cluster should survive; Zebra/Savanna/Grass should not."""
        subgraph = {"nodes": CLUSTER_NODES, "edges": CLUSTER_EDGES}
        ctx1, ctx2 = _mock_pcst_context()
        with ctx1, ctx2:
            result = pcst_prune(subgraph, CLUSTER_EMBEDDINGS, QUERY_EMBEDDING, max_nodes=3)

        survived_ids = {n["_id"] for n in result["nodes"]}
        # Relevant nodes should be present
        assert "apple" in survived_ids
        assert "beats" in survived_ids
        # At least the noise cluster should be reduced
        assert len(survived_ids & NOISE_IDS) < len(NOISE_IDS)

    def test_pruned_edges_only_connect_surviving_nodes(self):
        """Every edge in the result must connect two nodes that survived."""
        subgraph = {"nodes": CLUSTER_NODES, "edges": CLUSTER_EDGES}
        ctx1, ctx2 = _mock_pcst_context()
        with ctx1, ctx2:
            result = pcst_prune(subgraph, CLUSTER_EMBEDDINGS, QUERY_EMBEDDING, max_nodes=3)

        survived_ids = {n["_id"] for n in result["nodes"]}
        for edge in result["edges"]:
            assert edge["start"] in survived_ids, f"Dangling start: {edge}"
            assert edge["end"] in survived_ids, f"Dangling end: {edge}"

    def test_edge_types_preserved(self):
        """Pruning should not alter edge relationship types or props."""
        subgraph = {"nodes": CLUSTER_NODES, "edges": CLUSTER_EDGES}
        ctx1, ctx2 = _mock_pcst_context()
        with ctx1, ctx2:
            result = pcst_prune(subgraph, CLUSTER_EMBEDDINGS, QUERY_EMBEDDING, max_nodes=3)

        for edge in result["edges"]:
            assert "type" in edge
            assert "start" in edge
            assert "end" in edge

    def test_parallel_edges_both_survive(self):
        """Two different relationship types between the same nodes should both be kept."""
        nodes = [
            _make_node("a", "Alice", ["Person"]),
            _make_node("b", "Acme Corp", ["Company"]),
            # padding noise to exceed max_nodes
            *[_make_node(f"noise_{i}", f"Noise{i}") for i in range(20)],
        ]
        edges = [
            _make_edge("a", "b", "WORKS_AT"),
            _make_edge("a", "b", "FOUNDED"),
            # noise edges
            *[_make_edge(f"noise_{i}", f"noise_{i + 1}", "LINK") for i in range(19)],
        ]
        subgraph = {"nodes": nodes, "edges": edges}
        # Alice/Acme highly relevant, noise not
        embeddings = [[0.95, 0.05], [0.90, 0.10]] + [[0.05, 0.95]] * 20
        query = [1.0, 0.0]

        ctx1, ctx2 = _mock_pcst_context()
        with ctx1, ctx2:
            result = pcst_prune(subgraph, embeddings, query, max_nodes=5)

        survived_ids = {n["_id"] for n in result["nodes"]}
        assert "a" in survived_ids
        assert "b" in survived_ids

        # Both WORKS_AT and FOUNDED should survive if both nodes survived
        ab_edges = [e for e in result["edges"] if e["start"] == "a" and e["end"] == "b"]
        ab_types = {e["type"] for e in ab_edges}
        assert "WORKS_AT" in ab_types, f"WORKS_AT lost, got: {ab_types}"
        assert "FOUNDED" in ab_types, f"FOUNDED lost, got: {ab_types}"

    def test_self_referencing_edge(self):
        """A node with a self-loop edge should not crash the pruner."""
        nodes = [
            _make_node("a", "Recursive Thing"),
            *[_make_node(f"n{i}", f"N{i}") for i in range(24)],
        ]
        edges = [
            _make_edge("a", "a", "SELF_REF"),  # self-loop
            *[_make_edge(f"n{i}", f"n{i + 1}", "LINK") for i in range(23)],
        ]
        subgraph = {"nodes": nodes, "edges": edges}
        embeddings = [[0.9, 0.1]] + [[0.1, 0.9]] * 24
        query = [1.0, 0.0]

        ctx1, ctx2 = _mock_pcst_context()
        with ctx1, ctx2:
            result = pcst_prune(subgraph, embeddings, query, max_nodes=5)

        # Should not crash, and "a" should survive (highest prize)
        survived_ids = {n["_id"] for n in result["nodes"]}
        assert "a" in survived_ids

    def test_no_valid_edges_keeps_highest_prize_nodes(self):
        """When all edges are dangling, fallback to top-K by prize with correct nodes."""
        nodes = [_make_node(str(i), f"N{i}") for i in range(25)]
        # Edges that don't match any node IDs
        edges = [_make_edge("nonexistent_a", "nonexistent_b", "BROKEN")]
        subgraph = {"nodes": nodes, "edges": edges}
        # Node 0 gets highest prize, rest are noise
        embeddings = [[0.99, 0.01]] + [[0.01, 0.99]] * 24
        query = [1.0, 0.0]

        fake_pcst = types.ModuleType("pcst_fast")
        with (
            patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=True),
            patch.dict(sys.modules, {"pcst_fast": fake_pcst}),
        ):
            result = pcst_prune(subgraph, embeddings, query, max_nodes=10)

        assert len(result["nodes"]) == 10
        assert result["edges"] == []
        # The highest-prize node (idx 0) must be in the result
        survived_ids = {n["_id"] for n in result["nodes"]}
        assert "0" in survived_ids

    def test_pcst_receives_correct_arguments(self):
        """Verify pcst_fast gets the right edge array, prizes, and costs."""
        nodes = [
            _make_node("a", "Apple", ["Company"]),
            _make_node("b", "Beats", ["Company"]),
            _make_node("c", "Noise", ["Junk"]),
            *[_make_node(f"p{i}", f"Pad{i}") for i in range(20)],
        ]
        edges = [
            _make_edge("a", "b", "ACQUIRED"),
            _make_edge("b", "c", "RELATED"),
        ]
        subgraph = {"nodes": nodes, "edges": edges}
        embeddings = [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]] + [[0.05, 0.95]] * 20
        query = [1.0, 0.0]

        captured_args = {}

        fake_pcst = types.ModuleType("pcst_fast")

        def _capture_solver(edges_arr, prizes_arr, costs_arr, root, nc, pruning, verb):
            captured_args["edges"] = edges_arr.copy()
            captured_args["prizes"] = prizes_arr.copy()
            captured_args["costs"] = costs_arr.copy()
            captured_args["root"] = root
            captured_args["pruning"] = pruning
            # Return all nodes/edges to not interfere with assertions
            return np.arange(len(prizes_arr)), np.arange(len(edges_arr))

        fake_pcst.pcst_fast = _capture_solver  # type: ignore[attr-defined]

        with (
            patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=True),
            patch.dict(sys.modules, {"pcst_fast": fake_pcst}),
        ):
            pcst_prune(subgraph, embeddings, query, max_nodes=5)

        # Edge array should map our 2 edges to node indices
        assert captured_args["edges"].shape == (2, 2)
        # Node 'a' is idx 0, 'b' is idx 1, 'c' is idx 2
        assert list(captured_args["edges"][0]) == [0, 1]  # a→b
        assert list(captured_args["edges"][1]) == [1, 2]  # b→c

        # Prizes should have len == num nodes
        assert len(captured_args["prizes"]) == len(nodes)
        # Apple (idx 0) should have a higher prize than padding nodes
        assert captured_args["prizes"][0] > captured_args["prizes"][-1]

        # Costs should all be edge_cost (default 0.1)
        assert all(c == pytest.approx(0.1) for c in captured_args["costs"])

        # Unrooted
        assert captured_args["root"] == -1
        assert captured_args["pruning"] == "gw"

    def test_prize_boosting_actually_boosts_top_k(self):
        """The top-K prizes should be amplified compared to raw cosine similarity."""
        nodes = [_make_node(str(i), f"N{i}") for i in range(30)]
        # Spread embeddings: first 10 very relevant, rest noise
        embeddings = [[0.9, 0.1]] * 10 + [[0.1, 0.9]] * 20
        query = [1.0, 0.0]

        raw_prizes = compute_node_prizes(embeddings, query)

        captured_prizes = {}

        fake_pcst = types.ModuleType("pcst_fast")

        def _capture(edges_arr, prizes_arr, costs_arr, root, nc, pruning, verb):
            captured_prizes["boosted"] = prizes_arr.copy()
            return np.arange(len(prizes_arr)), np.arange(len(edges_arr))

        fake_pcst.pcst_fast = _capture  # type: ignore[attr-defined]

        edges = [_make_edge(str(i), str(i + 1), "LINK") for i in range(29)]
        subgraph = {"nodes": nodes, "edges": edges}

        with (
            patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=True),
            patch.dict(sys.modules, {"pcst_fast": fake_pcst}),
        ):
            pcst_prune(subgraph, embeddings, query, max_nodes=10)

        boosted = captured_prizes["boosted"]
        # Top-10 nodes (idx 0-9) should have boosted prizes > raw prizes
        for i in range(10):
            assert boosted[i] > raw_prizes[i], (
                f"Node {i} not boosted: {boosted[i]} vs {raw_prizes[i]}"
            )
        # Noise nodes should NOT be boosted
        for i in range(10, 30):
            assert boosted[i] == pytest.approx(raw_prizes[i]), f"Noise node {i} was boosted"

    def test_single_embedding_returns_single_prize(self):
        """compute_node_prizes returns one prize per embedding."""
        prizes = compute_node_prizes([[0.5, 0.5]], [1.0, 0.0])
        assert len(prizes) == 1
        assert 0.0 <= prizes[0] <= 1.0

    def test_hub_node_with_many_edges(self):
        """A hub connected to many spokes — tests that pcst_fast gets the right topology."""
        hub = _make_node("hub", "Central Hub", ["Hub"])
        spokes = [_make_node(f"s{i}", f"Spoke{i}") for i in range(24)]
        nodes = [hub, *spokes]
        edges = [_make_edge("hub", f"s{i}", "CONNECTED") for i in range(24)]
        subgraph = {"nodes": nodes, "edges": edges}
        # Hub is very relevant, half the spokes somewhat relevant, half noise
        embeddings = [[0.99, 0.01]] + [[0.7, 0.3]] * 12 + [[0.05, 0.95]] * 12
        query = [1.0, 0.0]

        captured = {}

        fake_pcst = types.ModuleType("pcst_fast")

        def _capture(edges_arr, prizes_arr, costs_arr, root, nc, pruning, verb):
            captured["edges"] = edges_arr.copy()
            # Select hub + first 9 relevant spokes
            sel_nodes = np.array([0, *list(range(1, 10))])
            sel_edges = np.array(list(range(9)))  # first 9 edges (hub→s0..s8)
            return sel_nodes, sel_edges

        fake_pcst.pcst_fast = _capture  # type: ignore[attr-defined]

        with (
            patch("gibsgraph.retrieval.pcst_pruner._pcst_available", return_value=True),
            patch.dict(sys.modules, {"pcst_fast": fake_pcst}),
        ):
            result = pcst_prune(subgraph, embeddings, query, max_nodes=10)

        # All edges in the pcst input should be star pattern from hub (idx 0)
        edge_arr = captured["edges"]
        assert all(int(edge_arr[i][0]) == 0 for i in range(len(edge_arr)))
        survived_ids = {n["_id"] for n in result["nodes"]}
        assert "hub" in survived_ids
        assert len(result["nodes"]) == 10


# ---------------------------------------------------------------------------
# _pcst_available
# ---------------------------------------------------------------------------


def test_pcst_available_returns_bool():
    result = _pcst_available()
    assert isinstance(result, bool)


def test_pcst_available_false_when_not_installed():
    with patch("importlib.util.find_spec", return_value=None):
        assert _pcst_available() is False
