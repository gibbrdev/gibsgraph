"""PCST subgraph pruning — keeps the most query-relevant connected subset.

Uses the Prize-Collecting Steiner Tree algorithm (via ``pcst-fast``) to prune
a neighbourhood subgraph down to the most relevant nodes for a given query.
Standalone module — no Neo4j or LLM dependencies.  Receives pre-computed
embeddings and operates purely on dicts.

Install the optional dependency::

    pip install "gibsgraph[gnn]"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Internal keys that should not contribute to node text
_INTERNAL_KEYS = frozenset({"_id", "_labels", "embedding", "vector"})

# Maximum length for a string property to be included in node text
_MAX_PROP_LEN = 200


def _pcst_available() -> bool:
    """Return True if the ``pcst_fast`` package is importable."""
    import importlib.util

    return importlib.util.find_spec("pcst_fast") is not None


def node_text(node: dict[str, Any]) -> str:
    """Extract a textual representation of a node for embedding.

    Combines labels, name/title, description, and short string properties
    into a single string suitable for embedding comparison.
    """
    parts: list[str] = []

    # Labels
    labels = node.get("_labels", [])
    if labels:
        parts.append(" ".join(str(lbl) for lbl in labels))

    # Priority fields
    for key in ("name", "title"):
        val = node.get(key)
        if isinstance(val, str) and val:
            parts.append(val)

    # Description gets its own slot
    desc = node.get("description")
    if isinstance(desc, str) and desc:
        parts.append(desc)

    # Other short string properties
    for key, val in node.items():
        if key in _INTERNAL_KEYS or key in ("name", "title", "description"):
            continue
        if isinstance(val, str) and 0 < len(val) <= _MAX_PROP_LEN:
            parts.append(val)

    return " ".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.  Returns 0.0 for zero vectors."""
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def compute_node_prizes(
    node_embeddings: list[list[float]],
    query_embedding: list[float],
) -> list[float]:
    """Compute a relevance prize per node embedding (cosine similarity, clamped to [0, 1])."""
    prizes: list[float] = []
    for emb in node_embeddings:
        sim = _cosine_similarity(emb, query_embedding)
        prizes.append(max(0.0, min(1.0, sim)))
    return prizes


def pcst_prune(
    subgraph: dict[str, Any],
    node_embeddings: list[list[float]],
    query_embedding: list[float],
    *,
    max_nodes: int = 20,
    edge_cost: float = 0.1,
) -> dict[str, Any]:
    """Prune *subgraph* to the most query-relevant connected subset via PCST.

    Parameters
    ----------
    subgraph:
        Dict with ``"nodes"`` and ``"edges"`` lists (as returned by the
        retriever's ``_fetch_neighbourhood``).
    node_embeddings:
        One embedding per node, same order as ``subgraph["nodes"]``.
    query_embedding:
        The query embedding to score relevance against.
    max_nodes:
        Target maximum number of nodes to keep.
    edge_cost:
        Base cost assigned to each edge (higher = more aggressive pruning).

    Returns
    -------
    dict
        Pruned subgraph with the same structure as *subgraph*.
    """
    nodes: list[dict[str, Any]] = subgraph.get("nodes", [])
    edges: list[dict[str, Any]] = subgraph.get("edges", [])

    # Short-circuit: nothing to prune
    if len(nodes) <= max_nodes:
        return subgraph

    # Check dependency
    if not _pcst_available():
        log.warning("pcst_pruner.pcst_fast_not_installed", hint="pip install 'gibsgraph[gnn]'")
        return subgraph

    import pcst_fast  # type: ignore[import-not-found]

    # Build node_id → index mapping (nodes must have "_id" from _fetch_neighbourhood)
    node_id_to_idx: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        nid = node.get("_id")
        if nid is None:
            log.warning("pcst_pruner.node_missing_id", index=idx)
            continue
        node_id_to_idx[str(nid)] = idx

    # Compute prizes
    prizes = compute_node_prizes(node_embeddings, query_embedding)
    prizes_arr = np.array(prizes, dtype=np.float64)

    # Scale top-K prizes up so PCST solver is guided toward target size
    if len(prizes) > max_nodes:
        top_k_indices = np.argsort(prizes_arr)[-max_nodes:]
        boost = float(np.mean(prizes_arr[top_k_indices])) + 1.0
        for i in top_k_indices:
            prizes_arr[i] *= boost

    # Build edge array and costs, tracking original-edge-index → pcst-edge-index
    edge_list: list[tuple[int, int]] = []
    orig_to_pcst: dict[int, int] = {}  # original edge idx → edge_list idx
    for orig_idx, e in enumerate(edges):
        src_idx = node_id_to_idx.get(e.get("start", ""))
        dst_idx = node_id_to_idx.get(e.get("end", ""))
        if src_idx is not None and dst_idx is not None:
            orig_to_pcst[orig_idx] = len(edge_list)
            edge_list.append((src_idx, dst_idx))

    if not edge_list:
        # No valid edges — return top-K nodes by prize
        top_indices = set(int(i) for i in np.argsort(prizes_arr)[-max_nodes:])
        return {
            "nodes": [n for i, n in enumerate(nodes) if i in top_indices],
            "edges": [],
        }

    edge_arr = np.array(edge_list, dtype=np.int64)
    cost_arr = np.full(len(edge_list), edge_cost, dtype=np.float64)

    # Run PCST
    selected_nodes, selected_edges = pcst_fast.pcst_fast(
        edge_arr, prizes_arr, cost_arr, -1, 1, "gw", 0
    )

    selected_node_set = set(int(i) for i in selected_nodes)
    selected_edge_set = set(int(i) for i in selected_edges)

    pruned_nodes = [n for i, n in enumerate(nodes) if i in selected_node_set]

    # Map selected pcst edge indices back to original edge dicts
    pruned_edge_dicts: list[dict[str, Any]] = []
    for orig_idx, pcst_idx in orig_to_pcst.items():
        if pcst_idx in selected_edge_set:
            pruned_edge_dicts.append(edges[orig_idx])

    log.info(
        "pcst_pruner.pruned",
        nodes_before=len(nodes),
        nodes_after=len(pruned_nodes),
        edges_before=len(edges),
        edges_after=len(pruned_edge_dicts),
    )

    return {"nodes": pruned_nodes, "edges": pruned_edge_dicts}
