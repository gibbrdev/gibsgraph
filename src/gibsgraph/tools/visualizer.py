"""Graph visualization — Mermaid diagrams + Neo4j Bloom URLs."""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

import structlog

from gibsgraph.config import Settings

log = structlog.get_logger(__name__)


class GraphVisualizer:
    """Generate Mermaid diagrams and Neo4j Bloom deep-link URLs from subgraphs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def to_mermaid(self, subgraph: dict[str, Any], *, max_nodes: int = 20) -> str:
        """Convert a subgraph dict to a Mermaid flowchart string."""
        nodes = subgraph.get("nodes", [])[:max_nodes]
        edges = subgraph.get("edges", [])

        lines = ["graph LR"]
        seen_nodes: set[str] = set()

        for node in nodes:
            node_id = str(node.get("id", node.get("name", "unknown")))
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", node_id)
            label = str(node.get("name", node_id))[:30].replace('"', "'")
            lines.append(f'    {safe_id}["{label}"]')
            seen_nodes.add(safe_id)

        for edge in edges:
            start = re.sub(r"[^a-zA-Z0-9_]", "_", str(edge.get("start", "")))
            end = re.sub(r"[^a-zA-Z0-9_]", "_", str(edge.get("end", "")))
            rel_type = edge.get("type", "RELATED")
            if start in seen_nodes and end in seen_nodes:
                lines.append(f"    {start} -->|{rel_type}| {end}")

        return "\n".join(lines)

    def bloom_url(self, subgraph: dict[str, Any]) -> str:
        """Generate a Neo4j Bloom deep-link URL for the subgraph."""
        node_ids = [str(n.get("id", "")) for n in subgraph.get("nodes", []) if n.get("id")]
        if not node_ids:
            return ""

        # Build Cypher-safe list with double quotes (not Python repr single quotes)
        id_list = "[" + ", ".join(f'"{nid}"' for nid in node_ids) + "]"
        cypher = f"MATCH (n) WHERE elementId(n) IN {id_list} RETURN n"
        encoded = urllib.parse.quote(cypher)
        base = "https://bloom.neo4j.io/index.html"
        url = f"{base}#search={encoded}"
        log.debug("visualizer.bloom_url", node_count=len(node_ids))
        return url

    def to_html_pyvis(self, subgraph: dict[str, Any]) -> str:
        """Generate an interactive PyVis HTML string (requires pyvis package)."""
        try:
            from pyvis.network import Network  # type: ignore[import-not-found]
        except ImportError as exc:
            msg = "Install pyvis: pip install gibsgraph[ui]"
            raise ImportError(msg) from exc

        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        for node in subgraph.get("nodes", []):
            node_id = str(node.get("_id", node.get("id", node.get("name", "?"))))
            label = str(node.get("name", node_id))[:40]
            net.add_node(node_id, label=label)
        for edge in subgraph.get("edges", []):
            net.add_edge(
                str(edge.get("start", "")), str(edge.get("end", "")), label=edge.get("type", "")
            )
        return str(net.generate_html())
