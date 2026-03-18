"""Build GibsGraph training pairs from Hetionet biomedical knowledge graph.

Source: https://github.com/hetio/hetionet
License: CC0 — fully open.
Size: 47,031 nodes (11 types), 2,250,197 edges (24 types).

Graph patterns:
  (:Compound)-[:TREATS]->(:Disease)
  (:Gene)-[:ASSOCIATES]->(:Disease)
  (:Compound)-[:BINDS]->(:Gene)
  (:Disease)-[:LOCALIZES]->(:Anatomy)
  (:Compound)-[:CAUSES]->(:SideEffect)
  + 19 more relationship types

This teaches the GNN biomedical graph traversal — drug repurposing,
disease-gene associations, and multi-hop reasoning over heterogeneous graphs.

Usage:
    python data/scripts/build_hetionet_pairs.py
    python data/scripts/build_hetionet_pairs.py --max-nodes 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

# Hetionet GitHub raw data URLs
HETIONET_NODES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"
HETIONET_EDGES_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
# JSON format is easier to parse
HETIONET_JSON_URL = "https://github.com/hetio/hetionet/raw/main/hetnet/json/hetionet-v1.0.json.bz2"
# Use the smaller TSV files
HETIONET_BASE = "https://github.com/hetio/hetionet/raw/main/hetnet/tsv/"

DEFAULT_OUTPUT = Path("data/training/hetionet_pairs.jsonl")
CACHE_DIR = Path("data/raw/hetionet")

# Hetionet metaedge abbreviations → readable types
METAEDGE_MAP: dict[str, tuple[str, str, str]] = {
    "AdG": ("Anatomy", "DOWNREGULATES", "Gene"),
    "AeG": ("Anatomy", "EXPRESSES", "Gene"),
    "AuG": ("Anatomy", "UPREGULATES", "Gene"),
    "CbG": ("Compound", "BINDS", "Gene"),
    "CcSE": ("Compound", "CAUSES", "SideEffect"),
    "CdG": ("Compound", "DOWNREGULATES", "Gene"),
    "CpD": ("Compound", "PALLIATES", "Disease"),
    "CrC": ("Compound", "RESEMBLES", "Compound"),
    "CtD": ("Compound", "TREATS", "Disease"),
    "CuG": ("Compound", "UPREGULATES", "Gene"),
    "DaG": ("Disease", "ASSOCIATES", "Gene"),
    "DdG": ("Disease", "DOWNREGULATES", "Gene"),
    "DlA": ("Disease", "LOCALIZES", "Anatomy"),
    "DpS": ("Disease", "PRESENTS", "Symptom"),
    "DrD": ("Disease", "RESEMBLES", "Disease"),
    "DuG": ("Disease", "UPREGULATES", "Gene"),
    "GcG": ("Gene", "COVARIES", "Gene"),
    "GiG": ("Gene", "INTERACTS", "Gene"),
    "GpBP": ("Gene", "PARTICIPATES_IN", "BiologicalProcess"),
    "GpCC": ("Gene", "PARTICIPATES_IN", "CellularComponent"),
    "GpMF": ("Gene", "PARTICIPATES_IN", "MolecularFunction"),
    "GpPW": ("Gene", "PARTICIPATES_IN", "Pathway"),
    "GrG": ("Gene", "REGULATES", "Gene"),
    "Gr>G": ("Gene", "REGULATES", "Gene"),
    "PCiC": ("PharmacologicClass", "INCLUDES", "Compound"),
}

# Edge weights by relationship importance for biomedical reasoning
EDGE_WEIGHTS: dict[str, float] = {
    "TREATS": 1.0,
    "CAUSES": 0.9,
    "BINDS": 0.85,
    "ASSOCIATES": 0.8,
    "LOCALIZES": 0.7,
    "PRESENTS": 0.7,
    "DOWNREGULATES": 0.75,
    "UPREGULATES": 0.75,
    "EXPRESSES": 0.7,
    "INTERACTS": 0.6,
    "PARTICIPATES_IN": 0.65,
    "RESEMBLES": 0.5,
    "COVARIES": 0.4,
    "REGULATES": 0.7,
    "PALLIATES": 0.85,
    "INCLUDES": 0.6,
}


def fetch_hetionet_tsv(cache_dir: Path) -> tuple[Path, Path]:
    """Download Hetionet TSV files."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = cache_dir / "nodes.tsv"
    edges_path = cache_dir / "edges.tsv"

    with httpx.Client(timeout=120, follow_redirects=True) as client:
        if not nodes_path.exists():
            print("  Downloading nodes TSV...")
            url = HETIONET_BASE + "hetionet-v1.0-nodes.tsv"
            resp = client.get(url)
            resp.raise_for_status()
            nodes_path.write_bytes(resp.content)
            print(f"  Saved {len(resp.content) / 1024:.1f} KB")

        if not edges_path.exists():
            print("  Downloading edges TSV...")
            url = HETIONET_BASE + "hetionet-v1.0-edges.sif.gz"
            resp = client.get(url)
            resp.raise_for_status()

            # Decompress gzip
            import gzip
            decompressed = gzip.decompress(resp.content)
            edges_path.write_bytes(decompressed)
            print(f"  Saved {len(decompressed) / 1024 / 1024:.1f} MB")

    return nodes_path, edges_path


def parse_nodes(nodes_path: Path) -> dict[str, dict]:
    """Parse Hetionet nodes TSV: id, name, kind."""
    nodes: dict[str, dict] = {}
    with open(nodes_path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            node_id = parts[0]
            name = parts[1]
            kind = parts[2]
            nodes[node_id] = {
                "id": node_id,
                "name": name,
                "kind": kind,
            }
    return nodes


def parse_edges(edges_path: Path) -> list[dict]:
    """Parse Hetionet edges SIF: source, metaedge, target."""
    edges = []
    with open(edges_path, encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            source = parts[0]
            metaedge = parts[1]
            target = parts[2]
            edges.append({
                "source": source,
                "metaedge": metaedge,
                "target": target,
            })
    return edges


def _kind_to_label(kind: str) -> str:
    """Convert Hetionet 'kind' to PascalCase Neo4j label."""
    mapping = {
        "Anatomy": "Anatomy",
        "Biological Process": "BiologicalProcess",
        "Cellular Component": "CellularComponent",
        "Compound": "Compound",
        "Disease": "Disease",
        "Gene": "Gene",
        "Molecular Function": "MolecularFunction",
        "Pathway": "Pathway",
        "Pharmacologic Class": "PharmacologicClass",
        "Side Effect": "SideEffect",
        "Symptom": "Symptom",
    }
    return mapping.get(kind, kind.replace(" ", ""))


def _metaedge_to_type(metaedge: str) -> str:
    """Convert Hetionet metaedge abbreviation to UPPER_SNAKE_CASE."""
    # Metaedges look like "Compound - treats - Disease" or abbreviated "CtD"
    # Also handle reverse: "DtC" → reversed
    # The SIF file uses the full format: "Compound - binds - Gene"
    parts = metaedge.split(" - ")
    if len(parts) == 3:
        rel = parts[1].strip().upper().replace(" ", "_").replace(">", "_")
        return rel

    # Abbreviated form
    if metaedge in METAEDGE_MAP:
        return METAEDGE_MAP[metaedge][1]

    # Clean up any non-alphanumeric chars
    import re
    cleaned = re.sub(r"[^A-Z0-9_]", "_", metaedge.upper())
    return cleaned


def build_training_pairs(
    nodes: dict[str, dict],
    edges: list[dict],
    *,
    max_nodes: int = 5000,
    min_edges: int = 1,
) -> list[dict]:
    """Convert Hetionet data into training pairs.

    Samples nodes with the most edges for richer training signal.
    """
    # Build edge maps
    forward: dict[str, list[dict]] = {}
    reverse: dict[str, list[dict]] = {}

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        forward.setdefault(src, []).append(edge)
        reverse.setdefault(tgt, []).append(edge)

    # Rank nodes by total edge count, take top max_nodes
    node_edge_count = {}
    for nid in nodes:
        fwd = len(forward.get(nid, []))
        rev = len(reverse.get(nid, []))
        node_edge_count[nid] = fwd + rev

    top_nodes = sorted(node_edge_count.items(), key=lambda x: -x[1])[:max_nodes]

    pairs = []
    for nid, edge_count in top_nodes:
        if edge_count < min_edges:
            continue

        node = nodes[nid]
        name = node["name"]
        kind = node["kind"]
        label = _kind_to_label(kind)

        # Build input text
        input_text = f"{name} ({kind})"

        # Forward edges (cap at 50 to keep pairs manageable)
        fwd_edges = forward.get(nid, [])[:50]
        rev_edges = reverse.get(nid, [])[:50]

        pair_edges = []
        target_nodes = []
        seen_targets = set()

        for edge in fwd_edges:
            tgt_id = edge["target"]
            tgt_node = nodes.get(tgt_id)
            if not tgt_node:
                continue

            rel_type = _metaedge_to_type(edge["metaedge"])
            tgt_label = _kind_to_label(tgt_node["kind"])

            pair_edges.append({
                "source": nid,
                "target": tgt_id,
                "type": rel_type,
                "weight": EDGE_WEIGHTS.get(rel_type, 0.5),
            })

            if tgt_id not in seen_targets:
                target_nodes.append({
                    "id": tgt_id,
                    "label": tgt_label,
                    "properties": {
                        "name": tgt_node["name"],
                        "kind": tgt_node["kind"],
                    },
                })
                seen_targets.add(tgt_id)

        pair_reverse_edges = []
        for edge in rev_edges:
            src_id = edge["source"]
            rel_type = _metaedge_to_type(edge["metaedge"])
            pair_reverse_edges.append({
                "source": src_id,
                "target": nid,
                "type": rel_type,
                "weight": EDGE_WEIGHTS.get(rel_type, 0.5),
            })

        pair = {
            "id": f"hetio_{nid}",
            "input_text": input_text,
            "metadata": {
                "domain": "hetionet",
                "node_kind": kind,
                "node_name": name,
                "hetionet_id": nid,
            },
            "expected_graph": {
                "source_node": {
                    "id": nid,
                    "label": label,
                    "properties": {
                        "name": name,
                        "kind": kind,
                    },
                },
                "target_nodes": target_nodes,
                "edges": pair_edges,
                "reverse_edges": pair_reverse_edges,
            },
            "quality": {
                "edge_count": len(pair_edges),
                "reverse_edge_count": len(pair_reverse_edges),
                "total_edges": len(pair_edges) + len(pair_reverse_edges),
                "text_length": len(input_text),
                "verified_source": "hetionet_v1.0",
            },
        }
        pairs.append(pair)

    return pairs


def print_stats(pairs: list[dict]) -> None:
    """Print dataset statistics."""
    total = len(pairs)
    if total == 0:
        print("No pairs generated.")
        return

    total_fwd = sum(p["quality"]["edge_count"] for p in pairs)
    total_rev = sum(p["quality"]["reverse_edge_count"] for p in pairs)
    avg_edges = sum(p["quality"]["total_edges"] for p in pairs) / total

    by_kind: dict[str, int] = {}
    by_edge_type: dict[str, int] = {}
    for p in pairs:
        kind = p["metadata"]["node_kind"]
        by_kind[kind] = by_kind.get(kind, 0) + 1
        for e in p["expected_graph"]["edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1
        for e in p["expected_graph"]["reverse_edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1

    print(f"\n{'=' * 60}")
    print("HETIONET TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:          {total}")
    print(f"  Total forward edges:  {total_fwd}")
    print(f"  Total reverse edges:  {total_rev}")
    print(f"  Avg edges per pair:   {avg_edges:.1f}")

    print("\n  By node kind:")
    for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
        print(f"    {kind}: {count}")

    print("\n  By edge type:")
    for et, count in sorted(by_edge_type.items(), key=lambda x: -x[1]):
        print(f"    {et}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from Hetionet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-nodes", type=int, default=5000,
                        help="Max nodes to include (ranked by edge count)")
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("Building Hetionet training pairs...")

    # Download/cache
    nodes_path, edges_path = fetch_hetionet_tsv(CACHE_DIR)

    # Parse
    print("\n  Parsing nodes...")
    nodes = parse_nodes(nodes_path)
    print(f"  {len(nodes)} nodes loaded")

    print("  Parsing edges...")
    edges = parse_edges(edges_path)
    print(f"  {len(edges)} edges loaded")

    # Build pairs
    pairs = build_training_pairs(
        nodes, edges,
        max_nodes=args.max_nodes,
        min_edges=args.min_edges,
    )
    print_stats(pairs)

    if args.stats_only:
        return

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} training pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nAttribution: Hetionet is CC0 — https://het.io/")


if __name__ == "__main__":
    main()
