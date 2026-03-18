"""Build GibsGraph training pairs from GLEIF LEI corporate ownership data.

Source: GLEIF Golden Copy — Level 1 (entity data) + Level 2 (who-owns-whom).
License: CC0 — fully open, no restrictions.

Graph patterns:
  (:LegalEntity)-[:DIRECT_PARENT]->(:LegalEntity)
  (:LegalEntity)-[:ULTIMATE_PARENT]->(:LegalEntity)
  (:LegalEntity)-[:REGISTERED_IN]->(:Jurisdiction)
  (:LegalEntity)-[:HAS_STATUS]->(:EntityStatus)

This teaches the GNN corporate ownership traversal patterns —
critical for KYC/AML compliance queries alongside EUR-Lex regulatory data.

Usage:
    python data/scripts/build_gleif_pairs.py
    python data/scripts/build_gleif_pairs.py --max-entities 5000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import httpx

# GLEIF Golden Copy API — used to discover latest download URLs dynamically
GLEIF_API_LATEST = "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/latest"

DEFAULT_OUTPUT = Path("data/training/gleif_pairs.jsonl")
CACHE_DIR = Path("data/raw/gleif")

# Edge weights for corporate ownership
EDGE_WEIGHTS: dict[str, float] = {
    "DIRECT_PARENT": 0.95,
    "ULTIMATE_PARENT": 1.0,
    "REGISTERED_IN": 0.6,
    "SAME_JURISDICTION": 0.3,
    "CHILD_OF": 0.95,
}


def fetch_gleif_relationships(cache_dir: Path) -> Path:
    """Download GLEIF Level 2 relationship CSV (who-owns-whom)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "rr-golden-copy.csv"

    if csv_path.exists():
        print(f"  Using cached: {csv_path}")
        return csv_path

    print("  Fetching GLEIF relationship data...")
    print("  This is a ~22 MB download. Please wait...")

    with httpx.Client(timeout=300, follow_redirects=True) as client:
        # Get latest download URL from API
        api_resp = client.get(GLEIF_API_LATEST)
        api_resp.raise_for_status()
        rr_data = api_resp.json()["data"]["rr"]["full_file"]["csv"]
        rr_url = rr_data["url"]
        record_count = rr_data["record_count"]

        print(f"  {record_count} relationship records available")
        print(f"  Downloading from: {rr_url}")
        resp = client.get(rr_url)
        resp.raise_for_status()

    # It's a ZIP containing a CSV
    zip_path = cache_dir / "rr-golden-copy.csv.zip"
    zip_path.write_bytes(resp.content)
    print(f"  Downloaded {len(resp.content) / 1024 / 1024:.1f} MB")

    # Extract CSV from ZIP
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if csv_names:
            with zf.open(csv_names[0]) as src, open(csv_path, "wb") as dst:
                dst.write(src.read())
            print(f"  Extracted: {csv_path}")
        else:
            # Maybe it's not zipped
            csv_path.write_bytes(resp.content)

    return csv_path


def parse_gleif_relationships(
    csv_path: Path,
    max_entities: int = 10000,
) -> tuple[dict[str, dict], list[dict]]:
    """Parse GLEIF relationship CSV into entities and relationships.

    Returns (entities_by_lei, relationships).
    """
    entities: dict[str, dict] = {}
    relationships: list[dict] = []
    count = 0

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if count >= max_entities:
                break

            # GLEIF RR CSV columns vary by version, handle common patterns
            start_lei = row.get("Relationship.StartNode.NodeID", row.get("StartNodeID", ""))
            end_lei = row.get("Relationship.EndNode.NodeID", row.get("EndNodeID", ""))
            rel_type = row.get("Relationship.RelationshipType", row.get("RelationshipType", ""))
            rel_status = row.get("Relationship.RelationshipStatus", row.get("RelationshipStatus", ""))
            start_name = row.get("Relationship.StartNode.NodeID.Name", "")
            end_name = row.get("Relationship.EndNode.NodeID.Name", "")

            if not start_lei or not end_lei or not rel_type:
                continue

            # Only active relationships
            if rel_status and rel_status.upper() != "ACTIVE":
                continue

            # Register entities
            if start_lei not in entities:
                entities[start_lei] = {
                    "lei": start_lei,
                    "name": start_name,
                }
            if end_lei not in entities:
                entities[end_lei] = {
                    "lei": end_lei,
                    "name": end_name,
                }

            # Normalize relationship type
            if "DIRECT" in rel_type.upper():
                edge_type = "DIRECT_PARENT"
            elif "ULTIMATE" in rel_type.upper():
                edge_type = "ULTIMATE_PARENT"
            else:
                edge_type = rel_type.upper().replace(" ", "_").replace("-", "_")

            relationships.append({
                "source": start_lei,
                "target": end_lei,
                "type": edge_type,
            })
            count += 1

    return entities, relationships


def build_training_pairs(
    entities: dict[str, dict],
    relationships: list[dict],
    *,
    min_edges: int = 0,
) -> list[dict]:
    """Convert GLEIF entities + relationships into training pairs."""
    # Build edge maps
    forward: dict[str, list[dict]] = {}
    reverse: dict[str, list[dict]] = {}

    for rel in relationships:
        forward.setdefault(rel["source"], []).append(rel)
        reverse.setdefault(rel["target"], []).append(rel)

    pairs = []
    for lei, entity in entities.items():
        fwd = forward.get(lei, [])
        rev = reverse.get(lei, [])

        if len(fwd) + len(rev) < min_edges:
            continue

        name = entity.get("name", "") or lei

        # Build edges first to enrich text
        edges = []
        target_nodes = []
        for rel in fwd:
            target_entity = entities.get(rel["target"], {})
            target_name = target_entity.get("name", "") or rel["target"]
            edge_type = rel["type"]

            edges.append({
                "source": lei,
                "target": rel["target"],
                "type": edge_type,
                "weight": EDGE_WEIGHTS.get(edge_type, 0.7),
            })
            target_nodes.append({
                "id": rel["target"],
                "label": "LegalEntity",
                "properties": {
                    "lei": rel["target"],
                    "name": target_name,
                },
            })

        reverse_edges = []
        for rel in rev:
            source_entity = entities.get(rel["source"], {})
            edge_type = "CHILD_OF" if "PARENT" in rel["type"] else rel["type"]
            reverse_edges.append({
                "source": rel["source"],
                "target": lei,
                "type": edge_type,
                "weight": EDGE_WEIGHTS.get(edge_type, 0.7),
            })

        # Build enriched text describing the entity's relationships
        rel_descriptions = []
        for rel in fwd:
            tgt_lei = rel["target"][:8]
            if "PARENT" in rel["type"]:
                rel_descriptions.append(f"subsidiary of entity {tgt_lei}")
            elif "FUND" in rel["type"]:
                rel_descriptions.append(f"fund managed by entity {tgt_lei}")
            elif "SUBFUND" in rel["type"]:
                rel_descriptions.append(f"subfund of entity {tgt_lei}")
            elif "BRANCH" in rel["type"]:
                rel_descriptions.append(f"international branch of entity {tgt_lei}")
        for rel in rev:
            src_lei = rel["source"][:8]
            if "PARENT" in rel["type"]:
                rel_descriptions.append(f"parent of entity {src_lei}")

        if rel_descriptions:
            input_text = f"Legal entity {lei[:12]}... — {'; '.join(rel_descriptions[:4])}"
        else:
            input_text = f"Legal entity {lei} with {len(fwd)} outgoing and {len(rev)} incoming ownership relationships"

        pair = {
            "id": f"gleif_{lei[:8]}",
            "input_text": input_text,
            "metadata": {
                "domain": "gleif_lei",
                "lei": lei,
                "entity_name": name,
                "relationship_types": list({r["type"] for r in fwd + rev}),
            },
            "expected_graph": {
                "source_node": {
                    "id": lei,
                    "label": "LegalEntity",
                    "properties": {
                        "lei": lei,
                        "name": name,
                    },
                },
                "target_nodes": target_nodes,
                "edges": edges,
                "reverse_edges": reverse_edges,
            },
            "quality": {
                "edge_count": len(edges),
                "reverse_edge_count": len(reverse_edges),
                "total_edges": len(edges) + len(reverse_edges),
                "text_length": len(input_text),
                "verified_source": "gleif_golden_copy",
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

    by_edge_type: dict[str, int] = {}
    for p in pairs:
        for e in p["expected_graph"]["edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1
        for e in p["expected_graph"]["reverse_edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1

    print(f"\n{'=' * 60}")
    print("GLEIF LEI TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:          {total}")
    print(f"  Total forward edges:  {total_fwd}")
    print(f"  Total reverse edges:  {total_rev}")
    print(f"  Avg edges per pair:   {avg_edges:.1f}")

    print("\n  By edge type:")
    for et, count in sorted(by_edge_type.items(), key=lambda x: -x[1]):
        print(f"    {et}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from GLEIF LEI data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-entities", type=int, default=10000,
                        help="Max relationship records to process")
    parser.add_argument("--min-edges", type=int, default=0)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("Building GLEIF LEI training pairs...")

    # Download/cache relationship data
    csv_path = fetch_gleif_relationships(CACHE_DIR)

    # Parse
    print(f"\n  Parsing relationships (max {args.max_entities})...")
    entities, relationships = parse_gleif_relationships(csv_path, max_entities=args.max_entities)
    print(f"  Entities: {len(entities)}")
    print(f"  Relationships: {len(relationships)}")

    # Build pairs
    pairs = build_training_pairs(entities, relationships, min_edges=args.min_edges)
    print_stats(pairs)

    if args.stats_only:
        return

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} training pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")
    print("\nAttribution: GLEIF data is CC0 — freely available for any purpose.")


if __name__ == "__main__":
    main()
