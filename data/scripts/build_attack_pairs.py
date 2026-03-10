"""Build GibsGraph training pairs from MITRE ATT&CK Enterprise STIX data.

Converts ATT&CK's STIX 2.1 bundle into supervised training pairs:
(input_text, expected_graph) where the graph is human-curated by MITRE
threat intelligence analysts — not LLM-generated.

Graph shape is structurally different from EUR-Lex:
- EUR-Lex: document → cross-references → document (mostly linear/hierarchical)
- ATT&CK: many-to-many web (groups use techniques, techniques mitigated by
  controls, software implements techniques, campaigns attributed to groups)

Training pair format (JSONL):
{
    "id": "T1566.001",
    "input_text": "Spearphishing Attachment — Adversaries may send ...",
    "metadata": {
        "domain": "mitre_attack",
        "object_type": "technique",
        "attack_id": "T1566.001",
        "tactics": ["initial-access"],
        "platforms": ["Windows", "macOS", "Linux"],
        "is_subtechnique": true
    },
    "expected_graph": {
        "source_node": {...},
        "target_nodes": [...],
        "edges": [...],
        "reverse_edges": [...]
    },
    "quality": {
        "edge_count": 42,
        "reverse_edge_count": 3,
        "has_subtechniques": true,
        "has_mitigations": true,
        "text_length": 1874,
        "verified_source": "mitre_attack_stix"
    }
}

Attribution:
© 2025 The MITRE Corporation. This work is reproduced and distributed
with the permission of The MITRE Corporation.

Usage:
    python data/scripts/build_attack_pairs.py
    python data/scripts/build_attack_pairs.py --min-edges 3 \
        --output data/training/attack_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTACK_PATH = Path("data/raw/attack/enterprise-attack.json")
DEFAULT_OUTPUT = Path("data/training/attack_pairs.jsonl")

# Edge weight scheme — tuned for threat intelligence graph traversal.
# Higher weight = more important for understanding attack chains.
EDGE_WEIGHTS: dict[str, float] = {
    "uses": 0.9,
    "mitigates": 0.85,
    "subtechnique-of": 1.0,
    "attributed-to": 0.8,
    "detects": 0.7,
    "revoked-by": 0.1,
}

# STIX type → Neo4j-style PascalCase label
STIX_TO_LABEL: dict[str, str] = {
    "attack-pattern": "Technique",
    "intrusion-set": "ThreatGroup",
    "malware": "Malware",
    "tool": "Tool",
    "course-of-action": "Mitigation",
    "campaign": "Campaign",
    "x-mitre-tactic": "Tactic",
    "x-mitre-data-component": "DataComponent",
    "x-mitre-data-source": "DataSource",
    "x-mitre-detection-strategy": "DetectionStrategy",
}


def load_bundle(path: Path) -> dict[str, dict]:
    """Load STIX bundle and index objects by ID."""
    with open(path, encoding="utf-8") as f:
        bundle = json.load(f)

    index: dict[str, dict] = {}
    for obj in bundle["objects"]:
        index[obj["id"]] = obj
    return index


def get_attack_id(obj: dict) -> str:
    """Extract ATT&CK ID (e.g., T1566.001, G0007, S0154) from external_references."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and "external_id" in ref:
            return ref["external_id"]
    return obj["id"].split("--")[1][:8]


def get_label(obj: dict) -> str:
    """Get Neo4j-style PascalCase label from STIX type."""
    return STIX_TO_LABEL.get(obj["type"], obj["type"].replace("-", " ").title().replace(" ", ""))


def build_node(obj: dict) -> dict:
    """Build a node dict from a STIX object."""
    attack_id = get_attack_id(obj)
    label = get_label(obj)

    props: dict[str, object] = {
        "stix_id": obj["id"],
        "attack_id": attack_id,
        "name": obj.get("name", ""),
        "type": obj["type"],
    }

    if obj["type"] == "attack-pattern":
        props["platforms"] = obj.get("x_mitre_platforms", [])
        props["is_subtechnique"] = obj.get("x_mitre_is_subtechnique", False)
        phases = obj.get("kill_chain_phases", [])
        props["tactics"] = [p["phase_name"] for p in phases]
    elif obj["type"] == "intrusion-set":
        props["aliases"] = obj.get("aliases", [])
    elif obj["type"] in ("malware", "tool"):
        props["platforms"] = obj.get("x_mitre_platforms", [])

    return {
        "id": attack_id,
        "label": label,
        "properties": props,
    }


def build_training_pairs(
    index: dict[str, dict],
    *,
    min_edges: int = 0,
    min_text_length: int = 50,
) -> list[dict]:
    """Convert ATT&CK STIX objects into GibsGraph training pairs."""
    # Separate objects and relationships
    node_objects: list[dict] = []
    relationships: list[dict] = []

    for obj in index.values():
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        if obj["type"] == "relationship":
            if obj.get("relationship_type") != "revoked-by":
                relationships.append(obj)
        elif obj["type"] in STIX_TO_LABEL:
            node_objects.append(obj)

    # Build forward/reverse edge maps
    forward: dict[str, list[dict]] = {}  # source_id -> [relationship]
    reverse: dict[str, list[dict]] = {}  # target_id -> [relationship]

    for rel in relationships:
        src = rel["source_ref"]
        tgt = rel["target_ref"]
        if src not in index or tgt not in index:
            continue
        forward.setdefault(src, []).append(rel)
        reverse.setdefault(tgt, []).append(rel)

    pairs = []
    for obj in node_objects:
        stix_id = obj["id"]
        text = obj.get("description", "").strip()
        name = obj.get("name", "")

        # Combine name + description as input text
        input_text = f"{name}\n\n{text}" if text else name
        if len(input_text) < min_text_length:
            continue

        # Forward edges (this object → targets)
        fwd_rels = forward.get(stix_id, [])
        # Reverse edges (sources → this object)
        rev_rels = reverse.get(stix_id, [])

        if len(fwd_rels) + len(rev_rels) < min_edges:
            continue

        edges = []
        target_nodes = []
        has_subtechniques = False
        has_mitigations = False

        for rel in fwd_rels:
            tgt_obj = index.get(rel["target_ref"])
            if not tgt_obj or tgt_obj.get("revoked") or tgt_obj.get("x_mitre_deprecated"):
                continue
            rel_type = rel["relationship_type"]
            edge = {
                "source": get_attack_id(obj),
                "target": get_attack_id(tgt_obj),
                "type": rel_type.upper().replace("-", "_"),
                "weight": EDGE_WEIGHTS.get(rel_type, 0.5),
            }
            # Include relationship description as edge context
            rel_desc = rel.get("description", "")
            if rel_desc:
                edge["description"] = rel_desc[:500]
            edges.append(edge)
            target_nodes.append(build_node(tgt_obj))

            if rel_type == "subtechnique-of":
                has_subtechniques = True
            if rel_type == "mitigates":
                has_mitigations = True

        reverse_edges = []
        for rel in rev_rels:
            src_obj = index.get(rel["source_ref"])
            if not src_obj or src_obj.get("revoked") or src_obj.get("x_mitre_deprecated"):
                continue
            rel_type = rel["relationship_type"]
            rev_edge = {
                "source": get_attack_id(src_obj),
                "target": get_attack_id(obj),
                "type": rel_type.upper().replace("-", "_"),
                "weight": EDGE_WEIGHTS.get(rel_type, 0.5),
            }
            rel_desc = rel.get("description", "")
            if rel_desc:
                rev_edge["description"] = rel_desc[:500]
            reverse_edges.append(rev_edge)

            if rel_type == "mitigates":
                has_mitigations = True

        attack_id = get_attack_id(obj)
        tactics = []
        platforms = []
        is_sub = False

        if obj["type"] == "attack-pattern":
            phases = obj.get("kill_chain_phases", [])
            tactics = [p["phase_name"] for p in phases]
            platforms = obj.get("x_mitre_platforms", [])
            is_sub = obj.get("x_mitre_is_subtechnique", False)

        pair = {
            "id": attack_id,
            "input_text": input_text,
            "metadata": {
                "domain": "mitre_attack",
                "object_type": get_label(obj).lower(),
                "attack_id": attack_id,
                "stix_type": obj["type"],
                "tactics": tactics,
                "platforms": platforms,
                "is_subtechnique": is_sub,
                "aliases": obj.get("aliases", []),
            },
            "expected_graph": {
                "source_node": build_node(obj),
                "target_nodes": target_nodes,
                "edges": edges,
                "reverse_edges": reverse_edges,
            },
            "quality": {
                "edge_count": len(edges),
                "reverse_edge_count": len(reverse_edges),
                "total_edges": len(edges) + len(reverse_edges),
                "has_subtechniques": has_subtechniques,
                "has_mitigations": has_mitigations,
                "text_length": len(input_text),
                "verified_source": "mitre_attack_stix",
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

    with_fwd = sum(1 for p in pairs if p["quality"]["edge_count"] > 0)
    with_rev = sum(1 for p in pairs if p["quality"]["reverse_edge_count"] > 0)
    total_fwd = sum(p["quality"]["edge_count"] for p in pairs)
    total_rev = sum(p["quality"]["reverse_edge_count"] for p in pairs)
    with_mitigations = sum(1 for p in pairs if p["quality"]["has_mitigations"])
    with_subtechniques = sum(1 for p in pairs if p["quality"]["has_subtechniques"])

    by_type: dict[str, int] = {}
    by_tactic: dict[str, int] = {}
    by_edge_type: dict[str, int] = {}

    for p in pairs:
        otype = p["metadata"]["object_type"]
        by_type[otype] = by_type.get(otype, 0) + 1
        for t in p["metadata"]["tactics"]:
            by_tactic[t] = by_tactic.get(t, 0) + 1
        for e in p["expected_graph"]["edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1
        for e in p["expected_graph"]["reverse_edges"]:
            et = e["type"]
            by_edge_type[et] = by_edge_type.get(et, 0) + 1

    text_lengths = [p["quality"]["text_length"] for p in pairs]
    avg_text = sum(text_lengths) / len(text_lengths)
    edge_counts = [p["quality"]["total_edges"] for p in pairs]
    avg_edges = sum(edge_counts) / len(edge_counts)

    print(f"\n{'=' * 60}")
    print("MITRE ATT&CK TRAINING PAIR STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Total pairs:              {total}")
    print(f"  With forward edges:       {with_fwd} ({with_fwd * 100 // total}%)")
    print(f"  With reverse edges:       {with_rev} ({with_rev * 100 // total}%)")
    print(f"  Total forward edges:      {total_fwd}")
    print(f"  Total reverse edges:      {total_rev}")
    print(f"  Avg edges per pair:       {avg_edges:.1f}")
    print(f"  With mitigations:         {with_mitigations}")
    print(f"  With subtechniques:       {with_subtechniques}")
    print(f"  Avg text length:          {avg_text:.0f} chars")

    print("\n  By object type:")
    for ot, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {ot}: {count}")

    if by_tactic:
        print("\n  By tactic (techniques only):")
        for t, count in sorted(by_tactic.items(), key=lambda x: -x[1]):
            print(f"    {t}: {count}")

    print("\n  By edge type:")
    for et, count in sorted(by_edge_type.items(), key=lambda x: -x[1]):
        print(f"    {et}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from MITRE ATT&CK data")
    parser.add_argument(
        "--input",
        type=Path,
        default=ATTACK_PATH,
        help="ATT&CK STIX JSON bundle path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--min-edges",
        type=int,
        default=0,
        help="Minimum total edges (forward + reverse) per object",
    )
    parser.add_argument(
        "--min-text",
        type=int,
        default=50,
        help="Minimum text length to include",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print statistics without writing output",
    )
    args = parser.parse_args()

    print(f"Loading ATT&CK bundle from {args.input}...")
    index = load_bundle(args.input)
    print(f"  {len(index)} STIX objects loaded")

    pairs = build_training_pairs(
        index,
        min_edges=args.min_edges,
        min_text_length=args.min_text,
    )

    print_stats(pairs)

    if args.stats_only:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(pairs)} training pairs to {args.output}")
    print(f"File size: {args.output.stat().st_size / 1024:.1f} KB")
    print(
        "\nAttribution: © 2025 The MITRE Corporation. Reproduced and distributed with permission."
    )


if __name__ == "__main__":
    main()
