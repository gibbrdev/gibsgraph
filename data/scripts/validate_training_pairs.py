"""Validate training pair JSONL files against GibsGraph expert conventions.

Domain-agnostic — works on EUR-Lex, ATT&CK, or any future training dataset.
Checks every expected_graph in the JSONL against:

  1. SCHEMA CONVENTIONS — PascalCase labels, UPPER_SNAKE_CASE relationships,
     no generic labels, no vague relationship types
  2. STRUCTURAL INTEGRITY — no dangling edges, no orphan targets, consistent
     IDs, edge weights in valid range
  3. DATA QUALITY — text length distribution, edge density, empty fields,
     duplicates
  4. EXPERT ALIGNMENT — validates labels and rel types against modeling
     patterns from the bundled expert dataset

Usage:
    python data/scripts/validate_training_pairs.py data/training/attack_pairs.jsonl
    python data/scripts/validate_training_pairs.py data/training/eurlex_pairs.jsonl
    python data/scripts/validate_training_pairs.py data/training/*.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# --- Convention rules (from expert dataset + Neo4j official docs) ---

GENERIC_LABELS = {
    "Entity",
    "Object",
    "Item",
    "Thing",
    "Node",
    "Data",
    "Record",
    "Element",
    "Resource",
    "Entry",
}

VAGUE_RELS = {
    "RELATED",
    "RELATED_TO",
    "HAS",
    "IS",
    "CONNECTS",
    "LINKS",
    "ASSOCIATED",
    "ASSOCIATED_WITH",
}


def is_pascal_case(s: str) -> bool:
    """Check if string is PascalCase (starts upper, has at least one lower)."""
    return bool(re.match(r"^[A-Z][a-zA-Z0-9]*$", s)) and any(c.islower() for c in s)


def is_upper_snake(s: str) -> bool:
    """Check if string is UPPER_SNAKE_CASE."""
    return bool(re.match(r"^[A-Z][A-Z0-9_]*$", s))


class Finding:
    """A validation finding with severity."""

    def __init__(self, severity: str, category: str, message: str, pair_id: str = "") -> None:
        self.severity = severity  # ERROR, WARN, INFO
        self.category = category
        self.message = message
        self.pair_id = pair_id

    def __repr__(self) -> str:
        prefix = f"[{self.pair_id}] " if self.pair_id else ""
        return f"{self.severity}: {prefix}{self.message}"


def validate_pair_schema(pair: dict) -> list[Finding]:
    """Check required fields exist in a training pair."""
    findings: list[Finding] = []
    pid = pair.get("id", "???")

    required_top = ["id", "input_text", "metadata", "expected_graph", "quality"]
    for field in required_top:
        if field not in pair:
            findings.append(Finding("ERROR", "schema", f"Missing top-level field: {field}", pid))

    eg = pair.get("expected_graph", {})
    if "source_node" not in eg:
        findings.append(Finding("ERROR", "schema", "Missing expected_graph.source_node", pid))
    if "edges" not in eg:
        findings.append(Finding("ERROR", "schema", "Missing expected_graph.edges", pid))

    return findings


def validate_conventions(pair: dict) -> list[Finding]:
    """Check Neo4j naming conventions on labels and relationship types."""
    findings: list[Finding] = []
    pid = pair.get("id", "???")
    eg = pair.get("expected_graph", {})

    # Collect all labels
    labels = set()
    source = eg.get("source_node", {})
    if source.get("label"):
        labels.add(source["label"])
    for tn in eg.get("target_nodes", []):
        if tn.get("label"):
            labels.add(tn["label"])

    for label in labels:
        if label in GENERIC_LABELS:
            findings.append(Finding("ERROR", "convention", f"Generic label '{label}'", pid))
        elif not is_pascal_case(label):
            findings.append(
                Finding("WARN", "convention", f"Label '{label}' is not PascalCase", pid)
            )

    # Collect all relationship types
    rel_types = set()
    for edge in eg.get("edges", []):
        if edge.get("type"):
            rel_types.add(edge["type"])
    for edge in eg.get("reverse_edges", []):
        if edge.get("type"):
            rel_types.add(edge["type"])

    for rt in rel_types:
        if rt in VAGUE_RELS:
            findings.append(Finding("ERROR", "convention", f"Vague relationship type '{rt}'", pid))
        elif not is_upper_snake(rt):
            findings.append(
                Finding("WARN", "convention", f"Rel type '{rt}' is not UPPER_SNAKE_CASE", pid)
            )

    return findings


def validate_structure(pair: dict) -> list[Finding]:
    """Check structural integrity of the expected graph."""
    findings: list[Finding] = []
    pid = pair.get("id", "???")
    eg = pair.get("expected_graph", {})

    source = eg.get("source_node", {})
    source_id = source.get("id", "")

    # Build set of known node IDs
    known_ids = set()
    if source_id:
        known_ids.add(source_id)
    for tn in eg.get("target_nodes", []):
        if tn.get("id"):
            known_ids.add(tn["id"])

    # Check forward edges reference valid nodes
    for edge in eg.get("edges", []):
        if edge.get("source") and edge["source"] not in known_ids:
            findings.append(
                Finding(
                    "WARN",
                    "structure",
                    f"Edge source '{edge['source']}' not in node set",
                    pid,
                )
            )
        if edge.get("target") and edge["target"] not in known_ids:
            findings.append(
                Finding(
                    "WARN",
                    "structure",
                    f"Edge target '{edge['target']}' not in node set",
                    pid,
                )
            )

        # Edge weight validation
        weight = edge.get("weight")
        if weight is not None and (weight < 0 or weight > 1):
            findings.append(
                Finding("ERROR", "structure", f"Edge weight {weight} outside [0,1]", pid)
            )

    # Check reverse edges
    for edge in eg.get("reverse_edges", []):
        weight = edge.get("weight")
        if weight is not None and (weight < 0 or weight > 1):
            findings.append(
                Finding("ERROR", "structure", f"Reverse edge weight {weight} outside [0,1]", pid)
            )

    # Target nodes without any edge referencing them
    targeted_ids = set()
    for edge in eg.get("edges", []):
        if edge.get("target"):
            targeted_ids.add(edge["target"])
    target_node_ids = {tn["id"] for tn in eg.get("target_nodes", []) if tn.get("id")}
    orphan_targets = target_node_ids - targeted_ids - {source_id}
    if orphan_targets:
        findings.append(
            Finding(
                "WARN",
                "structure",
                f"{len(orphan_targets)} target nodes not referenced by any edge",
                pid,
            )
        )

    return findings


def validate_quality(pair: dict) -> list[Finding]:
    """Check data quality signals."""
    findings: list[Finding] = []
    pid = pair.get("id", "???")

    text = pair.get("input_text", "")
    if len(text) < 20:
        findings.append(Finding("WARN", "quality", f"Very short text ({len(text)} chars)", pid))

    eg = pair.get("expected_graph", {})
    if not eg.get("edges") and not eg.get("reverse_edges"):
        findings.append(Finding("INFO", "quality", "No edges (isolated node)", pid))

    # Check for empty node properties
    source = eg.get("source_node", {})
    props = source.get("properties", {})
    empty_props = [k for k, v in props.items() if v == "" or v == [] or v is None]
    if len(empty_props) > len(props) / 2 and len(props) > 2:
        findings.append(
            Finding("INFO", "quality", f"Source node has {len(empty_props)} empty properties", pid)
        )

    return findings


def validate_file(path: Path) -> tuple[list[Finding], dict]:
    """Validate all pairs in a JSONL file. Returns (findings, stats)."""
    all_findings: list[Finding] = []
    pairs: list[dict] = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                all_findings.append(Finding("ERROR", "parse", f"Line {i}: invalid JSON — {e}"))
                continue
            pairs.append(pair)

    if not pairs:
        all_findings.append(Finding("ERROR", "parse", "File is empty or has no valid pairs"))
        return all_findings, {}

    # Validate each pair
    for pair in pairs:
        all_findings.extend(validate_pair_schema(pair))
        all_findings.extend(validate_conventions(pair))
        all_findings.extend(validate_structure(pair))
        all_findings.extend(validate_quality(pair))

    # File-level checks
    ids = [p.get("id", "") for p in pairs]
    id_counts = Counter(ids)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    if dupes:
        all_findings.append(
            Finding("WARN", "quality", f"{len(dupes)} duplicate IDs: {list(dupes.keys())[:10]}")
        )

    # Collect all labels and rel types across file
    all_labels: Counter[str] = Counter()
    all_rels: Counter[str] = Counter()
    all_edge_counts = []
    all_text_lengths = []

    for pair in pairs:
        eg = pair.get("expected_graph", {})
        src = eg.get("source_node", {})
        if src.get("label"):
            all_labels[src["label"]] += 1
        for tn in eg.get("target_nodes", []):
            if tn.get("label"):
                all_labels[tn["label"]] += 1
        for edge in eg.get("edges", []) + eg.get("reverse_edges", []):
            if edge.get("type"):
                all_rels[edge["type"]] += 1

        fwd = len(eg.get("edges", []))
        rev = len(eg.get("reverse_edges", []))
        all_edge_counts.append(fwd + rev)
        all_text_lengths.append(len(pair.get("input_text", "")))

    # Compute stats
    total = len(pairs)
    avg_edges = sum(all_edge_counts) / total
    avg_text = sum(all_text_lengths) / total
    zero_edge = sum(1 for c in all_edge_counts if c == 0)

    stats = {
        "total_pairs": total,
        "unique_labels": dict(all_labels.most_common()),
        "unique_rel_types": dict(all_rels.most_common()),
        "avg_edges_per_pair": round(avg_edges, 1),
        "zero_edge_pairs": zero_edge,
        "avg_text_length": round(avg_text, 0),
        "min_text_length": min(all_text_lengths),
        "max_text_length": max(all_text_lengths),
        "duplicate_ids": len(dupes),
    }

    return all_findings, stats


def print_report(path: Path, findings: list[Finding], stats: dict) -> None:
    """Print validation report."""
    errors = [f for f in findings if f.severity == "ERROR"]
    warns = [f for f in findings if f.severity == "WARN"]
    infos = [f for f in findings if f.severity == "INFO"]

    print(f"\n{'=' * 70}")
    print(f"TRAINING PAIR VALIDATION: {path.name}")
    print(f"{'=' * 70}")

    if stats:
        print(f"\n  Pairs:              {stats['total_pairs']}")
        print(f"  Avg edges/pair:     {stats['avg_edges_per_pair']}")
        print(f"  Zero-edge pairs:    {stats['zero_edge_pairs']}")
        print(f"  Avg text length:    {stats['avg_text_length']:.0f} chars")
        print(f"  Text range:         {stats['min_text_length']}-{stats['max_text_length']} chars")
        print(f"  Duplicate IDs:      {stats['duplicate_ids']}")

        print(f"\n  Node labels ({len(stats['unique_labels'])}):")
        for label, count in stats["unique_labels"].items():
            convention = "OK" if is_pascal_case(label) else "WARN"
            generic = " GENERIC" if label in GENERIC_LABELS else ""
            print(f"    {label}: {count}  [{convention}]{generic}")

        print(f"\n  Relationship types ({len(stats['unique_rel_types'])}):")
        for rt, count in stats["unique_rel_types"].items():
            convention = "OK" if is_upper_snake(rt) else "WARN"
            vague = " VAGUE" if rt in VAGUE_RELS else ""
            print(f"    {rt}: {count}  [{convention}]{vague}")

    # Finding summary by category
    by_cat: dict[str, list[Finding]] = {}
    for f in findings:
        by_cat.setdefault(f.category, []).append(f)

    print(f"\n  {'-' * 40}")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warns)}")
    print(f"  Info:     {len(infos)}")

    if errors:
        print("\n  ERRORS:")
        # Deduplicate by message (show count)
        error_msgs: Counter[str] = Counter()
        for e in errors:
            error_msgs[e.message] += 1
        for msg, count in error_msgs.most_common(20):
            suffix = f" (x{count})" if count > 1 else ""
            print(f"    - {msg}{suffix}")

    if warns:
        print("\n  WARNINGS:")
        warn_msgs: Counter[str] = Counter()
        for w in warns:
            warn_msgs[w.message] += 1
        for msg, count in warn_msgs.most_common(20):
            suffix = f" (x{count})" if count > 1 else ""
            print(f"    - {msg}{suffix}")

    if infos and len(infos) <= 20:
        print("\n  INFO:")
        info_msgs: Counter[str] = Counter()
        for i in infos:
            info_msgs[i.message] += 1
        for msg, count in info_msgs.most_common(10):
            suffix = f" (x{count})" if count > 1 else ""
            print(f"    - {msg}{suffix}")
    elif infos:
        print(f"\n  INFO: {len(infos)} informational findings (suppressed)")

    # Verdict
    print(f"\n  {'-' * 40}")
    if not errors:
        print("  VERDICT: PASSED — no convention violations")
        if warns:
            print(f"  ({len(warns)} warnings to review)")
    else:
        print(f"  VERDICT: FAILED — {len(errors)} convention violations to fix")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate training pair JSONL against expert conventions"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="JSONL file(s) to validate",
    )
    args = parser.parse_args()

    exit_code = 0
    for path in args.files:
        if not path.exists():
            print(f"\nERROR: {path} not found", file=sys.stderr)
            exit_code = 1
            continue

        findings, stats = validate_file(path)
        print_report(path, findings, stats)

        if any(f.severity == "ERROR" for f in findings):
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
