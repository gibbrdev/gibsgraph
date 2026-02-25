"""Load pre-computed embeddings into Neo4j and create vector indexes.

Reads embeddings.npz + embeddings_meta.jsonl from data/processed/ and:
  1. Matches each embedding to its corresponding Neo4j node
  2. Sets the `embedding` property (float[] 384-dim)
  3. Adds an :Expert label to all embedded nodes
  4. Creates a vector index for semantic search

Usage:
  python data/scripts/load_embeddings.py [--uri bolt://localhost:7687] [--password ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
VECTOR_DIMS = 384
VECTOR_INDEX_NAME = "expert_embedding"


def load_meta() -> list[dict]:
    """Load embedding metadata."""
    path = PROCESSED_DIR / "embeddings_meta.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def set_embeddings_by_name(
    session, label: str, key: str, name_to_vec: dict[str, list[float]]
) -> int:
    """Set embedding property on nodes matched by a name/title field."""
    result = session.run(
        f"""
        UNWIND $entries AS entry
        MATCH (n:{label} {{{key}: entry.name}})
        SET n.embedding = entry.vec, n:Expert
        RETURN count(n) AS cnt
        """,
        entries=[{"name": n, "vec": v} for n, v in name_to_vec.items()],
    )
    return result.single()["cnt"]


def normalize_ws(s: str) -> str:
    """Collapse all whitespace to single spaces for comparison."""
    return " ".join(s.split())


def set_example_embeddings(
    session, examples: list[dict], embeddings: np.ndarray
) -> tuple[int, int]:
    """Match CypherExample nodes by cypher text extracted from embedding text.

    Matching is done in Python with whitespace normalization, then written
    back to Neo4j by elementId.
    """
    # Build a lookup: normalized cypher → embedding vector
    marker = " Cypher: "
    cypher_to_vec: dict[str, list[float]] = {}
    for ex in examples:
        text = ex["text"]
        idx = text.find(marker)
        if idx == -1:
            continue
        cypher = text[idx + len(marker) :]
        key = normalize_ws(cypher)
        vec = embeddings[ex["embedding_index"]].tolist()
        cypher_to_vec[key] = vec

    # Fetch all CypherExample nodes from Neo4j
    result = session.run(
        "MATCH (e:CypherExample) RETURN elementId(e) AS eid, e.cypher AS cypher"
    )
    matches: list[dict[str, object]] = []
    for r in result:
        key = normalize_ws(r["cypher"])
        if key in cypher_to_vec:
            matches.append({"eid": r["eid"], "vec": cypher_to_vec[key]})

    # Write embeddings back by elementId
    chunk_size = 50
    for i in range(0, len(matches), chunk_size):
        chunk = matches[i : i + chunk_size]
        session.run(
            """
            UNWIND $entries AS entry
            MATCH (n) WHERE elementId(n) = entry.eid
            SET n.embedding = entry.vec, n:Expert
            """,
            entries=chunk,
        )

    return len(matches), len(cypher_to_vec)


def create_vector_index(session) -> None:
    """Create a vector index on :Expert(embedding) for semantic search."""
    session.run(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (n:Expert) ON (n.embedding)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )


def create_fulltext_index(session) -> None:
    """Create a fulltext index for keyword search across expert nodes."""
    session.run(
        """
        CREATE FULLTEXT INDEX expert_fulltext IF NOT EXISTS
        FOR (n:CypherClause|CypherFunction|CypherExample|ModelingPattern|BestPractice)
        ON EACH [n.name, n.description, n.title, n.cypher, n.signature]
        """
    )


def print_results(session) -> None:
    """Print final stats."""
    result = session.run(
        """
        MATCH (n:Expert)
        WHERE n.embedding IS NOT NULL
        RETURN labels(n)[0] AS label, count(n) AS cnt
        ORDER BY cnt DESC
        """
    )
    print("\n  Nodes with embeddings:")
    total = 0
    for r in result:
        lbl = r["label"] if r["label"] != "Expert" else "(Expert only)"
        print(f"    {lbl}: {r['cnt']}")
        total += r["cnt"]
    print(f"    TOTAL: {total}")

    result = session.run(
        """
        SHOW INDEXES YIELD name, type, state
        WHERE type = 'VECTOR'
        RETURN name, state
        """
    )
    print("\n  Vector indexes:")
    for r in result:
        print(f"    {r['name']}: {r['state']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load embeddings into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="neo4j")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading Embeddings into Neo4j Expert Graph")
    print("=" * 60)

    # Load files
    print("\n[1/5] Loading embedding files...")
    embeddings = np.load(PROCESSED_DIR / "embeddings.npz")["embeddings"]
    meta = load_meta()
    print(f"  {len(meta)} vectors x {embeddings.shape[1]} dims")

    # Group by type
    by_type: dict[str, list[dict]] = {}
    for m in meta:
        by_type.setdefault(m["type"], []).append(m)
    for t, items in sorted(by_type.items()):
        print(f"  {t}: {len(items)}")

    # Connect
    auth = (args.username, args.password) if args.password else None
    driver = GraphDatabase.driver(args.uri, auth=auth)

    with driver.session(database=args.database) as session:
        # Clauses — match by name
        print("\n[2/5] Loading clause embeddings...")
        clauses = {
            m["name"]: embeddings[m["embedding_index"]].tolist()
            for m in by_type.get("cypher_clause", [])
        }
        cnt = set_embeddings_by_name(session, "CypherClause", "name", clauses)
        print(f"  Matched {cnt}/{len(clauses)}")

        # Functions — match by name
        print("  Loading function embeddings...")
        functions = {
            m["name"]: embeddings[m["embedding_index"]].tolist()
            for m in by_type.get("cypher_function", [])
        }
        cnt = set_embeddings_by_name(session, "CypherFunction", "name", functions)
        print(f"  Matched {cnt}/{len(functions)}")

        # Patterns — match by name
        print("  Loading pattern embeddings...")
        patterns = {
            m["name"]: embeddings[m["embedding_index"]].tolist()
            for m in by_type.get("modeling_pattern", [])
        }
        cnt = set_embeddings_by_name(session, "ModelingPattern", "name", patterns)
        print(f"  Matched {cnt}/{len(patterns)}")

        # Best practices — match by title (name in meta = title in Neo4j)
        print("  Loading best practice embeddings...")
        practices = {
            m["name"]: embeddings[m["embedding_index"]].tolist()
            for m in by_type.get("best_practice", [])
        }
        cnt = set_embeddings_by_name(session, "BestPractice", "title", practices)
        print(f"  Matched {cnt}/{len(practices)}")

        # Examples — match by cypher text
        print("\n[3/5] Loading example embeddings...")
        matched, attempted = set_example_embeddings(
            session, by_type.get("cypher_example", []), embeddings
        )
        print(f"  Matched {matched}/{attempted}")

        # Create indexes
        print("\n[4/6] Creating vector index...")
        create_vector_index(session)
        print(f"  Index '{VECTOR_INDEX_NAME}' created")

        print("\n[5/6] Creating fulltext index...")
        create_fulltext_index(session)
        print("  Index 'expert_fulltext' created")

        # Stats
        print("\n[6/6] Results")
        print_results(session)

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
