"""Load processed JSONL data into Neo4j as the expert knowledge graph.

Reads from data/processed/*.jsonl and creates:
  - (:CypherClause) nodes with examples
  - (:CypherFunction) nodes with signatures
  - (:CypherExample) nodes linked to their context
  - (:ModelingPattern) nodes with labels and relationships
  - (:BestPractice) nodes categorized by type
  - (:Source) nodes for provenance tracking
  - (:Industry) and (:Domain) taxonomy nodes

Usage:
  python data/scripts/load_expert_graph.py [--uri bolt://localhost:7687] [--password ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase

PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def load_jsonl(filepath: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    if not filepath.exists():
        print(f"  SKIP: {filepath.name} not found")
        return []
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def create_constraints_and_indexes(session) -> None:
    """Create uniqueness constraints and performance indexes for the expert graph."""
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CypherClause) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:CypherFunction) REQUIRE f.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:ModelingPattern) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BestPractice) REQUIRE b.title IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.path IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:FunctionCategory) REQUIRE cat.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:PracticeCategory) REQUIRE cat.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
    ]
    for c in constraints:
        session.run(c)
    print("  Constraints created")

    # Performance indexes for real query patterns:
    # - ExpertStore searches examples by category (clause vs function)
    # - Retriever filters by authority_level for confidence-weighted results
    # - Source.type enables filtering official_docs vs knowledge_base
    indexes = [
        "CREATE INDEX IF NOT EXISTS FOR (ex:CypherExample) ON (ex.category)",
        "CREATE INDEX IF NOT EXISTS FOR (bp:BestPractice) ON (bp.authority_level)",
        "CREATE INDEX IF NOT EXISTS FOR (s:Source) ON (s.type)",
    ]
    for idx in indexes:
        session.run(idx)
    print("  Indexes created")


def load_clauses(session, clauses: list[dict]) -> int:
    """Load Cypher clauses into the expert graph."""
    result = session.run("""
        UNWIND $clauses AS c
        MERGE (clause:CypherClause {name: c.name})
        SET clause.description = c.description,
            clause.source_file = c.source_file,
            clause.authority_level = c.authority_level,
            clause.example_count = size(c.syntax_examples),
            clause.section_count = size(c.sections)
        MERGE (src:Source {path: c.source_file})
        SET src.type = 'official_docs',
            src.authority_level = 1
        MERGE (clause)-[:SOURCED_FROM]->(src)
        RETURN count(clause) AS cnt
    """, clauses=clauses)
    return result.single()["cnt"]


def load_functions(session, functions: list[dict]) -> int:
    """Load Cypher functions into the expert graph."""
    result = session.run("""
        UNWIND $functions AS f
        MERGE (func:CypherFunction {name: f.name})
        SET func.description = f.description,
            func.signature = f.signature,
            func.returns = f.returns,
            func.source_file = f.source_file,
            func.authority_level = f.authority_level,
            func.example_count = size(f.examples)
        MERGE (cat:FunctionCategory {name: f.category})
        MERGE (func)-[:BELONGS_TO]->(cat)
        MERGE (src:Source {path: f.source_file})
        SET src.type = 'official_docs',
            src.authority_level = 1
        MERGE (func)-[:SOURCED_FROM]->(src)
        RETURN count(func) AS cnt
    """, functions=functions)
    return result.single()["cnt"]


def load_examples(session, examples: list[dict]) -> int:
    """Load Cypher examples, linking to their clause/function context.

    Bug fix: The original CALL { WITH x WITH x WHERE ... } subquery pattern
    silently produced zero matches in some Neo4j versions. Using direct
    conditional MATCH instead.
    """
    # Deduplicate by cypher text
    seen = set()
    unique = []
    for ex in examples:
        if ex["cypher"] not in seen:
            seen.add(ex["cypher"])
            unique.append(ex)

    # Create example nodes first
    session.run("""
        UNWIND $examples AS e
        CREATE (ex:CypherExample {
            cypher: e.cypher,
            description: e.description,
            context: e.context,
            category: e.category,
            source_file: e.source_file,
            authority_level: e.authority_level
        })
    """, examples=unique)

    # Link clause examples via DEMONSTRATES
    clause_result = session.run("""
        MATCH (ex:CypherExample)
        WHERE ex.category = 'clause'
        WITH ex
        MATCH (clause:CypherClause {name: ex.context})
        MERGE (ex)-[:DEMONSTRATES]->(clause)
        RETURN count(*) AS cnt
    """)
    clause_cnt = clause_result.single()["cnt"]
    print(f"    Linked {clause_cnt} clause examples via DEMONSTRATES")

    # Link function examples via DEMONSTRATES
    func_result = session.run("""
        MATCH (ex:CypherExample)
        WHERE ex.category = 'function'
        WITH ex
        MATCH (func:CypherFunction {name: ex.context})
        MERGE (ex)-[:DEMONSTRATES]->(func)
        RETURN count(*) AS cnt
    """)
    func_cnt = func_result.single()["cnt"]
    print(f"    Linked {func_cnt} function examples via DEMONSTRATES")

    # Return total example count
    result = session.run("MATCH (ex:CypherExample) RETURN count(ex) AS cnt")
    return result.single()["cnt"]


def load_patterns(session, patterns: list[dict]) -> int:
    """Load modeling patterns into the expert graph.

    Bug fix: source_file can be None in parsed data. Using COALESCE to
    default to 'neo4j-modeling-docs' instead of failing silently on
    MERGE with a null key.
    """
    result = session.run("""
        UNWIND $patterns AS p
        MERGE (pat:ModelingPattern {name: p.name})
        SET pat.description = p.description,
            pat.when_to_use = p.when_to_use,
            pat.anti_pattern = p.anti_pattern,
            pat.node_labels = p.node_labels,
            pat.relationship_types = p.relationship_types,
            pat.source_file = p.source_file,
            pat.authority_level = p.authority_level,
            pat.example_count = size(p.cypher_examples)
        WITH pat, COALESCE(p.source_file, 'neo4j-modeling-docs') AS src_path,
             p.authority_level AS auth
        MERGE (src:Source {path: src_path})
        SET src.type = 'official_docs',
            src.authority_level = COALESCE(auth, 1)
        MERGE (pat)-[:SOURCED_FROM]->(src)
        RETURN count(pat) AS cnt
    """, patterns=patterns)
    return result.single()["cnt"]


def load_practices(session, practices: list[dict]) -> int:
    """Load best practices into the expert graph.

    Bug fix: source_file can be None in parsed data. Using COALESCE to
    default to 'neo4j-knowledge-base' instead of failing silently on
    MERGE with a null key.
    """
    # Deduplicate by title
    seen = set()
    unique = []
    for p in practices:
        if p["title"] not in seen:
            seen.add(p["title"])
            unique.append(p)

    result = session.run("""
        UNWIND $practices AS p
        MERGE (bp:BestPractice {title: p.title})
        SET bp.description = p.description,
            bp.source_file = p.source_file,
            bp.authority_level = p.authority_level,
            bp.example_count = size(p.cypher_examples)
        MERGE (cat:PracticeCategory {name: p.category})
        MERGE (bp)-[:BELONGS_TO]->(cat)
        WITH bp, p,
             COALESCE(p.source_file, 'neo4j-knowledge-base') AS src_path
        MERGE (src:Source {path: src_path})
        SET src.type = CASE
            WHEN src_path CONTAINS 'knowledge-base' THEN 'knowledge_base'
            ELSE 'official_docs'
            END,
            src.authority_level = COALESCE(p.authority_level, 1)
        MERGE (bp)-[:SOURCED_FROM]->(src)
        RETURN count(bp) AS cnt
    """, practices=unique)
    return result.single()["cnt"]


def create_industry_nodes(session) -> int:
    """Create industry taxonomy nodes and link patterns."""
    industries = [
        "Financial Services", "Healthcare", "Cybersecurity",
        "Supply Chain", "Compliance", "E-commerce",
        "HR & Workforce", "Media & Content", "IT Operations",
        "Government", "Life Sciences", "Telecommunications",
    ]
    session.run("""
        UNWIND $industries AS name
        MERGE (:Industry {name: name})
    """, industries=industries)
    print(f"  Created {len(industries)} industry nodes")
    return len(industries)


def print_stats(session) -> None:
    """Print expert graph statistics."""
    result = session.run("""
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    print("\n  Expert Graph Stats:")
    total = 0
    for r in result:
        print(f"    {r['label']}: {r['count']}")
        total += r["count"]

    result = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY count DESC
    """)
    print("\n  Relationships:")
    rel_total = 0
    for r in result:
        print(f"    {r['type']}: {r['count']}")
        rel_total += r["count"]

    print(f"\n  Total: {total} nodes, {rel_total} relationships")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load expert graph into Neo4j")
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--username", default="neo4j")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--clear", action="store_true", help="Clear existing expert data first")
    args = parser.parse_args()

    print("=" * 60)
    print("Loading Expert Knowledge Graph into Neo4j")
    print("=" * 60)

    # Connect
    auth = (args.username, args.password) if args.password else None
    driver = GraphDatabase.driver(args.uri, auth=auth)

    with driver.session(database=args.database) as session:
        if args.clear:
            print("\n  Clearing existing expert data...")
            session.run("MATCH (n) WHERE n:CypherClause OR n:CypherFunction OR n:CypherExample "
                        "OR n:ModelingPattern OR n:BestPractice OR n:Source "
                        "OR n:FunctionCategory OR n:PracticeCategory OR n:Industry "
                        "DETACH DELETE n")

        print("\n[1/6] Creating constraints and indexes...")
        create_constraints_and_indexes(session)

        print("\n[2/6] Loading Cypher clauses...")
        clauses = load_jsonl(PROCESSED_DIR / "cypher_clauses.jsonl")
        if clauses:
            cnt = load_clauses(session, clauses)
            print(f"  Loaded {cnt} clauses")

        print("\n[3/6] Loading Cypher functions...")
        functions = load_jsonl(PROCESSED_DIR / "cypher_functions.jsonl")
        if functions:
            cnt = load_functions(session, functions)
            print(f"  Loaded {cnt} functions")

        print("\n[4/6] Loading Cypher examples...")
        examples = load_jsonl(PROCESSED_DIR / "cypher_examples.jsonl")
        if examples:
            cnt = load_examples(session, examples)
            print(f"  Loaded {cnt} examples")

        print("\n[5/6] Loading modeling patterns & best practices...")
        patterns = load_jsonl(PROCESSED_DIR / "modeling_patterns.jsonl")
        if patterns:
            cnt = load_patterns(session, patterns)
            print(f"  Loaded {cnt} patterns")

        practices = load_jsonl(PROCESSED_DIR / "best_practices.jsonl")
        if practices:
            cnt = load_practices(session, practices)
            print(f"  Loaded {cnt} practices")

        print("\n[6/6] Creating industry taxonomy...")
        create_industry_nodes(session)

        print_stats(session)

    driver.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
