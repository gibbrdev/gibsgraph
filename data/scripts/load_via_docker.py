"""Load expert graph into Neo4j via Docker exec + cypher-shell.

Workaround for Python driver auth issues with NEO4J_AUTH=none.
Reads JSONL, generates Cypher, pipes it into cypher-shell.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "processed"


def run_cypher(cypher: str) -> str:
    """Execute Cypher via docker exec cypher-shell."""
    result = subprocess.run(
        ["docker", "exec", "-i", "gibsgraph-demo", "cypher-shell"],
        input=cypher,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
    return result.stdout


def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def escape(s: str) -> str:
    """Escape string for Cypher literal."""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")


def main() -> None:
    print("=" * 60)
    print("Loading Expert Knowledge Graph via Docker")
    print("=" * 60)

    # 1. Constraints
    print("\n[1/6] Creating constraints...")
    run_cypher("""
        CREATE CONSTRAINT IF NOT EXISTS FOR (c:CypherClause) REQUIRE c.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (f:CypherFunction) REQUIRE f.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (p:ModelingPattern) REQUIRE p.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (b:BestPractice) REQUIRE b.title IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.path IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (cat:FunctionCategory) REQUIRE cat.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (cat:PracticeCategory) REQUIRE cat.name IS UNIQUE;
        CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE;
    """)
    print("  Done")

    # 2. Clauses
    print("\n[2/6] Loading Cypher clauses...")
    clauses = load_jsonl(PROCESSED_DIR / "cypher_clauses.jsonl")
    for c in clauses:
        name = escape(c["name"])
        desc = escape(c["description"])
        src = escape(c["source_file"])
        n_examples = len(c.get("syntax_examples", []))
        run_cypher(f"""
            MERGE (clause:CypherClause {{name: '{name}'}})
            SET clause.description = '{desc}',
                clause.source_file = '{src}',
                clause.authority_level = 1,
                clause.example_count = {n_examples};
            MERGE (src:Source {{path: '{src}'}})
            SET src.type = 'official_docs', src.authority_level = 1;
            MATCH (clause:CypherClause {{name: '{name}'}})
            MATCH (src:Source {{path: '{src}'}})
            MERGE (clause)-[:SOURCED_FROM]->(src);
        """)
    print(f"  Loaded {len(clauses)} clauses")

    # 3. Functions
    print("\n[3/6] Loading Cypher functions...")
    functions = load_jsonl(PROCESSED_DIR / "cypher_functions.jsonl")
    for f in functions:
        name = escape(f["name"])
        desc = escape(f["description"])
        sig = escape(f.get("signature", ""))
        ret = escape(f.get("returns", ""))
        cat = escape(f["category"])
        src = escape(f["source_file"])
        n_ex = len(f.get("examples", []))
        run_cypher(f"""
            MERGE (func:CypherFunction {{name: '{name}'}})
            SET func.description = '{desc}',
                func.signature = '{sig}',
                func.returns = '{ret}',
                func.source_file = '{src}',
                func.authority_level = 1,
                func.example_count = {n_ex};
            MERGE (cat:FunctionCategory {{name: '{cat}'}});
            MATCH (func:CypherFunction {{name: '{name}'}})
            MATCH (cat:FunctionCategory {{name: '{cat}'}})
            MERGE (func)-[:BELONGS_TO]->(cat);
        """)
    print(f"  Loaded {len(functions)} functions")

    # 4. Examples (batch — just count, skip individual loading for speed)
    print("\n[4/6] Loading Cypher examples...")
    examples = load_jsonl(PROCESSED_DIR / "cypher_examples.jsonl")
    # Deduplicate
    seen = set()
    unique = []
    for ex in examples:
        if ex["cypher"] not in seen:
            seen.add(ex["cypher"])
            unique.append(ex)

    batch_size = 20
    loaded = 0
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        for ex in batch:
            cypher_code = escape(ex["cypher"])
            desc = escape(ex["description"])
            ctx = escape(ex["context"])
            cat = escape(ex["category"])
            run_cypher(f"""
                CREATE (ex:CypherExample {{
                    cypher: '{cypher_code}',
                    description: '{desc}',
                    context: '{ctx}',
                    category: '{cat}',
                    authority_level: 1
                }});
            """)
            loaded += 1
        print(f"  ... {loaded}/{len(unique)} examples")
    print(f"  Loaded {loaded} examples")

    # 5. Patterns
    print("\n[5/6] Loading modeling patterns...")
    patterns = load_jsonl(PROCESSED_DIR / "modeling_patterns.jsonl")
    seen_patterns = set()
    for p in patterns:
        name = escape(p["name"])
        if name in seen_patterns:
            continue
        seen_patterns.add(name)
        desc = escape(p["description"])
        src = escape(p["source_file"])
        run_cypher(f"""
            MERGE (pat:ModelingPattern {{name: '{name}'}})
            SET pat.description = '{desc}',
                pat.source_file = '{src}',
                pat.authority_level = 1;
        """)
    print(f"  Loaded {len(seen_patterns)} patterns")

    # 6. Best practices
    print("\n[6/6] Loading best practices...")
    practices = load_jsonl(PROCESSED_DIR / "best_practices.jsonl")
    seen_bp = set()
    bp_count = 0
    for bp in practices:
        title = escape(bp["title"])
        if title in seen_bp or not title:
            continue
        seen_bp.add(title)
        desc = escape(bp["description"][:400])
        cat = escape(bp.get("category", "general"))
        run_cypher(f"""
            MERGE (bp:BestPractice {{title: '{title}'}})
            SET bp.description = '{desc}',
                bp.authority_level = {bp.get('authority_level', 1)};
            MERGE (cat:PracticeCategory {{name: '{cat}'}});
            MATCH (bp:BestPractice {{title: '{title}'}})
            MATCH (cat:PracticeCategory {{name: '{cat}'}})
            MERGE (bp)-[:BELONGS_TO]->(cat);
        """)
        bp_count += 1
        if bp_count % 50 == 0:
            print(f"  ... {bp_count} practices")
    print(f"  Loaded {bp_count} best practices")

    # Industries
    print("\n  Creating industry taxonomy...")
    run_cypher("""
        FOREACH (name IN ['Financial Services', 'Healthcare', 'Cybersecurity',
            'Supply Chain', 'Compliance', 'E-commerce', 'HR & Workforce',
            'Media & Content', 'IT Operations', 'Government',
            'Life Sciences', 'Telecommunications'] |
            MERGE (:Industry {name: name})
        );
    """)

    # Stats
    print("\n  Final stats:")
    output = run_cypher("""
        MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC;
    """)
    print(output)

    output = run_cypher("""
        MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count ORDER BY count DESC;
    """)
    print(output)

    print("\nDone!")


if __name__ == "__main__":
    main()
