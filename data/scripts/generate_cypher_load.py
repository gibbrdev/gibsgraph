"""Generate a single Cypher script to load all expert data into Neo4j.

Output: /tmp/load_expert.cypher — pipe directly into cypher-shell.
"""

from __future__ import annotations

import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "processed"
OUTPUT = Path("/tmp/load_expert.cypher")


def load_jsonl(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def esc(s: str) -> str:
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ").replace("\r", "")


def main() -> None:
    lines: list[str] = []

    # Constraints
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (c:CypherClause) REQUIRE c.name IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (f:CypherFunction) REQUIRE f.name IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (p:ModelingPattern) REQUIRE p.name IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (b:BestPractice) REQUIRE b.title IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.path IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (cat:FunctionCategory) REQUIRE cat.name IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (cat:PracticeCategory) REQUIRE cat.name IS UNIQUE;")
    lines.append("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE;")

    # Clauses
    clauses = load_jsonl(PROCESSED_DIR / "cypher_clauses.jsonl")
    for c in clauses:
        name = esc(c["name"])
        desc = esc(c["description"])
        src = esc(c["source_file"])
        n_ex = len(c.get("syntax_examples", []))
        lines.append(
            f"MERGE (c:CypherClause {{name: '{name}'}}) "
            f"SET c.description = '{desc}', c.source_file = '{src}', "
            f"c.authority_level = 1, c.example_count = {n_ex};"
        )
    print(f"Clauses: {len(clauses)}")

    # Functions
    functions = load_jsonl(PROCESSED_DIR / "cypher_functions.jsonl")
    for f in functions:
        name = esc(f["name"])
        desc = esc(f["description"])
        sig = esc(f.get("signature", ""))
        ret = esc(f.get("returns", ""))
        cat = esc(f["category"])
        n_ex = len(f.get("examples", []))
        lines.append(
            f"MERGE (f:CypherFunction {{name: '{name}'}}) "
            f"SET f.description = '{desc}', f.signature = '{sig}', "
            f"f.returns = '{ret}', f.authority_level = 1, f.example_count = {n_ex};"
        )
        lines.append(f"MERGE (:FunctionCategory {{name: '{cat}'}});")
        lines.append(
            f"MATCH (f:CypherFunction {{name: '{name}'}}) "
            f"MATCH (cat:FunctionCategory {{name: '{cat}'}}) "
            f"MERGE (f)-[:BELONGS_TO]->(cat);"
        )
    print(f"Functions: {len(functions)}")

    # Examples (deduplicated, limited to 200 for speed)
    examples = load_jsonl(PROCESSED_DIR / "cypher_examples.jsonl")
    seen = set()
    ex_count = 0
    for ex in examples:
        cypher = ex["cypher"]
        if cypher in seen or ex_count >= 200:
            continue
        seen.add(cypher)
        cypher_esc = esc(cypher)
        desc = esc(ex["description"][:200])
        ctx = esc(ex["context"])
        cat = esc(ex["category"])
        lines.append(
            f"CREATE (:CypherExample {{cypher: '{cypher_esc}', "
            f"description: '{desc}', context: '{ctx}', "
            f"category: '{cat}', authority_level: 1}});"
        )
        ex_count += 1
    print(f"Examples: {ex_count}")

    # Patterns
    patterns = load_jsonl(PROCESSED_DIR / "modeling_patterns.jsonl")
    seen_p = set()
    p_count = 0
    for p in patterns:
        name = esc(p["name"])
        if name in seen_p:
            continue
        seen_p.add(name)
        desc = esc(p["description"][:300])
        lines.append(
            f"MERGE (:ModelingPattern {{name: '{name}', description: '{desc}', authority_level: 1}});"
        )
        p_count += 1
    print(f"Patterns: {p_count}")

    # Best practices
    practices = load_jsonl(PROCESSED_DIR / "best_practices.jsonl")
    seen_bp = set()
    bp_count = 0
    for bp in practices:
        title = esc(bp["title"])
        if title in seen_bp or not title:
            continue
        seen_bp.add(title)
        desc = esc(bp["description"][:300])
        cat = esc(bp.get("category", "general"))
        lines.append(
            f"MERGE (bp:BestPractice {{title: '{title}'}}) "
            f"SET bp.description = '{desc}', bp.authority_level = 1;"
        )
        lines.append(f"MERGE (:PracticeCategory {{name: '{cat}'}});")
        lines.append(
            f"MATCH (bp:BestPractice {{title: '{title}'}}) "
            f"MATCH (cat:PracticeCategory {{name: '{cat}'}}) "
            f"MERGE (bp)-[:BELONGS_TO]->(cat);"
        )
        bp_count += 1
    print(f"Best practices: {bp_count}")

    # Industries
    industries = [
        "Financial Services", "Healthcare", "Cybersecurity",
        "Supply Chain", "Compliance", "E-commerce",
        "HR & Workforce", "Media & Content", "IT Operations",
        "Government", "Life Sciences", "Telecommunications",
    ]
    for ind in industries:
        lines.append(f"MERGE (:Industry {{name: '{ind}'}});")
    print(f"Industries: {len(industries)}")

    # Write
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nGenerated {len(lines)} Cypher statements -> {OUTPUT}")


if __name__ == "__main__":
    main()
