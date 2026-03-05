"""Add grounded records for index, constraint, subquery, and vector search gaps.

All Cypher syntax verified against official Neo4j docs:
  - Indexes: https://neo4j.com/docs/cypher-manual/current/indexes/syntax/
  - Constraints: https://neo4j.com/docs/cypher-manual/current/constraints/syntax/
  - Subqueries: https://neo4j.com/docs/cypher-manual/current/subqueries/
  - Vector: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/

Usage:
  python data/scripts/add_grounded_records.py
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1].parent / "src" / "gibsgraph" / "data"

# ── New cypher_examples ──────────────────────────────────────────────────────

NEW_EXAMPLES: list[dict[str, object]] = [
    # RANGE indexes
    {
        "cypher": "CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)",
        "description": "Create a range index on a single node property for fast exact-match lookups.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE INDEX idx_person_name_born FOR (n:Person) ON (n.name, n.born)",
        "description": "Create a composite range index on multiple node properties. Useful when queries filter on both properties together.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE INDEX idx_acted_role FOR ()-[r:ACTED_IN]-() ON (r.role)",
        "description": "Create a range index on a relationship property.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE INDEX idx_person_email IF NOT EXISTS FOR (n:Person) ON (n.email)",
        "description": "Create a range index only if it does not already exist. Idempotent, safe to run multiple times.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # TEXT index
    {
        "cypher": "CREATE TEXT INDEX idx_movie_title FOR (n:Movie) ON (n.title)",
        "description": "Create a text index for string property lookups. Optimized for STARTS WITH, ENDS WITH, and CONTAINS predicates.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE TEXT INDEX idx_rel_comment FOR ()-[r:REVIEWED]-() ON (r.comment)",
        "description": "Create a text index on a relationship string property.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # POINT index
    {
        "cypher": "CREATE POINT INDEX idx_location FOR (n:Location) ON (n.coordinates)",
        "description": "Create a point index for spatial queries using point.distance() and point.withinBBox().",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # FULLTEXT index
    {
        "cypher": "CREATE FULLTEXT INDEX idx_movie_search FOR (n:Movie) ON EACH [n.title, n.description]",
        "description": "Create a fulltext index across multiple properties for approximate text search. Queried with db.index.fulltext.queryNodes().",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE FULLTEXT INDEX idx_review_text FOR ()-[r:REVIEWED]-() ON EACH [r.summary, r.body]",
        "description": "Create a fulltext index on relationship properties. Queried with db.index.fulltext.queryRelationships().",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # VECTOR index
    {
        "cypher": (
            "CREATE VECTOR INDEX idx_doc_embeddings FOR (n:Document) ON (n.embedding)\n"
            "OPTIONS {indexConfig: {\n"
            "  `vector.similarity_function`: 'cosine',\n"
            "  `vector.dimensions`: 1536\n"
            "}}"
        ),
        "description": "Create a vector index for semantic similarity search. Requires specifying dimensions and similarity function (cosine or euclidean).",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "CREATE VECTOR INDEX idx_chunk_embed IF NOT EXISTS\n"
            "FOR (n:Chunk) ON (n.embedding)\n"
            "OPTIONS {indexConfig: {\n"
            "  `vector.similarity_function`: 'euclidean',\n"
            "  `vector.dimensions`: 768\n"
            "}}"
        ),
        "description": "Create a vector index with euclidean distance for RAG chunk retrieval. 768 dimensions matches common sentence-transformer models.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # LOOKUP index
    {
        "cypher": "CREATE LOOKUP INDEX idx_node_labels FOR (n) ON EACH labels(n)",
        "description": "Create a token lookup index on node labels. Speeds up queries that filter by label without property predicates.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE LOOKUP INDEX idx_rel_types FOR ()-[r]-() ON EACH type(r)",
        "description": "Create a token lookup index on relationship types.",
        "context": "index creation",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    # SHOW / DROP indexes
    {
        "cypher": (
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state\n"
            "WHERE state = 'ONLINE'"
        ),
        "description": "List all online indexes with their type, target labels/types, and indexed properties.",
        "context": "index management",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "DROP INDEX idx_person_name IF EXISTS",
        "description": "Drop an index by name. IF EXISTS makes it idempotent.",
        "context": "index management",
        "category": "index",
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },

    # ── Constraints ──────────────────────────────────────────────────────────

    {
        "cypher": "CREATE CONSTRAINT uniq_person_email FOR (n:Person) REQUIRE n.email IS UNIQUE",
        "description": "Create a uniqueness constraint ensuring no two Person nodes share the same email. Also implicitly creates a range index.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "CREATE CONSTRAINT uniq_person_name FOR (n:Person)\n"
            "REQUIRE (n.firstName, n.lastName) IS UNIQUE"
        ),
        "description": "Create a composite uniqueness constraint on multiple properties. The combination must be unique, not each property individually.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE CONSTRAINT uniq_order_id FOR ()-[r:PURCHASED]-() REQUIRE r.orderId IS UNIQUE",
        "description": "Create a uniqueness constraint on a relationship property.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE CONSTRAINT exists_person_name FOR (n:Person) REQUIRE n.name IS NOT NULL",
        "description": "Create an existence constraint ensuring every Person node has a name property. Enterprise Edition only.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE CONSTRAINT exists_acted_role FOR ()-[r:ACTED_IN]-() REQUIRE r.role IS NOT NULL",
        "description": "Create an existence constraint on a relationship property. Every ACTED_IN relationship must have a role.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "CREATE CONSTRAINT type_person_born FOR (n:Person) REQUIRE n.born IS :: INTEGER",
        "description": "Create a property type constraint ensuring the born property is always an integer. Enterprise Edition only.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "CREATE CONSTRAINT key_person FOR (n:Person)\n"
            "REQUIRE (n.firstName, n.lastName) IS NODE KEY"
        ),
        "description": "Create a node key constraint combining existence + uniqueness. All listed properties must exist and their combination must be unique. Enterprise Edition.",
        "context": "constraint creation",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties\n"
            "WHERE type = 'UNIQUENESS'"
        ),
        "description": "List all uniqueness constraints with their target labels and properties.",
        "context": "constraint management",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": "DROP CONSTRAINT uniq_person_email IF EXISTS",
        "description": "Drop a constraint by name. IF EXISTS makes it idempotent.",
        "context": "constraint management",
        "category": "constraint",
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },

    # ── Subquery patterns ────────────────────────────────────────────────────

    {
        "cypher": (
            "MATCH (p:Person)\n"
            "WHERE EXISTS {\n"
            "  MATCH (p)-[:ACTED_IN]->(m:Movie)\n"
            "  WHERE m.rating > 8.0\n"
            "}\n"
            "RETURN p.name"
        ),
        "description": "EXISTS subquery: find persons who acted in at least one movie rated above 8.0. The subquery is a filter, it does not return data.",
        "context": "subquery patterns",
        "category": "subquery",
        "source_file": "modules\\ROOT\\pages\\subqueries\\existential.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "MATCH (p:Person)\n"
            "RETURN p.name,\n"
            "  COUNT {\n"
            "    MATCH (p)-[:DIRECTED]->(m:Movie)\n"
            "  } AS directedCount"
        ),
        "description": "COUNT subquery: count how many movies each person directed without expanding the result set.",
        "context": "subquery patterns",
        "category": "subquery",
        "source_file": "modules\\ROOT\\pages\\subqueries\\count.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "MATCH (p:Person)\n"
            "RETURN p.name,\n"
            "  COLLECT {\n"
            "    MATCH (p)-[:ACTED_IN]->(m:Movie)\n"
            "    RETURN m.title\n"
            "  } AS movies"
        ),
        "description": "COLLECT subquery: collect movie titles per person into a list. Alternative to MATCH + collect() that avoids grouping complexity.",
        "context": "subquery patterns",
        "category": "subquery",
        "source_file": "modules\\ROOT\\pages\\subqueries\\collect.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "MATCH (p:Person)\n"
            "CALL (p) {\n"
            "  MATCH (p)-[:ACTED_IN]->(m:Movie)\n"
            "  RETURN m ORDER BY m.rating DESC LIMIT 3\n"
            "}\n"
            "RETURN p.name, m.title, m.rating"
        ),
        "description": "CALL subquery with variable scope clause: find each person's top 3 rated movies. The (p) imports the variable into the subquery scope.",
        "context": "subquery patterns",
        "category": "subquery",
        "source_file": "modules\\ROOT\\pages\\subqueries\\call-subquery.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS row\n"
            "CALL (row) {\n"
            "  MERGE (p:Person {id: row.id})\n"
            "  SET p.name = row.name\n"
            "} IN TRANSACTIONS OF 1000 ROWS"
        ),
        "description": "CALL subquery IN TRANSACTIONS: batch import CSV data in chunks of 1000 rows. Prevents out-of-memory on large imports.",
        "context": "subquery patterns",
        "category": "subquery",
        "source_file": "modules\\ROOT\\pages\\subqueries\\call-subquery.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },

    # ── Vector search patterns ───────────────────────────────────────────────

    {
        "cypher": (
            "CALL db.index.vector.queryNodes('idx_chunks', 5, $queryEmbedding)\n"
            "YIELD node, score\n"
            "RETURN node.text, score"
        ),
        "description": "Vector similarity search using the procedure API. Returns top 5 most similar chunks with their cosine similarity scores.",
        "context": "vector search",
        "category": "vector",
        "source_file": "modules\\ROOT\\pages\\indexes\\semantic-indexes\\vector-indexes.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "cypher": (
            "CALL db.index.vector.queryNodes('idx_chunks', 10, $queryEmbedding)\n"
            "YIELD node, score\n"
            "WHERE score > 0.8\n"
            "MATCH (node)-[:PART_OF]->(doc:Document)\n"
            "RETURN doc.title, node.text, score\n"
            "ORDER BY score DESC"
        ),
        "description": "Vector search with score threshold and graph traversal. Find similar chunks, filter by minimum similarity, then traverse to parent documents.",
        "context": "vector search",
        "category": "vector",
        "source_file": "modules\\ROOT\\pages\\indexes\\semantic-indexes\\vector-indexes.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
]

# ── New best_practices ───────────────────────────────────────────────────────

NEW_PRACTICES: list[dict[str, object]] = [
    {
        "title": "Index strategy: choose the right index type",
        "description": "Neo4j 5.x offers six index types. RANGE (default) handles most equality and range lookups. TEXT is optimized for STARTS WITH, ENDS WITH, CONTAINS on strings. POINT is for spatial distance queries. FULLTEXT enables approximate text search across multiple properties. VECTOR enables semantic similarity search with embeddings. LOOKUP speeds up label/type-based scans. Always create the most specific index type for your query pattern.",
        "category": "performance",
        "cypher_examples": [
            "CREATE INDEX idx_person_name FOR (n:Person) ON (n.name)",
            "CREATE TEXT INDEX idx_movie_title FOR (n:Movie) ON (n.title)",
            "CREATE POINT INDEX idx_location FOR (n:Location) ON (n.coordinates)",
            "CREATE FULLTEXT INDEX idx_search FOR (n:Article) ON EACH [n.title, n.body]",
            "CREATE VECTOR INDEX idx_embed FOR (n:Chunk) ON (n.embedding)\nOPTIONS {indexConfig: {`vector.similarity_function`: 'cosine', `vector.dimensions`: 1536}}",
        ],
        "source_file": "modules\\ROOT\\pages\\indexes\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "title": "Constraint strategy: enforce data integrity at the database level",
        "description": "Use uniqueness constraints on natural keys (email, SKU, external ID) to prevent duplicate nodes. Use existence constraints (Enterprise) on required properties to catch missing data at write time. Use node key constraints when you need both existence and uniqueness on a composite key. Constraints also create implicit indexes, so you do not need a separate index on constrained properties.",
        "category": "modeling",
        "cypher_examples": [
            "CREATE CONSTRAINT uniq_person_email FOR (n:Person) REQUIRE n.email IS UNIQUE",
            "CREATE CONSTRAINT exists_person_name FOR (n:Person) REQUIRE n.name IS NOT NULL",
            "CREATE CONSTRAINT key_product FOR (n:Product) REQUIRE (n.sku, n.region) IS NODE KEY",
        ],
        "source_file": "modules\\ROOT\\pages\\constraints\\syntax.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "title": "Batch imports with CALL IN TRANSACTIONS",
        "description": "When importing large datasets (100k+ rows), wrap writes in CALL {} IN TRANSACTIONS to commit in batches. This prevents transaction memory from growing unbounded. Typical batch size is 1000-10000 rows. Always use MERGE (not CREATE) when the data might contain duplicates, and back it with a uniqueness constraint for safety.",
        "category": "performance",
        "cypher_examples": [
            "LOAD CSV WITH HEADERS FROM 'file:///data.csv' AS row\nCALL (row) {\n  MERGE (p:Person {id: row.id})\n  SET p.name = row.name, p.email = row.email\n} IN TRANSACTIONS OF 5000 ROWS",
            "UNWIND $batch AS item\nCALL (item) {\n  MERGE (n:Product {sku: item.sku})\n  SET n += item.properties\n} IN TRANSACTIONS OF 1000 ROWS",
        ],
        "source_file": "modules\\ROOT\\pages\\subqueries\\call-subquery.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
    {
        "title": "Vector search for RAG and semantic retrieval",
        "description": "Neo4j vector indexes enable approximate nearest-neighbor search on embeddings stored as node properties. Create a vector index specifying dimensions and similarity function (cosine for normalized embeddings, euclidean for raw). Common pattern: store document chunks as nodes with embedding properties, then retrieve the top-k most similar chunks for a given query embedding.",
        "category": "cypher",
        "cypher_examples": [
            "CREATE VECTOR INDEX idx_chunks FOR (n:Chunk) ON (n.embedding)\nOPTIONS {indexConfig: {`vector.similarity_function`: 'cosine', `vector.dimensions`: 1536}}",
            "CALL db.index.vector.queryNodes('idx_chunks', 5, $queryEmbedding)\nYIELD node, score\nRETURN node.text, score",
        ],
        "source_file": "modules\\ROOT\\pages\\indexes\\semantic-indexes\\vector-indexes.adoc",
        "authority_level": 1,
        "quality_tier": "high",
    },
]


def main() -> None:
    # Append to cypher_examples.jsonl
    ex_path = DATA_DIR / "cypher_examples.jsonl"
    with open(ex_path, "a", encoding="utf-8") as f:
        for rec in NEW_EXAMPLES:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Append to best_practices.jsonl
    bp_path = DATA_DIR / "best_practices.jsonl"
    with open(bp_path, "a", encoding="utf-8") as f:
        for rec in NEW_PRACTICES:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    ex_total = sum(1 for _ in open(ex_path, encoding="utf-8"))
    bp_total = sum(1 for _ in open(bp_path, encoding="utf-8"))

    print(f"Added {len(NEW_EXAMPLES)} cypher examples (total: {ex_total})")
    print(f"Added {len(NEW_PRACTICES)} best practices (total: {bp_total})")


if __name__ == "__main__":
    main()
