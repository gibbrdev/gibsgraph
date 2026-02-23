"""Graph retrieval — auto-adapts to any Neo4j graph.

Strategy:
- If vector index exists → vector search + neighbourhood + optional PCST pruning
- If no vector index → text-to-Cypher using auto-discovered schema
- Both paths return the same RetrievalResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from neo4j import GraphDatabase

from gibsgraph.config import Settings

log = structlog.get_logger(__name__)


@dataclass
class GraphSchema:
    """Auto-discovered schema of a Neo4j graph."""

    labels: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    relationship_patterns: list[str] = field(default_factory=list)
    property_keys: dict[str, list[str]] = field(default_factory=dict)
    sample_values: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    indexes: list[dict[str, str]] = field(default_factory=list)
    node_count: int = 0
    has_vector_index: bool = False
    vector_index_name: str = ""

    def to_prompt(self) -> str:
        """Serialize schema for LLM prompt context."""
        lines = ["Neo4j Graph Schema:"]
        lines.append(f"  Node labels: {', '.join(self.labels)}")
        lines.append("  Relationship patterns (how nodes connect):")
        for pat in self.relationship_patterns:
            lines.append(f"    {pat}")
        for label, props in self.property_keys.items():
            lines.append(f"  :{label} properties: {', '.join(props)}")
            if label in self.sample_values:
                for prop, vals in self.sample_values[label].items():
                    lines.append(f"    sample {prop}: {vals}")
        lines.append(f"  Total nodes: {self.node_count}")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Result from the graph retriever."""

    subgraph: dict[str, Any] = field(default_factory=dict)
    context: str = ""
    cypher: str = ""
    nodes_count: int = 0
    edges_count: int = 0
    strategy: str = ""


class GraphRetriever:
    """Retrieves the most relevant subgraph for a query.

    Auto-detects the best retrieval strategy based on what the graph supports:
    - Vector index present → vector similarity + neighbourhood expansion
    - No vector index → LLM generates Cypher from discovered schema

    Security: ALL Cypher uses $parameters — never f-strings.
    """

    _VECTOR_SEARCH_CYPHER = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    WHERE score > $min_score
    RETURN node, score
    ORDER BY score DESC
    """

    _NEIGHBOURHOOD_CYPHER = """
    MATCH (n)-[r]-(m)
    WHERE elementId(n) IN $node_ids
    RETURN n, r, m
    LIMIT $limit
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
            max_connection_lifetime=settings.neo4j_max_connection_lifetime,
        )
        self._schema: GraphSchema | None = None

    def discover_schema(self) -> GraphSchema:
        """Auto-discover the graph's labels, relationships, properties, and indexes."""
        if self._schema is not None:
            return self._schema

        log.info("retriever.discovering_schema")
        with self._driver.session(database=self.settings.neo4j_database) as session:
            # Labels
            labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]

            # Relationship types
            rel_types = [
                r["relationshipType"]
                for r in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
            ]

            # Node count
            record = session.run("MATCH (n) RETURN count(n) AS c").single()
            node_count: int = record["c"] if record else 0

            # Property keys per label (sample 5 nodes each)
            property_keys: dict[str, list[str]] = {}
            for label in labels:
                records = list(
                    session.run(
                        "MATCH (n) WHERE $label IN labels(n) RETURN keys(n) AS k LIMIT 5",
                        label=label,
                    )
                )
                all_keys: set[str] = set()
                for r in records:
                    all_keys.update(r["k"])
                if all_keys:
                    property_keys[label] = sorted(all_keys)

            # Sample property values — so LLM knows data conventions
            sample_values: dict[str, dict[str, list[str]]] = {}
            for label in labels:
                records = list(
                    session.run(
                        "MATCH (n) WHERE $label IN labels(n) RETURN properties(n) AS p LIMIT 5",
                        label=label,
                    )
                )
                if not records:
                    continue
                label_samples: dict[str, list[str]] = {}
                for r in records:
                    for k, v in r["p"].items():
                        if isinstance(v, str) and len(v) < 80:
                            label_samples.setdefault(k, [])
                            if v not in label_samples[k]:
                                label_samples[k].append(v)
                # Keep only properties with interesting distinct values (< 20)
                sample_values[label] = {
                    k: vs[:5]
                    for k, vs in label_samples.items()
                    if 1 < len(vs) <= 20 or k in ("name", "regulation", "title")
                }

            # Relationship patterns — discover (source)-[:TYPE]->(target)
            rel_patterns: list[str] = []
            pattern_records = list(
                session.run(
                    "MATCH (a)-[r]->(b) "
                    "RETURN DISTINCT labels(a)[0] AS from_l, type(r) AS rel, "
                    "labels(b)[0] AS to_l "
                    "LIMIT 50"
                )
            )
            seen_patterns: set[str] = set()
            for r in pattern_records:
                pat = f"(:{r['from_l']})-[:{r['rel']}]->(:{r['to_l']})"
                if pat not in seen_patterns:
                    seen_patterns.add(pat)
                    rel_patterns.append(pat)

            # Indexes — check for vector indexes
            has_vector = False
            vector_index_name = ""
            indexes: list[dict[str, str]] = []
            try:
                for r in session.run("SHOW INDEXES YIELD name, type, labelsOrTypes, properties"):
                    idx = {
                        "name": r["name"],
                        "type": r["type"],
                        "labels": str(r["labelsOrTypes"] or []),
                        "properties": str(r["properties"] or []),
                    }
                    indexes.append(idx)
                    if r["type"] == "VECTOR":
                        has_vector = True
                        vector_index_name = r["name"]
            except Exception:
                log.debug("retriever.show_indexes_not_supported")

        self._schema = GraphSchema(
            labels=labels,
            relationship_types=rel_types,
            relationship_patterns=rel_patterns,
            property_keys=property_keys,
            sample_values=sample_values,
            indexes=indexes,
            node_count=node_count,
            has_vector_index=has_vector,
            vector_index_name=vector_index_name,
        )
        log.info(
            "retriever.schema_discovered",
            labels=len(labels),
            rels=len(rel_types),
            nodes=node_count,
            has_vector=has_vector,
        )
        return self._schema

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float = 0.75,
        neighbourhood_limit: int = 100,
    ) -> RetrievalResult:
        """Run retrieval — auto-selects strategy based on graph capabilities."""
        log.info("retriever.retrieve", query=query[:80])
        schema = self.discover_schema()

        if schema.has_vector_index:
            result = self._retrieve_vector(
                query,
                schema=schema,
                top_k=top_k,
                min_score=min_score,
                neighbourhood_limit=neighbourhood_limit,
            )
            # Fall back to text-to-Cypher if vector search found nothing
            if result.nodes_count == 0 and result.edges_count == 0:
                log.info("retriever.vector_fallback_to_cypher")
                return self._retrieve_cypher(query, schema=schema)
            return result
        return self._retrieve_cypher(query, schema=schema)

    # ------------------------------------------------------------------
    # Strategy 1: Vector search + neighbourhood
    # ------------------------------------------------------------------

    def _retrieve_vector(
        self,
        query: str,
        *,
        schema: GraphSchema,
        top_k: int,
        min_score: float,
        neighbourhood_limit: int,
    ) -> RetrievalResult:
        """Vector similarity search → neighbourhood expansion."""
        log.info("retriever.strategy_vector")
        embedding = self._embed(query)

        candidate_ids = self._vector_search(
            embedding=embedding,
            index_name=schema.vector_index_name,
            top_k=top_k,
            min_score=min_score,
        )

        if not candidate_ids:
            log.warning("retriever.no_vector_candidates")
            return RetrievalResult(
                context="No relevant nodes found via vector search.",
                strategy="vector",
            )

        subgraph = self._fetch_neighbourhood(node_ids=candidate_ids, limit=neighbourhood_limit)
        context = self._serialize_context(subgraph)

        return RetrievalResult(
            subgraph=subgraph,
            context=context,
            cypher=self._NEIGHBOURHOOD_CYPHER,
            nodes_count=len(subgraph.get("nodes", [])),
            edges_count=len(subgraph.get("edges", [])),
            strategy="vector",
        )

    # ------------------------------------------------------------------
    # Strategy 2: Text-to-Cypher via LLM
    # ------------------------------------------------------------------

    def _retrieve_cypher(
        self, query: str, *, schema: GraphSchema, max_retries: int = 1
    ) -> RetrievalResult:
        """Generate Cypher from natural language using the graph schema."""
        log.info("retriever.strategy_cypher")

        cypher = self._generate_cypher(query, schema=schema)
        if not cypher:
            return RetrievalResult(
                context="Could not generate a query for this question.",
                strategy="cypher",
            )

        # Execute the generated Cypher (read-only), retry on failure
        subgraph, error = self._execute_read_cypher(cypher)

        for attempt in range(max_retries):
            if not error:
                break
            log.info("retriever.cypher_retry", attempt=attempt + 1, error=error[:120])
            cypher = self._generate_cypher(
                query, schema=schema, error=error, previous_cypher=cypher
            )
            if not cypher:
                break
            subgraph, error = self._execute_read_cypher(cypher)

        context = self._serialize_context(subgraph)

        return RetrievalResult(
            subgraph=subgraph,
            context=context,
            cypher=cypher,
            nodes_count=len(subgraph.get("nodes", [])),
            edges_count=len(subgraph.get("edges", [])),
            strategy="cypher",
        )

    _CYPHER_SYSTEM_PROMPT = (
        "You are a Neo4j 5 Cypher expert. Generate a single READ-ONLY Cypher query "
        "to answer the user's question.\n\n"
        "Neo4j 5 syntax rules (IMPORTANT):\n"
        "- Use COUNT {{ pattern }} instead of size(pattern). Example: "
        "COUNT {{ (p)--() }} not size((p)--())\n"
        "- For shortest path: MATCH p = shortestPath((a)-[*]-(b)) RETURN p\n"
        "- For variable-length paths use [*..N] with a bound, e.g. [*..6]\n"
        "- Use elementId(n) not id(n)\n"
        "- String matching: use toLower() or CONTAINS, not regex unless needed\n\n"
        "Rules:\n"
        "- ONLY read queries (MATCH, RETURN, WITH, WHERE, ORDER BY, LIMIT)\n"
        "- NEVER use CREATE, MERGE, SET, DELETE, DETACH, DROP, CALL {{...}}\n"
        "- Use properties from the schema provided\n"
        "- Return useful columns (node properties, counts, paths)\n"
        "- LIMIT results to 25 max\n"
        "- Return ONLY the Cypher query, no explanation, no markdown\n"
    )

    def _generate_cypher(
        self,
        query: str,
        *,
        schema: GraphSchema,
        error: str = "",
        previous_cypher: str = "",
    ) -> str:
        """Use LLM to generate read-only Cypher from a natural language query."""
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=0.0,
            max_retries=self.settings.llm_max_retries,
        )

        prompt = f"{self._CYPHER_SYSTEM_PROMPT}\n{schema.to_prompt()}\n\n"

        if error and previous_cypher:
            prompt += (
                f"Your previous Cypher query failed:\n"
                f"Query: {previous_cypher}\n"
                f"Error: {error}\n\n"
                f"Fix the query to resolve this error.\n\n"
            )

        prompt += f"Question: {query}"

        response = llm.invoke(prompt)
        cypher = str(response.content).strip()

        # Strip markdown code fences if present
        if cypher.startswith("```"):
            lines = cypher.split("\n")
            cypher = "\n".join(line for line in lines if not line.startswith("```")).strip()

        log.info("retriever.cypher_generated", cypher=cypher[:200])
        return cypher

    def _execute_read_cypher(self, cypher: str) -> tuple[dict[str, Any], str]:
        """Execute a read-only Cypher query. Returns (subgraph, error_message)."""
        from gibsgraph.tools.cypher_validator import CypherValidator

        empty: dict[str, Any] = {"nodes": [], "edges": [], "records": []}

        validator = CypherValidator()
        if not validator.validate(cypher):
            log.warning("retriever.cypher_rejected", cypher=cypher[:200])
            return empty, "Cypher rejected by validator (possible write operation)"

        try:
            with self._driver.session(database=self.settings.neo4j_database) as session:
                records = list(session.run(cypher))

            # Convert records to a structured result
            nodes: list[dict[str, Any]] = []
            edges: list[dict[str, Any]] = []
            tabular: list[dict[str, Any]] = []

            for record in records:
                row: dict[str, Any] = {}
                for key in record.keys():
                    val = record[key]
                    # Neo4j Node
                    if hasattr(val, "labels"):
                        node_dict = self._clean_props(dict(val))
                        node_dict["_labels"] = list(val.labels)
                        node_dict["_id"] = val.element_id
                        nodes.append(node_dict)
                        row[key] = node_dict
                    # Neo4j Relationship
                    elif hasattr(val, "type"):
                        edge_dict: dict[str, Any] = {
                            "type": val.type,
                            "start": val.start_node.element_id,
                            "end": val.end_node.element_id,
                            "props": self._clean_props(dict(val)),
                        }
                        edges.append(edge_dict)
                        row[key] = edge_dict
                    else:
                        row[key] = val
                tabular.append(row)

            return {"nodes": nodes, "edges": edges, "records": tabular}, ""

        except Exception as exc:
            log.error("retriever.cypher_execution_failed", error=str(exc))
            return empty, str(exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Embed text using configured embedding model."""
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=self.settings.embedding_model,
            dimensions=self.settings.embedding_dimensions,
        )
        return embeddings.embed_query(text)

    def _vector_search(
        self,
        *,
        embedding: list[float],
        index_name: str,
        top_k: int,
        min_score: float,
    ) -> list[str]:
        """Return elementIds of candidate nodes via vector index."""
        try:
            with self._driver.session(database=self.settings.neo4j_database) as session:
                records = session.run(
                    self._VECTOR_SEARCH_CYPHER,
                    index_name=index_name,
                    top_k=top_k,
                    embedding=embedding,
                    min_score=min_score,
                )
                return [r["node"].element_id for r in records]
        except Exception as exc:
            log.warning("retriever.vector_search_failed", error=str(exc))
            return []

    @staticmethod
    def _clean_props(props: dict[str, Any]) -> dict[str, Any]:
        """Remove embedding vectors and other large properties from node/edge dicts."""
        return {k: v for k, v in props.items() if not isinstance(v, list) or len(v) < 50}

    def _fetch_neighbourhood(self, *, node_ids: list[str], limit: int) -> dict[str, Any]:
        """Fetch 1-hop neighbourhood for given node IDs."""
        with self._driver.session(database=self.settings.neo4j_database) as session:
            records = list(
                session.run(
                    self._NEIGHBOURHOOD_CYPHER,
                    node_ids=node_ids,
                    limit=limit,
                )
            )
        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []
        for r in records:
            for key in ("n", "m"):
                n = r[key]
                nodes[n.element_id] = self._clean_props(dict(n))
            edges.append(
                {
                    "type": r["r"].type,
                    "start": r["r"].start_node.element_id,
                    "end": r["r"].end_node.element_id,
                    "props": self._clean_props(dict(r["r"])),
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}

    def _serialize_context(self, subgraph: dict[str, Any]) -> str:
        """Convert subgraph to a context string for LLM prompting."""
        lines: list[str] = []

        # Tabular records (from Cypher path)
        records = subgraph.get("records", [])
        if records:
            lines.append(f"Query results ({len(records)} rows):")
            for row in records[:25]:
                parts: list[str] = []
                for k, v in row.items():
                    if isinstance(v, dict) and "_labels" in v:
                        label = v["_labels"][0] if v["_labels"] else "Node"
                        props = {pk: pv for pk, pv in v.items() if not pk.startswith("_")}
                        parts.append(f"{label}: {props}")
                    elif isinstance(v, dict) and "type" in v:
                        parts.append(f"-[:{v['type']}]->")
                    else:
                        parts.append(f"{k}={v}")
                lines.append(f"  {', '.join(parts)}")
            return "\n".join(lines)

        # Node/edge subgraph (from vector path)
        nodes = subgraph.get("nodes", [])
        edges = subgraph.get("edges", [])
        if nodes:
            lines.append(f"Nodes ({len(nodes)}):")
            for n in nodes[:20]:
                lines.append(f"  - {n}")
        if edges:
            lines.append(f"Relationships ({len(edges)}):")
            for e in edges[:30]:
                lines.append(f"  - ({e['start']})-[:{e['type']}]->({e['end']})")
        return "\n".join(lines) if lines else "No results found."

    def close(self) -> None:
        self._driver.close()
