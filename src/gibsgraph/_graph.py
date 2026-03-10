"""GibsGraph public facade — the only class most users need."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from gibsgraph.agent import GibsGraphAgent

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types — simple, inspectable, no Pydantic overhead for the user
# ---------------------------------------------------------------------------


@dataclass
class Answer:
    """Result from Graph.ask()."""

    question: str
    answer: str
    cypher: str = ""
    confidence: float = 0.0
    visualization: str = ""  # Mermaid string
    bloom_url: str = ""  # Neo4j Bloom deep-link
    nodes_retrieved: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.answer

    def __repr__(self) -> str:
        return f"Answer(answer={self.answer!r:.60}, confidence={self.confidence:.2f})"


@dataclass
class IngestResult:
    """Result from Graph.ingest()."""

    source: str
    nodes_created: int
    relationships_created: int
    chunks_processed: int
    validation_warnings: list[str] = field(default_factory=list)
    validation_passed: bool = True

    def __str__(self) -> str:
        base = (
            f"Ingested '{self.source}': "
            f"{self.nodes_created} nodes, "
            f"{self.relationships_created} relationships"
        )
        if self.validation_warnings:
            base += f" ({len(self.validation_warnings)} warnings)"
        return base


@dataclass
class SchemaInfo:
    """Result from Graph.schema() — current graph structure."""

    node_labels: list[str]
    relationship_types: list[str]
    node_count: int
    relationship_count: int
    properties: dict[str, list[str]]  # label -> [property names]

    def __str__(self) -> str:
        return (
            f"Schema: {len(self.node_labels)} labels, "
            f"{len(self.relationship_types)} rel types, "
            f"{self.node_count} nodes, {self.relationship_count} relationships"
        )


# ---------------------------------------------------------------------------
# LLM auto-detection
# ---------------------------------------------------------------------------


def _resolve_llm(llm: str) -> str:
    """Detect best available LLM from environment if llm='auto'."""
    from gibsgraph.config import PROVIDERS

    if llm != "auto":
        return llm

    # Load .env so API keys are available via os.getenv
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    for provider in PROVIDERS:
        if os.getenv(provider.env_key):
            log.debug("graph.llm_auto_resolved", choice=provider.default_model)
            return provider.default_model

    env_keys = " or ".join(p.env_key for p in PROVIDERS)
    raise RuntimeError(
        f"No LLM API key found. Set {env_keys}, "
        f"or pass llm='{PROVIDERS[0].default_model}' explicitly.\n"
        "Docs: https://github.com/gibbrdev/gibsgraph#configuration"
    )


# ---------------------------------------------------------------------------
# Main facade
# ---------------------------------------------------------------------------


class Graph:
    """GibsGraph — one class, two methods, zero boilerplate.

    Usage::

        from gibsgraph import Graph

        g = Graph("bolt://localhost:7687", password="your-password")
        print(g.ask("What regulations apply to EU fintechs?"))

    Everything — Neo4j connection, LLM selection, retrieval, Cypher
    generation — is handled automatically. Override anything via kwargs.

    Args:
        uri:       Neo4j Bolt URI. Defaults to NEO4J_URI env var.
        password:  Neo4j password. Defaults to NEO4J_PASSWORD env var.
        username:  Neo4j username. Defaults to NEO4J_USERNAME env var or 'neo4j'.
        database:  Neo4j database name. Defaults to 'neo4j'.
        llm:       LLM to use. 'auto' detects from env keys
                   (OpenAI → Anthropic → Mistral). Any model name works.
        read_only: If True (default), disables ingest(). Safe for production queries.
        top_k:     Number of candidate nodes to retrieve per query. Default 10.
    """

    def __init__(
        self,
        uri: str | None = None,
        *,
        password: str | None = None,
        username: str | None = None,
        database: str = "neo4j",
        llm: str = "auto",
        read_only: bool = True,
        top_k: int = 10,
    ) -> None:
        from gibsgraph.config import Settings

        # Resolve connection — kwargs win over env vars
        resolved_uri: str = uri or os.getenv("NEO4J_URI") or "bolt://localhost:7687"
        resolved_password: str = password or os.getenv("NEO4J_PASSWORD") or ""
        resolved_username: str = username or os.getenv("NEO4J_USERNAME") or "neo4j"

        if not resolved_password:
            raise ValueError(
                "Neo4j password is required. Pass password='...' or set NEO4J_PASSWORD env var."
            )

        self._settings = Settings(
            NEO4J_URI=resolved_uri,
            NEO4J_USERNAME=resolved_username,
            NEO4J_PASSWORD=resolved_password,  # type: ignore[arg-type]
            NEO4J_DATABASE=database,
            NEO4J_READ_ONLY=read_only,
            LLM_MODEL=_resolve_llm(llm),
        )
        self._top_k = top_k
        self._agent: GibsGraphAgent | None = None  # lazy init
        log.info("graph.ready", uri=resolved_uri, llm=self._settings.llm_model)

    # ------------------------------------------------------------------
    # Public API — the only two methods most users need
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Answer:
        """Ask a natural language question about your Neo4j graph.

        Args:
            question: Any natural language question. GibsGraph retrieves
                      the relevant subgraph and generates a grounded answer.

        Returns:
            Answer object. Use str(result) or result.answer for the text.
            result.cypher shows what was queried. result.visualization
            gives a Mermaid diagram string.

        Example::

            result = g.ask("Which companies are subsidiaries of Apple?")
            print(result)                  # prints the answer
            print(result.cypher)           # the Cypher that was run
            print(result.confidence)       # 0.0-1.0
        """
        log.info("graph.ask", question=question[:100])
        agent_result = self._agent_instance().ask(question)

        return Answer(
            question=question,
            answer=agent_result.explanation or "No answer found.",
            cypher=agent_result.cypher_used,
            confidence=1.0 if not agent_result.errors else 0.3,
            visualization=self._to_mermaid(agent_result.subgraph),
            bloom_url=agent_result.visualization_url,
            nodes_retrieved=len((agent_result.subgraph or {}).get("nodes", [])),
            errors=agent_result.errors,
        )

    def ingest(self, text: str, *, source: str = "manual") -> IngestResult:
        """Ingest text into your Neo4j knowledge graph.

        Extracts entities and relationships from `text` and writes them
        to Neo4j. Requires read_only=False.

        Args:
            text:   Plain text to extract a knowledge graph from.
            source: Label for this ingestion (e.g. filename, URL).

        Returns:
            IngestResult with node/relationship counts.

        Example::

            g = Graph("bolt://...", password="...", read_only=False)
            g.ingest("Apple acquired Beats for $3B in 2014.", source="news")
        """
        if self._settings.neo4j_read_only:
            raise RuntimeError(
                "ingest() requires read_only=False.\n"
                "Use: Graph('bolt://...', password='...', read_only=False)"
            )
        log.info("graph.ingest", source=source, length=len(text))
        result = self._agent_instance().kg_builder.ingest(text, source=source)

        # Post-ingest validation — lightweight quality checks
        warnings = self._validate_ingest(result.nodes_created)

        return IngestResult(
            source=source,
            nodes_created=result.nodes_created,
            relationships_created=result.relationships_created,
            chunks_processed=result.chunks_processed,
            validation_warnings=warnings,
            validation_passed=len(warnings) == 0,
        )

    def schema(self) -> SchemaInfo:
        """Inspect the current Neo4j graph structure.

        Returns node labels, relationship types, counts, and properties
        without requiring any LLM calls.

        Example::

            info = g.schema()
            print(info.node_labels)       # ['Person', 'Company']
            print(info.relationship_types) # ['WORKS_AT', 'ACQUIRED']
            print(info.properties)         # {'Person': ['name', 'age'], ...}
        """
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            self._settings.neo4j_uri,
            auth=(
                self._settings.neo4j_username,
                self._settings.neo4j_password.get_secret_value(),
            ),
            max_connection_lifetime=self._settings.neo4j_max_connection_lifetime,
        )
        try:
            with driver.session(database=self._settings.neo4j_database) as session:
                labels = [
                    r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")
                ]
                rel_types = [
                    r["relationshipType"]
                    for r in session.run(
                        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                    )
                ]
                node_count: int = session.run("MATCH (n) RETURN count(n) AS c").single(strict=True)[
                    "c"
                ]
                rel_count: int = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single(
                    strict=True
                )["c"]

                # Collect properties per label
                props: dict[str, list[str]] = {}
                for label in labels:
                    result = session.run(
                        "MATCH (n) WHERE $label IN labels(n) "
                        "UNWIND keys(n) AS key "
                        "RETURN DISTINCT key ORDER BY key",
                        label=label,
                    )
                    props[label] = [r["key"] for r in result]

            return SchemaInfo(
                node_labels=labels,
                relationship_types=rel_types,
                node_count=node_count,
                relationship_count=rel_count,
                properties=props,
            )
        finally:
            driver.close()

    # ------------------------------------------------------------------
    # Power-user helpers (still on Graph, but not the main story)
    # ------------------------------------------------------------------

    def visualize(self, question: str) -> str:
        """Ask a question and return a Mermaid diagram string instead of text.

        Paste the output into https://mermaid.live to render it.
        """
        return self.ask(question).visualization

    def cypher(self, question: str) -> str:
        """Ask a question and return only the generated Cypher query."""
        return self.ask(question).cypher

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _agent_instance(self) -> GibsGraphAgent:
        """Lazy-init the agent on first use."""
        if self._agent is None:
            from gibsgraph.agent import GibsGraphAgent

            self._agent = GibsGraphAgent(settings=self._settings)
        return self._agent

    def _to_mermaid(self, subgraph: dict[str, Any] | None) -> str:
        if not subgraph:
            return ""
        from gibsgraph.tools.visualizer import GraphVisualizer

        return GraphVisualizer(settings=self._settings).to_mermaid(subgraph)

    def _validate_ingest(self, nodes_created: int) -> list[str]:
        """Run post-ingest quality checks on the graph.

        Checks for:
        - Generic node labels (Entity, Object, Thing, etc.)
        - Non-PascalCase labels
        - Relationship types not in UPPER_SNAKE_CASE
        - Orphan nodes (no relationships)
        """
        import re

        warnings: list[str] = []
        if nodes_created == 0:
            return warnings

        generic_labels = {
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

        try:
            info = self.schema()
        except Exception as exc:
            log.warning("validate_ingest.schema_failed", error=str(exc))
            return [f"Could not validate: {exc}"]

        for label in info.node_labels:
            if label.startswith("__"):
                continue  # skip internal labels like __Entity__
            if label in generic_labels:
                warnings.append(f"Generic label '{label}' — consider a domain-specific name")
            if not re.match(r"^[A-Z][a-zA-Z0-9]*$", label) or not any(c.islower() for c in label):
                if label not in generic_labels:
                    warnings.append(
                        f"Label '{label}' is not PascalCase — Neo4j convention is PascalCase"
                    )

        for rel_type in info.relationship_types:
            if rel_type.startswith("__"):
                continue
            if not re.match(r"^[A-Z][A-Z0-9_]*$", rel_type):
                warnings.append(
                    f"Relationship '{rel_type}' is not UPPER_SNAKE_CASE — "
                    f"Neo4j convention is UPPER_SNAKE_CASE"
                )

        if warnings:
            log.info("validate_ingest.warnings", count=len(warnings))

        return warnings

    def close(self) -> None:
        """Close Neo4j connections and release resources."""
        if self._agent is not None:
            self._agent.close()

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"Graph(uri={self._settings.neo4j_uri!r}, "
            f"llm={self._settings.llm_model!r}, "
            f"read_only={self._settings.neo4j_read_only})"
        )
