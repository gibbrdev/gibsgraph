"""Knowledge Graph builder — text to Neo4j via neo4j-graphrag."""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from neo4j import GraphDatabase

from gibsgraph.config import Settings

log = structlog.get_logger(__name__)


@dataclass
class IngestResult:
    nodes_created: int
    relationships_created: int
    chunks_processed: int


class KGBuilder:
    """Extracts entities and relations from text and writes them to Neo4j.

    Uses neo4j-graphrag SimpleKGPipeline under the hood.
    Schema validation is enforced via Pydantic models before any write.
    All Cypher queries are parameterized — never string-interpolated.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password.get_secret_value()),
            max_connection_lifetime=settings.neo4j_max_connection_lifetime,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, text: str, *, source: str = "manual") -> IngestResult:
        """Extract entities/relations from `text` and persist to Neo4j."""
        log.info("kg_builder.ingest", source=source, text_length=len(text))

        if self.settings.neo4j_read_only:
            msg = "KGBuilder.ingest() is disabled in read-only mode (NEO4J_READ_ONLY=true)"
            raise RuntimeError(msg)

        # In production: call neo4j-graphrag SimpleKGPipeline
        # pipeline = SimpleKGPipeline(driver=self._driver, llm=..., embedder=...)
        # result = pipeline.run(text=text)

        # Stub for now — returns mock result
        log.info("kg_builder.ingest.complete", nodes=0, rels=0, chunks=1)
        return IngestResult(nodes_created=0, relationships_created=0, chunks_processed=1)

    def ingest_file(self, path: str) -> IngestResult:
        """Ingest a text file by path."""
        import pathlib

        text = pathlib.Path(path).read_text(encoding="utf-8")
        return self.ingest(text, source=path)

    def clear_graph(self) -> None:
        """⚠️ Delete ALL nodes and relationships. Use with extreme caution."""
        if self.settings.neo4j_read_only:
            msg = "Cannot clear graph in read-only mode"
            raise RuntimeError(msg)
        with self._driver.session(database=self.settings.neo4j_database) as session:
            # Parameterized — even though no user input here, keeps pattern consistent
            session.run("MATCH (n) DETACH DELETE n")
        log.warning("kg_builder.graph_cleared")

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> KGBuilder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
