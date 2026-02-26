"""Knowledge Graph builder — text to Neo4j via neo4j-graphrag."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from neo4j import GraphDatabase

from gibsgraph.config import Settings, provider_for_model

if TYPE_CHECKING:
    from neo4j_graphrag.embeddings import Embedder
    from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
    from neo4j_graphrag.llm import LLMInterface

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# neo4j-graphrag LLM / embedder factories
# ---------------------------------------------------------------------------


def _make_kg_llm(settings: Settings) -> LLMInterface:
    """Create a neo4j-graphrag LLM instance from settings.

    Routes to the correct neo4j-graphrag LLM class using the provider
    registry.  Same routing logic as ``agent.py:_make_llm`` but produces
    neo4j-graphrag LLM objects instead of LangChain chat models.
    """
    model = settings.llm_model
    provider = provider_for_model(model)

    if provider and provider.name == "anthropic":
        from neo4j_graphrag.llm import AnthropicLLM

        if not settings.anthropic_api_key:
            msg = "ANTHROPIC_API_KEY is required for Anthropic models"
            raise RuntimeError(msg)
        return AnthropicLLM(
            model_name=model,
            model_params={"temperature": settings.llm_temperature, "max_tokens": 4096},
            api_key=settings.anthropic_api_key.get_secret_value(),
        )

    if provider and provider.name == "mistral":
        from neo4j_graphrag.llm import MistralAILLM

        if not settings.mistral_api_key:
            msg = "MISTRAL_API_KEY is required for Mistral models"
            raise RuntimeError(msg)
        return MistralAILLM(
            model_name=model,
            model_params={"temperature": settings.llm_temperature},
            api_key=settings.mistral_api_key.get_secret_value(),
        )

    # OpenAI-compatible (xAI/Grok or native OpenAI)
    from neo4j_graphrag.llm import OpenAILLM

    if provider and provider.base_url:
        # xAI / Grok — OpenAI-compatible endpoint
        api_key_field = getattr(settings, f"{provider.name}_api_key", None)
        if not api_key_field:
            msg = f"{provider.env_key} is required for {provider.name} models"
            raise RuntimeError(msg)
        return OpenAILLM(
            model_name=model,
            model_params={"temperature": settings.llm_temperature},
            api_key=api_key_field.get_secret_value(),
            base_url=provider.base_url,
        )

    # Default: native OpenAI
    if not settings.openai_api_key:
        msg = "OPENAI_API_KEY is required for OpenAI models"
        raise RuntimeError(msg)
    return OpenAILLM(
        model_name=model,
        model_params={"temperature": settings.llm_temperature},
        api_key=settings.openai_api_key.get_secret_value(),
    )


def _make_kg_embedder(settings: Settings) -> Embedder:
    """Create an OpenAI embedder for neo4j-graphrag entity embeddings."""
    from neo4j_graphrag.embeddings import OpenAIEmbeddings

    if not settings.openai_api_key:
        msg = (
            "OPENAI_API_KEY is required for embeddings during ingestion. "
            "Set it in your environment or .env file."
        )
        raise RuntimeError(msg)
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )


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

        if not text.strip():
            return IngestResult(nodes_created=0, relationships_created=0, chunks_processed=0)

        llm = _make_kg_llm(self.settings)
        embedder = _make_kg_embedder(self.settings)
        pipeline = self._build_pipeline(llm, embedder)

        nodes_before, rels_before = self._count_graph_entities()

        try:
            asyncio.run(pipeline.run_async(text=text))
        except RuntimeError as exc:
            if "cannot be called from a running event loop" in str(exc).lower():
                msg = (
                    "Cannot run ingest() inside an existing event loop "
                    "(e.g. Jupyter). Install nest_asyncio and call "
                    "nest_asyncio.apply() before using g.ingest()."
                )
                raise RuntimeError(msg) from exc
            raise

        nodes_after, rels_after = self._count_graph_entities()

        # Estimate chunks from text length (default splitter uses ~4000 char chunks)
        chunks_processed = max(1, -(-len(text) // 4000))

        result = IngestResult(
            nodes_created=max(0, nodes_after - nodes_before),
            relationships_created=max(0, rels_after - rels_before),
            chunks_processed=chunks_processed,
        )
        log.info(
            "kg_builder.ingest.done",
            nodes_created=result.nodes_created,
            relationships_created=result.relationships_created,
            chunks_processed=result.chunks_processed,
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, llm: LLMInterface, embedder: Embedder) -> SimpleKGPipeline:
        """Create a SimpleKGPipeline for entity/relation extraction."""
        from neo4j_graphrag.experimental.pipeline.kg_builder import (
            SimpleKGPipeline as _SimpleKGPipeline,
        )

        return _SimpleKGPipeline(
            llm=llm,
            driver=self._driver,
            embedder=embedder,
            from_pdf=False,
            on_error="IGNORE",
            perform_entity_resolution=True,
            neo4j_database=self.settings.neo4j_database,
        )

    def _count_graph_entities(self) -> tuple[int, int]:
        """Count __Entity__ nodes and relationships in the graph."""
        with self._driver.session(database=self.settings.neo4j_database) as session:
            node_result = session.run("MATCH (n:__Entity__) RETURN count(n) AS c")
            node_count: int = node_result.single(strict=True)["c"]
            rel_result = session.run("MATCH (:__Entity__)-[r]->() RETURN count(r) AS c")
            rel_count: int = rel_result.single(strict=True)["c"]
        return node_count, rel_count

    def ingest_file(self, path: str) -> IngestResult:
        """Ingest a text file by path."""
        import pathlib

        text = pathlib.Path(path).read_text(encoding="utf-8")
        return self.ingest(text, source=path)

    def clear_graph(self) -> None:
        """Delete ALL nodes and relationships. Use with extreme caution."""
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
