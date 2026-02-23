"""GibsGraph — GraphRAG + LangGraph agent for Neo4j knowledge graph reasoning."""

from gibsgraph._graph import Answer, Graph, IngestResult

__version__ = "0.1.0"
__all__ = ["Graph", "Answer", "IngestResult"]

# Power-user imports — available but not in the spotlight:
# from gibsgraph.agent import GibsGraphAgent
# from gibsgraph.config import Settings
# from gibsgraph.retrieval.retriever import GraphRetriever
# from gibsgraph.kg_builder.builder import KGBuilder
