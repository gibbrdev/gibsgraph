"""GibsGraph — GraphRAG + LangGraph agent for Neo4j knowledge graph reasoning."""

from gibsgraph._graph import Answer, Graph, IngestResult, SchemaInfo

__version__ = "0.4.1"
__all__ = ["Answer", "Graph", "IngestResult", "SchemaInfo"]

# Power-user imports — available but not in the spotlight:
# from gibsgraph.agent import GibsGraphAgent
# from gibsgraph.config import Settings
# from gibsgraph.retrieval.retriever import GraphRetriever
# from gibsgraph.kg_builder.builder import KGBuilder
