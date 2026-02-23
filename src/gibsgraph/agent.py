"""Core LangGraph agent for GibsGraph."""

from __future__ import annotations

from typing import Any

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from gibsgraph.config import Settings, get_settings
from gibsgraph.kg_builder.builder import KGBuilder
from gibsgraph.retrieval.retriever import GraphRetriever
from gibsgraph.tools.cypher_validator import CypherValidator
from gibsgraph.tools.visualizer import GraphVisualizer

log = structlog.get_logger(__name__)


def _make_llm(settings: Settings) -> BaseChatModel:
    """Create the appropriate LLM client based on the configured model.

    Uses the provider registry in config.py to detect which LangChain
    class to instantiate.  Providers with a ``base_url`` (e.g. xAI/Grok)
    are OpenAI-compatible and reuse ``ChatOpenAI``.  Falls back to OpenAI
    for unknown model names.
    """
    import os

    from gibsgraph.config import provider_for_model

    model = settings.llm_model
    provider = provider_for_model(model)

    if provider and provider.name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,  # type: ignore[call-arg]
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
        )
    if provider and provider.name == "mistral":
        from langchain_mistralai import ChatMistralAI  # type: ignore[import-not-found]

        return ChatMistralAI(  # type: ignore[no-any-return]
            model=model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
        )
    # OpenAI-compatible providers (xAI/Grok, etc.) — same class, custom base_url
    from langchain_openai import ChatOpenAI

    if provider and provider.base_url:
        return ChatOpenAI(
            model=model,
            temperature=settings.llm_temperature,
            max_retries=settings.llm_max_retries,
            base_url=provider.base_url,
            api_key=os.getenv(provider.env_key) or "",  # type: ignore[arg-type]
        )
    # Default: native OpenAI (also handles unknown model names)
    return ChatOpenAI(
        model=model,
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Immutable-style state passed between agent nodes."""

    query: str
    usecase: str = ""
    subgraph: dict[str, Any] | None = None
    schema_cypher: str | None = None
    retrieved_context: str = ""
    explanation: str = ""
    cypher_used: str = ""
    visualization_url: str = ""
    errors: list[str] = Field(default_factory=list)
    steps: int = 0
    requires_human_review: bool = False


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def retrieve_subgraph(
    state: AgentState, *, settings: Settings, retriever: GraphRetriever
) -> dict[str, Any]:
    """Run retrieval and return the subgraph + context."""
    log.info("retrieve_subgraph")
    try:
        result = retriever.retrieve(query=state.query)
        return {
            "subgraph": result.subgraph,
            "retrieved_context": result.context,
            "cypher_used": result.cypher,
            "steps": state.steps + 1,
        }
    except Exception as exc:
        log.error("retrieve_subgraph_failed", error=str(exc))
        return {"errors": [*state.errors, str(exc)], "steps": state.steps + 1}


def generate_explanation(state: AgentState, *, settings: Settings) -> dict[str, Any]:
    """Generate a natural language explanation from the retrieved context."""
    log.info("generate_explanation")

    if not state.retrieved_context or state.retrieved_context == "No results found.":
        return {
            "explanation": "No relevant information found in the knowledge graph.",
            "steps": state.steps + 1,
        }

    llm = _make_llm(settings)

    response = llm.invoke(
        "You are a knowledge graph analyst. Answer the user's question based "
        "ONLY on the graph data provided below. Be specific and cite the "
        "exact nodes, articles, or relationships from the data.\n\n"
        "If the data doesn't contain enough information, say so clearly.\n\n"
        f"Question: {state.query}\n\n"
        f"Graph data:\n{state.retrieved_context}\n\n"
        f"Cypher used: {state.cypher_used}"
    )

    explanation = str(response.content).strip()
    log.info("generate_explanation.done", length=len(explanation))
    return {"explanation": explanation, "steps": state.steps + 1}


def validate_output(state: AgentState) -> dict[str, Any]:
    """Validate Cypher and flag if human review is needed."""
    validator = CypherValidator()
    is_valid = validator.validate(state.cypher_used) if state.cypher_used else True
    requires_review = not is_valid or bool(state.errors)
    return {"requires_human_review": requires_review, "steps": state.steps + 1}


def visualize(state: AgentState, *, settings: Settings) -> dict[str, Any]:
    """Generate Mermaid / Neo4j Bloom visualization URL."""
    if not state.subgraph:
        return {"steps": state.steps + 1}
    viz = GraphVisualizer(settings=settings)
    url = viz.bloom_url(state.subgraph)
    return {"visualization_url": url, "steps": state.steps + 1}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------


def should_continue(state: AgentState, max_steps: int = 10) -> str:
    """Route: stop on error/max_steps, human-review, or continue."""
    if state.steps >= max_steps or len(state.errors) >= 3:
        return END
    if state.requires_human_review:
        return "human_review"
    return "visualize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    settings: Settings | None = None, *, retriever: GraphRetriever | None = None
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Compile and return the LangGraph agent."""
    settings = settings or get_settings()
    _retriever = retriever or GraphRetriever(settings=settings)

    def _retrieve(state: AgentState) -> dict[str, Any]:
        return retrieve_subgraph(state, settings=settings, retriever=_retriever)

    def _explain(state: AgentState) -> dict[str, Any]:
        return generate_explanation(state, settings=settings)

    def _visualize(state: AgentState) -> dict[str, Any]:
        return visualize(state, settings=settings)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("explain", _explain)
    graph.add_node("validate", validate_output)
    graph.add_node("visualize", _visualize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "validate")
    graph.add_conditional_edges(
        "validate",
        should_continue,
        {"visualize": "visualize", "human_review": END, END: END},
    )
    graph.add_edge("visualize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class GibsGraphAgent:
    """High-level entrypoint for GibsGraph."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._retriever = GraphRetriever(settings=self.settings)
        self._graph = build_graph(self.settings, retriever=self._retriever)
        self.kg_builder = KGBuilder(settings=self.settings)

    @classmethod
    def from_env(cls) -> GibsGraphAgent:
        """Create agent from environment variables / .env file."""
        return cls(settings=get_settings())

    def ask(self, query: str) -> AgentState:
        """Run the full agent pipeline for a natural language query."""
        log.info("agent.ask", query=query[:120])
        initial = AgentState(query=query)
        result = self._graph.invoke(initial)
        return AgentState(**result)

    async def ask_async(self, query: str) -> AgentState:
        """Async version of ask()."""
        initial = AgentState(query=query)
        result = await self._graph.ainvoke(initial)
        return AgentState(**result)

    def close(self) -> None:
        """Close Neo4j connections."""
        self._retriever.close()
        self.kg_builder.close()
