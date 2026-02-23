"""Core LangGraph agent for GibsGraph."""

from __future__ import annotations

import structlog
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from gibsgraph.config import Settings, get_settings
from gibsgraph.kg_builder.builder import KGBuilder
from gibsgraph.retrieval.retriever import GraphRetriever
from gibsgraph.tools.cypher_validator import CypherValidator
from gibsgraph.tools.visualizer import GraphVisualizer

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """Immutable-style state passed between agent nodes."""

    query: str
    usecase: str = ""
    subgraph: dict | None = None
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


def classify_usecase(state: AgentState, *, settings: Settings) -> dict:
    """Classify the query into a usecase category via LLM."""
    log.info("classify_usecase", query=state.query[:80])

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        max_retries=settings.llm_max_retries,
    )

    response = llm.invoke(
        "Classify this question into exactly one category. "
        "Reply with ONLY the category name, nothing else.\n\n"
        "Categories:\n"
        "- graph_structure (about nodes, relationships, schema, counts)\n"
        "- cross_reference (comparing or linking across entities)\n"
        "- compliance (regulatory requirements, obligations, rules)\n"
        "- general (anything else)\n\n"
        f"Question: {state.query}"
    )
    usecase = response.content.strip().lower().replace(" ", "_")
    log.info("classify_usecase.result", usecase=usecase)
    return {"usecase": usecase, "steps": state.steps + 1}


def retrieve_subgraph(state: AgentState, *, settings: Settings) -> dict:
    """Run retrieval and return the subgraph + context."""
    log.info("retrieve_subgraph", usecase=state.usecase)
    retriever = GraphRetriever(settings=settings)
    try:
        result = retriever.retrieve(query=state.query)
        return {
            "subgraph": result.subgraph,
            "retrieved_context": result.context,
            "cypher_used": result.cypher,
            "steps": state.steps + 1,
        }
    except Exception as exc:  # noqa: BLE001
        log.error("retrieve_subgraph_failed", error=str(exc))
        return {"errors": [*state.errors, str(exc)], "steps": state.steps + 1}
    finally:
        retriever.close()


def generate_explanation(state: AgentState, *, settings: Settings) -> dict:
    """Generate a natural language explanation from the retrieved context."""
    log.info("generate_explanation")

    if not state.retrieved_context or state.retrieved_context == "No results found.":
        return {
            "explanation": "No relevant information found in the knowledge graph.",
            "steps": state.steps + 1,
        }

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        max_retries=settings.llm_max_retries,
    )

    response = llm.invoke(
        "You are a knowledge graph analyst. Answer the user's question based "
        "ONLY on the graph data provided below. Be specific and cite the "
        "exact nodes, articles, or relationships from the data.\n\n"
        "If the data doesn't contain enough information, say so clearly.\n\n"
        f"Question: {state.query}\n\n"
        f"Graph data:\n{state.retrieved_context}\n\n"
        f"Cypher used: {state.cypher_used}"
    )

    explanation = response.content.strip()
    log.info("generate_explanation.done", length=len(explanation))
    return {"explanation": explanation, "steps": state.steps + 1}


def validate_output(state: AgentState) -> dict:
    """Validate Cypher and flag if human review is needed."""
    validator = CypherValidator()
    is_valid = validator.validate(state.cypher_used) if state.cypher_used else True
    requires_review = not is_valid or bool(state.errors)
    return {"requires_human_review": requires_review, "steps": state.steps + 1}


def visualize(state: AgentState, *, settings: Settings) -> dict:
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


def build_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Compile and return the LangGraph agent."""
    settings = settings or get_settings()

    def _classify(state: AgentState) -> dict:
        return classify_usecase(state, settings=settings)

    def _retrieve(state: AgentState) -> dict:
        return retrieve_subgraph(state, settings=settings)

    def _explain(state: AgentState) -> dict:
        return generate_explanation(state, settings=settings)

    def _visualize(state: AgentState) -> dict:
        return visualize(state, settings=settings)

    graph = StateGraph(AgentState)
    graph.add_node("classify", _classify)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("explain", _explain)
    graph.add_node("validate", validate_output)
    graph.add_node("visualize", _visualize)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", "validate")
    graph.add_conditional_edges(
        "validate", should_continue, {"visualize": "visualize", END: END}
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
        self._graph = build_graph(self.settings)
        self.kg_builder = KGBuilder(settings=self.settings)

    @classmethod
    def from_env(cls) -> "GibsGraphAgent":
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
