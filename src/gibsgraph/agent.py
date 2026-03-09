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


class IntentClassification(BaseModel):
    """Structured output from the intent classifier node.

    Extracts structured context from free-form user input so downstream
    nodes (retrieval, explanation) work with clean, typed data instead
    of raw NL strings.
    """

    action: str = Field(
        default="ask",
        description="What the user wants: 'ask' (query existing graph), "
        "'build' (create/ingest a graph), or 'schema' (inspect structure).",
    )
    industry: str = Field(
        default="",
        description="Detected industry (e.g. 'insurance', 'fintech', 'healthcare'). "
        "Empty if not determinable.",
    )
    region: str = Field(
        default="",
        description="Geographic region or jurisdiction (e.g. 'sweden', 'eu', 'us'). "
        "Empty if not determinable.",
    )
    regulations: list[str] = Field(
        default_factory=list,
        description="Applicable regulations inferred from industry + region "
        "(e.g. ['GDPR', 'IDD', 'Solvency II']). Empty if none detected.",
    )
    data_type: str = Field(
        default="",
        description="Type of data the user is working with "
        "(e.g. 'TOS documents', 'transaction logs', 'research papers').",
    )
    goal: str = Field(
        default="",
        description="The user's objective in plain language "
        "(e.g. 'map customer psychology', 'detect fraud patterns').",
    )
    enriched_query: str = Field(
        default="",
        description="The original query rewritten with extracted context for better "
        "retrieval. Includes industry, regulation, and goal context.",
    )


class AgentState(BaseModel):
    """Immutable-style state passed between agent nodes."""

    query: str
    usecase: str = ""
    intent: IntentClassification = Field(default_factory=IntentClassification)
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


_CLASSIFY_PROMPT = (
    "You are an intent classifier for a Neo4j knowledge graph system. "
    "Analyze the user's input and extract structured context.\n\n"
    "Rules:\n"
    "- action: 'ask' if querying data, 'build' if creating/ingesting, "
    "'schema' if inspecting structure\n"
    "- industry: use lowercase single words (insurance, fintech, healthcare, etc.)\n"
    "- region: use lowercase (sweden, eu, us, etc.)\n"
    "- regulations: infer from industry + region. "
    "EU insurance → GDPR, IDD, Solvency II, DORA. "
    "EU fintech → GDPR, PSD2, MiCA, DORA. "
    "US healthcare → HIPAA. Only include when clearly applicable.\n"
    "- enriched_query: rewrite the input as a clear, specific query that "
    "includes the detected context. This is used for graph retrieval.\n"
    "- Leave fields empty (or empty list) when not determinable. "
    "Do not guess.\n\n"
    "User input: {query}"
)


def classify_intent(state: AgentState, *, settings: Settings) -> dict[str, Any]:
    """Extract structured intent from the user's free-form input.

    Uses LLM with structured output to parse industry, region,
    regulations, data type, and goal from messy NL input. One LLM call.
    """
    log.info("classify_intent", query=state.query[:100])

    llm = _make_llm(settings)

    try:
        structured_llm = llm.with_structured_output(IntentClassification)
        result = structured_llm.invoke(_CLASSIFY_PROMPT.format(query=state.query))

        if not isinstance(result, IntentClassification):
            log.warning("classify_intent.unexpected_type", type=type(result).__name__)
            return {"steps": state.steps + 1}

        log.info(
            "classify_intent.done",
            action=result.action,
            industry=result.industry,
            region=result.region,
            regulations=result.regulations,
        )
        return {
            "intent": result,
            "usecase": f"{result.industry}/{result.goal}" if result.industry else "",
            "steps": state.steps + 1,
        }
    except Exception as exc:
        log.warning("classify_intent.failed", error=str(exc))
        return {"steps": state.steps + 1}


def retrieve_subgraph(
    state: AgentState, *, settings: Settings, retriever: GraphRetriever
) -> dict[str, Any]:
    """Run retrieval using the enriched query when available."""
    log.info("retrieve_subgraph")
    # Use the enriched query from intent classification if available
    query = state.intent.enriched_query or state.query
    try:
        result = retriever.retrieve(query=query)
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

    # Build context-aware prompt using classified intent
    intent_context = ""
    if state.intent.industry:
        intent_context += f"Industry: {state.intent.industry}\n"
    if state.intent.region:
        intent_context += f"Region: {state.intent.region}\n"
    if state.intent.regulations:
        intent_context += f"Regulations: {', '.join(state.intent.regulations)}\n"
    if state.intent.goal:
        intent_context += f"User goal: {state.intent.goal}\n"

    prompt = (
        "You are a knowledge graph analyst. Answer the user's question based "
        "ONLY on the graph data provided below. Be specific and cite the "
        "exact nodes, articles, or relationships from the data.\n\n"
        "If the data doesn't contain enough information, say so clearly.\n\n"
    )
    if intent_context:
        prompt += f"Context:\n{intent_context}\n"
    prompt += (
        f"Question: {state.query}\n\n"
        f"Graph data:\n{state.retrieved_context}\n\n"
        f"Cypher used: {state.cypher_used}"
    )

    response = llm.invoke(prompt)

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

    def _classify(state: AgentState) -> dict[str, Any]:
        return classify_intent(state, settings=settings)

    def _retrieve(state: AgentState) -> dict[str, Any]:
        return retrieve_subgraph(state, settings=settings, retriever=_retriever)

    def _explain(state: AgentState) -> dict[str, Any]:
        return generate_explanation(state, settings=settings)

    def _visualize(state: AgentState) -> dict[str, Any]:
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
