# GibsGraph — Claude Code Agent Instructions

This file tells Claude Code how to work in this repo. Read this before making changes.

## What GibsGraph actually is

Not a wrapper. A knowledge machine.

**The one-sentence version:**
GibsGraph lets anyone ask complex questions against a Neo4j graph in plain
language — and get answers they can trust, with source citations, without
knowing Cypher.

**The full vision (see ROADMAP_INTERNAL.md):**
Natural language → GibsGraph → professional Neo4j graph, automatically.
The system learns from every graph built in production. The more it's used,
the better it gets. The vallgrav (moat) grows every day.

## Architecture — four layers

```
LAYER 1: EXPERT GRAPH (source of truth)
Neo4j docs + research papers + blogs + best practices

LAYER 2: VALIDATION SUITE
LangGraph agent comparing new graphs against expert graph
Syntactic → Structural → Semantic → Domain validation

LAYER 3: TRAINING DATA
Validated graphs logged automatically
Synthetic use cases per industry (100 × Tier 1 industries)
GNN trained on validated data

LAYER 4: LEARNING LOOP
Every production graph from real users → training data
GNN retrained periodically
```

## Version we're building NOW: v1.0

Focus: `g.ask()` works against a real Neo4j graph. PCST implemented (not stub).
Demo runs against gibs.dev regulatory graph. `pip install gibsgraph` works.

See ROADMAP_INTERNAL.md for full version strategy.

## Project overview

GibsGraph is a **GraphRAG + LangGraph agent** for Neo4j knowledge graph reasoning.
Stack: Python 3.12, LangGraph, neo4j-graphrag, Pydantic v2, ruff, mypy strict.

## Public API (what users see)

```python
from gibsgraph import Graph          # ← THE primary import. One class.
from gibsgraph import Answer         # for type hints only
from gibsgraph import IngestResult   # for type hints only
```

The `Graph` facade in `src/gibsgraph/_graph.py` is what users interact with.
Internal classes (`GibsGraphAgent`, `KGBuilder`, `GraphRetriever`) are implementation
detail — never expose them in user-facing docs or examples.

```python
# Correct — what users write
g = Graph("bolt://localhost:7687", password="secret")
result = g.ask("Who acquired Beats?")
g.ingest("Apple acquired Beats for $3B", read_only=False)

# Wrong — internal, don't surface these to users
from gibsgraph.agent import GibsGraphAgent  # power users only
```

## Extras naming convention

Named after what you DO, not the stack:
- `pip install gibsgraph[llm]`  — not `[openai]`
- `pip install gibsgraph[gnn]`  — not `[torch,torch-geometric]`
- `pip install gibsgraph[ui]`   — not `[streamlit]`
- `pip install gibsgraph[full]` — everything

## Development commands

```bash
# Install
pip install -e ".[dev]"
pre-commit install

# Lint + type check
ruff check src/ tests/
mypy src/gibsgraph

# Test
pytest tests/unit/          # fast, no Neo4j needed
pytest tests/integration/   # requires NEO4J_PASSWORD env var

# Run demo
streamlit run app/streamlit_demo.py

# Docker
docker compose up
```

## CRITICAL rules — never violate

1. **Cypher security**: ALWAYS use `$parameters`. NEVER f-strings in Cypher queries.
2. **No secrets**: NEVER hardcode credentials. Always use Settings / env vars.
3. **Read-only by default**: Check `settings.neo4j_read_only` before any write.
4. **Pydantic for state**: All inter-node data via `AgentState` fields.
5. **Parameterized typing**: `mypy --strict` must pass. No bare `Any`.
6. **structlog not print**: Use `log = structlog.get_logger(__name__)` for all logging.

## Adding a new agent node

1. Define a function `def my_node(state: AgentState) -> dict:` in `agent.py`
2. Return only the fields you're updating (partial state update)
3. Add `graph.add_node("my_node", my_node)` and wire edges
4. Write unit tests in `tests/unit/test_agent_state.py`

## Adding a new tool

1. Create `src/gibsgraph/tools/my_tool.py`
2. Add a class with `__init__(self, settings: Settings)`
3. Add `__init__.py` export
4. Write tests in `tests/unit/test_my_tool.py`

## PR checklist before pushing

- [ ] `ruff check . --fix` run
- [ ] `mypy src/gibsgraph` passes  
- [ ] Tests written for new code
- [ ] No `.env` or secrets in diff
- [ ] CHANGELOG.md updated

## Where stubs need real implementation

Search for `# In production:` comments — these mark where stub code needs
real LLM/embedding/model calls wired in.

Key stubs:
- `retrieval/retriever.py` → `_embed()`, `_pcst_prune()`  
- `kg_builder/builder.py` → `ingest()` (wire neo4j-graphrag SimpleKGPipeline)
- `agent.py` → `classify_usecase()`, `generate_explanation()` (wire LLM calls)
- `gnn/g_retriever.py` → `predict()` (load real model weights)
