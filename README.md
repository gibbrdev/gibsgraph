# GibsGraph

> **Natural language queries for any Neo4j graph — automatically.**
> Built by [V.Gibson](https://gibs.dev) at Gibbr AB

[![CI](https://github.com/gibbrdev/gibsgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/gibbrdev/gibsgraph/actions)
[![Coverage](https://codecov.io/gh/gibbrdev/gibsgraph/branch/main/graph/badge.svg)](https://codecov.io/gh/gibbrdev/gibsgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

GibsGraph connects to any Neo4j knowledge graph, auto-discovers its schema, and lets
you ask questions in plain English. It generates Cypher automatically, retrieves the
relevant subgraph, and returns a grounded answer — with the Cypher shown so you can
verify what was queried.

**[Changelog](CHANGELOG.md)** | **[Contributing](CONTRIBUTING.md)**

---

## What it does (v0.3.1)

- **Natural language queries** — ask anything about your Neo4j graph
- **Expert knowledge graph** — 715 nodes: 36 clauses, 122 functions, 383 examples, 309 best practices
- **4-stage validation** — syntactic → structural → semantic → domain, with enterprise severity levels
- **Auto schema discovery** — connects and learns your graph structure automatically
- **Dual retrieval** — vector search (when index exists) with text-to-Cypher fallback
- **Cypher self-healing** — if generated Cypher fails, the error is sent back to the LLM for correction
- **Cypher transparency** — see exactly what was queried
- **Visualization** — Mermaid diagrams and Neo4j Bloom deep links
- **Secure by default** — read-only transactions, parameterized Cypher, injection validation
- **LLM flexibility** — OpenAI, Anthropic, Mistral, xAI/Grok — auto-detected from env keys

### Planned (not yet implemented)

- Text-to-graph ingestion (`g.ingest()`)
- G-Retriever GNN reasoning
- PCST subgraph pruning

---

## Quick start

```bash
pip install gibsgraph
```

```python
from gibsgraph import Graph

# Connect — auto-detects LLM from your env keys
g = Graph("bolt://localhost:7687", password="your-password")

# Ask anything
result = g.ask("What movies did Tom Hanks act in?")
print(result)              # the answer
print(result.cypher)       # Cypher that was run
print(result.confidence)   # 0.0-1.0
print(result.visualization) # Mermaid diagram string

g.close()  # or use: with Graph(...) as g:
```

---

## Configuration

Set environment variables (or pass them directly):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=your_password
export OPENAI_API_KEY=sk-...     # or ANTHROPIC_API_KEY, MISTRAL_API_KEY, XAI_API_KEY
```

Or use a `.env` file — copy `.env.example` to get started.

---

## Installation

```bash
pip install gibsgraph
```

Optional extras:

```bash
pip install "gibsgraph[mistral]"  # Mistral LLM support
pip install "gibsgraph[gnn]"      # G-Retriever GNN (needs GPU/CUDA)
pip install "gibsgraph[ui]"       # Streamlit demo UI
pip install "gibsgraph[full]"     # everything including dev tools
```

### Docker

```bash
cp .env.example .env
# Edit .env with your Neo4j password and API key
docker compose up
# Open http://localhost:8501
```

---

## Architecture

```
User Query
    |
    v
LangGraph Agent (agent.py)
    +-- retrieval/     <- Auto schema discovery + text-to-Cypher + vector search
    +-- tools/         <- Cypher validator, Mermaid visualizer
    +-- training/      <- 4-stage validation (syntactic/structural/semantic/domain)
    +-- expert.py      <- ExpertStore — 715 nodes, vector + fulltext indexes
    +-- kg_builder/    <- Text to Neo4j (planned)
    +-- gnn/           <- G-Retriever inference (planned)
    |
    v
Neo4j Knowledge Graph
```

---

## Examples

| Example | Description |
|---------|-------------|
| [`examples/regulatory_kg.py`](examples/regulatory_kg.py) | EU regulatory knowledge graph (gibs.dev use case) |

---

## Testing

216 unit tests, 78% coverage.

```bash
pytest                          # All tests
pytest tests/unit/              # Unit only
pytest tests/integration/       # Integration (requires Neo4j)
pytest --cov --cov-report=html  # With coverage report
```

---

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/gibsgraph
pytest
```

---

## License

MIT — see [LICENSE](LICENSE)

---

## Acknowledgements

Built on:
- [neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [G-Retriever](https://arxiv.org/abs/2402.07630) (He et al., 2024)
