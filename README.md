# GibsGraph

> **From natural language to production Neo4j graph — automatically.**
> Built by [Victor Gibson](https://gibs.dev) — creator of gibs.dev EU regulatory compliance API

[![CI](https://github.com/vibecoder/gibsgraph/actions/workflows/ci.yml/badge.svg)](https://github.com/vibecoder/gibsgraph/actions)
[![Coverage](https://codecov.io/gh/vibecoder/gibsgraph/branch/main/graph/badge.svg)](https://codecov.io/gh/vibecoder/gibsgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/gibsgraph)](https://pypi.org/project/gibsgraph/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord)](https://discord.gg/gibsgraph)
[![Neo4j Certified](https://img.shields.io/badge/Neo4j-Certified%20Professional-blue?logo=neo4j)](https://graphacademy.neo4j.com/)

GibsGraph lets anyone ask complex questions against a Neo4j knowledge graph in plain
language — and get answers they can trust, with source citations, without knowing Cypher.

Today: ask questions. Soon: describe your company, get your graph.

**[See the full roadmap](ROADMAP.md)**

---

## The problem

Companies implement AI by sticking it on the front door. A chatbot here, a summary there.
It looks like AI. It is not.

Real AI needs structured, connected data. Knowledge graphs. But knowledge graphs have always
required Neo4j experts, Cypher knowledge, and months of work.

GibsGraph removes that barrier.

---

## What it does today (v1.0)

- **Natural language queries** — ask anything about your Neo4j graph
- **PCST retrieval** — Prize-Collecting Steiner Tree pruning, no hallucinated paths
- **Source-cited answers** — every answer references exact graph nodes
- **Cypher transparency** — see exactly what was queried
- **Visualization** — Mermaid diagrams and Neo4j Bloom deep links
- **Secure by default** — parameterized Cypher, read-only mode, no hardcoded credentials
- **EU-native options** — Mistral support, full local mode via LLaMA 3

## 🚀 30-second demo

```bash
pip install gibsgraph
gibsgraph ask "What regulations apply to fintech companies in the EU?"
```

Or with Docker:

```bash
docker compose up
# Open http://localhost:8501
```

---

## 📦 Installation

```bash
pip install gibsgraph
```

That's it for most users. Optional extras, named after what you want to do:

```bash
pip install gibsgraph[llm]    # OpenAI + Anthropic LLM support
pip install gibsgraph[gnn]    # G-Retriever GNN reasoning (needs GPU/CUDA)
pip install gibsgraph[ui]     # Streamlit demo UI
pip install gibsgraph[full]   # everything
```

---

## ⚙️ Configuration

Set two environment variables (or pass them directly — see Quick start):

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_PASSWORD=your_password
export OPENAI_API_KEY=sk-...     # or ANTHROPIC_API_KEY
```

Or use a `.env` file — copy `.env.example` to get started.

---

## 🏃 Quick start

```python
from gibsgraph import Graph

# Connect — auto-detects LLM from your env keys
g = Graph("bolt://localhost:7687", password="secret")

# Ask anything
result = g.ask("What regulations apply to EU fintech companies?")
print(result)              # the answer
print(result.cypher)       # Cypher that was run
print(result.confidence)   # 0.0–1.0
print(result.visualization) # Mermaid diagram string
```

**Build a knowledge graph from text:**

```python
g = Graph("bolt://localhost:7687", password="secret", read_only=False)
g.ingest("Apple acquired Beats Electronics for $3B in 2014.")
g.ingest("Tim Cook announced the deal at WWDC.")

result = g.ask("What did Apple acquire and for how much?")
print(result)
```

**That's the whole API for 90% of use cases.** Power users can import internals directly — see [Advanced usage](docs/advanced.md).

---

## 🗺️ Examples

| Example | Description |
|---------|-------------|
| [`examples/regulatory_kg.py`](examples/regulatory_kg.py) | EU regulatory knowledge graph (gibs.dev usecase) |
| [`examples/company_graph.py`](examples/company_graph.py) | M&A and company relationship graph |
| [`examples/academic_graph.py`](examples/academic_graph.py) | Research paper citation network |
| [`examples/qa_benchmark.py`](examples/qa_benchmark.py) | WebQSP + SceneGraphs benchmarks |
| [`examples/human_in_loop.py`](examples/human_in_loop.py) | Human-in-the-loop agent with interrupts |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
LangGraph Agent (core/agent.py)
    ├── kg_builder/    ← Text → Neo4j (neo4j-graphrag)
    ├── retrieval/     ← PCST subgraph pruning + vector search
    ├── gnn/           ← G-Retriever inference
    └── tools/         ← Cypher validator, Mermaid visualizer
    │
    ▼
Neo4j Knowledge Graph
```

---

## 🧪 Testing

```bash
pytest                          # All tests
pytest tests/unit/              # Unit only
pytest tests/integration/       # Integration (requires Neo4j)
pytest --cov --cov-report=html  # With coverage report
```

---

## 🤝 Contributing

We love contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

Good first issues are labeled [`good first issue`](https://github.com/vibecoder/gibsgraph/issues?q=is%3Aopen+label%3A%22good+first+issue%22).

```bash
git checkout -b feat/your-feature
# Make changes
pre-commit run --all-files
pytest
git push origin feat/your-feature
# Open a PR!
```

---

## 📄 License

MIT — see [LICENSE](LICENSE)

---

## 🙏 Acknowledgements

Built on the shoulders of:
- [neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [G-Retriever](https://arxiv.org/abs/2402.07630) (He et al., 2024)
- [neo4j-gnn-llm-example](https://github.com/neo4j-product-examples/neo4j-gnn-llm-example)
