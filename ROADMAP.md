# GibsGraph Roadmap

> Built by Victor Gibson — creator of [gibs.dev](https://gibs.dev)

---

## Where we're going

GibsGraph starts as the simplest possible interface to Neo4j GraphRAG.
It grows into a knowledge machine that gets smarter every time someone uses it.

---

## Shipped

- `g.ask()` — natural language queries against any Neo4j graph
- Auto schema discovery — connects and learns your graph structure
- Dual retrieval — vector search with text-to-Cypher fallback
- Cypher self-healing — failed queries auto-corrected by LLM
- 4 LLM providers — OpenAI, Anthropic, Mistral, xAI/Grok (auto-detected)
- Expert knowledge graph — 715 nodes (36 clauses, 122 functions, 383 examples, 309 best practices)
- 4-stage validation suite — syntactic → structural → semantic → domain
- Enterprise severity levels (ERROR/WARNING/INFO) on validation findings
- Source-cited answers with Cypher transparency
- Mermaid diagrams + Neo4j Bloom visualization
- `pip install gibsgraph` on PyPI
- Streamlit demo UI + Docker one-command setup

---

## v1.0 — Foundation
**Theme: It works. It impresses.**

- `g.ingest()` — text-to-graph ingestion via neo4j-graphrag SimpleKGPipeline
- PCST subgraph pruning for precise, hallucination-free retrieval
- Use case generation per industry
- Demo runs against gibs.dev regulatory graph

---

## v1.5 — Smart routing
**Theme: Cheap to run. Energy efficient.**

- Automatic model routing — simple questions use small models, complex ones use large
- 60-80% cost reduction for typical workloads
- Full local mode — no data leaves your server (LLaMA 3)
- Users see none of this — just faster, cheaper answers

---

## v2.5 — Trained specialist
**Theme: Not a generalist. A Neo4j expert.**

- Fine-tuned on 1,000+ validated graphs across 8 industries
- GNN validation layer — structural pattern recognition
- Industry coverage: fintech, compliance, supply chain, healthcare,
  cybersecurity, HR, e-commerce, media
- 95%+ precision on supported industries

---

## v3.0 — The knowledge machine
**Theme: Describe your company. Get your graph.**

- `g.map()` — natural language to production Neo4j graph in seconds
- Learning loop — every production graph improves the system
- The more it's used, the better it gets
- Enterprise schemas, constraints, indexes, GDS projections — all automatic

---

## Always

- Open source (MIT)
- Security first — parameterized Cypher, read-only by default, no hardcoded credentials
- EU-native options at every version
- Your data stays yours

---

## How to influence this roadmap

Open an issue. Label it `roadmap`.

The features that get built are the ones the community actually needs.
We track everything in [GitHub Projects](https://github.com/gibbrdev/gibsgraph/projects).
