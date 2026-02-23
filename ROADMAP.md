# GibsGraph Roadmap

> Built by Victor Gibson — creator of [gibs.dev](https://gibs.dev)

---

## Where we're going

GibsGraph starts as the simplest possible interface to Neo4j GraphRAG.
It grows into a knowledge machine that gets smarter every time someone uses it.

---

## v1.0 — Foundation (March 2026)
**Theme: It works. It impresses.**

- `g.ask()` — natural language queries against any Neo4j graph
- Auto-detects your LLM from environment (OpenAI or Anthropic)
- PCST subgraph pruning for precise, hallucination-free retrieval
- Source-cited answers with Cypher transparency
- Mermaid diagrams + Neo4j Bloom visualization
- `pip install gibsgraph` — zero config to get started
- Streamlit demo UI
- Docker one-command setup

---

## v1.5 — Smart routing (April 2026)
**Theme: Cheap to run. Energy efficient.**

- Automatic model routing — simple questions use small models, complex ones use large
- 60-80% cost reduction for typical workloads
- EU-native LLM support (Mistral as default option)
- Full local mode — no data leaves your server (LLaMA 3)
- Users see none of this — just faster, cheaper answers

---

## v2.0 — Expert knowledge (Q2 2026)
**Theme: Precision that professionals trust.**

- Expert graph built from Neo4j docs, research papers, and production use cases
- Validation suite — every generated graph checked against expert knowledge
- 4-layer validation: syntactic → structural → semantic → domain
- Confidence scores with explainability
- `result.to_dataframe()` for data science workflows

---

## v2.5 — Trained specialist (Q3 2026)
**Theme: Not a generalist. A Neo4j expert.**

- Fine-tuned on 1,000+ validated graphs across 8 industries
- GNN validation layer — structural pattern recognition
- Industry coverage: fintech, compliance, supply chain, healthcare,
  cybersecurity, HR, e-commerce, media
- 95%+ precision on supported industries

---

## v3.0 — The knowledge machine (Q4 2026)
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

## Not on the roadmap

We won't build a hosted service, a managed database, or a visual graph editor.
We're a library. Other tools do those things well.

---

## How to influence this roadmap

Open an issue. Label it `roadmap`.

The features that get built are the ones the community actually needs.
We track everything in [GitHub Projects](https://github.com/gibbrdev/gibsgraph/projects).
