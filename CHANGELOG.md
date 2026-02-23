# Changelog

All notable changes to GibsGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-23

### Added
- `Graph` facade — one class, two methods (`ask`, `ingest`), zero boilerplate
- LangGraph agent pipeline: classify, retrieve, explain, validate, visualize
- Auto schema discovery — connects to any Neo4j graph and learns its structure
- Dual retrieval strategy: vector search (when available) or text-to-Cypher
- LLM-powered Cypher generation from natural language using discovered schema
- LLM-powered answer generation with source citations from graph data
- Query classification (graph_structure, cross_reference, compliance, general)
- Cypher injection validator with parameterized query enforcement
- Neo4j Bloom deep links + Mermaid diagram visualization
- KGBuilder for text-to-graph ingestion (stub, SimpleKGPipeline pending)
- G-Retriever GNN inference wrapper (stub, model weights pending)
- Streamlit demo UI with ask, ingest, and visualization tabs
- CLI: `gibsgraph ask "question"` and `gibsgraph ingest file.txt`
- Docker + docker-compose with Neo4j
- CI pipeline: ruff, mypy, pytest, bandit, trivy
- Pydantic v2 settings with `.env` file support
- LLM auto-detection from environment (OpenAI / Anthropic)
- Read-only mode by default — safe for production queries
- structlog for structured logging throughout

### Project structure
- `src/gibsgraph/` — src-layout with hatch build backend
- `src/gibsgraph/retrieval/` — graph retrieval with schema discovery
- `src/gibsgraph/kg_builder/` — knowledge graph builder
- `src/gibsgraph/gnn/` — G-Retriever GNN package
- `src/gibsgraph/tools/` — Cypher validator, visualizer
- `tests/unit/` and `tests/integration/` — pytest test suites
- `app/` — Streamlit demo
- `examples/` — usage examples (regulatory KG)
- `.github/` — CI workflows, issue templates, dependabot
