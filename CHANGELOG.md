# Changelog

All notable changes to GibsGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-02-23

### Added
- Neo4j 5 syntax rules in Cypher generation prompt (`COUNT{}`, `shortestPath`, `elementId`, bounded variable-length paths)
- Cypher self-healing retry — when generated Cypher fails execution, the error is sent back to the LLM for automatic correction (1 retry)

### Fixed
- `human_review` route missing from LangGraph conditional edges (caused KeyError on validation failures)
- All `response.content.strip()` calls wrapped with `str()` to handle LangChain's `str | list` union return type
- All `dict` type annotations upgraded to `dict[str, Any]` for mypy strict compliance
- `TYPE_CHECKING` import pattern for `GibsGraphAgent` to avoid circular imports
- Dockerfile now copies `LICENSE` file (was causing `OSError` during build)
- Renamed ambiguous variable `l` to `line` in retriever
- Sorted `__all__` exports in `__init__.py` files (ruff RUF022)
- Removed deprecated `ANN101`/`ANN102` rules from pyproject.toml
- Fixed EN DASH character in docstring (ruff RUF002)

### Testing
- Rewrote `test_agent_state.py` with proper LLM mocking (`ChatOpenAI` patched)
- Rewrote `test_agent_integration.py` with dual LLM mock fixture
- Added `test_retriever.py` — 6 unit tests (schema serialization, sample values, defaults, close, context serialization)
- **22 tests passing**, 47% code coverage
- Stress-tested against Neo4j Movies dataset (10/10 queries passing):
  - Simple lookups: actor filmography, director identification, node counts
  - Cross-references: co-actors across multiple people, director-producer overlap
  - Aggregations: oldest movie with cast, most connected person
  - Path finding: shortest path between actors (Kevin Bacon test)
  - Adversarial: off-topic questions handled gracefully, Cypher injection neutralized, empty queries rejected cleanly

### CI
- All 4 CI jobs green: lint & type check, Python 3.12 tests, Python 3.13 tests, Docker build & Trivy scan
- mypy strict: 0 errors across 15 source files
- ruff: 0 lint errors, 0 format issues
- bandit: 0 security findings

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
