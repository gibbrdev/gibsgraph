# Changelog

All notable changes to GibsGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-02-23

### Added
- Neo4j 5 syntax rules in Cypher generation prompt (`COUNT{}`, `shortestPath`, `elementId`, bounded variable-length paths)
- Cypher self-healing retry — when generated Cypher fails execution, the error is sent back to the LLM for automatic correction (1 retry)
- Vector search with embedding cleanup and automatic Cypher fallback when vector returns no results
- Stress test framework (`tests/stress/`) with hybrid assertions: behavioral + ground truth + anti-hallucination
- `Graph.close()` and context manager support (`with Graph(...) as g:`)
- Anthropic LLM support — `ChatAnthropic` used when model name starts with `claude`
- CLI `--help` and `--version` flags

### Fixed
- `pydantic-settings` added to base dependencies (was dev-only, broke `pip install gibsgraph`)
- CypherValidator now blocks `CREATE`, `SET`, `MERGE`, `REMOVE`, `DELETE`, `FOREACH` (was only blocking `;DROP`)
- Generated Cypher uses `session.execute_read()` for server-enforced read-only transactions
- `COUNT { }` double-brace escaping in Cypher prompt — was sending literal `{{ }}` to LLM
- Removed dead classification LLM call that burned tokens without changing pipeline behavior
- Neo4j driver and schema now reused across queries instead of recreating per call
- Bloom URL Cypher now uses double-quoted string literals (valid Cypher, not Python repr)
- PyVis nodes use element ID as key instead of label (fixes edge mismatches)
- Mermaid labels escaped for double-quote characters
- `ingest()` stub now raises `NotImplementedError` instead of silently returning zeros
- CLI file handle leak in `_cmd_ingest` — now uses `Path.read_text(encoding="utf-8")`
- CLI uses `with Graph() as g:` for proper connection cleanup
- Dockerfile healthcheck uses Python instead of `curl` (not available in `python:3.12-slim`)
- docker-compose: removed deprecated `version` key, Neo4j ports bound to localhost only, fixed config namespace
- README rewritten — removed false feature claims, dead example links, fake badges, incorrect version
- CLAUDE.md: removed reference to confidential ROADMAP_INTERNAL.md
- pyproject.toml: fixed author email, removed duplicate `[llm]` extra, corrected GitHub URLs
- `human_review` route missing from LangGraph conditional edges (caused KeyError on validation failures)
- All `response.content.strip()` calls wrapped with `str()` to handle LangChain's `str | list` union return type
- All `dict` type annotations upgraded to `dict[str, Any]` for mypy strict compliance
- `TYPE_CHECKING` import pattern for `GibsGraphAgent` to avoid circular imports
- Dockerfile now copies `LICENSE` file (was causing `OSError` during build)
- Renamed ambiguous variable `l` to `line` in retriever
- Sorted `__all__` exports in `__init__.py` files (ruff RUF022)
- Removed deprecated `ANN101`/`ANN102` rules from pyproject.toml
- Fixed EN DASH character in docstring (ruff RUF002)

### Security
- CypherValidator upgraded from semicolon-only blocklist to full write-operation blocklist
- Read transactions enforced at Neo4j driver level (`execute_read`)
- Bloom URL Cypher no longer uses `repr()` f-string interpolation
- docker-compose: Neo4j ports bound to `127.0.0.1` (was `0.0.0.0`)

### Testing
- **18 unit tests passing**, stress test 11/11 (100%) on Neo4j Movies dataset
- Stress test categories: simple lookups, cross-references, aggregations, path finding, adversarial
- Hybrid test assertions: behavioral (no crash, has data, has cypher) + ground truth (DB facts only) + anti-hallucination

### CI
- All 4 CI jobs green: lint & type check, Python 3.12 tests, Python 3.13 tests, Docker build & Trivy scan
- mypy strict: 0 errors across 15 source files
- ruff: 0 lint errors, 0 format issues
- bandit: 0 security findings
- Merged 4 dependabot PRs (actions/checkout@v6, setup-python@v6, setup-uv@v7, codecov-action@v5)

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
