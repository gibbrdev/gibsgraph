# Changelog

All notable changes to GibsGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4] - 2026-02-26

### Added
- `g.ingest()` — text-to-graph ingestion via neo4j-graphrag `SimpleKGPipeline`
- Free-form entity/relationship extraction from any text (no schema required)
- Automatic entity resolution to prevent duplicates on repeated ingestion
- Multi-provider LLM support for ingestion: OpenAI, Anthropic, Mistral, xAI/Grok
- Friendly error message when running ingest inside an existing event loop (Jupyter)

### Testing
- 12 new unit tests for `KGBuilder.ingest()`, LLM/embedder factories, and facade delegation

## [0.3.3] - 2026-02-26

### Added
- 92 curated expert records: cross-reference joins, aggregation chains, path queries, subqueries, APOC procedures, GDS algorithms, industry modeling patterns, error recovery practices, index guidance
- 11 APOC/GDS procedure signatures (apoc.path.expandConfig, gds.pageRank.stream, gds.louvain.stream, etc.)
- 8 industry modeling patterns: supply chain, social network, fraud detection, e-commerce, IT infrastructure, healthcare, content management, organizational hierarchy
- 10 best practices: 5 error recovery (cartesian products, type mismatch, eager aggregation, missing UNWIND, relationship direction) + 5 index guidance (RANGE, TEXT, VECTOR, composite, when NOT to index)
- Quality-tier filtering in `BundledExpertStore._load()` — skips `quality_tier: "low"` records
- Expanded keyword search: `category`, `when_to_use`, and embedded `cypher_examples` now indexed

### Fixed
- Invalid nested `collect(collect(...))` in APOC collection example (two-step aggregation)
- Null end node in `apoc.create.vRelationship` example (now passes actual Region node)
- Deprecated `gds.graph.project.cypher` replaced with `gds.graph.filter` (GDS 2.5+)
- Removed 5 duplicate modeling patterns and 1 duplicate best practice
- Re-tagged ~88 irrelevant security/ops articles as `quality_tier: "low"`

### Changed
- Expert data: 920 records on disk, ~849 loaded after quality filtering (was 834/834)

### Testing
- 150 unit tests passing (was 224 in v0.3.2 count, now 150 excluding training property tests), 52% unit coverage
- 44 expert-specific tests (was 36): quality-tier filtering, cross-reference search, GDS/APOC search, industry patterns, deduplication checks, cypher_examples extraction

## [0.3.2] - 2026-02-26

### Added
- Bundled expert data in package — 5 JSONL files (~750K) ship with `pip install gibsgraph`
- `BundledExpertStore` — keyword-based fallback when Neo4j expert index is unavailable
- Expert knowledge works out of the box without loading data into Neo4j first
- `src/gibsgraph/data/` package: cypher_clauses, cypher_functions, cypher_examples, modeling_patterns, best_practices

### Changed
- `ExpertStore.search()` falls back to bundled JSONL instead of returning empty when Neo4j unavailable
- `ExpertStore.search()` falls back to bundled JSONL on Neo4j query exceptions (was returning empty)

### Testing
- 224 unit tests passing (was 216), 79% coverage
- 15 new tests: `BundledExpertStore`, bundled data file validation, tokenizer, fallback wiring

## [0.3.1] - 2026-02-26

### Added
- Enterprise severity levels for validation findings (ERROR/WARNING/INFO)
- `Finding` model with severity, stage, and message fields
- `ValidationResult.errors`, `.warnings`, `.infos` convenience properties
- 9 missing Cypher clauses: YIELD, DISTINCT, UNION, UNION ALL, CASE, WHEN, THEN, ELSE, SHOW (27 → 36 total)
- `quality_tier` field on best practices data (high/medium/low — 30%/63%/6%)
- Independent golden test fixtures: 3 known-good, 3 known-bad, 3 adversarial schemas
- Hypothesis property-based tests for structural and Cypher scoring
- Data quality checker script (`validate_expert_graph.py`) — queries live Neo4j instead of self-validation
- `hypothesis` added to `[dev]` extras

### Changed
- Semantic validation now queries live Neo4j for real data quality (labels exist, rel types exist, orphan nodes, property completeness) instead of circular keyword overlap against expert graph
- Semantic score without Neo4j driver is 0.0, not 0.5 — no benefit-of-the-doubt scoring
- Approval now requires live database connection (overall < 0.7 without semantic checks)
- Empty schema structural score: 0.167 (was 0.667 due to vacuous truth bug)

### Fixed
- Vacuous truth in `score_structural()` — `all([])` returns True in Python, empty schemas got unearned credit on 8 of 12 checks
- DEMONSTRATES relationships: 0 → 200 (CALL subquery bug in Neo4j silently returned 0 rows)
- SOURCED_FROM null source_file for BestPractice (308) and ModelingPattern (15)
- Generic labels downgraded from ERROR → WARNING severity (don't block syntactic gate)
- Expert graph indexes now filtered from user schema discovery in retriever

### Removed
- Circular semantic validation (expert graph searching itself for keyword overlap)
- `ExpertStore` dependency in `SchemaValidator` (replaced with direct Neo4j queries)
- Benefit-of-the-doubt 0.5 score for unavailable semantic validation

### Testing
- 216 unit tests passing (was 199), 78% coverage
- Golden fixture tests: supply chain, social network, fraud detection, plus adversarial schemas
- Property-based tests verify scoring invariants hold for random inputs
- All adversarial schemas correctly blocked from approval

## [0.3.0] - 2026-02-25

### Added
- Expert knowledge graph — 715 nodes (27 clauses, 122 functions, 383 examples, 20 patterns, 309 best practices)
- Expert embeddings — 849 vectors × 384 dims (all-MiniLM-L6-v2), vector + fulltext indexes
- 4-stage validation suite (syntactic → structural → semantic → domain)
- ExpertStore fulltext search wired into `g.ask()`
- `SchemaValidator`, `GraphSchema`, `Finding` models in `training/`

### Testing
- 187 tests, 77% coverage

## [0.2.0] - 2026-02-23

### Added
- Centralized LLM provider registry — OpenAI, Anthropic, Mistral, xAI/Grok auto-detected

### Changed
- Test coverage 37% → 70% (88 new tests across 5 files)
- GitHub URLs migrated (buildsyncinc/vibecoder → gibbrdev)

### Fixed
- mypy errors in `_make_llm` for Mistral and xAI providers

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

[Unreleased]: https://github.com/gibbrdev/gibsgraph/compare/v0.3.4...HEAD
[0.3.4]: https://github.com/gibbrdev/gibsgraph/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/gibbrdev/gibsgraph/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/gibbrdev/gibsgraph/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/gibbrdev/gibsgraph/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/gibbrdev/gibsgraph/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/gibbrdev/gibsgraph/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/gibbrdev/gibsgraph/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/gibbrdev/gibsgraph/releases/tag/v0.1.0
