# GibsGraph — Task Tracker

> Source of truth for what's done, what's in progress, and what's next.
> Updated every session. Every task has a date.

---

## The Vision

Build a **Neo4j expert system** — not a generic RAG tool.

1. Build an expert knowledge graph from Neo4j docs, research papers, Cypher patterns, and real use cases
2. Use that expert to generate knowledge graphs for any domain
3. Generate hundreds of validated use cases, train on them
4. End result: describe your business → get a production knowledge graph

---

## Completed

| Date | Task | Notes |
|------|------|-------|
| 2026-02-23 | Code review + security audit | 23 issues fixed across 14 files |
| 2026-02-23 | Tagged v0.1.1 | First stable release |
| 2026-02-23 | GitHub profile migration | All URLs updated from buildsyncinc/vibecoder → gibbrdev |
| 2026-02-23 | Made repo public | `gh repo edit --visibility public` |
| 2026-02-23 | Test coverage 37% → 70% | 88 new tests across 5 test files |
| 2026-02-23 | Centralized LLM provider registry | OpenAI, Anthropic, Mistral, xAI/Grok support |
| 2026-02-23 | CI lint/mypy fixes | ruff format + 3 mypy errors in _make_llm |
| 2026-02-23 | PyPI published v0.2.0 | Trusted publisher via OIDC, `pip install gibsgraph` works |
| 2026-02-23 | Created 6 GitHub issues | good-first-issue labels, contributor-friendly |
| 2026-02-23 | Loaded Epoch AI dataset (removed) | Demo exercise, removed to keep graph pure |
| 2026-02-24 | Cloned 5 Neo4j doc repos | docs-cypher, docs-getting-started, docs-operations, docs-drivers, knowledge-base (168 MB) |
| 2026-02-24 | Built AsciiDoc parser pipeline | parse_cypher_docs.py + parse_modeling_docs.py |
| 2026-02-24 | Parsed expert knowledge to JSONL | 27 clauses, 122 functions, 383 examples, 20 patterns, 309 best practices |
| 2026-02-24 | Loaded expert graph into Neo4j | 715 nodes of pure Neo4j expertise, 0 noise |
| 2026-02-24 | Generated expert embeddings | 849 vectors x 384 dims (all-MiniLM-L6-v2), semantic search verified |
| 2026-02-25 | Expert graph wired into g.ask() | ExpertStore (fulltext search), expert context injected into Cypher generation, expert labels/rels/indexes filtered from user schema discovery. 126 tests, 72% coverage. |
| 2026-02-25 | Loaded embeddings into Neo4j | 663/849 embeddings loaded, vector index `expert_embedding` (384-dim cosine) + fulltext index `expert_fulltext` created and verified. |
| 2026-02-25 | Layer 2: Validation Suite built | 4-stage pipeline (syntactic→structural→semantic→domain). 4 source files: models.py, prompts.py, scorer.py, validator.py. Socratic LLM scoring with 12 yes/no questions. |
| 2026-02-25 | Layer 2 tests: 61 unit tests | test_models.py (14), test_scorer.py (21), test_validator.py (26). Real schemas, edge cases, no doped tests. 187 total tests, 77% coverage. |

## In Progress

| Date | Task | Status |
|------|------|--------|
| 2026-02-25 | Expert graph integration test with live embeddings | PENDING — need live Neo4j to test vector/fulltext search end-to-end |

## Resume Next Session

> Pick up here. Layer 2 validation suite built. Next: wire into pipeline + start use case generation.

1. **Expert graph integration test** — test ExpertStore with real Neo4j (715 nodes, 663 embeddings, fulltext + vector indexes)
2. **Publish expert dataset** — HuggingFace or similar. Files ready in `data/processed/`
3. **Phase 2: `g.ingest()`** — documents → knowledge graph, guided by the expert

## Next — The Real Product

### Phase 1: Expert Knowledge Graph (the foundation) — MOSTLY DONE

> Without this, GibsGraph is just another RAG wrapper.

- [x] Scrape/parse Neo4j official documentation into structured data
- [x] Parse Cypher reference manual — every function, clause, pattern
- [x] Collect graph modeling best practices (Graphacademy, Neo4j blog, books)
- [x] Design the expert graph schema (CypherClause, CypherFunction, CypherExample, ModelingPattern, BestPractice, etc.)
- [x] Load expert knowledge into Neo4j as the "expert graph"
- [x] Generate embeddings for vector search (849 vectors, 384 dimensions)
- [x] Wire expert graph into GibsGraph runtime (fulltext search, expert-guided Cypher generation)
- [x] Load embeddings into Neo4j (663 vectors, vector index + fulltext index)
- [ ] Expert graph integration test with live embeddings (vector search end-to-end)
- [ ] Publish expert dataset (first open Neo4j expert embedding dataset)

### Phase 2: Ingest Pipeline (`g.ingest()`)

> "Give me your documents, I'll build your knowledge graph."

- [ ] Implement `KGBuilder.ingest()` — text/PDF/CSV → entities + relationships → Neo4j
- [ ] LLM-powered entity extraction with expert graph guidance
- [ ] Schema suggestion based on expert knowledge ("for compliance data, use this pattern")
- [ ] Validation: every generated graph checked against expert patterns

### Phase 3: Use Case Generation + Training

> "Generate 1,000 validated graphs. Train on them."

- [ ] Generate synthetic use cases per industry using expert graph
- [x] Validate each generated graph (syntactic → structural → semantic → domain) — Layer 2 built
- [ ] Build training dataset from validated graphs
- [ ] Fine-tune or train GNN on validated patterns

### Phase 4: The Expert (`g.map()`)

> "Describe your company. Get your graph."

- [ ] `g.map("We're a fintech processing cross-border payments")` → production schema
- [ ] Learning loop: every production graph feeds back
- [ ] The more it's used, the better it gets

---

## Backlog (infrastructure, not urgent)

- [ ] README rewrite with clear product positioning
- [ ] Demo GIF/screenshot for README
- [ ] CHANGELOG.md
- [ ] Tackle GitHub issues (#5-#10)
- [ ] Streamlit app improvements

---

## Decisions Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-02-23 | Paused infrastructure work | Was drifting from vision. CI/tests/PyPI done — foundation is solid. |
| 2026-02-23 | Expert knowledge graph is the #1 priority | This is what differentiates GibsGraph from every other GraphRAG tool. |
| 2026-02-24 | Removed Epoch AI data from graph | Keep expert graph pure — only Neo4j knowledge, no demo noise. |
| 2026-02-24 | Chose all-MiniLM-L6-v2 for embeddings | 384-dim, fast, good quality. First open Neo4j expert embedding dataset. |
| 2026-02-25 | Fulltext search as default for OSS | Zero extra deps (no sentence-transformers/torch). Vector index pre-computed for future upgrade path. |
| 2026-02-25 | Renamed `schema` → `graph_schema` in training models | Avoids Pydantic BaseModel.schema() collision. |
| 2026-02-25 | Spec moved to docs/specs/ (gitignored) | USECASE_PIPELINE_SPEC_V2.md — keeps repo clean, internal docs out of public. |
