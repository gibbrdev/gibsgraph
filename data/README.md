# Expert Knowledge Graph — Data

This directory contains the raw and processed data that powers GibsGraph's expert knowledge graph.

## Structure

```
data/
├── raw/                  # Unprocessed source material
│   ├── docs/             # Neo4j official docs (cloned AsciiDoc repos)
│   ├── papers/           # Research papers (PDFs + extracted text)
│   ├── use_cases/        # Production use cases and case studies
│   └── schemas/          # Documented graph schemas by industry
├── processed/            # Cleaned, structured JSONL ready for Neo4j loading
├── scripts/              # Parsing and loading scripts
└── README.md
```

## Authority Levels

| Level | Source Type | Trust |
|-------|-----------|-------|
| 1 | Neo4j official documentation | Highest |
| 2 | Peer-reviewed research papers | High |
| 3 | Production case studies | High |
| 4 | Community examples (GraphGists, forums) | Medium |

## Data is NOT committed to git

Raw docs and papers are large. Clone/download them locally using:
```bash
python data/scripts/fetch_sources.py
```
