# Contributing to GibsGraph

First off — thank you! 🎉 GibsGraph is built by the community.

## Quick start for contributors

```bash
git clone https://github.com/gibbrdev/gibsgraph
cd gibsgraph
pip install -e ".[dev]"
pre-commit install
cp .env.example .env  # fill in your Neo4j + OpenAI keys
```

## Good First Issues

New here? Look for issues labeled [`good first issue`](https://github.com/gibbrdev/gibsgraph/issues?q=is%3Aopen+label%3A%22good+first+issue%22). These are:

- 📝 Adding docstrings to existing functions
- ✅ Writing unit tests for edge cases  
- 📖 Improving example notebooks
- 🐛 Small, well-defined bugs

## Workflow

1. **Fork** the repo and create a branch: `git checkout -b feat/my-feature`
2. **Write code** — follow existing patterns
3. **Write tests** — aim to keep coverage above 80%
4. **Lint**: `ruff check . && mypy src/`
5. **Test**: `pytest tests/unit/`
6. **Commit**: use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`)
7. **Open a PR** against `main` — fill in the PR template

## Code style

- **Ruff** for linting and formatting (line length 100)
- **mypy** strict mode — no `Any` unless justified
- **Pydantic** models for all data structures
- **Parameterized Cypher only** — never f-strings in queries
- **structlog** for logging — no bare `print()` in library code

## Security

If you find a security vulnerability, please **do not** open a public issue. See [SECURITY.md](SECURITY.md).

## Questions?

Open a [Discussion](https://github.com/gibbrdev/gibsgraph/discussions) or join [Discord](https://discord.gg/gibsgraph).
