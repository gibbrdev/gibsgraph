# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅        |

## Reporting a Vulnerability

GibsGraph follows **Coordinated Disclosure** via GitHub Security Advisories.

**Please do NOT open a public GitHub issue for security vulnerabilities.**

Instead:
1. Go to [Security Advisories](https://github.com/vibecoder/gibsgraph/security/advisories/new)
2. Describe the vulnerability, steps to reproduce, and impact
3. We will respond within **48 hours** and coordinate a fix + CVE if needed

## Security practices in this codebase

- All Cypher queries use `$parameters` — never string interpolation
- Neo4j connection defaults to read-only (`NEO4J_READ_ONLY=true`)
- Credentials via `.env` only — never hardcoded
- Dependencies scanned weekly via Dependabot + Trivy in CI
- CodeQL analysis enabled on all PRs
- `bandit` and `detect-secrets` in pre-commit hooks
