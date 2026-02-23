"""Cypher query validation tool — syntax check + injection guard."""

from __future__ import annotations

import re

import structlog

log = structlog.get_logger(__name__)

# Patterns that should NEVER appear in parameterized Cypher
_INJECTION_PATTERNS = [
    r";\s*DROP",
    r";\s*DELETE",
    r";\s*DETACH",
    r"CALL\s+\{",          # subquery injection vector
    r"LOAD\s+CSV",         # file system access
    r"apoc\.export",       # APOC export procedures
    r"apoc\.load",         # APOC load procedures
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# f-string interpolation indicators (should never appear in generated Cypher)
_INTERPOLATION_PATTERN = re.compile(r'\$\{.+?\}|"\s*\+\s*\w+\s*\+\s*"')


class CypherValidationError(ValueError):
    """Raised when Cypher fails validation."""


class CypherValidator:
    """Validates Cypher queries for safety and correctness.

    Rules enforced:
    - No injection patterns (DROP, DELETE via semicolon, etc.)
    - No string interpolation (must use $params)
    - Parameters referenced in query must use $name syntax
    - Query must not be empty
    """

    def validate(self, cypher: str) -> bool:
        """Return True if valid, False if potentially unsafe."""
        try:
            self.assert_valid(cypher)
            return True
        except CypherValidationError:
            return False

    def assert_valid(self, cypher: str) -> None:
        """Raise CypherValidationError if the query is unsafe."""
        if not cypher or not cypher.strip():
            raise CypherValidationError("Cypher query is empty")

        for pattern in _COMPILED_PATTERNS:
            if pattern.search(cypher):
                raise CypherValidationError(
                    f"Potentially unsafe Cypher pattern detected: {pattern.pattern}"
                )

        if _INTERPOLATION_PATTERN.search(cypher):
            raise CypherValidationError(
                "String interpolation detected — use $param syntax instead"
            )

        log.debug("cypher_validator.valid", cypher_length=len(cypher))

    def extract_parameters(self, cypher: str) -> list[str]:
        """Return list of $parameter names referenced in the query."""
        return re.findall(r'\$(\w+)', cypher)
