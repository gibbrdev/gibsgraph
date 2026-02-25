"""Expert knowledge store — searches the Neo4j expert graph.

The expert graph contains 715 nodes of Neo4j best practices, Cypher
patterns, function signatures, and modeling guidance parsed from
official documentation.

This module searches that knowledge and returns formatted context
for the LLM to generate better Cypher queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from neo4j import Driver

log = structlog.get_logger(__name__)

FULLTEXT_INDEX = "expert_fulltext"
VECTOR_INDEX = "expert_embedding"


@dataclass
class ExpertHit:
    """A single expert knowledge result."""

    label: str
    name: str
    score: float
    description: str = ""
    cypher: str = ""
    signature: str = ""


@dataclass
class ExpertContext:
    """Expert knowledge relevant to a user query."""

    hits: list[ExpertHit] = field(default_factory=list)
    query: str = ""

    def to_prompt(self) -> str:
        """Format expert knowledge for inclusion in an LLM prompt."""
        if not self.hits:
            return ""

        sections: list[str] = [
            "Neo4j Expert Knowledge (use these patterns to write better Cypher):"
        ]

        examples: list[str] = []
        practices: list[str] = []
        functions: list[str] = []
        patterns: list[str] = []

        for h in self.hits:
            if h.label == "CypherExample" and h.cypher:
                examples.append(f"  {h.cypher}")
            elif h.label == "BestPractice":
                desc = h.description[:200] if h.description else ""
                practices.append(f"  - {h.name}: {desc}")
            elif h.label == "CypherFunction" and h.signature:
                functions.append(f"  - {h.name}: {h.signature}")
            elif h.label == "CypherClause":
                desc = h.description[:150] if h.description else ""
                practices.append(f"  - {h.name}: {desc}")
            elif h.label == "ModelingPattern":
                desc = h.description[:200] if h.description else ""
                patterns.append(f"  - {h.name}: {desc}")

        if examples:
            sections.append("Relevant Cypher examples:")
            sections.extend(examples[:5])
        if practices:
            sections.append("Best practices:")
            sections.extend(practices[:3])
        if functions:
            sections.append("Relevant functions:")
            sections.extend(functions[:3])
        if patterns:
            sections.append("Modeling patterns:")
            sections.extend(patterns[:2])

        return "\n".join(sections)


class ExpertStore:
    """Searches the expert knowledge graph for query-relevant context.

    Uses Neo4j fulltext index for keyword-based search. The expert graph
    lives in the same Neo4j instance as the user's data — expert nodes
    have distinct labels (CypherClause, CypherFunction, etc.) that don't
    clash with user data.
    """

    _FULLTEXT_QUERY = """
    CALL db.index.fulltext.queryNodes($index, $search_text, {limit: $limit})
    YIELD node, score
    RETURN labels(node)[0] AS label,
           coalesce(node.name, node.title) AS name,
           score,
           coalesce(node.description, '') AS description,
           coalesce(node.cypher, '') AS cypher,
           coalesce(node.signature, '') AS signature
    """

    def __init__(self, driver: Driver, *, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check if the expert fulltext index exists and is online."""
        if self._available is not None:
            return self._available

        try:
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    "SHOW INDEXES YIELD name, type, state "
                    "WHERE name = $name AND type = 'FULLTEXT' AND state = 'ONLINE' "
                    "RETURN count(*) AS cnt",
                    name=FULLTEXT_INDEX,
                )
                record = result.single()
                self._available = bool(record and record["cnt"] > 0)
        except Exception:
            log.debug("expert.index_check_failed")
            self._available = False

        log.info("expert.available", available=self._available)
        return self._available

    def search(self, query: str, *, top_k: int = 8) -> ExpertContext:
        """Search expert knowledge for patterns relevant to the query.

        Args:
            query: The user's natural language question.
            top_k: Maximum number of expert results to return.

        Returns:
            ExpertContext with matched hits, ready for prompt injection.
        """
        if not self.is_available():
            return ExpertContext(query=query)

        try:
            # Lucene query: escape special chars, use AND for better precision
            lucene_query = _to_lucene(query)

            with self._driver.session(database=self._database) as session:
                result = session.run(
                    self._FULLTEXT_QUERY,
                    index=FULLTEXT_INDEX,
                    search_text=lucene_query,
                    limit=top_k,
                )
                hits = [
                    ExpertHit(
                        label=r["label"],
                        name=r["name"] or "",
                        score=r["score"],
                        description=r["description"],
                        cypher=r["cypher"],
                        signature=r["signature"],
                    )
                    for r in result
                ]

            log.info("expert.search", query=query[:60], hits=len(hits))
            return ExpertContext(hits=hits, query=query)

        except Exception as exc:
            log.warning("expert.search_failed", error=str(exc))
            return ExpertContext(query=query)


def _to_lucene(query: str) -> str:
    """Convert a natural language query to a Lucene fulltext query.

    Escapes special characters and joins words with spaces (OR semantics).
    """
    # Lucene special characters that need escaping
    special = set('+-&|!(){}[]^"~*?:\\/>')
    cleaned = []
    for ch in query:
        if ch in special:
            cleaned.append(f"\\{ch}")
        else:
            cleaned.append(ch)
    return "".join(cleaned)
