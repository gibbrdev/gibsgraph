"""Four-stage schema validation pipeline.

Validates generated Neo4j schemas against the expert knowledge graph:

  Stage 1 — SYNTACTIC:  Is the Cypher parseable? No dangerous keywords?
  Stage 2 — STRUCTURAL: Required elements present? All justified?
  Stage 3 — SEMANTIC:   Do labels/relationships align with expert patterns?
  Stage 4 — DOMAIN:     Is it domain-complete? Regulatory-correct? (LLM)

Stages 1-3 are fully deterministic (no LLM). Stage 4 uses Socratic
scoring via Haiku when available, falls back to deterministic-only.
"""

from __future__ import annotations

import re

import structlog
from neo4j import Driver

from gibsgraph.expert import ExpertStore
from gibsgraph.training.models import GraphSchema, SynthesisResult, ValidationResult
from gibsgraph.training.prompts import score_cypher_quality, score_structural
from gibsgraph.training.scorer import QualityScorer

log = structlog.get_logger(__name__)

# Labels that are too generic — a red flag for quality
GENERIC_LABELS = {
    "Entity",
    "Object",
    "Item",
    "Thing",
    "Node",
    "Data",
    "Record",
    "Element",
    "Resource",
    "Entry",
}

# Cypher keywords that must never appear in a schema setup script.
# Shared with prompts.score_cypher_quality — keep in sync.
FORBIDDEN_KEYWORDS = {"DELETE", "DETACH", "DROP", "REMOVE", "FOREACH", "CALL {"}


class SchemaValidator:
    """Validates generated graph schemas in 4 stages.

    Usage::

        validator = SchemaValidator(neo4j_driver, database="neo4j")
        result = validator.validate(schema)
        print(result.overall_score, result.findings)
    """

    def __init__(
        self,
        driver: Driver | None = None,
        *,
        database: str = "neo4j",
    ) -> None:
        self._expert: ExpertStore | None = None
        if driver is not None:
            self._expert = ExpertStore(driver, database=database)

    def validate(self, schema: GraphSchema) -> ValidationResult:
        """Run all 4 validation stages. Returns ValidationResult."""
        findings: list[str] = []

        # Stage 1: Syntactic
        syntactic_ok, syntactic_findings = self._validate_syntactic(schema)
        findings.extend(syntactic_findings)

        # Stage 2: Structural
        structural_score, structural_findings = score_structural(schema)
        findings.extend(structural_findings)

        # Stage 3: Semantic (expert graph alignment)
        semantic_score, semantic_findings = self._validate_semantic(schema)
        findings.extend(semantic_findings)

        # Stage 4: Domain (deterministic only — LLM scoring via QualityScorer)
        cypher_score, cypher_findings = score_cypher_quality(schema.cypher_setup)
        findings.extend(cypher_findings)

        # Combine scores
        overall = self._compute_overall(
            syntactic=syntactic_ok,
            structural=structural_score,
            semantic=semantic_score,
            cypher=cypher_score,
        )

        return ValidationResult(
            syntactic=syntactic_ok,
            structural_score=structural_score,
            semantic_score=semantic_score,
            domain_score=cypher_score,
            overall_score=overall,
            findings=findings,
            approved_for_training=overall >= 0.7 and syntactic_ok,
        )

    def validate_full(
        self,
        synthesis: SynthesisResult,
        *,
        research_context: str,
        expert_patterns: list[str],
        industry: str,
        differentiators: list[str],
        settings: object | None = None,
    ) -> ValidationResult:
        """Full validation including LLM Socratic scoring (Stage 4).

        Requires Settings with an LLM API key configured.
        """
        from gibsgraph.config import Settings

        if not isinstance(settings, Settings):
            # Fall back to deterministic-only
            return self.validate(synthesis.graph_schema)

        # Run deterministic stages
        base_result = self.validate(synthesis.graph_schema)

        # Run LLM-assisted scoring for domain dimension
        scorer = QualityScorer(settings)
        overall, breakdown, scorer_findings = scorer.score(
            synthesis,
            research_context=research_context,
            expert_patterns=expert_patterns,
            industry=industry,
            differentiators=differentiators,
        )

        all_findings = base_result.findings + scorer_findings

        return ValidationResult(
            syntactic=base_result.syntactic,
            structural_score=base_result.structural_score,
            semantic_score=base_result.semantic_score,
            domain_score=breakdown.get("regulatory_coverage", 0.0),
            overall_score=overall,
            findings=all_findings,
            approved_for_training=overall >= 0.7 and base_result.syntactic,
        )

    # ------------------------------------------------------------------
    # Stage 1: Syntactic validation
    # ------------------------------------------------------------------

    def _validate_syntactic(self, schema: GraphSchema) -> tuple[bool, list[str]]:
        """Check that the Cypher setup is syntactically safe."""
        findings: list[str] = []
        ok = True

        if not schema.cypher_setup.strip():
            findings.append("SYNTACTIC: Cypher setup script is empty")
            return False, findings

        # Check for forbidden write operations
        upper = schema.cypher_setup.upper()
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in upper:
                findings.append(f"SYNTACTIC: Forbidden keyword '{keyword}' in Cypher setup")
                ok = False

        # Check that constraints reference existing node labels
        schema_labels = {n.label for n in schema.nodes}
        for constraint in schema.constraints:
            # Extract label from constraint: FOR (x:Label)
            match = re.search(r"FOR\s*\(\w+:(\w+)\)", constraint, re.IGNORECASE)
            if match:
                label = match.group(1)
                if label not in schema_labels:
                    findings.append(f"SYNTACTIC: Constraint references unknown label '{label}'")
                    ok = False

        # Check for generic labels
        for node in schema.nodes:
            if node.label in GENERIC_LABELS:
                findings.append(
                    f"SYNTACTIC: Generic label '{node.label}' — use a domain-specific name"
                )
                ok = False

        return ok, findings

    # ------------------------------------------------------------------
    # Stage 3: Semantic validation (expert graph alignment)
    # ------------------------------------------------------------------

    def _validate_semantic(self, schema: GraphSchema) -> tuple[float, list[str]]:
        """Check if schema elements align with expert knowledge."""
        findings: list[str] = []

        if self._expert is None or not self._expert.is_available():
            findings.append("SEMANTIC: Expert graph not available — skipped")
            return 0.5, findings  # neutral score when unavailable

        total_checks = 0
        passed_checks = 0

        # Check if relationship types match known patterns
        for rel in schema.relationships:
            total_checks += 1
            query = f"{rel.from_label} {rel.type} {rel.to_label} graph modeling"
            ctx = self._expert.search(query, top_k=3)
            if ctx.hits and ctx.hits[0].score > 0.5:
                passed_checks += 1
            else:
                findings.append(
                    f"SEMANTIC: Relationship (:{rel.from_label})-[:{rel.type}]->"
                    f"(:{rel.to_label}) has no expert pattern match"
                )

        # Check if constraint patterns follow best practices
        if schema.constraints:
            total_checks += 1
            ctx = self._expert.search("uniqueness constraint best practice", top_k=2)
            if ctx.hits:
                passed_checks += 1

        # Check if indexes follow query pattern best practices
        if schema.indexes:
            total_checks += 1
            ctx = self._expert.search("index query performance best practice", top_k=2)
            if ctx.hits:
                passed_checks += 1

        score = round(passed_checks / total_checks, 3) if total_checks > 0 else 0.0
        return score, findings

    # ------------------------------------------------------------------
    # Score combination
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall(
        *,
        syntactic: bool,
        structural: float,
        semantic: float,
        cypher: float,
    ) -> float:
        """Combine stage scores into overall score.

        Syntactic is a gate — if it fails, overall is capped at 0.3.
        """
        if not syntactic:
            return round(min(0.3, (structural + semantic + cypher) / 3), 3)

        # Weighted combination of the other stages
        combined = structural * 0.40 + semantic * 0.35 + cypher * 0.25
        return round(combined, 3)
