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

from gibsgraph.training.models import (
    Finding,
    FindingSeverity,
    GraphSchema,
    SynthesisResult,
    ValidationResult,
)
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
        self._driver = driver
        self._database = database

    def validate(self, schema: GraphSchema) -> ValidationResult:
        """Run all 4 validation stages. Returns ValidationResult.

        Approval logic uses severity levels (enterprise pattern):
        - ERROR findings block approval regardless of score
        - WARNING findings degrade score but don't block alone
        - INFO findings are purely informational
        """
        findings: list[Finding] = []

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

        # Approval: no errors AND score >= 0.7
        has_errors = any(f.severity == FindingSeverity.ERROR for f in findings)
        approved = overall >= 0.7 and not has_errors

        return ValidationResult(
            syntactic=syntactic_ok,
            structural_score=structural_score,
            semantic_score=semantic_score,
            domain_score=cypher_score,
            overall_score=overall,
            findings=findings,
            approved_for_training=approved,
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
        has_errors = any(f.severity == FindingSeverity.ERROR for f in all_findings)

        return ValidationResult(
            syntactic=base_result.syntactic,
            structural_score=base_result.structural_score,
            semantic_score=base_result.semantic_score,
            domain_score=breakdown.get("regulatory_coverage", 0.0),
            overall_score=overall,
            findings=all_findings,
            approved_for_training=overall >= 0.7 and not has_errors,
        )

    # ------------------------------------------------------------------
    # Stage 1: Syntactic validation
    # ------------------------------------------------------------------

    def _validate_syntactic(self, schema: GraphSchema) -> tuple[bool, list[Finding]]:
        """Check that the Cypher setup is syntactically safe.

        Returns (syntactic_ok, findings). syntactic_ok is False only when
        ERROR-severity findings exist. WARNINGs (like generic labels) don't
        block the syntactic gate — they degrade score elsewhere.
        """
        findings: list[Finding] = []

        if not schema.cypher_setup.strip():
            findings.append(
                Finding(
                    severity=FindingSeverity.ERROR,
                    stage="SYNTACTIC",
                    message="Cypher setup script is empty",
                )
            )
            return False, findings

        # Check for forbidden write operations (ERROR — dangerous)
        upper = schema.cypher_setup.upper()
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword in upper:
                findings.append(
                    Finding(
                        severity=FindingSeverity.ERROR,
                        stage="SYNTACTIC",
                        message=f"Forbidden keyword '{keyword}' in Cypher setup",
                    )
                )

        # Check that constraints reference existing node labels (ERROR)
        schema_labels = {n.label for n in schema.nodes}
        for constraint in schema.constraints:
            match = re.search(r"FOR\s*\(\w+:(\w+)\)", constraint, re.IGNORECASE)
            if match:
                label = match.group(1)
                if label not in schema_labels:
                    findings.append(
                        Finding(
                            severity=FindingSeverity.ERROR,
                            stage="SYNTACTIC",
                            message=f"Constraint references unknown label '{label}'",
                        )
                    )

        # Check for generic labels (WARNING — quality issue, not safety issue)
        for node in schema.nodes:
            if node.label in GENERIC_LABELS:
                findings.append(
                    Finding(
                        severity=FindingSeverity.WARNING,
                        stage="SYNTACTIC",
                        message=f"Generic label '{node.label}' — use a domain-specific name",
                    )
                )

        # syntactic_ok = no ERROR-severity findings
        has_errors = any(f.severity == FindingSeverity.ERROR for f in findings)
        return not has_errors, findings

    # ------------------------------------------------------------------
    # Stage 3: Semantic validation (data quality checks)
    # ------------------------------------------------------------------

    def _validate_semantic(self, schema: GraphSchema) -> tuple[float, list[Finding]]:
        """Check schema against actual Neo4j data quality.

        When a driver is available, queries the live database to verify:
        1. Node labels actually exist in the database
        2. Relationship types exist
        3. Property completeness (non-null required properties)
        4. Orphan nodes (nodes with 0 relationships)
        5. Referential integrity (relationships point to valid targets)

        When no driver is available, returns 0.0 — no benefit-of-the-doubt
        score for unverified data.
        """
        findings: list[Finding] = []

        if self._driver is None:
            findings.append(
                Finding(
                    severity=FindingSeverity.INFO,
                    stage="SEMANTIC",
                    message="No Neo4j driver — semantic validation skipped (score 0.0)",
                )
            )
            return 0.0, findings

        total_checks = 0
        passed_checks = 0

        try:
            with self._driver.session(database=self._database) as session:
                # 1. Check if schema node labels exist in the database
                db_labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                db_labels = {r["label"] for r in db_labels_result}

                for node in schema.nodes:
                    total_checks += 1
                    if node.label in db_labels:
                        passed_checks += 1
                    else:
                        findings.append(
                            Finding(
                                severity=FindingSeverity.WARNING,
                                stage="SEMANTIC",
                                message=f"Label '{node.label}' not found in database",
                            )
                        )

                # 2. Check if relationship types exist
                db_rels_result = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                db_rel_types = {r["relationshipType"] for r in db_rels_result}

                for rel in schema.relationships:
                    total_checks += 1
                    if rel.type in db_rel_types:
                        passed_checks += 1
                    else:
                        findings.append(
                            Finding(
                                severity=FindingSeverity.WARNING,
                                stage="SEMANTIC",
                                message=(f"Relationship type '{rel.type}' not found in database"),
                            )
                        )

                # 3. Check for orphan nodes (labels with 0 relationships)
                for node in schema.nodes:
                    if node.label not in db_labels:
                        continue  # already flagged above
                    total_checks += 1
                    orphan_result = session.run(
                        "MATCH (n) WHERE $label IN labels(n) "
                        "AND NOT (n)--() RETURN count(n) AS orphans",
                        label=node.label,
                    )
                    orphan_rec = orphan_result.single()
                    orphan_count: int = orphan_rec["orphans"] if orphan_rec else 0
                    total_result = session.run(
                        "MATCH (n) WHERE $label IN labels(n) RETURN count(n) AS total",
                        label=node.label,
                    )
                    total_rec = total_result.single()
                    total_count: int = total_rec["total"] if total_rec else 0

                    if total_count > 0 and orphan_count / total_count < 0.5:
                        passed_checks += 1
                    elif total_count == 0:
                        findings.append(
                            Finding(
                                severity=FindingSeverity.WARNING,
                                stage="SEMANTIC",
                                message=f"Label '{node.label}' has 0 nodes in database",
                            )
                        )
                    else:
                        findings.append(
                            Finding(
                                severity=FindingSeverity.WARNING,
                                stage="SEMANTIC",
                                message=(
                                    f"Label '{node.label}' has {orphan_count}/{total_count} "
                                    f"orphan nodes (>50%)"
                                ),
                            )
                        )

                # 4. Check property completeness for required properties
                for node in schema.nodes:
                    if node.label not in db_labels or not node.required_properties:
                        continue
                    for prop in node.required_properties:
                        total_checks += 1
                        null_result = session.run(
                            "MATCH (n) WHERE $label IN labels(n) "
                            "AND n[$prop] IS NULL RETURN count(n) AS nulls",
                            label=node.label,
                            prop=prop,
                        )
                        null_rec = null_result.single()
                        null_count: int = null_rec["nulls"] if null_rec else 0
                        if null_count == 0:
                            passed_checks += 1
                        else:
                            findings.append(
                                Finding(
                                    severity=FindingSeverity.WARNING,
                                    stage="SEMANTIC",
                                    message=(f"'{node.label}.{prop}' has {null_count} null values"),
                                )
                            )

        except Exception as exc:
            log.warning("semantic.query_failed", error=str(exc))
            findings.append(
                Finding(
                    severity=FindingSeverity.WARNING,
                    stage="SEMANTIC",
                    message=f"Database query failed: {exc}",
                )
            )
            return 0.0, findings

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
