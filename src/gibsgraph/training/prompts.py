"""Versioned prompts for the training pipeline.

All LLM prompts live here, separate from logic. When a prompt improves,
change this file and log the version. This makes prompt engineering
auditable and reproducible.

Prompt version: 1.0 (2026-02-25)
"""

from __future__ import annotations

from gibsgraph.training.models import (
    Finding,
    FindingSeverity,
    GraphSchema,
    SynthesisResult,
)


def build_socratic_scoring_prompt(
    synthesis: SynthesisResult,
    research_context: str,
    expert_patterns: list[str],
    industry: str,
    differentiators: list[str],
) -> str:
    """Build the Socratic yes/no scoring prompt for Haiku.

    Technique: Instead of asking for a score (invites vague answers),
    ask 12 specific yes/no questions. Score = fraction of YES answers
    per dimension. This makes scoring deterministic and auditable.
    """
    node_summary = [
        f"{n.label} (required: {n.required_properties})" for n in synthesis.graph_schema.nodes
    ]
    rel_summary = [(r.from_label, r.type, r.to_label) for r in synthesis.graph_schema.relationships]

    return f"""You are a strict Neo4j schema reviewer.
Answer each question with exactly "YES" or "NO". No explanation.

SCHEMA UNDER REVIEW:
Industry: {industry}
Differentiators: {differentiators}
Nodes: {node_summary}
Relationships: {rel_summary}
Constraints: {synthesis.graph_schema.constraints}
Indexes: {synthesis.graph_schema.indexes}
Claimed regulations: {synthesis.regulatory_requirements}

RESEARCH CONTEXT (what the schema should address):
{research_context[:2000]}

EXPERT PATTERNS (what Neo4j experts say this domain needs):
{expert_patterns}

REGULATORY COVERAGE (answer YES or NO for each):
R1: Does the schema have a node that can store the primary regulated entity for {industry}?
R2: Are all claimed regulatory requirements actually addressable with this schema?
R3: Is there a relationship pattern that enables the primary compliance query?
R4: Do the constraints reflect the regulatory uniqueness requirements?

EXPERT ALIGNMENT (answer YES or NO for each):
E1: Do relationship directions match domain logic (money flows, ownership, causality)?
E2: Are the node labels domain-specific (not generic like Entity or Object)?
E3: Do the indexes reflect the primary query patterns for this domain?
E4: Are the expert patterns from the knowledge base reflected in the schema?

COMPLETENESS (answer YES or NO for each):
C1: Are there at least 4 node types that represent distinct domain concepts?
C2: Is the primary domain risk pattern (fraud ring, compliance violation, etc.) detectable?
C3: Are temporal properties present where the domain requires audit trails?
C4: Would a domain expert recognize this schema as specific to {industry}?

Respond ONLY with valid JSON. No other text:
{{"R1": "YES/NO", "R2": "YES/NO", "R3": "YES/NO", "R4": "YES/NO",
  "E1": "YES/NO", "E2": "YES/NO", "E3": "YES/NO", "E4": "YES/NO",
  "C1": "YES/NO", "C2": "YES/NO", "C3": "YES/NO", "C4": "YES/NO"}}"""


SOCRATIC_DIMENSIONS = {
    "regulatory_coverage": ["R1", "R2", "R3", "R4"],
    "expert_alignment": ["E1", "E2", "E3", "E4"],
    "completeness": ["C1", "C2", "C3", "C4"],
}


def compute_score_from_socratic(answers: dict[str, str]) -> dict[str, float]:
    """Convert yes/no answers to dimension scores (0.0-1.0)."""
    breakdown: dict[str, float] = {}
    for dimension, keys in SOCRATIC_DIMENSIONS.items():
        yes_count = sum(1 for k in keys if answers.get(k, "").upper() == "YES")
        breakdown[dimension] = round(yes_count / len(keys), 3)
    return breakdown


def score_structural(schema: GraphSchema) -> tuple[float, list[Finding]]:
    """Deterministic structural checks with severity levels.

    Returns (score, findings) where each finding has a severity:
    - ERROR: Broken schema (dangling refs, inconsistent properties)
    - WARNING: Quality issues (missing justifications, low counts)
    - INFO: Awareness (disconnected taxonomy nodes)
    """
    findings: list[Finding] = []
    checks: dict[str, bool] = {}

    # --- Minimum counts (WARNING: quality, not broken) ---
    checks["Has at least 3 node types"] = len(schema.nodes) >= 3
    checks["Has at least 2 relationship types"] = len(schema.relationships) >= 2
    checks["Has at least 1 constraint"] = len(schema.constraints) >= 1
    checks["Has at least 1 index"] = len(schema.indexes) >= 1

    # --- Field presence (WARNING: quality) ---
    # Guard against vacuous truth: all([]) == True in Python, but an empty
    # schema should NOT get credit for "all nodes have X" when there are 0 nodes.
    checks["All relationships have direction rationale"] = (
        len(schema.relationships) > 0
        and all(r.direction_rationale for r in schema.relationships)
    )
    checks["All nodes have required properties"] = (
        len(schema.nodes) > 0 and all(n.required_properties for n in schema.nodes)
    )
    checks["All nodes justified by research/pattern"] = (
        len(schema.nodes) > 0 and all(n.justified_by for n in schema.nodes)
    )
    checks["All relationships justified by research/pattern"] = (
        len(schema.relationships) > 0
        and all(r.justified_by for r in schema.relationships)
    )

    # --- Field quality (INFO: nice-to-have) ---
    min_justification_len = 20
    has_justifiable = len(schema.nodes) > 0 or len(schema.relationships) > 0
    checks["Justifications are substantive (>20 chars)"] = has_justifiable and all(
        len(n.justified_by) >= min_justification_len for n in schema.nodes
    ) and all(len(r.justified_by) >= min_justification_len for r in schema.relationships)

    # --- Consistency: required_properties must be subset of properties (ERROR) ---
    checks["Required properties are subset of properties"] = (
        len(schema.nodes) > 0
        and all(set(n.required_properties) <= set(n.properties) for n in schema.nodes)
    )

    # --- Relationship endpoints reference real node labels (ERROR) ---
    node_labels = {n.label for n in schema.nodes}
    orphan_endpoints = []
    for r in schema.relationships:
        if r.from_label not in node_labels:
            orphan_endpoints.append(f"{r.type}: from_label '{r.from_label}'")
        if r.to_label not in node_labels:
            orphan_endpoints.append(f"{r.type}: to_label '{r.to_label}'")
    checks["Relationship endpoints reference existing nodes"] = len(orphan_endpoints) == 0
    for ep in orphan_endpoints:
        findings.append(
            Finding(
                severity=FindingSeverity.ERROR,
                stage="STRUCTURAL",
                message=f"Dangling endpoint — {ep}",
            )
        )

    # --- Connectivity: nodes should participate in at least one relationship ---
    connected_labels = set()
    for r in schema.relationships:
        connected_labels.add(r.from_label)
        connected_labels.add(r.to_label)
    disconnected = node_labels - connected_labels
    # Allow up to 20% disconnected (taxonomy nodes are legitimate)
    max_disconnected = max(1, len(schema.nodes) // 5)
    checks["Most nodes participate in relationships"] = len(disconnected) <= max_disconnected
    if disconnected:
        findings.append(
            Finding(
                severity=FindingSeverity.INFO,
                stage="STRUCTURAL",
                message=(
                    f"{len(disconnected)} node types have no relationships: {sorted(disconnected)}"
                ),
            )
        )

    # Severity mapping for check failures
    _error_checks = {
        "Required properties are subset of properties",
        "Relationship endpoints reference existing nodes",
    }
    _info_checks = {
        "Justifications are substantive (>20 chars)",
    }

    for check_name, passed in checks.items():
        if not passed:
            if check_name in _error_checks:
                severity = FindingSeverity.ERROR
            elif check_name in _info_checks:
                severity = FindingSeverity.INFO
            else:
                severity = FindingSeverity.WARNING
            findings.append(
                Finding(
                    severity=severity,
                    stage="STRUCTURAL",
                    message=f"{check_name} — FAILED",
                )
            )

    score = round(sum(checks.values()) / len(checks), 3) if checks else 0.0
    return score, findings


def score_cypher_quality(cypher: str) -> tuple[float, list[Finding]]:
    """Deterministic Cypher setup quality checks with severity levels."""
    findings: list[Finding] = []

    if not cypher.strip():
        findings.append(
            Finding(
                severity=FindingSeverity.ERROR,
                stage="CYPHER",
                message="Setup script is empty",
            )
        )
        return 0.0, findings

    from gibsgraph.training.validator import FORBIDDEN_KEYWORDS

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in cypher.upper():
            findings.append(
                Finding(
                    severity=FindingSeverity.ERROR,
                    stage="CYPHER",
                    message=f"Contains dangerous keyword '{keyword}'",
                )
            )
            return 0.0, findings

    score = 0.0
    if "CREATE CONSTRAINT" in cypher.upper():
        score += 0.5
    else:
        findings.append(
            Finding(
                severity=FindingSeverity.WARNING,
                stage="CYPHER",
                message="No CREATE CONSTRAINT in setup",
            )
        )

    if "CREATE INDEX" in cypher.upper():
        score += 0.5
    else:
        findings.append(
            Finding(
                severity=FindingSeverity.WARNING,
                stage="CYPHER",
                message="No CREATE INDEX in setup",
            )
        )

    return score, findings
