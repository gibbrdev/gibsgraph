"""Tests for the quality scorer.

Tests deterministic scoring logic with realistic schemas —
good, bad, and edge cases. No LLM calls (Socratic tests are separate).
"""

from unittest.mock import MagicMock

from gibsgraph.training.models import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
    SynthesisResult,
)
from gibsgraph.training.prompts import (
    compute_score_from_socratic,
    score_cypher_quality,
    score_structural,
)
from gibsgraph.training.scorer import QualityScorer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_schema() -> GraphSchema:
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Account", properties=["id", "iban"],
                required_properties=["id"], description="Bank account",
                justified_by="PSD2 requires account entity",
            ),
            NodeSchema(
                label="Transaction", properties=["id", "amount", "timestamp"],
                required_properties=["id", "timestamp"],
                description="Payment", justified_by="PSD2 RTS Art.2",
            ),
            NodeSchema(
                label="Merchant", properties=["id", "name"],
                required_properties=["id"], description="Merchant",
                justified_by="TRA requires merchant context",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="SENT", from_label="Account", to_label="Transaction",
                properties=[], description="Account sent payment",
                direction_rationale="Account is actor",
                justified_by="Fraud ring detection",
            ),
            RelationshipSchema(
                type="TO", from_label="Transaction", to_label="Merchant",
                properties=[], description="Payment to merchant",
                direction_rationale="Money flows to merchant",
                justified_by="TRA merchant scoring",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX tx_ts IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;\n"
            "CREATE INDEX tx_ts IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp);"
        ),
    )


def _bad_schema() -> GraphSchema:
    """Missing justifications, no constraints, no indexes."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Thing", properties=["id"],
                required_properties=[], description="A thing",
                justified_by="",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="RELATES_TO", from_label="Thing", to_label="Thing",
                properties=[], description="Generic",
                direction_rationale="",
                justified_by="",
            ),
        ],
        constraints=[],
        indexes=[],
        cypher_setup="",
    )


def _synth(schema: GraphSchema) -> SynthesisResult:
    return SynthesisResult(
        model="test-model", scenario="test", design_rationale="test",
        graph_schema=schema, regulatory_requirements=["PSD2 Art.2"],
        expert_patterns_used=["fraud ring"], findings_used=["finding 1"],
        quality_score=0.0, score_breakdown={}, file_path="test.json",
    )


# ---------------------------------------------------------------------------
# Structural scoring
# ---------------------------------------------------------------------------


class TestScoreStructural:
    def test_good_schema_scores_high(self):
        score, findings = score_structural(_good_schema())
        assert score == 1.0
        assert findings == []

    def test_bad_schema_scores_low(self):
        score, findings = score_structural(_bad_schema())
        assert score < 0.5
        assert len(findings) >= 3  # missing constraints, indexes, justifications

    def test_missing_single_justification_drops_score(self):
        schema = _good_schema()
        schema.nodes[0].justified_by = ""
        score, findings = score_structural(schema)
        assert score < 1.0
        assert any("justified" in f.lower() for f in findings)

    def test_missing_direction_rationale_drops_score(self):
        schema = _good_schema()
        schema.relationships[0].direction_rationale = ""
        score, findings = score_structural(schema)
        assert score < 1.0
        assert any("direction" in f.lower() for f in findings)

    def test_two_nodes_is_not_enough(self):
        """3 node types is the minimum for a real schema."""
        schema = _good_schema()
        schema.nodes = schema.nodes[:2]
        score, findings = score_structural(schema)
        assert score < 1.0
        assert any("3 node" in f for f in findings)

    def test_one_relationship_is_not_enough(self):
        schema = _good_schema()
        schema.relationships = schema.relationships[:1]
        score, findings = score_structural(schema)
        assert score < 1.0
        assert any("2 relationship" in f for f in findings)


# ---------------------------------------------------------------------------
# Cypher quality scoring
# ---------------------------------------------------------------------------


class TestScoreCypherQuality:
    def test_good_cypher_scores_1(self):
        score, findings = score_cypher_quality(_good_schema().cypher_setup)
        assert score == 1.0
        assert findings == []

    def test_empty_cypher_scores_0(self):
        score, findings = score_cypher_quality("")
        assert score == 0.0
        assert any("empty" in f.lower() for f in findings)

    def test_dangerous_detach_delete_scores_0(self):
        score, findings = score_cypher_quality("MATCH (n) DETACH DELETE n")
        assert score == 0.0
        # Could match DELETE or DETACH first (set iteration order)
        assert any("dangerous keyword" in f.lower() for f in findings)

    def test_dangerous_drop_scores_0(self):
        score, _findings = score_cypher_quality("DROP INDEX my_index")
        assert score == 0.0

    def test_constraint_only_scores_half(self):
        score, _ = score_cypher_quality(
            "CREATE CONSTRAINT x IF NOT EXISTS FOR (n:X) REQUIRE n.id IS UNIQUE"
        )
        assert score == 0.5

    def test_index_only_scores_half(self):
        score, _ = score_cypher_quality(
            "CREATE INDEX x IF NOT EXISTS FOR (n:X) ON (n.id)"
        )
        assert score == 0.5

    def test_case_insensitive_keyword_detection(self):
        """DELETE in lowercase should still be caught."""
        score, _ = score_cypher_quality("match (n) delete n")
        assert score == 0.0


# ---------------------------------------------------------------------------
# Socratic score computation
# ---------------------------------------------------------------------------


class TestComputeSocraticScore:
    def test_all_yes_scores_1(self):
        answers = {k: "YES" for k in
                   ["R1", "R2", "R3", "R4", "E1", "E2", "E3", "E4", "C1", "C2", "C3", "C4"]}
        breakdown = compute_score_from_socratic(answers)
        assert breakdown["regulatory_coverage"] == 1.0
        assert breakdown["expert_alignment"] == 1.0
        assert breakdown["completeness"] == 1.0

    def test_all_no_scores_0(self):
        answers = {k: "NO" for k in
                   ["R1", "R2", "R3", "R4", "E1", "E2", "E3", "E4", "C1", "C2", "C3", "C4"]}
        breakdown = compute_score_from_socratic(answers)
        assert all(v == 0.0 for v in breakdown.values())

    def test_mixed_answers(self):
        answers = {
            "R1": "YES", "R2": "YES", "R3": "NO", "R4": "NO",
            "E1": "YES", "E2": "YES", "E3": "YES", "E4": "NO",
            "C1": "YES", "C2": "NO", "C3": "NO", "C4": "YES",
        }
        breakdown = compute_score_from_socratic(answers)
        assert breakdown["regulatory_coverage"] == 0.5
        assert breakdown["expert_alignment"] == 0.75
        assert breakdown["completeness"] == 0.5

    def test_missing_keys_count_as_no(self):
        """If the LLM omits an answer, it should count as NO."""
        answers = {"R1": "YES"}  # only 1 out of 12
        breakdown = compute_score_from_socratic(answers)
        assert breakdown["regulatory_coverage"] == 0.25  # 1 of 4

    def test_case_insensitive_yes(self):
        answers = {"R1": "yes", "R2": "Yes", "R3": "YES", "R4": "yEs"}
        breakdown = compute_score_from_socratic(answers)
        assert breakdown["regulatory_coverage"] == 1.0


# ---------------------------------------------------------------------------
# Deterministic scorer (no LLM)
# ---------------------------------------------------------------------------


class TestQualityScorerDeterministic:
    def test_good_schema_scores_high(self):
        scorer = QualityScorer(MagicMock())
        overall, _breakdown, findings = scorer.score_deterministic(_synth(_good_schema()))
        assert overall >= 0.9
        assert findings == []

    def test_bad_schema_scores_low(self):
        scorer = QualityScorer(MagicMock())
        overall, _breakdown, findings = scorer.score_deterministic(_synth(_bad_schema()))
        assert overall < 0.5
        assert len(findings) > 0

    def test_deterministic_does_not_call_llm(self):
        """score_deterministic must never invoke the LLM."""
        settings = MagicMock()
        scorer = QualityScorer(settings)
        scorer.score_deterministic(_synth(_good_schema()))
        # _make_llm is only called in _score_socratic which score_deterministic skips
        settings.llm_model.assert_not_called()
