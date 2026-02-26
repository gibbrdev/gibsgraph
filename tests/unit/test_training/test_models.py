"""Tests for training pipeline Pydantic models.

Tests real serialization, validation boundaries, and property logic —
not just that fields exist.
"""

import pytest
from pydantic import ValidationError

from gibsgraph.training.models import (
    Differentiator,
    Finding,
    FindingSeverity,
    GraphSchema,
    Industry,
    NodeSchema,
    RelationshipSchema,
    SynthesisResult,
    UseCaseRecord,
    ValidationResult,
)

# ---------------------------------------------------------------------------
# Fixtures — realistic schemas for reuse
# ---------------------------------------------------------------------------


def fintech_schema() -> GraphSchema:
    """A realistic fintech fraud detection schema."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Account",
                properties=["id", "iban", "country", "opened_at"],
                required_properties=["id", "iban"],
                description="Bank account subject to PSD2 monitoring",
                justified_by="PSD2 RTS Art.2 requires transaction monitoring per account",
            ),
            NodeSchema(
                label="Transaction",
                properties=["id", "amount", "currency", "timestamp", "channel"],
                required_properties=["id", "amount", "timestamp"],
                description="Payment event between accounts",
                justified_by="Fraud ring detection requires transaction-level granularity",
            ),
            NodeSchema(
                label="Merchant",
                properties=["id", "name", "mcc_code", "country"],
                required_properties=["id", "mcc_code"],
                description="Payment recipient",
                justified_by="TRA requires merchant category risk scoring",
            ),
            NodeSchema(
                label="FraudCase",
                properties=["id", "opened_at", "status", "total_amount"],
                required_properties=["id", "opened_at"],
                description="Aggregated fraud investigation",
                justified_by="Regulatory reporting requires case-level tracking",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="SENT",
                from_label="Account",
                to_label="Transaction",
                properties=["initiated_at"],
                description="Account initiated payment",
                direction_rationale="Account is the actor initiating the transaction",
                justified_by="Fraud rings require sender→transaction→receiver paths",
            ),
            RelationshipSchema(
                type="RECEIVED_BY",
                from_label="Transaction",
                to_label="Merchant",
                properties=[],
                description="Payment destination",
                direction_rationale="Money flows from transaction to merchant",
                justified_by="TRA requires merchant risk context per transaction",
            ),
            RelationshipSchema(
                type="FLAGGED_IN",
                from_label="Transaction",
                to_label="FraudCase",
                properties=["flagged_at", "reason"],
                description="Transaction flagged as suspicious",
                direction_rationale="Transaction is the evidence, case is the investigation",
                justified_by="PSD2 requires audit trail from transaction to investigation",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX account_iban IF NOT EXISTS FOR (a:Account) ON (a.iban)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;\n"
            "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE;\n"
            "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp);\n"
            "CREATE INDEX account_iban IF NOT EXISTS FOR (a:Account) ON (a.iban);"
        ),
    )


def empty_schema() -> GraphSchema:
    """A completely empty schema — should fail every check."""
    return GraphSchema(nodes=[], relationships=[], constraints=[], indexes=[], cypher_setup="")


def generic_label_schema() -> GraphSchema:
    """Schema with generic labels that should be flagged."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Entity",
                properties=["id"],
                required_properties=["id"],
                description="A thing",
                justified_by="needed",
            ),
            NodeSchema(
                label="Object",
                properties=["name"],
                required_properties=["name"],
                description="Another thing",
                justified_by="needed",
            ),
        ],
        relationships=[],
        constraints=[],
        indexes=[],
        cypher_setup="CREATE CONSTRAINT e IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;",
    )


# ---------------------------------------------------------------------------
# Model instantiation and serialization
# ---------------------------------------------------------------------------


class TestGraphSchema:
    def test_fintech_schema_round_trips_through_json(self):
        schema = fintech_schema()
        json_str = schema.model_dump_json()
        restored = GraphSchema.model_validate_json(json_str)
        assert len(restored.nodes) == 4
        assert len(restored.relationships) == 3
        assert "account_id" in restored.constraints[0]

    def test_empty_schema_is_valid_pydantic(self):
        """Empty schema is valid Pydantic — validation is the validator's job."""
        schema = empty_schema()
        assert schema.nodes == []
        assert schema.cypher_setup == ""


class TestValidationResult:
    def test_score_boundaries(self):
        """Scores must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            ValidationResult(
                syntactic=True,
                structural_score=1.5,
                overall_score=0.5,
                findings=[],
            )
        with pytest.raises(ValidationError):
            ValidationResult(
                syntactic=True,
                structural_score=0.5,
                overall_score=-0.1,
                findings=[],
            )

    def test_approval_defaults_to_false(self):
        result = ValidationResult(
            syntactic=True,
            structural_score=0.9,
            overall_score=0.85,
            findings=[],
        )
        assert result.approved_for_training is False


class TestUseCaseRecord:
    def test_auto_generates_id(self):
        r1 = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[Differentiator.EU, Differentiator.PSD2],
        )
        r2 = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[Differentiator.EU],
        )
        assert r1.id != r2.id
        assert len(r1.id) == 8

    def test_is_approved_false_when_no_validation(self):
        r = UseCaseRecord(
            industry=Industry.HEALTHCARE,
            sub_industry="patient",
            differentiators=[Differentiator.HIPAA],
        )
        assert r.is_approved is False

    def test_is_approved_true_when_validation_approves(self):
        r = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[Differentiator.EU],
            validation=ValidationResult(
                syntactic=True,
                structural_score=0.9,
                overall_score=0.88,
                findings=[],
                approved_for_training=True,
                approved_by="reviewer",
            ),
        )
        assert r.is_approved is True

    def test_is_approved_false_when_validation_rejects(self):
        r = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[Differentiator.EU],
            validation=ValidationResult(
                syntactic=False,
                structural_score=0.3,
                overall_score=0.2,
                findings=[
                    Finding(
                        severity=FindingSeverity.ERROR,
                        stage="SYNTACTIC",
                        message="bad schema",
                    )
                ],
                approved_for_training=False,
            ),
        )
        assert r.is_approved is False

    def test_winning_synthesis_returns_correct_winner(self):
        schema = fintech_schema()
        synth_a = SynthesisResult(
            model="model-a",
            scenario="A scenario",
            design_rationale="A rationale",
            graph_schema=schema,
            regulatory_requirements=[],
            expert_patterns_used=[],
            findings_used=[],
            quality_score=0.8,
            score_breakdown={},
            file_path="a.json",
        )
        synth_b = SynthesisResult(
            model="model-b",
            scenario="B scenario",
            design_rationale="B rationale",
            graph_schema=schema,
            regulatory_requirements=[],
            expert_patterns_used=[],
            findings_used=[],
            quality_score=0.6,
            score_breakdown={},
            file_path="b.json",
        )

        r = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[Differentiator.EU],
            synthesis_a=synth_a,
            synthesis_b=synth_b,
            winner="a",
        )
        assert r.winning_synthesis is synth_a

        r.winner = "b"
        assert r.winning_synthesis is synth_b

    def test_winning_synthesis_none_when_no_winner(self):
        r = UseCaseRecord(
            industry=Industry.FINTECH,
            sub_industry="fraud",
            differentiators=[],
        )
        assert r.winning_synthesis is None

    def test_full_record_serializes_to_json(self):
        """Entire record including nested models must survive JSON round-trip."""
        r = UseCaseRecord(
            industry=Industry.CYBERSECURITY,
            sub_industry="threat_detection",
            differentiators=[Differentiator.ENTERPRISE, Differentiator.REALTIME],
            validation=ValidationResult(
                syntactic=True,
                structural_score=0.75,
                overall_score=0.72,
                findings=[
                    Finding(
                        severity=FindingSeverity.INFO,
                        stage="STRUCTURAL",
                        message="minor issue",
                    )
                ],
                approved_for_training=True,
            ),
        )
        json_str = r.model_dump_json()
        restored = UseCaseRecord.model_validate_json(json_str)
        assert restored.industry == Industry.CYBERSECURITY
        assert restored.is_approved is True
        assert len(restored.validation.findings) == 1  # type: ignore[union-attr]


class TestEnums:
    def test_industry_values_are_lowercase(self):
        for ind in Industry:
            assert ind.value == ind.value.lower()

    def test_differentiator_used_as_string(self):
        d = Differentiator.GDPR
        assert f"regulation: {d}" == "regulation: gdpr"

    def test_unknown_industry_raises(self):
        with pytest.raises(ValueError):
            Industry("blockchain")
