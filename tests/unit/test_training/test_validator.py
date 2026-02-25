"""Tests for the 4-stage schema validation pipeline.

Tests each validation stage in isolation AND the full pipeline.
Uses realistic schemas from real domains — not toy data.
"""

from unittest.mock import MagicMock, patch

from gibsgraph.training.models import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
)
from gibsgraph.training.validator import (
    FORBIDDEN_KEYWORDS,
    GENERIC_LABELS,
    SchemaValidator,
)

# ---------------------------------------------------------------------------
# Helpers — realistic domain schemas
# ---------------------------------------------------------------------------


def _fintech_schema() -> GraphSchema:
    """Well-formed fintech fraud detection schema. Should pass all stages."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Account",
                properties=["id", "iban", "country", "opened_at"],
                required_properties=["id", "iban"],
                description="Bank account subject to PSD2 monitoring",
                justified_by="PSD2 RTS Art.2 requires account entity",
            ),
            NodeSchema(
                label="Transaction",
                properties=["id", "amount", "currency", "timestamp"],
                required_properties=["id", "amount", "timestamp"],
                description="Payment event between accounts",
                justified_by="Fraud ring detection requires tx-level granularity",
            ),
            NodeSchema(
                label="Merchant",
                properties=["id", "name", "mcc_code"],
                required_properties=["id", "mcc_code"],
                description="Payment recipient",
                justified_by="TRA requires merchant category risk scoring",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="SENT",
                from_label="Account",
                to_label="Transaction",
                properties=["initiated_at"],
                description="Account initiated payment",
                direction_rationale="Account is the actor initiating the tx",
                justified_by="Fraud rings require sender→tx→receiver paths",
            ),
            RelationshipSchema(
                type="RECEIVED_BY",
                from_label="Transaction",
                to_label="Merchant",
                properties=[],
                description="Payment destination",
                direction_rationale="Money flows from tx to merchant",
                justified_by="TRA requires merchant context per transaction",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT tx_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT account_id IF NOT EXISTS "
            "FOR (a:Account) REQUIRE a.id IS UNIQUE;\n"
            "CREATE CONSTRAINT tx_id IF NOT EXISTS "
            "FOR (t:Transaction) REQUIRE t.id IS UNIQUE;\n"
            "CREATE INDEX tx_timestamp IF NOT EXISTS "
            "FOR (t:Transaction) ON (t.timestamp);"
        ),
    )


def _empty_schema() -> GraphSchema:
    """Empty schema — should fail badly."""
    return GraphSchema(
        nodes=[],
        relationships=[],
        constraints=[],
        indexes=[],
        cypher_setup="",
    )


def _generic_label_schema() -> GraphSchema:
    """Schema using labels from the GENERIC_LABELS set."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Entity",
                properties=["id"],
                required_properties=["id"],
                description="Something",
                justified_by="needed",
            ),
            NodeSchema(
                label="Object",
                properties=["name"],
                required_properties=["name"],
                description="Another thing",
                justified_by="needed",
            ),
            NodeSchema(
                label="Item",
                properties=["type"],
                required_properties=["type"],
                description="A third thing",
                justified_by="needed",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="RELATES_TO",
                from_label="Entity",
                to_label="Object",
                properties=[],
                description="A relation",
                direction_rationale="Flow",
                justified_by="needed",
            ),
            RelationshipSchema(
                type="HAS",
                from_label="Object",
                to_label="Item",
                properties=[],
                description="Ownership",
                direction_rationale="Owner to item",
                justified_by="needed",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX obj_name IF NOT EXISTS FOR (o:Object) ON (o.name)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE;\n"
            "CREATE INDEX obj_name IF NOT EXISTS "
            "FOR (o:Object) ON (o.name);"
        ),
    )


def _orphan_constraint_schema() -> GraphSchema:
    """Constraint references a label not in the schema's node list."""
    return GraphSchema(
        nodes=[
            NodeSchema(
                label="Account",
                properties=["id"],
                required_properties=["id"],
                description="Account",
                justified_by="needed",
            ),
            NodeSchema(
                label="Transaction",
                properties=["id"],
                required_properties=["id"],
                description="Transaction",
                justified_by="needed",
            ),
            NodeSchema(
                label="Merchant",
                properties=["id"],
                required_properties=["id"],
                description="Merchant",
                justified_by="needed",
            ),
        ],
        relationships=[
            RelationshipSchema(
                type="SENT",
                from_label="Account",
                to_label="Transaction",
                properties=[],
                description="Sent payment",
                direction_rationale="Account is actor",
                justified_by="needed",
            ),
            RelationshipSchema(
                type="TO",
                from_label="Transaction",
                to_label="Merchant",
                properties=[],
                description="Payment to merchant",
                direction_rationale="Money flows out",
                justified_by="needed",
            ),
        ],
        constraints=[
            "CREATE CONSTRAINT ghost_id IF NOT EXISTS FOR (g:Ghost) REQUIRE g.id IS UNIQUE",
        ],
        indexes=[
            "CREATE INDEX acct IF NOT EXISTS FOR (a:Account) ON (a.id)",
        ],
        cypher_setup=(
            "CREATE CONSTRAINT ghost_id IF NOT EXISTS "
            "FOR (g:Ghost) REQUIRE g.id IS UNIQUE;\n"
            "CREATE INDEX acct IF NOT EXISTS FOR (a:Account) ON (a.id);"
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1: Syntactic validation
# ---------------------------------------------------------------------------


class TestSyntacticValidation:
    """Tests for _validate_syntactic — the gatekeeper stage."""

    def test_good_schema_passes_syntactic(self):
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.syntactic is True

    def test_empty_cypher_fails_syntactic(self):
        v = SchemaValidator()
        result = v.validate(_empty_schema())
        assert result.syntactic is False
        assert any("empty" in f.lower() for f in result.findings)

    def test_forbidden_keywords_all_caught(self):
        """Every keyword in FORBIDDEN_KEYWORDS must fail syntactic."""
        v = SchemaValidator()
        for keyword in FORBIDDEN_KEYWORDS:
            schema = _fintech_schema()
            schema.cypher_setup = f"{keyword} something"
            result = v.validate(schema)
            assert result.syntactic is False, f"Keyword '{keyword}' was not caught"
            assert any(keyword in f for f in result.findings)

    def test_forbidden_keyword_case_insensitive(self):
        """'delete' in lowercase should still be caught."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "delete all"
        result = v.validate(schema)
        assert result.syntactic is False

    def test_generic_labels_flagged(self):
        """All labels in GENERIC_LABELS should be caught."""
        v = SchemaValidator()
        result = v.validate(_generic_label_schema())
        assert result.syntactic is False
        generic_findings = [f for f in result.findings if "Generic label" in f]
        # Entity, Object, Item — all three should be flagged
        assert len(generic_findings) == 3

    def test_generic_labels_are_comprehensive(self):
        """Verify the constant covers the most common offenders."""
        expected = {
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
        assert GENERIC_LABELS == expected

    def test_constraint_referencing_unknown_label(self):
        """A constraint on 'Ghost' should fail when no Ghost node exists."""
        v = SchemaValidator()
        result = v.validate(_orphan_constraint_schema())
        assert result.syntactic is False
        assert any("Ghost" in f for f in result.findings)

    def test_constraint_label_extraction_case_sensitive(self):
        """FOR (a:Account) should match Account exactly."""
        v = SchemaValidator()
        schema = _fintech_schema()
        # Replace constraint with different case in label
        schema.constraints = [
            "CREATE CONSTRAINT x IF NOT EXISTS FOR (a:ACCOUNT) REQUIRE a.id IS UNIQUE"
        ]
        result = v.validate(schema)
        # ACCOUNT != Account, so it should be flagged as unknown
        assert any("ACCOUNT" in f for f in result.findings)


# ---------------------------------------------------------------------------
# Stage 2 & 4 together: Structural + Cypher quality
# (tested individually in test_scorer.py, here we test their effect
# on the overall pipeline result)
# ---------------------------------------------------------------------------


class TestStructuralInPipeline:
    """Structural checks affect overall score through the pipeline."""

    def test_good_schema_high_structural_score(self):
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.structural_score == 1.0

    def test_empty_schema_low_structural_score(self):
        v = SchemaValidator()
        result = v.validate(_empty_schema())
        # Empty fails count checks (nodes, rels, constraints, indexes)
        # but all() on empty iterables is True (vacuous truth), so
        # justification/rationale checks pass trivially → 4/8 = 0.5
        assert result.structural_score == 0.5


# ---------------------------------------------------------------------------
# Stage 3: Semantic validation (expert graph alignment)
# ---------------------------------------------------------------------------


class TestSemanticValidation:
    """Tests for _validate_semantic — expert graph alignment."""

    def test_no_driver_returns_neutral_score(self):
        """Without a Neo4j driver, semantic = 0.5 (benefit of doubt)."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.semantic_score == 0.5
        assert any("Expert graph not available" in f for f in result.findings)

    def test_unavailable_expert_returns_neutral(self):
        """If ExpertStore.is_available() is False, semantic = 0.5."""
        mock_driver = MagicMock()
        with patch("gibsgraph.training.validator.ExpertStore") as mock_cls:
            store = MagicMock()
            store.is_available.return_value = False
            mock_cls.return_value = store
            v = SchemaValidator(mock_driver, database="test")
            result = v.validate(_fintech_schema())
        assert result.semantic_score == 0.5

    def test_expert_match_boosts_semantic_score(self):
        """When expert search returns high-confidence hits, score > 0.5."""
        mock_driver = MagicMock()
        with patch("gibsgraph.training.validator.ExpertStore") as mock_cls:
            store = MagicMock()
            store.is_available.return_value = True
            # Return a hit with score > 0.5 for every search
            hit = MagicMock()
            hit.score = 0.9
            ctx = MagicMock()
            ctx.hits = [hit]
            store.search.return_value = ctx
            mock_cls.return_value = store

            v = SchemaValidator(mock_driver, database="test")
            result = v.validate(_fintech_schema())

        # 2 relationship checks + 1 constraint check + 1 index check = 4 total
        # All should pass (score > 0.5)
        assert result.semantic_score > 0.5

    def test_no_expert_hits_drops_semantic_score(self):
        """When expert search returns nothing, score drops."""
        mock_driver = MagicMock()
        with patch("gibsgraph.training.validator.ExpertStore") as mock_cls:
            store = MagicMock()
            store.is_available.return_value = True
            ctx = MagicMock()
            ctx.hits = []  # no matches
            store.search.return_value = ctx
            mock_cls.return_value = store

            v = SchemaValidator(mock_driver, database="test")
            result = v.validate(_fintech_schema())

        # No hits for any relationship check → 0 passed out of N
        assert result.semantic_score == 0.0
        assert any("no expert pattern match" in f for f in result.findings)


# ---------------------------------------------------------------------------
# Overall score combination
# ---------------------------------------------------------------------------


class TestOverallScoring:
    def test_syntactic_failure_caps_overall_at_03(self):
        """If syntactic fails, overall is capped at 0.3 regardless of others."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DETACH DELETE everything"
        result = v.validate(schema)
        assert result.syntactic is False
        assert result.overall_score <= 0.3

    def test_perfect_schema_gets_high_overall(self):
        """A schema passing all deterministic checks should score well."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        # structural=1.0, semantic=0.5 (no expert), cypher=1.0
        # = 1.0*0.40 + 0.5*0.35 + 1.0*0.25 = 0.825
        assert result.overall_score == 0.825

    def test_approval_requires_07_and_syntactic(self):
        """approved_for_training needs overall >= 0.7 AND syntactic pass."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.approved_for_training is True
        assert result.overall_score >= 0.7
        assert result.syntactic is True

    def test_approval_denied_when_overall_too_low(self):
        v = SchemaValidator()
        result = v.validate(_empty_schema())
        assert result.approved_for_training is False

    def test_approval_denied_on_syntactic_fail_even_if_score_ok(self):
        """Generic labels fail syntactic but structural could be fine."""
        v = SchemaValidator()
        result = v.validate(_generic_label_schema())
        assert result.syntactic is False
        assert result.approved_for_training is False


# ---------------------------------------------------------------------------
# Edge cases and boundary tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_cypher_with_only_whitespace_is_empty(self):
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "   \n\t  "
        result = v.validate(schema)
        assert result.syntactic is False

    def test_constraint_regex_handles_no_match(self):
        """A constraint without FOR (x:Label) pattern should not crash."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.constraints = ["THIS IS NOT A REAL CONSTRAINT"]
        result = v.validate(schema)
        # Should not raise, regex just doesn't match
        assert result is not None

    def test_multiple_forbidden_keywords_all_reported(self):
        """If cypher has DELETE and DROP, both should appear in findings."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DETACH DELETE n; DROP INDEX x"
        result = v.validate(schema)
        assert result.syntactic is False
        keyword_findings = [f for f in result.findings if "Forbidden keyword" in f]
        found_keywords = {f.split("'")[1] for f in keyword_findings}
        assert "DELETE" in found_keywords
        assert "DETACH" in found_keywords
        assert "DROP" in found_keywords

    def test_schema_with_mixed_good_and_bad_nodes(self):
        """One generic label among domain-specific ones still fails."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.nodes.append(
            NodeSchema(
                label="Thing",
                properties=["id"],
                required_properties=["id"],
                description="Unknown",
                justified_by="unclear",
            ),
        )
        result = v.validate(schema)
        assert result.syntactic is False
        assert any("Thing" in f for f in result.findings)

    def test_findings_are_strings(self):
        """Every finding must be a plain string — no objects or None."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        for f in result.findings:
            assert isinstance(f, str)

        result_bad = v.validate(_empty_schema())
        for f in result_bad.findings:
            assert isinstance(f, str)


# ---------------------------------------------------------------------------
# validate_full (LLM-assisted path)
# ---------------------------------------------------------------------------


class TestValidateFull:
    def test_without_settings_falls_back_to_deterministic(self):
        """validate_full with settings=None should return same as validate."""
        from gibsgraph.training.models import SynthesisResult

        v = SchemaValidator()
        schema = _fintech_schema()
        synth = SynthesisResult(
            model="test-model",
            scenario="test",
            design_rationale="test",
            graph_schema=schema,
            regulatory_requirements=["PSD2"],
            expert_patterns_used=["fraud ring"],
            findings_used=["finding 1"],
            quality_score=0.0,
            score_breakdown={},
            file_path="test.json",
        )
        result = v.validate_full(
            synth,
            research_context="fintech fraud detection",
            expert_patterns=["fraud ring", "account monitoring"],
            industry="fintech",
            differentiators=["eu", "psd2"],
            settings=None,
        )
        # Should be same as v.validate(schema)
        expected = v.validate(schema)
        assert result.syntactic == expected.syntactic
        assert result.structural_score == expected.structural_score

    def test_with_non_settings_object_falls_back(self):
        """Passing a random object (not Settings) falls back safely."""
        from gibsgraph.training.models import SynthesisResult

        v = SchemaValidator()
        synth = SynthesisResult(
            model="test-model",
            scenario="test",
            design_rationale="test",
            graph_schema=_fintech_schema(),
            regulatory_requirements=[],
            expert_patterns_used=[],
            findings_used=[],
            quality_score=0.0,
            score_breakdown={},
            file_path="test.json",
        )
        result = v.validate_full(
            synth,
            research_context="test",
            expert_patterns=[],
            industry="test",
            differentiators=[],
            settings="not a Settings object",
        )
        assert result is not None
        assert isinstance(result.overall_score, float)
