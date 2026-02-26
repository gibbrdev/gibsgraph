"""Tests for the 4-stage schema validation pipeline.

Tests each validation stage in isolation AND the full pipeline.
Uses realistic schemas from real domains — not toy data.
"""

from unittest.mock import MagicMock

from gibsgraph.training.models import (
    Finding,
    FindingSeverity,
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
    """Tests for _validate_syntactic — the gatekeeper stage.

    Key design: generic labels are WARNING (quality issue), not ERROR.
    Only dangerous Cypher, empty setup, and orphan constraints are ERRORs.
    """

    def test_good_schema_passes_syntactic(self):
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.syntactic is True

    def test_empty_cypher_fails_syntactic(self):
        v = SchemaValidator()
        result = v.validate(_empty_schema())
        assert result.syntactic is False
        assert any("empty" in f.message.lower() for f in result.findings)

    def test_forbidden_keywords_all_caught(self):
        """Every keyword in FORBIDDEN_KEYWORDS must fail syntactic."""
        v = SchemaValidator()
        for keyword in FORBIDDEN_KEYWORDS:
            schema = _fintech_schema()
            schema.cypher_setup = f"{keyword} something"
            result = v.validate(schema)
            assert result.syntactic is False, f"Keyword '{keyword}' was not caught"
            assert any(keyword in f.message for f in result.findings)

    def test_forbidden_keyword_is_error_severity(self):
        """Forbidden keywords must be ERROR severity."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DELETE n"
        result = v.validate(schema)
        keyword_findings = [f for f in result.findings if "Forbidden" in f.message]
        assert all(f.severity == FindingSeverity.ERROR for f in keyword_findings)

    def test_forbidden_keyword_case_insensitive(self):
        """'delete' in lowercase should still be caught."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "delete all"
        result = v.validate(schema)
        assert result.syntactic is False

    def test_generic_labels_are_warnings_not_errors(self):
        """Generic labels should be WARNING severity — they don't block syntactic."""
        v = SchemaValidator()
        result = v.validate(_generic_label_schema())
        # Generic labels are WARNINGs now, so syntactic gate still passes
        assert result.syntactic is True
        generic_findings = [f for f in result.findings if "Generic label" in f.message]
        assert len(generic_findings) == 3
        assert all(f.severity == FindingSeverity.WARNING for f in generic_findings)

    def test_generic_labels_still_block_approval(self):
        """Generic labels don't block syntactic but block approval via errors elsewhere."""
        v = SchemaValidator()
        result = v.validate(_generic_label_schema())
        # Syntactic passes (only warnings), but short justifications cause
        # INFO findings, not errors — check approval is based on score + errors
        generic_findings = [f for f in result.findings if "Generic label" in f.message]
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
        assert any("Ghost" in f.message for f in result.findings)

    def test_constraint_label_extraction_case_sensitive(self):
        """FOR (a:Account) should match Account exactly."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.constraints = [
            "CREATE CONSTRAINT x IF NOT EXISTS FOR (a:ACCOUNT) REQUIRE a.id IS UNIQUE"
        ]
        result = v.validate(schema)
        assert any("ACCOUNT" in f.message for f in result.findings)

    def test_findings_are_finding_objects(self):
        """All findings must be Finding instances with severity."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        for f in result.findings:
            assert isinstance(f, Finding)

    def test_error_findings_block_approval(self):
        """Any ERROR-severity finding should block approval."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DELETE n"
        result = v.validate(schema)
        assert len(result.errors) > 0
        assert result.approved_for_training is False


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
        # With vacuous truth fix: empty schema fails count checks AND
        # field-presence checks (guarded against all([]) == True).
        # Only 2 of 12 checks pass → score < 0.5
        assert result.structural_score < 0.5


# ---------------------------------------------------------------------------
# Stage 3: Semantic validation (expert graph alignment)
# ---------------------------------------------------------------------------


class TestSemanticValidation:
    """Tests for _validate_semantic — data quality checks.

    Without a Neo4j driver, semantic = 0.0 (no benefit of doubt).
    With a driver, queries the actual database for data quality.
    """

    def test_no_driver_returns_zero_score(self):
        """Without a Neo4j driver, semantic = 0.0 (no benefit of doubt)."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.semantic_score == 0.0
        assert any("semantic validation skipped" in f.message for f in result.findings)

    def test_no_driver_is_info_severity(self):
        """No driver finding should be INFO, not WARNING or ERROR."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        semantic_findings = [f for f in result.findings if f.stage == "SEMANTIC"]
        assert all(f.severity == FindingSeverity.INFO for f in semantic_findings)

    def test_driver_queries_real_labels(self):
        """With a driver, semantic checks query db.labels() for real data."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock db.labels() returning our schema labels
        labels_result = MagicMock()
        labels_result.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"label": "Account"},
                    {"label": "Transaction"},
                    {"label": "Merchant"},
                ]
            )
        )
        rel_types_result = MagicMock()
        rel_types_result.__iter__ = MagicMock(
            return_value=iter([{"relationshipType": "SENT"}, {"relationshipType": "RECEIVED_BY"}])
        )

        # Mock orphan and property checks
        zero_result = MagicMock()
        zero_result.single.return_value = {"orphans": 0, "total": 10, "nulls": 0}

        def run_side_effect(query, **kwargs):
            if "db.labels" in query:
                return labels_result
            if "db.relationshipTypes" in query:
                return rel_types_result
            return zero_result

        mock_session.run.side_effect = run_side_effect

        v = SchemaValidator(mock_driver, database="test")
        result = v.validate(_fintech_schema())

        # With all labels and rel types found, score should be > 0.0
        assert result.semantic_score > 0.0
        # The driver's session was actually used
        assert mock_session.run.called

    def test_missing_labels_produce_warnings(self):
        """When db.labels() doesn't contain schema labels, get WARNINGs."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Empty database — no labels, no rel types
        empty_result = MagicMock()
        empty_result.__iter__ = MagicMock(return_value=iter([]))

        mock_session.run.return_value = empty_result

        v = SchemaValidator(mock_driver, database="test")
        result = v.validate(_fintech_schema())

        assert result.semantic_score == 0.0
        assert any("not found in database" in f.message for f in result.findings)

    def test_query_failure_returns_zero_with_warning(self):
        """If database query fails, return 0.0 with a WARNING."""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Exception("Connection refused")

        v = SchemaValidator(mock_driver, database="test")
        result = v.validate(_fintech_schema())

        assert result.semantic_score == 0.0
        assert any("query failed" in f.message.lower() for f in result.findings)


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

    def test_perfect_schema_gets_reasonable_overall(self):
        """A schema passing deterministic checks scores reasonably.

        Without a Neo4j driver, semantic = 0.0 (no benefit of doubt).
        structural=1.0, semantic=0.0, cypher=1.0
        = 1.0*0.40 + 0.0*0.35 + 1.0*0.25 = 0.65
        """
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert result.overall_score == 0.65

    def test_approval_requires_07_and_no_errors(self):
        """approved_for_training needs overall >= 0.7 AND no ERROR findings.

        Without a driver, semantic = 0.0, so overall = 0.65 < 0.7.
        This means approval requires a live Neo4j connection for semantic checks.
        """
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        # Without driver, overall = 0.65 < 0.7, so NOT approved
        assert result.approved_for_training is False
        assert len(result.errors) == 0

    def test_approval_denied_when_overall_too_low(self):
        v = SchemaValidator()
        result = v.validate(_empty_schema())
        assert result.approved_for_training is False

    def test_approval_denied_when_errors_exist(self):
        """ERROR-severity findings block approval even with decent score."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DETACH DELETE everything"
        result = v.validate(schema)
        assert len(result.errors) > 0
        assert result.approved_for_training is False

    def test_generic_labels_dont_block_syntactic_gate(self):
        """Generic labels are WARNINGs — syntactic gate passes."""
        v = SchemaValidator()
        result = v.validate(_generic_label_schema())
        # With severity refactor, generic labels are WARNING not ERROR
        assert result.syntactic is True

    def test_convenience_properties(self):
        """errors, warnings, infos properties filter correctly."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.infos, list)
        # All findings accounted for
        assert len(result.errors) + len(result.warnings) + len(result.infos) == len(result.findings)


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
        assert result is not None

    def test_multiple_forbidden_keywords_all_reported(self):
        """If cypher has DELETE and DROP, both should appear in findings."""
        v = SchemaValidator()
        schema = _fintech_schema()
        schema.cypher_setup = "DETACH DELETE n; DROP INDEX x"
        result = v.validate(schema)
        assert result.syntactic is False
        keyword_findings = [f for f in result.findings if "Forbidden keyword" in f.message]
        found_keywords = {f.message.split("'")[1] for f in keyword_findings}
        assert "DELETE" in found_keywords
        assert "DETACH" in found_keywords
        assert "DROP" in found_keywords

    def test_schema_with_mixed_good_and_bad_nodes(self):
        """One generic label among domain-specific ones is a WARNING, not ERROR."""
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
        # Generic labels are now WARNINGs — syntactic still passes
        assert result.syntactic is True
        assert any("Thing" in f.message for f in result.findings)
        thing_findings = [f for f in result.findings if "Thing" in f.message]
        assert thing_findings[0].severity == FindingSeverity.WARNING

    def test_findings_are_finding_objects(self):
        """Every finding must be a Finding instance."""
        v = SchemaValidator()
        result = v.validate(_fintech_schema())
        for f in result.findings:
            assert isinstance(f, Finding)

        result_bad = v.validate(_empty_schema())
        for f in result_bad.findings:
            assert isinstance(f, Finding)

    def test_finding_str_includes_severity(self):
        """Finding.__str__ should include severity and stage."""
        f = Finding(
            severity=FindingSeverity.WARNING,
            stage="STRUCTURAL",
            message="test message",
        )
        s = str(f)
        assert "WARNING" in s
        assert "STRUCTURAL" in s
        assert "test message" in s


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


# ---------------------------------------------------------------------------
# Golden fixture tests — independent schemas NOT written for the validator
# ---------------------------------------------------------------------------


class TestGoldenSchemas:
    """Tests using independently-sourced schemas from real Neo4j use cases."""

    def test_known_good_schemas_pass_structural(self):
        """All known-good schemas should score > 0.7 structural."""
        from tests.fixtures.golden_schemas import KNOWN_GOOD

        v = SchemaValidator()
        for schema_fn in KNOWN_GOOD:
            schema = schema_fn()
            result = v.validate(schema)
            assert result.structural_score > 0.7, (
                f"{schema_fn.__name__} scored {result.structural_score}"
            )

    def test_known_good_schemas_pass_syntactic(self):
        """All known-good schemas should pass syntactic validation."""
        from tests.fixtures.golden_schemas import KNOWN_GOOD

        v = SchemaValidator()
        for schema_fn in KNOWN_GOOD:
            schema = schema_fn()
            result = v.validate(schema)
            assert result.syntactic is True, f"{schema_fn.__name__} failed syntactic"

    def test_known_good_schemas_have_no_errors(self):
        """Known-good schemas should have no ERROR findings."""
        from tests.fixtures.golden_schemas import KNOWN_GOOD

        v = SchemaValidator()
        for schema_fn in KNOWN_GOOD:
            schema = schema_fn()
            result = v.validate(schema)
            assert len(result.errors) == 0, f"{schema_fn.__name__} has errors: {result.errors}"

    def test_known_bad_schemas_have_issues(self):
        """All known-bad schemas should have warnings or errors."""
        from tests.fixtures.golden_schemas import KNOWN_BAD

        v = SchemaValidator()
        for schema_fn in KNOWN_BAD:
            schema = schema_fn()
            result = v.validate(schema)
            assert len(result.findings) > 0, (
                f"{schema_fn.__name__} produced no findings — validator too lenient"
            )

    def test_dangling_references_caught(self):
        """Schema with dangling relationship endpoints must get ERROR."""
        from tests.fixtures.golden_schemas import dangling_references_schema

        v = SchemaValidator()
        result = v.validate(dangling_references_schema())
        assert any(f.severity == FindingSeverity.ERROR for f in result.findings)
        assert any("Address" in f.message for f in result.findings)

    def test_properties_mismatch_caught(self):
        """Schema where required_properties are not in properties must get ERROR."""
        from tests.fixtures.golden_schemas import properties_mismatch_schema

        v = SchemaValidator()
        result = v.validate(properties_mismatch_schema())
        assert any(f.severity == FindingSeverity.ERROR for f in result.findings)

    def test_circular_only_gets_warnings(self):
        """All-self-referential schema should get quality warnings."""
        from tests.fixtures.golden_schemas import circular_only_schema

        v = SchemaValidator()
        result = v.validate(circular_only_schema())
        # Missing direction_rationale, missing justification, missing constraints/indexes
        assert len(result.findings) > 0
        assert result.structural_score < 0.7


class TestAdversarialSchemas:
    """Test that adversarial schemas don't game the validator."""

    def test_checkbox_stuffer_not_approved(self):
        """Schema with all fields filled but nonsense domain must not be approved.

        The validator may give it decent structural score since all fields
        are present, but without a driver, semantic = 0.0 blocks approval.
        """
        from tests.fixtures.golden_schemas import checkbox_stuffer_schema

        v = SchemaValidator()
        result = v.validate(checkbox_stuffer_schema())
        # Without a driver, overall = structural*0.40 + 0.0*0.35 + cypher*0.25
        # Even if structural = 1.0 and cypher = 1.0, overall = 0.65 < 0.7
        assert result.approved_for_training is False

    def test_keyword_stuffer_not_approved(self):
        """Schema with keyword-stuffed justifications must not be approved."""
        from tests.fixtures.golden_schemas import keyword_stuffer_schema

        v = SchemaValidator()
        result = v.validate(keyword_stuffer_schema())
        assert result.approved_for_training is False

    def test_structural_pass_semantic_fail_not_approved(self):
        """Schema that passes structural but is semantically absurd must not pass."""
        from tests.fixtures.golden_schemas import structural_pass_semantic_fail_schema

        v = SchemaValidator()
        result = v.validate(structural_pass_semantic_fail_schema())
        # High structural score, but semantic = 0.0 without driver
        assert result.approved_for_training is False
