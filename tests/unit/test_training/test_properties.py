"""Property-based tests for the training pipeline using Hypothesis.

These tests verify invariants that must hold for ALL possible inputs,
not just hand-picked examples. Hypothesis generates random schemas
and checks that the scoring functions never violate their contracts.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from gibsgraph.training.models import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
)
from gibsgraph.training.prompts import score_cypher_quality, score_structural

# ---------------------------------------------------------------------------
# Strategies for generating random schema components
# ---------------------------------------------------------------------------

_label_st = st.text(
    alphabet=st.characters(categories=("Lu", "Ll")),
    min_size=1,
    max_size=20,
)

_property_st = st.text(
    alphabet=st.characters(categories=("Ll",)),
    min_size=1,
    max_size=15,
)

_sentence_st = st.text(min_size=0, max_size=100)


@st.composite
def node_schema_st(draw):
    """Generate a random NodeSchema."""
    label = draw(_label_st)
    props = draw(st.lists(_property_st, min_size=0, max_size=5, unique=True))
    # required_properties is a subset of properties
    req = draw(st.lists(st.sampled_from(props) if props else st.nothing(), max_size=len(props)))
    return NodeSchema(
        label=label,
        properties=props,
        required_properties=list(set(req)),
        description=draw(_sentence_st),
        justified_by=draw(_sentence_st),
    )


@st.composite
def relationship_schema_st(draw, labels=None):
    """Generate a random RelationshipSchema."""
    if labels and len(labels) > 0:
        from_label = draw(st.sampled_from(labels))
        to_label = draw(st.sampled_from(labels))
    else:
        from_label = draw(_label_st)
        to_label = draw(_label_st)
    return RelationshipSchema(
        type=draw(_label_st),
        from_label=from_label,
        to_label=to_label,
        properties=draw(st.lists(_property_st, max_size=3)),
        description=draw(_sentence_st),
        direction_rationale=draw(_sentence_st),
        justified_by=draw(_sentence_st),
    )


@st.composite
def graph_schema_st(draw):
    """Generate a random GraphSchema."""
    nodes = draw(st.lists(node_schema_st(), min_size=0, max_size=8))
    labels = [n.label for n in nodes]
    rels = draw(st.lists(relationship_schema_st(labels=labels), min_size=0, max_size=6))
    return GraphSchema(
        nodes=nodes,
        relationships=rels,
        constraints=draw(st.lists(st.text(min_size=0, max_size=80), max_size=3)),
        indexes=draw(st.lists(st.text(min_size=0, max_size=80), max_size=3)),
        cypher_setup=draw(st.text(min_size=0, max_size=200)),
    )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestStructuralScoreProperties:
    @given(schema=graph_schema_st())
    @settings(max_examples=200, deadline=None)
    def test_score_always_bounded_0_to_1(self, schema):
        """Structural score must always be in [0.0, 1.0] for any input."""
        score, _ = score_structural(schema)
        assert 0.0 <= score <= 1.0

    def test_empty_schema_never_scores_high(self):
        """If schema has 0 nodes and 0 rels, structural score must be < 0.5."""
        schema = GraphSchema(
            nodes=[], relationships=[], constraints=[], indexes=[], cypher_setup=""
        )
        score, _ = score_structural(schema)
        assert score < 0.5

    @given(schema=graph_schema_st())
    @settings(max_examples=100, deadline=None)
    def test_scoring_is_deterministic(self, schema):
        """Same input must always produce same output."""
        s1, f1 = score_structural(schema)
        s2, f2 = score_structural(schema)
        assert s1 == s2
        assert len(f1) == len(f2)

    @given(schema=graph_schema_st())
    @settings(max_examples=200, deadline=None)
    def test_findings_are_always_valid(self, schema):
        """All findings must have valid stage and message."""
        _, findings = score_structural(schema)
        for f in findings:
            assert f.stage == "STRUCTURAL"
            assert len(f.message) > 0

    def test_no_nodes_means_no_vacuous_truth(self):
        """With 0 nodes and 0 rels, field-presence checks should all fail."""
        schema = GraphSchema(
            nodes=[], relationships=[], constraints=[], indexes=[], cypher_setup=""
        )
        score, _findings = score_structural(schema)
        # With zero nodes and zero rels, most checks fail
        assert score < 0.5


class TestCypherQualityProperties:
    @given(cypher=st.text(min_size=0, max_size=500))
    @settings(max_examples=200, deadline=None)
    def test_score_always_bounded_0_to_1(self, cypher):
        """Cypher quality score must always be in [0.0, 1.0]."""
        score, _ = score_cypher_quality(cypher)
        assert 0.0 <= score <= 1.0

    @given(cypher=st.text(min_size=0, max_size=500))
    @settings(max_examples=100, deadline=None)
    def test_scoring_is_deterministic(self, cypher):
        """Same Cypher must produce same score."""
        s1, _ = score_cypher_quality(cypher)
        s2, _ = score_cypher_quality(cypher)
        assert s1 == s2

    @given(cypher=st.from_regex(r"^\s*$", fullmatch=True))
    @settings(max_examples=50, deadline=None)
    def test_empty_or_whitespace_scores_zero(self, cypher):
        """Empty or whitespace-only Cypher must score 0."""
        score, _ = score_cypher_quality(cypher)
        assert score == 0.0
