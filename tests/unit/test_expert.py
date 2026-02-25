"""Tests for the expert knowledge store."""

from unittest.mock import MagicMock

from gibsgraph.expert import ExpertContext, ExpertHit, ExpertStore, _to_lucene


class TestToLucene:
    """Test Lucene query escaping."""

    def test_plain_text_unchanged(self):
        assert _to_lucene("shortest path") == "shortest path"

    def test_escapes_special_chars(self):
        result = _to_lucene('query with "quotes" and (parens)')
        assert '\\"' in result
        assert "\\(" in result
        assert "\\)" in result

    def test_escapes_operators(self):
        result = _to_lucene("foo + bar - baz")
        assert "\\+" in result
        assert "\\-" in result

    def test_empty_string(self):
        assert _to_lucene("") == ""

    def test_colon_escaped(self):
        result = _to_lucene("MATCH (n:Person)")
        assert "\\:" in result


class TestExpertHit:
    """Test ExpertHit dataclass."""

    def test_defaults(self):
        hit = ExpertHit(label="CypherClause", name="MATCH", score=0.95)
        assert hit.description == ""
        assert hit.cypher == ""
        assert hit.signature == ""


class TestExpertContext:
    """Test ExpertContext formatting."""

    def test_empty_hits_returns_empty_prompt(self):
        ctx = ExpertContext(hits=[], query="test")
        assert ctx.to_prompt() == ""

    def test_cypher_example_in_prompt(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(
                    label="CypherExample",
                    name="example",
                    score=0.9,
                    cypher="MATCH (n)-[:ACTED_IN]->(m) RETURN n, m",
                )
            ],
            query="actors in movies",
        )
        prompt = ctx.to_prompt()
        assert "MATCH (n)-[:ACTED_IN]->(m) RETURN n, m" in prompt
        assert "Relevant Cypher examples:" in prompt

    def test_best_practice_in_prompt(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(
                    label="BestPractice",
                    name="Use indexes",
                    score=0.8,
                    description="Always create indexes on frequently queried properties.",
                )
            ],
            query="performance",
        )
        prompt = ctx.to_prompt()
        assert "Use indexes" in prompt
        assert "Best practices:" in prompt

    def test_function_in_prompt(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(
                    label="CypherFunction",
                    name="shortestPath",
                    score=0.85,
                    signature="shortestPath(path) :: PATH",
                )
            ],
            query="shortest path",
        )
        prompt = ctx.to_prompt()
        assert "shortestPath" in prompt
        assert "Relevant functions:" in prompt

    def test_clause_in_prompt(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(
                    label="CypherClause",
                    name="OPTIONAL MATCH",
                    score=0.7,
                    description="Matches patterns optionally.",
                )
            ],
            query="optional",
        )
        prompt = ctx.to_prompt()
        assert "OPTIONAL MATCH" in prompt

    def test_modeling_pattern_in_prompt(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(
                    label="ModelingPattern",
                    name="Intermediate node",
                    score=0.75,
                    description="Use intermediate nodes for many-to-many relationships.",
                )
            ],
            query="modeling",
        )
        prompt = ctx.to_prompt()
        assert "Intermediate node" in prompt
        assert "Modeling patterns:" in prompt

    def test_mixed_hits_all_sections(self):
        ctx = ExpertContext(
            hits=[
                ExpertHit(label="CypherExample", name="ex", score=0.9, cypher="MATCH (n) RETURN n"),
                ExpertHit(label="BestPractice", name="bp", score=0.8, description="Do this."),
                ExpertHit(label="CypherFunction", name="fn", score=0.7, signature="fn() :: INT"),
                ExpertHit(label="ModelingPattern", name="mp", score=0.6, description="Model this."),
            ],
            query="test",
        )
        prompt = ctx.to_prompt()
        assert "Relevant Cypher examples:" in prompt
        assert "Best practices:" in prompt
        assert "Relevant functions:" in prompt
        assert "Modeling patterns:" in prompt

    def test_limits_examples_to_five(self):
        hits = [
            ExpertHit(label="CypherExample", name=f"ex{i}", score=0.9, cypher=f"RETURN {i}")
            for i in range(10)
        ]
        ctx = ExpertContext(hits=hits, query="test")
        prompt = ctx.to_prompt()
        # Should only include 5 examples
        assert prompt.count("RETURN") == 5


class TestExpertStore:
    """Test ExpertStore with mocked Neo4j driver."""

    def _make_store(self):
        driver = MagicMock()
        return ExpertStore(driver, database="neo4j"), driver

    def test_is_available_true(self):
        store, driver = self._make_store()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        record = MagicMock()
        record.__getitem__ = lambda self, key: 1
        result = MagicMock()
        result.single.return_value = record
        session.run.return_value = result

        assert store.is_available() is True

    def test_is_available_false_when_no_index(self):
        store, driver = self._make_store()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        record = MagicMock()
        record.__getitem__ = lambda self, key: 0
        result = MagicMock()
        result.single.return_value = record
        session.run.return_value = result

        assert store.is_available() is False

    def test_is_available_caches_result(self):
        store, driver = self._make_store()
        store._available = True
        # Should not call driver at all
        assert store.is_available() is True
        driver.session.assert_not_called()

    def test_search_returns_empty_when_unavailable(self):
        store, _ = self._make_store()
        store._available = False
        result = store.search("test query")
        assert result.hits == []
        assert result.query == "test query"

    def test_search_returns_hits(self):
        store, driver = self._make_store()
        store._available = True

        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = {
            "label": "CypherClause",
            "name": "MATCH",
            "score": 0.95,
            "description": "Pattern matching",
            "cypher": "",
            "signature": "",
        }
        result_mock = MagicMock()
        result_mock.__iter__ = MagicMock(return_value=iter([mock_record]))
        session.run.return_value = result_mock

        result = store.search("match patterns")
        assert len(result.hits) == 1
        assert result.hits[0].label == "CypherClause"
        assert result.hits[0].name == "MATCH"

    def test_search_handles_exception(self):
        store, driver = self._make_store()
        store._available = True

        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.side_effect = Exception("connection lost")

        result = store.search("test")
        assert result.hits == []
