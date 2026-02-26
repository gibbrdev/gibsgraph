"""Tests for the expert knowledge store."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from gibsgraph.expert import (
    BundledExpertStore,
    ExpertContext,
    ExpertHit,
    ExpertStore,
    _to_lucene,
    _tokenize,
)


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

    def test_search_uses_bundled_when_unavailable(self):
        store, _ = self._make_store()
        store._available = False
        result = store.search("MATCH clause")
        # Falls back to bundled store — should get hits
        assert len(result.hits) > 0
        assert result.query == "MATCH clause"

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
        # Falls back to bundled — should still get hits
        assert result.query == "test"

    def test_search_falls_back_to_bundled_when_unavailable(self):
        store, _ = self._make_store()
        store._available = False
        result = store.search("MATCH clause")
        # Should get bundled hits instead of empty
        assert len(result.hits) > 0
        assert result.query == "MATCH clause"

    def test_search_falls_back_to_bundled_on_exception(self):
        store, driver = self._make_store()
        store._available = True

        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        session.run.side_effect = Exception("connection lost")

        result = store.search("shortestPath")
        assert len(result.hits) > 0
        assert result.query == "shortestPath"


class TestTokenize:
    """Test keyword tokenizer."""

    def test_basic_tokens(self):
        tokens = _tokenize("MATCH shortest path")
        assert "match" in tokens
        assert "shortest" in tokens
        assert "path" in tokens

    def test_stop_words_excluded(self):
        tokens = _tokenize("the quick brown fox is a test")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens

    def test_empty_string(self):
        assert _tokenize("") == set()


class TestBundledDataFiles:
    """Test that bundled JSONL files exist and are valid."""

    DATA_DIR = Path(__file__).resolve().parents[2] / "src" / "gibsgraph" / "data"
    EXPECTED_FILES = (
        "cypher_clauses.jsonl",
        "cypher_functions.jsonl",
        "cypher_examples.jsonl",
        "modeling_patterns.jsonl",
        "best_practices.jsonl",
    )

    def test_all_files_exist(self):
        for name in self.EXPECTED_FILES:
            path = self.DATA_DIR / name
            assert path.exists(), f"Missing bundled data file: {name}"

    def test_files_are_valid_jsonl(self):
        for name in self.EXPECTED_FILES:
            path = self.DATA_DIR / name
            lines = path.read_text(encoding="utf-8").splitlines()
            assert len(lines) > 0, f"Empty data file: {name}"
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise AssertionError(f"{name} line {i + 1}: invalid JSON: {e}") from e

    def test_init_py_exists(self):
        assert (self.DATA_DIR / "__init__.py").exists()


class TestBundledExpertStore:
    """Test the bundled JSONL fallback store."""

    def test_loads_records(self):
        store = BundledExpertStore()
        records = store._load()
        assert len(records) > 100  # Should have hundreds of records

    def test_lazy_load_caches(self):
        store = BundledExpertStore()
        first = store._load()
        second = store._load()
        assert first is second

    def test_search_match_returns_hits(self):
        store = BundledExpertStore()
        result = store.search("MATCH clause")
        assert len(result.hits) > 0
        assert result.query == "MATCH clause"

    def test_search_avg_returns_function(self):
        store = BundledExpertStore()
        result = store.search("avg average aggregating")
        assert len(result.hits) > 0
        labels = {h.label for h in result.hits}
        assert "CypherFunction" in labels

    def test_search_respects_top_k(self):
        store = BundledExpertStore()
        result = store.search("MATCH RETURN WHERE", top_k=3)
        assert len(result.hits) <= 3

    def test_search_empty_query_returns_empty(self):
        store = BundledExpertStore()
        result = store.search("")
        assert result.hits == []

    def test_search_stop_words_only_returns_empty(self):
        store = BundledExpertStore()
        result = store.search("the is a")
        assert result.hits == []

    def test_hits_have_correct_types(self):
        store = BundledExpertStore()
        result = store.search("aggregating functions avg count")
        for hit in result.hits:
            assert isinstance(hit, ExpertHit)
            assert isinstance(hit.score, float)
            assert 0.0 < hit.score <= 1.0
            assert hit.label in {
                "CypherClause",
                "CypherFunction",
                "CypherExample",
                "ModelingPattern",
                "BestPractice",
            }
