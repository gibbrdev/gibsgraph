"""Unit tests for CypherValidator."""

import pytest

from gibsgraph.tools.cypher_validator import CypherValidator, CypherValidationError


@pytest.fixture
def validator() -> CypherValidator:
    return CypherValidator()


def test_valid_parameterized_query(validator):
    cypher = "MATCH (n:Entity {name: $name}) RETURN n LIMIT $limit"
    assert validator.validate(cypher) is True


def test_empty_query_fails(validator):
    assert validator.validate("") is False
    assert validator.validate("   ") is False


def test_injection_drop_fails(validator):
    assert validator.validate("MATCH (n) RETURN n; DROP DATABASE neo4j") is False


def test_injection_delete_fails(validator):
    assert validator.validate("MATCH (n) RETURN n; DELETE n") is False


def test_load_csv_blocked(validator):
    assert validator.validate("LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row") is False


def test_apoc_export_blocked(validator):
    assert validator.validate("CALL apoc.export.csv.all('dump.csv', {})") is False


def test_extract_parameters(validator):
    cypher = "MATCH (n:Entity {name: $name}) WHERE n.age > $min_age RETURN n"
    params = validator.extract_parameters(cypher)
    assert "name" in params
    assert "min_age" in params


def test_assert_valid_raises_on_injection(validator):
    with pytest.raises(CypherValidationError):
        validator.assert_valid("MATCH (n) RETURN n; DROP DATABASE neo4j")
