"""Unit tests for Settings configuration."""

import pytest

from gibsgraph.config import PROVIDERS, Settings, provider_for_model


def test_settings_defaults():
    s = Settings(NEO4J_PASSWORD="test")
    assert s.neo4j_uri == "bolt://localhost:7687"
    assert s.neo4j_username == "neo4j"
    assert s.neo4j_database == "neo4j"
    assert s.neo4j_read_only is True
    assert s.llm_model == "gpt-4o-mini"
    assert s.llm_temperature == 0.0


def test_settings_custom_values():
    s = Settings(
        NEO4J_URI="bolt://custom:7688",
        NEO4J_PASSWORD="secret",
        NEO4J_USERNAME="admin",
        NEO4J_DATABASE="mydb",
        NEO4J_READ_ONLY=False,
        LLM_MODEL="claude-3-5-sonnet",
        LLM_TEMPERATURE=0.5,
    )
    assert s.neo4j_uri == "bolt://custom:7688"
    assert s.neo4j_username == "admin"
    assert s.neo4j_database == "mydb"
    assert s.neo4j_read_only is False
    assert s.llm_model == "claude-3-5-sonnet"
    assert s.llm_temperature == 0.5


def test_settings_invalid_uri():
    with pytest.raises(ValueError, match="NEO4J_URI must start with"):
        Settings(NEO4J_URI="http://bad:7687", NEO4J_PASSWORD="test")


def test_settings_valid_uri_schemes():
    for scheme in ("bolt://", "bolt+s://", "neo4j://", "neo4j+s://"):
        s = Settings(NEO4J_URI=f"{scheme}host:7687", NEO4J_PASSWORD="test")
        assert s.neo4j_uri.startswith(scheme)


def test_settings_password_is_secret():
    s = Settings(NEO4J_PASSWORD="supersecret")
    assert s.neo4j_password.get_secret_value() == "supersecret"
    # Should not expose password in repr
    assert "supersecret" not in repr(s)


# --- Provider registry ---


def test_provider_for_model_openai():
    p = provider_for_model("gpt-4o-mini")
    assert p is not None
    assert p.name == "openai"


def test_provider_for_model_anthropic():
    p = provider_for_model("claude-3-5-sonnet-20241022")
    assert p is not None
    assert p.name == "anthropic"


def test_provider_for_model_mistral():
    p = provider_for_model("mistral-small-latest")
    assert p is not None
    assert p.name == "mistral"


def test_provider_for_model_xai():
    p = provider_for_model("grok-3")
    assert p is not None
    assert p.name == "xai"
    assert p.base_url == "https://api.x.ai/v1"


def test_provider_for_model_unknown():
    p = provider_for_model("some-random-model")
    assert p is None


def test_providers_order():
    assert PROVIDERS[0].name == "openai"
    assert PROVIDERS[1].name == "anthropic"
    assert PROVIDERS[2].name == "mistral"
    assert PROVIDERS[3].name == "xai"


def test_default_llm_model_matches_first_provider():
    s = Settings(NEO4J_PASSWORD="test")
    assert s.llm_model == PROVIDERS[0].default_model


# --- PCST settings ---


def test_pcst_defaults():
    s = Settings(NEO4J_PASSWORD="test")
    assert s.pcst_enabled is False
    assert s.pcst_max_nodes == 20
    assert s.pcst_edge_cost == 0.1


def test_pcst_custom_values():
    s = Settings(
        NEO4J_PASSWORD="test",
        PCST_ENABLED=True,
        PCST_MAX_NODES=50,
        PCST_EDGE_COST=0.25,
    )
    assert s.pcst_enabled is True
    assert s.pcst_max_nodes == 50
    assert s.pcst_edge_cost == 0.25
