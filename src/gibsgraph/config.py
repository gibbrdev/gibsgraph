"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """GibsGraph configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: SecretStr = Field(..., alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    neo4j_read_only: bool = Field(default=True, alias="NEO4J_READ_ONLY")
    neo4j_max_connection_lifetime: int = Field(default=3600, alias="NEO4J_MAX_CONNECTION_LIFETIME")

    # LLM
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # Embeddings
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(default=1536, alias="EMBEDDING_DIMENSIONS")

    # Agent
    agent_max_steps: int = Field(default=10, alias="AGENT_MAX_STEPS")
    agent_checkpoint_db: str = Field(
        default="sqlite:///checkpoints.db", alias="AGENT_CHECKPOINT_DB"
    )

    # Observability (optional)
    langsmith_api_key: SecretStr | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="gibsgraph", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    # Security
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_PER_MINUTE")

    @field_validator("neo4j_uri")
    @classmethod
    def validate_neo4j_uri(cls, v: str) -> str:
        """Ensure Neo4j URI uses an accepted scheme."""
        allowed = ("bolt://", "bolt+s://", "neo4j://", "neo4j+s://")
        if not any(v.startswith(s) for s in allowed):
            msg = f"NEO4J_URI must start with one of {allowed}"
            raise ValueError(msg)
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()  # type: ignore[call-arg]
