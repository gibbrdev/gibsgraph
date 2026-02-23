"""Application configuration via environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# LLM provider registry — single source of truth for model → provider mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LLMProvider:
    """Describes one LLM provider: its env key, default model, and prefix.

    Providers with ``base_url`` are OpenAI-compatible and use ``ChatOpenAI``
    with a custom endpoint.  Native providers (base_url=None) get their own
    LangChain class in ``_make_llm``.
    """

    name: str
    env_key: str
    default_model: str
    model_prefixes: tuple[str, ...]
    base_url: str | None = None


PROVIDERS: tuple[LLMProvider, ...] = (
    LLMProvider(
        name="openai",
        env_key="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
        model_prefixes=("gpt-", "o1-", "o3-"),
    ),
    LLMProvider(
        name="anthropic",
        env_key="ANTHROPIC_API_KEY",
        default_model="claude-3-haiku-20240307",
        model_prefixes=("claude-",),
    ),
    LLMProvider(
        name="mistral",
        env_key="MISTRAL_API_KEY",
        default_model="mistral-small-latest",
        model_prefixes=("mistral-", "open-mistral-", "open-mixtral-"),
    ),
    LLMProvider(
        name="xai",
        env_key="XAI_API_KEY",
        default_model="grok-3",
        model_prefixes=("grok-",),
        base_url="https://api.x.ai/v1",
    ),
)

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 1536


def provider_for_model(model: str) -> LLMProvider | None:
    """Return the provider that owns a given model name, or None."""
    for p in PROVIDERS:
        if any(model.startswith(prefix) for prefix in p.model_prefixes):
            return p
    return None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


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
    mistral_api_key: SecretStr | None = Field(default=None, alias="MISTRAL_API_KEY")
    xai_api_key: SecretStr | None = Field(default=None, alias="XAI_API_KEY")
    llm_model: str = Field(default=PROVIDERS[0].default_model, alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")

    # Embeddings
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSIONS, alias="EMBEDDING_DIMENSIONS"
    )

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
