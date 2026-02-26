"""Pydantic models for the training pipeline.

These models define the data structures for use case generation,
schema synthesis, validation, and scoring. Every generated use case
flows through these models from research to approved training data.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


def _short_id() -> str:
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FindingSeverity(StrEnum):
    """Severity level for validation findings.

    Enterprise pattern: ERRORs block, WARNINGs degrade, INFOs inform.
    """

    ERROR = "error"  # Must fix — blocks approval
    WARNING = "warning"  # Should fix — degrades quality score
    INFO = "info"  # Good to know — no score impact


class Industry(StrEnum):
    """Target industries for use case generation."""

    FINTECH = "fintech"
    INSURANCE = "insurance"
    SUPPLY_CHAIN = "supply_chain"
    HEALTHCARE = "healthcare"
    CYBERSECURITY = "cybersecurity"
    HR = "hr"
    ECOMMERCE = "ecommerce"
    MEDIA = "media"
    COMPLIANCE = "compliance"
    SOCIAL_NETWORK = "social_network"


class Differentiator(StrEnum):
    """Context modifiers that shape schema design."""

    STARTUP = "startup"
    ENTERPRISE = "enterprise"
    EU = "eu"
    US = "us"
    GLOBAL = "global"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PSD2 = "psd2"
    SOX = "sox"
    SOLVENCY_II = "solvency_ii"
    DORA = "dora"
    REALTIME = "realtime"
    BATCH = "batch"
    HIGH_VOLUME = "high_volume"
    SENSITIVE_DATA = "sensitive_data"
    LEGACY_INTEGRATION = "legacy_integration"


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


class NodeSchema(BaseModel):
    """A node type in a generated graph schema."""

    label: str
    properties: list[str]
    required_properties: list[str]
    description: str
    justified_by: str  # which finding or expert pattern requires this node


class RelationshipSchema(BaseModel):
    """A relationship type in a generated graph schema."""

    type: str
    from_label: str
    to_label: str
    properties: list[str]
    description: str
    direction_rationale: str
    justified_by: str  # which finding or pattern requires this relationship


class GraphSchema(BaseModel):
    """A complete Neo4j graph schema ready for validation."""

    nodes: list[NodeSchema]
    relationships: list[RelationshipSchema]
    constraints: list[str]
    indexes: list[str]
    cypher_setup: str


class Finding(BaseModel):
    """A single validation finding with severity level.

    Enterprise pattern (SonarQube, ESLint): categorize findings by severity
    so teams can prioritize fixes and avoid blocking on non-critical issues.
    """

    severity: FindingSeverity
    stage: str  # SYNTACTIC, STRUCTURAL, SEMANTIC, CYPHER
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.stage}: {self.message}"


# ---------------------------------------------------------------------------
# Research models
# ---------------------------------------------------------------------------


class ResearchSource(BaseModel):
    """Research collected from one source."""

    source_type: Literal["perplexity", "grok", "expert_graph", "manual"]
    fetched_at: datetime = Field(default_factory=_utcnow)
    raw_content: str
    file_path: str
    quality_score: float = Field(ge=0.0, le=1.0)
    key_findings: list[str]
    token_count: int = 0


class ExpertGraphResult(BaseModel):
    """Results from querying the expert knowledge graph."""

    fetched_at: datetime = Field(default_factory=_utcnow)
    query_used: str
    nodes_retrieved: int
    patterns_used: list[str]
    similarity_scores: list[float]
    context_text: str


# ---------------------------------------------------------------------------
# Synthesis + Validation
# ---------------------------------------------------------------------------


class SynthesisResult(BaseModel):
    """Output from one LLM synthesizing a graph schema."""

    model: str
    synthesized_at: datetime = Field(default_factory=_utcnow)
    scenario: str
    design_rationale: str
    graph_schema: GraphSchema
    regulatory_requirements: list[str]
    expert_patterns_used: list[str]
    findings_used: list[str]
    quality_score: float = Field(ge=0.0, le=1.0)
    score_breakdown: dict[str, float]
    file_path: str
    tokens_used: int = 0
    cost_usd: float = 0.0


class ValidationResult(BaseModel):
    """Result of the 4-stage validation pipeline."""

    syntactic: bool
    structural_score: float = Field(ge=0.0, le=1.0)
    semantic_score: float = Field(ge=0.0, le=1.0, default=0.0)
    domain_score: float = Field(ge=0.0, le=1.0, default=0.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    findings: list[Finding]
    approved_for_training: bool = False
    approved_by: str | None = None
    approved_at: datetime | None = None

    @property
    def errors(self) -> list[Finding]:
        """ERROR-severity findings — must fix, block approval."""
        return [f for f in self.findings if f.severity == FindingSeverity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        """WARNING-severity findings — should fix, degrade score."""
        return [f for f in self.findings if f.severity == FindingSeverity.WARNING]

    @property
    def infos(self) -> list[Finding]:
        """INFO-severity findings — awareness only."""
        return [f for f in self.findings if f.severity == FindingSeverity.INFO]


# ---------------------------------------------------------------------------
# Master record
# ---------------------------------------------------------------------------


class UseCaseRecord(BaseModel):
    """One complete use case — from research to approved training data."""

    id: str = Field(default_factory=_short_id)
    created_at: datetime = Field(default_factory=_utcnow)
    industry: Industry
    sub_industry: str
    differentiators: list[Differentiator]

    research: dict[str, ResearchSource] = Field(default_factory=dict)
    expert_graph: ExpertGraphResult | None = None

    synthesis_a: SynthesisResult | None = None  # Model A (e.g. Sonnet)
    synthesis_b: SynthesisResult | None = None  # Model B (e.g. Gemini)
    winner: Literal["a", "b"] | None = None
    selection_reason: str | None = None

    validation: ValidationResult | None = None

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    generation_time_seconds: float = 0.0
    notes: str = ""

    @property
    def is_approved(self) -> bool:
        """Whether this use case passed validation and is approved for training."""
        return self.validation is not None and self.validation.approved_for_training

    @property
    def winning_synthesis(self) -> SynthesisResult | None:
        """Return the winning synthesis result, if any."""
        if self.winner == "a":
            return self.synthesis_a
        if self.winner == "b":
            return self.synthesis_b
        return None
