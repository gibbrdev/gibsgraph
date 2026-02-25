"""Quality scorer for generated graph schemas.

Scores on five weighted dimensions:
- regulatory_coverage (0.25) — LLM Socratic
- expert_alignment (0.25) — LLM Socratic
- structural_validity (0.20) — deterministic
- completeness (0.20) — LLM Socratic
- cypher_quality (0.10) — deterministic
"""

from __future__ import annotations

import json

import structlog

from gibsgraph.config import Settings
from gibsgraph.training.models import SynthesisResult
from gibsgraph.training.prompts import (
    build_socratic_scoring_prompt,
    compute_score_from_socratic,
    score_cypher_quality,
    score_structural,
)

log = structlog.get_logger(__name__)

SCORE_WEIGHTS: dict[str, float] = {
    "regulatory_coverage": 0.25,
    "expert_alignment": 0.25,
    "structural_validity": 0.20,
    "completeness": 0.20,
    "cypher_quality": 0.10,
}


class QualityScorer:
    """Scores a synthesis result on five dimensions.

    Two deterministic (no LLM needed):
    - structural_validity: constraints, indexes, direction rationale present
    - cypher_quality: syntax check on Cypher setup script

    Three LLM-assisted (Haiku, Socratic yes/no):
    - regulatory_coverage
    - expert_alignment
    - completeness
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def score(
        self,
        synthesis: SynthesisResult,
        *,
        research_context: str,
        expert_patterns: list[str],
        industry: str,
        differentiators: list[str],
    ) -> tuple[float, dict[str, float], list[str]]:
        """Score a synthesis result. Returns (overall, breakdown, findings)."""
        breakdown: dict[str, float] = {}
        all_findings: list[str] = []

        # Deterministic checks
        structural, struct_findings = score_structural(synthesis.graph_schema)
        breakdown["structural_validity"] = structural
        all_findings.extend(struct_findings)

        cypher, cypher_findings = score_cypher_quality(synthesis.graph_schema.cypher_setup)
        breakdown["cypher_quality"] = cypher
        all_findings.extend(cypher_findings)

        # LLM-assisted Socratic scoring
        socratic = self._score_socratic(
            synthesis=synthesis,
            research_context=research_context,
            expert_patterns=expert_patterns,
            industry=industry,
            differentiators=differentiators,
        )
        breakdown.update(socratic)

        # Weighted overall score
        overall = sum(
            breakdown.get(dim, 0.0) * weight
            for dim, weight in SCORE_WEIGHTS.items()
        )
        overall = round(overall, 3)

        log.info("scorer.scored", overall=overall, breakdown=breakdown)
        return overall, breakdown, all_findings

    def score_deterministic(
        self, synthesis: SynthesisResult
    ) -> tuple[float, dict[str, float], list[str]]:
        """Score only deterministic dimensions (no LLM calls).

        Useful for fast local validation without API keys.
        """
        breakdown: dict[str, float] = {}
        all_findings: list[str] = []

        structural, struct_findings = score_structural(synthesis.graph_schema)
        breakdown["structural_validity"] = structural
        all_findings.extend(struct_findings)

        cypher, cypher_findings = score_cypher_quality(synthesis.graph_schema.cypher_setup)
        breakdown["cypher_quality"] = cypher
        all_findings.extend(cypher_findings)

        # Weight only the deterministic dimensions
        det_weights = {
            "structural_validity": SCORE_WEIGHTS["structural_validity"],
            "cypher_quality": SCORE_WEIGHTS["cypher_quality"],
        }
        total_weight = sum(det_weights.values())
        overall = sum(
            breakdown.get(dim, 0.0) * weight
            for dim, weight in det_weights.items()
        ) / total_weight if total_weight > 0 else 0.0

        return round(overall, 3), breakdown, all_findings

    def _score_socratic(
        self,
        synthesis: SynthesisResult,
        research_context: str,
        expert_patterns: list[str],
        industry: str,
        differentiators: list[str],
    ) -> dict[str, float]:
        """LLM-assisted Socratic scoring via Haiku."""
        from gibsgraph.agent import _make_llm

        prompt = build_socratic_scoring_prompt(
            synthesis=synthesis,
            research_context=research_context,
            expert_patterns=expert_patterns,
            industry=industry,
            differentiators=differentiators,
        )

        try:
            llm = _make_llm(self.settings)
            response = llm.invoke(prompt)
            text = str(response.content).strip()

            # Strip markdown fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    line for line in lines if not line.startswith("```")
                ).strip()

            answers: dict[str, str] = json.loads(text)
            return compute_score_from_socratic(answers)
        except Exception as exc:
            log.error("scorer.socratic_failed", error=str(exc))
            return {
                "regulatory_coverage": 0.0,
                "expert_alignment": 0.0,
                "completeness": 0.0,
            }
