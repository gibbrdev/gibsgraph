"""Example: EU Regulatory Knowledge Graph.

This example demonstrates GibsGraph for the gibs.dev regulatory compliance usecase.
Builds a KG from EU fintech regulations and answers compliance questions.

Usage:
    cp .env.example .env  # set NEO4J_READ_ONLY=false for ingestion
    python examples/regulatory_kg.py
"""

from __future__ import annotations

from gibsgraph import Graph

REGULATORY_TEXT = """
The EU Payment Services Directive 2 (PSD2) requires all payment service providers
operating in the European Economic Area to implement Strong Customer Authentication (SCA)
for electronic payment transactions. SCA requires at least two of three authentication
factors: something the user knows (password/PIN), something the user has (phone/token),
and something the user is (biometric).

MiCA (Markets in Crypto-Assets Regulation) applies to crypto-asset service providers (CASPs)
operating in the EU. CASPs must obtain authorization from their national competent authority
before providing services. The regulation covers issuers of asset-referenced tokens (ARTs)
and e-money tokens (EMTs).

DORA (Digital Operational Resilience Act) mandates that financial entities including banks,
insurance companies, and investment firms implement ICT risk management frameworks,
report major ICT incidents, and conduct digital operational resilience testing.
"""

QUESTIONS = [
    "What authentication requirements does PSD2 impose on payment providers?",
    "Which regulation governs crypto asset service providers in the EU?",
    "What must financial entities do under DORA?",
    "What are the three authentication factors under SCA?",
]


def main() -> None:
    print("🕸️  GibsGraph — EU Regulatory KG Example\n")

    # Ingest (requires read_only=False)
    g_write = Graph(read_only=False)
    try:
        print("📥 Ingesting regulatory text...")
        result = g_write.ingest(REGULATORY_TEXT, source="eu_regulations")
        print(f"   ✓ {result}\n")
    except RuntimeError as e:
        print(f"   ℹ️  Skipping ingest: {e}\n")

    # Ask questions (read-only, default)
    g = Graph()
    for question in QUESTIONS:
        print(f"❓ {question}")
        result = g.ask(question)
        print(f"   💬 {result}")
        if result.errors:
            print(f"   ⚠️  {result.errors}")
        print()


if __name__ == "__main__":
    main()
