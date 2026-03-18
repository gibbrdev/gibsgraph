"""Build GibsGraph training pairs from DORA (Digital Operational Resilience Act).

CELEX: 32022R2554 — Regulation (EU) 2022/2554
64 articles, 9 chapters, ~80 recitals.
Dense cross-references to MiFID II, Solvency II, GDPR, NIS2, PSD2, CRD/CRR.

Usage:
    python data/scripts/build_dora_pairs.py
    python data/scripts/build_dora_pairs.py --stats-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch_eurlex import build_regulation_pairs

CELEX = "32022R2554"
REGULATION = "dora"
DEFAULT_OUTPUT = Path("data/training/dora_pairs.jsonl")
CACHE_DIR = Path("data/raw/eurlex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from DORA")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-edges", type=int, default=0)
    parser.add_argument("--min-text", type=int, default=50)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    cache = None if args.no_cache else CACHE_DIR

    build_regulation_pairs(
        CELEX,
        REGULATION,
        args.output,
        cache_dir=cache,
        min_edges=args.min_edges,
        min_text_length=args.min_text,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
