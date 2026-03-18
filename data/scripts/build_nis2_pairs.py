"""Build GibsGraph training pairs from NIS2 Directive.

CELEX: 32022L2555 — Directive (EU) 2022/2555
46 articles, ~144 recitals, 3 annexes.
Cross-refs to DORA (Art 4 financial sector carve-out), CER, GDPR, eIDAS.

Usage:
    python data/scripts/build_nis2_pairs.py
    python data/scripts/build_nis2_pairs.py --stats-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch_eurlex import build_regulation_pairs

CELEX = "32022L2555"
REGULATION = "nis2"
DEFAULT_OUTPUT = Path("data/training/nis2_pairs.jsonl")
CACHE_DIR = Path("data/raw/eurlex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from NIS2")
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
