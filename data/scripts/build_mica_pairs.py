"""Build GibsGraph training pairs from MiCA (Markets in Crypto-Assets).

CELEX: 32023R1114 — Regulation (EU) 2023/1114
~149 articles, 9 titles, ~109 recitals, 7 annexes.
Largest EU fintech regulation. Dense cross-refs to MiFID II, MAR, AMLD, PSD2, DORA.

Usage:
    python data/scripts/build_mica_pairs.py
    python data/scripts/build_mica_pairs.py --stats-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fetch_eurlex import build_regulation_pairs

CELEX = "32023R1114"
REGULATION = "mica"
DEFAULT_OUTPUT = Path("data/training/mica_pairs.jsonl")
CACHE_DIR = Path("data/raw/eurlex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training pairs from MiCA")
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
