"""ARCHIVED — Hub probe that found no source. Replaced by prep_scitechbanro.py (GitHub xlsx).

Prepare SciTechBanRO (~11K Romanian science/tech articles, domain classification).

Sourcing notes:
  - readerbench/SciTechBanRO
  - dumitrescustefan/scitechban-ro
"""

from __future__ import annotations

import argparse

from romteb.data_prep._generic_classification_prep import run_generic_prep


CANDIDATE_SOURCES = [
    "readerbench/SciTechBanRO",
    "readerbench/sci-tech-ban-ro",
    "dumitrescustefan/scitechban-ro",
    "SciTechBanRO",
]
TARGET_REPO = "alina0195/romteb-scitechbanro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_generic_prep(
        name="SciTechBanRO",
        candidate_sources=CANDIDATE_SOURCES,
        target_repo=TARGET_REPO,
        text_keys=("text", "content", "body", "article", "abstract"),
        label_keys=("label", "domain", "class", "category", "topic"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
