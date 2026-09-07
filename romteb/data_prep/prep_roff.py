"""Prepare ROFF (~5K Romanian tweets, offensive vs. non-offensive).

Sourcing notes:
  - readerbench/ROFF
  - readerbench/ro-twitter-offense
  - Andrei7/ROFF
"""

from __future__ import annotations

import argparse

from romteb.data_prep._generic_classification_prep import run_generic_prep


CANDIDATE_SOURCES = [
    "readerbench/ROFF",
    "readerbench/ro-twitter-offense",
    "Andrei7/ROFF",
    "ROFF",
]
TARGET_REPO = "alina0195/romteb-roff"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_generic_prep(
        name="ROFF",
        candidate_sources=CANDIDATE_SOURCES,
        target_repo=TARGET_REPO,
        text_keys=("tweet", "text", "content", "body"),
        label_keys=("label", "offensive", "class", "category"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
