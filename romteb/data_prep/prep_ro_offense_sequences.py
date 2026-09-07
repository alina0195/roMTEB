"""Prepare RO-Offense-Sequences (~4800 Romanian comments, offense classification).

Sourcing notes:
  - readerbench/ro-offense-sequences
  - readerbench/RO-Offense-Sequences
"""

from __future__ import annotations

import argparse

from romteb.data_prep._generic_classification_prep import run_generic_prep


CANDIDATE_SOURCES = [
    "readerbench/ro-offense-sequences",
    "readerbench/RO-Offense-Sequences",
    "RO-Offense-Sequences",
]
TARGET_REPO = "alina0195/romteb-ro-offense-sequences"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_generic_prep(
        name="RO-Offense-Sequences",
        candidate_sources=CANDIDATE_SOURCES,
        target_repo=TARGET_REPO,
        text_keys=("text", "content", "comment", "body"),
        label_keys=("label", "offensive", "class", "category"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
