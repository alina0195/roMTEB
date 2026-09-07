"""Prepare Fakenews-RO (~5K Romanian news articles, fake vs. real).

Sourcing notes:
  - cipriantruica/Romanian-Fake-News (~14K articles)
  - readerbench/ro-fake-news (mirror)
  - dumitrescustefan/fake-news-ro

If none of these resolve in 30 minutes, document the drop in
romteb/tasks/_inventory.md and remove the FakenewsRoClassification entry
from benchmark.py.
"""

from __future__ import annotations

import argparse

from romteb.data_prep._generic_classification_prep import run_generic_prep


CANDIDATE_SOURCES = [
    "cipriantruica/Romanian-Fake-News",
    "readerbench/ro-fake-news",
    "dumitrescustefan/fake-news-ro",
    "FakeRom",
]
TARGET_REPO = "alina0195/romteb-fakenews-ro"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    run_generic_prep(
        name="Fakenews-RO",
        candidate_sources=CANDIDATE_SOURCES,
        target_repo=TARGET_REPO,
        text_keys=("text", "content", "body", "article", "news"),
        label_keys=("label", "labels", "class", "is_fake", "fake"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
