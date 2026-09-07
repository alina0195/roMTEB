"""Prepare FB-RO-Offense (~4000 Facebook comments, offense classification).

LICENSE WARNING: Facebook content is subject to platform ToS. This dataset
likely cannot be redistributed on HuggingFace. Confirm the original
distribution license; if unclear, DROP this task from v1 and document in
romteb/tasks/_inventory.md.

Sourcing notes:
  - readerbench/fb-ro-offense
  - readerbench/FB-RO-Offense
  - any non-FB-redistribution-restricted Romanian comment corpus
"""

from __future__ import annotations

import argparse

from romteb.data_prep._generic_classification_prep import run_generic_prep


CANDIDATE_SOURCES = [
    "readerbench/fb-ro-offense",
    "readerbench/FB-RO-Offense",
    "FB-RO-Offense",
]
TARGET_REPO = "alina0195/romteb-fb-ro-offense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i_have_verified_license",
        action="store_true",
        help="Require explicit acknowledgement before uploading FB-derived content.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not args.i_have_verified_license:
        raise SystemExit(
            "Refusing to upload Facebook-derived content without "
            "--i_have_verified_license. Verify the source dataset's license "
            "and Facebook ToS first; if redistribution is forbidden, drop "
            "this task and record the reason in _inventory.md."
        )

    run_generic_prep(
        name="FB-RO-Offense",
        candidate_sources=CANDIDATE_SOURCES,
        target_repo=TARGET_REPO,
        text_keys=("text", "comment", "content", "body"),
        label_keys=("label", "offensive", "class", "category"),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
