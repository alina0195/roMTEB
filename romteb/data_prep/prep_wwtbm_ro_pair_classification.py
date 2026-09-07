"""RETIRED — WWTBM is now Retrieval + Reranking.

The previous PairClassification pipeline is kept as
``prep_wwtbm_ro_pair_classification_archived.py`` (do not delete).
Current command: ``romteb/data_prep/prep_wwtbm_ro.py --also_retrieval``
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "RETIRED: WWTBM PairClassification is no longer in RoMTEB.\n"
        "  archived: romteb/data_prep/prep_wwtbm_ro_pair_classification_archived.py\n"
        "  current:  romteb/data_prep/prep_wwtbm_ro.py --also_retrieval",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
