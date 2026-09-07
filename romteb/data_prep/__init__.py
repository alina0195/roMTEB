"""One-time data-preparation scripts.

Each `prep_*.py` is a standalone CLI: load raw data, normalize to MTEB
schema, push to `alina0195/romteb-<name>`. Run inside the romteb.sif
image:

    ./apptainer-exec-romteb.sh romteb/data_prep/prep_ro_sts.py
    ./apptainer-exec-romteb.sh romteb/data_prep/prep_roretrieval_clustering.py --input PATH --dry_run
"""
