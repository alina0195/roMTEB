#!/bin/bash
#
# One-time installer: populate a host-side directory with the Python
# packages RoMTEB needs but that are NOT already present in
# transformers_flash_attn.sif. The runner then bind-mounts this dir into
# the container and prepends it to PYTHONPATH.
#
# This is the workaround for HPC nodes without a /etc/subuid entry that
# cannot build a new SIF from a localimage SIF source.
#
# Usage:
#   bash romteb/scripts/install_romteb_libs.sh
#
# To force a clean reinstall, delete $ROMTEB_LIBS first:
#   rm -rf $ROMTEB_LIBS && bash romteb/scripts/install_romteb_libs.sh

set -euo pipefail

ROMTEB_SIF=${ROMTEB_SIF:-/export/home/proiecte/aux/alina.gheorghe2505/transformers_flash_attn.sif}
ROMTEB_LIBS=${ROMTEB_LIBS:-/export/projects/nlp/users/alina_gheorghe/roMTEB/pylibs}

if [ ! -f "${ROMTEB_SIF}" ]; then
  echo "ERROR: SIF not found at ${ROMTEB_SIF}" >&2
  exit 2
fi

mkdir -p "${ROMTEB_LIBS}"
echo "Image:      ${ROMTEB_SIF}"
echo "Target dir: ${ROMTEB_LIBS}"
echo

# Heavy packages that are already in the SIF (torch, sentence-transformers,
# transformers, datasets, etc.). We use --no-deps to avoid downloading
# them again. The handful of small deps that mteb needs but the base SIF
# does not provide are listed in EXTRA_DEPS.
#
# If at runtime you hit ImportError for a package, add it to EXTRA_DEPS
# and rerun this script (no need to delete ROMTEB_LIBS first).
EXTRA_DEPS=(
  "pytrec_eval-terrier"
  "tabulate>=0.9"
  "jsonlines"
)

echo "==> Removing stale packages from previous installs (if any)"
apptainer exec --nv \
  -B "${ROMTEB_LIBS}:${ROMTEB_LIBS}" \
  "${ROMTEB_SIF}" \
  pip uninstall --target="${ROMTEB_LIBS}" --yes polars 2>/dev/null || true
rm -rf "${ROMTEB_LIBS}/polars" "${ROMTEB_LIBS}/polars"*.dist-info 2>/dev/null || true

echo "==> Installing mteb + extra deps into ${ROMTEB_LIBS}"
apptainer exec --nv \
  -B "${ROMTEB_LIBS}:${ROMTEB_LIBS}" \
  "${ROMTEB_SIF}" \
  pip install \
    --target="${ROMTEB_LIBS}" \
    --upgrade \
    --no-cache-dir \
    --no-deps \
    "mteb>=2.0,<3.0" "${EXTRA_DEPS[@]}"

echo
echo "==> Verifying imports inside the container"
apptainer exec --nv \
  -B "${ROMTEB_LIBS}:/opt/romteb_libs" \
  --env "PYTHONPATH=/opt/romteb_libs" \
  "${ROMTEB_SIF}" \
  python - <<'PY'
import sys
print("python:", sys.version.split()[0])

import mteb
print("mteb:", mteb.__version__)

import tabulate
print("tabulate:", tabulate.__version__)

# MTEB 2.x: abstract task classes from mteb.abstasks, TaskMetadata from mteb.
from mteb.abstasks import (
    AbsTaskSTS,
    AbsTaskClassification,
    AbsTaskPairClassification,
    AbsTaskRetrieval,
)
from mteb import TaskMetadata
print("mteb abstask imports OK")

# Verify mteb.get_tasks is callable (core API used by benchmark.py).
tasks = mteb.get_tasks(languages=["ron"])
print(f"Romanian MTEB tasks found: {len(tasks)}")
PY

echo
echo "Done. Run a smoke test next:"
echo "  ./apptainer-exec-romteb.sh romteb/scripts/inventory_mteb_ron.py"
