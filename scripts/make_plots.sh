#!/bin/bash
# Rebuild RoMTEB tables from result JSONs, then write figures.
#
#   ./apptainer-exec-romteb.sh scripts/make_plots.sh
#   ./apptainer-exec-romteb.sh scripts/make_plots.sh results
#
# Output: <results_dir>/romteb_results_*.csv/.md  and  <results_dir>/plots/*.png

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT}"
OUTPUT_DIR=${1:-"${ROOT}/results"}

python -u "${ROOT}/scripts/aggregate_results.py" --results_dir "${OUTPUT_DIR}"
python -u "${ROOT}/scripts/plot_results.py" --results_dir "${OUTPUT_DIR}" --out_dir "${OUTPUT_DIR}/plots"
echo "Plots: ${OUTPUT_DIR}/plots/"
