#!/bin/bash
# After a full eval has written result JSONs + IR predictions:
# JuRo pool size, aggregate tables, bootstrap CIs, plots.
#
# Do not start this while run_all_models.sh is still writing the same
# results/ tree. Submit with --dependency=afterok:<eval_jobid>.
#
#   sbatch -A phd --gres=gpu:1 --partition=dgxa100 \
#     --dependency=afterok:259253 scripts/sbatch_post_eval.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${ROOT}"

OUTPUT_DIR=${1:-"${ROOT}/results"}
RUNNER="${ROOT}/apptainer-exec-romteb.sh"

echo "=== RoMTEB post-eval report ==="
echo "results=${OUTPUT_DIR}"
echo "host=$(hostname)  date=$(date)"
nvidia-smi -L 2>/dev/null || echo "(no nvidia-smi; continuing)"

echo
echo "=== 1. JuRo mean(|top_ranked|) (report only; do not rewrite eval_config) ==="
"${RUNNER}" "${ROOT}/scripts/juro_topk_stats.py" \
  --out "${OUTPUT_DIR}/juro_topk_stats.txt"

echo
echo "=== 2. Aggregate tables (by task / domain / model, Borda Overall) ==="
"${RUNNER}" "${ROOT}/scripts/aggregate_results.py" --results_dir "${OUTPUT_DIR}"

echo
echo "=== 3. Bootstrap IR CIs + E5 vs bge-m3 (incl. RoD-TAL) ==="
rm -f "${OUTPUT_DIR}/romteb_ir_pairwise.csv"
"${RUNNER}" "${ROOT}/scripts/bootstrap_ir.py" \
  --results_dir "${OUTPUT_DIR}" \
  --pair intfloat/multilingual-e5-large BAAI/bge-m3 \
  | tee "${OUTPUT_DIR}/romteb_ir_bootstrap.log"

if [ -f "${OUTPUT_DIR}/romteb_results_summary.csv" ]; then
  TOP_PAIR=$(python3 - "${OUTPUT_DIR}/romteb_results_summary.csv" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
col = "Overall (Borda rank)"
ranked = []
for row in rows:
    raw = (row.get(col) or "").strip()
    model = (row.get("Model") or "").strip()
    if not raw or raw == "-" or model.startswith("romteb/"):
        continue
    try:
        ranked.append((float(raw), model))
    except ValueError:
        continue
ranked.sort()
if len(ranked) >= 2:
    print(ranked[0][1], ranked[1][1])
PY
) || true
  DEFAULT_A="intfloat/multilingual-e5-large"
  DEFAULT_B="BAAI/bge-m3"
  if [ -n "${TOP_PAIR:-}" ]; then
    TOP_A=${TOP_PAIR%% *}
    TOP_B=${TOP_PAIR#* }
    if [ "${TOP_A}" != "${DEFAULT_A}" ] || [ "${TOP_B}" != "${DEFAULT_B}" ]; then
      echo
      echo "=== 3b. Pairwise on RoD-TAL: ${TOP_A} vs ${TOP_B} ==="
      "${RUNNER}" "${ROOT}/scripts/bootstrap_ir.py" \
        --results_dir "${OUTPUT_DIR}" \
        --pair "${TOP_A}" "${TOP_B}" \
        --task RoDTALLawsRetrieval \
        | tee -a "${OUTPUT_DIR}/romteb_ir_bootstrap.log"
    fi
  fi
fi

echo
echo "=== 4. Figures (per domain / task / model) ==="
"${RUNNER}" "${ROOT}/scripts/plot_results.py" --results_dir "${OUTPUT_DIR}"

echo
echo "Tables: ${OUTPUT_DIR}/romteb_results_*.md"
echo "Plots:  ${OUTPUT_DIR}/plots/"
echo "Done at $(date)"
